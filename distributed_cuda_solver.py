import sys
import json
import math
import re
import hashlib
import subprocess
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
UP, DOWN, LEFT, RIGHT = range(4)
OPPOSITE = (DOWN, UP, RIGHT, LEFT)


def get_gpu_uuid():
    return subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        encoding='utf-8'
    ).strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(f'{github_user.strip()}::{gpu_uuid}'.encode('utf-8')).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f'Expected exactly 4 MPI ranks, got {size}')
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    seed = derive_student_seed(github_user, gpu_uuid)
    cart = comm.Create_cart([2, 2], periods=[False, False], reorder=False)
    row_rank, col_rank = cart.Get_coords(cart.Get_rank())
    nbr_up, nbr_down = cart.Shift(0, 1)
    nbr_left, nbr_right = cart.Shift(1, 1)
    return {
        'cart': cart, 'rank': cart.Get_rank(), 'size': size,
        'coords': (row_rank, col_rank),
        'neighbors': (nbr_up, nbr_down, nbr_left, nbr_right),
        'gpu_uuid': gpu_uuid, 'seed': seed, 'github_username': github_user,
        'ny': GLOBAL_NY // 2, 'nx': GLOBAL_NX // 2,
    }


def initialize_subdomains(ctx):
    ny, nx = ctx['ny'], ctx['nx']
    row_rank, col_rank = ctx['coords']
    gy = row_rank * ny + np.arange(ny + 2, dtype=np.float64)
    gx = col_rank * nx + np.arange(nx + 2, dtype=np.float64)
    y = gy[:, None] / (GLOBAL_NY + 1)
    x = gx[None, :] / (GLOBAL_NX + 1)
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    seed = ctx['seed']
    cx1 = 0.25 + ((seed >> 0) & 255) / 255.0 * 0.15
    cy1 = 0.25 + ((seed >> 8) & 255) / 255.0 * 0.15
    cx2 = 0.60 + ((seed >> 16) & 255) / 255.0 * 0.15
    cy2 = 0.60 + ((seed >> 24) & 255) / 255.0 * 0.15
    amp1 = 0.8 + ((seed >> 32) & 255) / 255.0 * 0.5
    amp2 = 0.5 + ((seed >> 40) & 255) / 255.0 * 0.4
    u = (amp1 * np.exp(-((x-cx1)**2 + (y-cy1)**2) / 0.018)
         + amp2 * np.exp(-((x-cx2)**2 + (y-cy2)**2) / 0.012))
    source = 0.0001 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)
    u = np.ascontiguousarray(u, dtype=np.float64)
    kappa = np.ascontiguousarray(kappa, dtype=np.float64)
    source = np.ascontiguousarray(np.broadcast_to(source, (ny+2, nx+2)))
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u, kappa, source


@cuda.jit
def pack_halos_kernel(u, out, direction, ny, nx):
    t = cuda.grid(1)
    if direction == UP and t < nx:
        out[t] = u[1, t + 1]
    elif direction == DOWN and t < nx:
        out[t] = u[ny, t + 1]
    elif direction == LEFT and t < ny:
        out[t] = u[t + 1, 1]
    elif direction == RIGHT and t < ny:
        out[t] = u[t + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, incoming, direction, ny, nx):
    t = cuda.grid(1)
    if direction == UP and t < nx:
        u[0, t + 1] = incoming[t]
    elif direction == DOWN and t < nx:
        u[ny + 1, t + 1] = incoming[t]
    elif direction == LEFT and t < ny:
        u[t + 1, 0] = incoming[t]
    elif direction == RIGHT and t < ny:
        u[t + 1, nx + 1] = incoming[t]


@cuda.jit
def tiled_stencil_kernel(u, v, kappa, source, dt, dx2, dy2,
                         r_start, r_end, c_start, c_end):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    r = r_start + cuda.blockIdx.y * TILE_DIM + ty
    c = c_start + cuda.blockIdx.x * TILE_DIM + tx
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    max_r = u.shape[0] - 1
    max_c = u.shape[1] - 1
    rr = min(max(r, 0), max_r)
    cc = min(max(c, 0), max_c)
    smem[ty + 1, tx + 1] = u[rr, cc]
    if tx == 0:
        smem[ty + 1, 0] = u[rr, max(cc - 1, 0)]
    if tx == TILE_DIM - 1:
        smem[ty + 1, TILE_DIM + 1] = u[rr, min(cc + 1, max_c)]
    if ty == 0:
        smem[0, tx + 1] = u[max(rr - 1, 0), cc]
    if ty == TILE_DIM - 1:
        smem[TILE_DIM + 1, tx + 1] = u[min(rr + 1, max_r), cc]
    cuda.syncthreads()
    if r_start <= r < r_end and c_start <= c < c_end:
        center = smem[ty + 1, tx + 1]
        north = smem[ty, tx + 1]
        south = smem[ty + 2, tx + 1]
        west = smem[ty + 1, tx]
        east = smem[ty + 1, tx + 2]
        kn = 0.5 * (kappa[r, c] + kappa[r - 1, c])
        ks = 0.5 * (kappa[r, c] + kappa[r + 1, c])
        kw = 0.5 * (kappa[r, c] + kappa[r, c - 1])
        ke = 0.5 * (kappa[r, c] + kappa[r, c + 1])
        diffusion = (ks * (south-center) - kn * (center-north)) / dy2
        diffusion += (ke * (east-center) - kw * (center-west)) / dx2
        v[r, c] = center + dt * (diffusion - LAMBDA_DAMP * center**3 + source[r, c])


def launch_box(u, v, k, s, dt, dx2, dy2, bounds, stream):
    r0, r1, c0, c1 = bounds
    if r1 <= r0 or c1 <= c0:
        return
    blocks = (math.ceil((c1-c0) / TILE_DIM), math.ceil((r1-r0) / TILE_DIM))
    tiled_stencil_kernel[blocks, (TILE_DIM, TILE_DIM), stream](
        u, v, k, s, dt, dx2, dy2, r0, r1, c0, c1)


def allocate_halos(ctx):
    halos = {}
    ny, nx = ctx['ny'], ctx['nx']
    for direction in (UP, DOWN, LEFT, RIGHT):
        length = nx if direction in (UP, DOWN) else ny
        halos[direction] = {
            'd_send': cuda.device_array(length, dtype=np.float64),
            'd_recv': cuda.device_array(length, dtype=np.float64),
            'h_send': cuda.pinned_array(length, dtype=np.float64),
            'h_recv': cuda.pinned_array(length, dtype=np.float64),
            'length': length,
        }
    return halos


def exchange_halos_async(ctx, u, halos, stream_boundary):
    neighbors = ctx['neighbors']
    for direction in range(4):
        if neighbors[direction] == MPI.PROC_NULL:
            continue
        h = halos[direction]
        blocks = math.ceil(h['length'] / 128)
        pack_halos_kernel[blocks, 128, stream_boundary](
            u, h['d_send'], direction, ctx['ny'], ctx['nx'])
        h['d_send'].copy_to_host(h['h_send'], stream=stream_boundary)
    stream_boundary.synchronize()
    requests = []
    cart = ctx['cart']
    for direction in range(4):
        neighbor = neighbors[direction]
        if neighbor != MPI.PROC_NULL:
            requests.append(cart.Irecv(halos[direction]['h_recv'],
                                       source=neighbor, tag=OPPOSITE[direction]))
    for direction in range(4):
        neighbor = neighbors[direction]
        if neighbor != MPI.PROC_NULL:
            requests.append(cart.Isend(halos[direction]['h_send'],
                                       dest=neighbor, tag=direction))
    return requests


def complete_halos(ctx, u, halos, requests, stream_boundary):
    MPI.Request.Waitall(requests)
    for direction in range(4):
        if ctx['neighbors'][direction] == MPI.PROC_NULL:
            continue
        h = halos[direction]
        h['d_recv'].copy_to_device(h['h_recv'], stream=stream_boundary)
        unpack_halos_kernel[math.ceil(h['length']/128), 128, stream_boundary](
            u, h['d_recv'], direction, ctx['ny'], ctx['nx'])


def step_overlapped(ctx, u, v, k, s, halos, dt, dx2, dy2,
                    stream_boundary, stream_interior, overlapped):
    ny, nx = ctx['ny'], ctx['nx']
    requests = exchange_halos_async(ctx, u, halos, stream_boundary)
    compute_stream = stream_interior if overlapped else stream_boundary
    if overlapped:
        launch_box(u, v, k, s, dt, dx2, dy2, (2, ny, 2, nx), compute_stream)
    complete_halos(ctx, u, halos, requests, stream_boundary)
    if not overlapped:
        launch_box(u, v, k, s, dt, dx2, dy2, (2, ny, 2, nx), compute_stream)
    launch_box(u, v, k, s, dt, dx2, dy2, (1, 2, 1, nx+1), stream_boundary)
    launch_box(u, v, k, s, dt, dx2, dy2, (ny, ny+1, 1, nx+1), stream_boundary)
    launch_box(u, v, k, s, dt, dx2, dy2, (2, ny, 1, 2), stream_boundary)
    launch_box(u, v, k, s, dt, dx2, dy2, (2, ny, nx, nx+1), stream_boundary)
    stream_boundary.synchronize()
    stream_interior.synchronize()
    return v, u


def global_diagnostics(ctx, u, previous, dx, dy):
    current = u.copy_to_host()[1:-1, 1:-1]
    old = previous.copy_to_host()[1:-1, 1:-1]
    energy_local = np.array([np.sum(current) * dx * dy], dtype=np.float64)
    energy_global = np.zeros(1, dtype=np.float64)
    ctx['cart'].Allreduce(energy_local, energy_global, op=MPI.SUM)
    residual_local = np.array([np.sum((current-old)**2)], dtype=np.float64)
    residual_global = np.zeros(1, dtype=np.float64)
    ctx['cart'].Allreduce(residual_local, residual_global, op=MPI.SUM)
    return float(energy_global[0]), float(np.sqrt(residual_global[0]))


def run_mode(ctx, initial, k, source, dt, dx, dy, overlapped):
    u = cuda.to_device(initial)
    v = cuda.to_device(np.zeros_like(initial))
    halos = allocate_halos(ctx)
    boundary_stream = cuda.stream()
    interior_stream = cuda.stream()
    cart = ctx['cart']
    trace = []
    init_energy = np.array([np.sum(initial[1:-1, 1:-1])*dx*dy], dtype=np.float64)
    global_energy = np.zeros(1, dtype=np.float64)
    cart.Allreduce(init_energy, global_energy, op=MPI.SUM)
    trace.append({'step': 0, 'energy': float(global_energy[0]), 'l2_residual': 0.0})
    cart.Barrier()
    begin = cuda.event(timing=True)
    end = cuda.event(timing=True)
    begin.record(boundary_stream)
    start = MPI.Wtime()
    for step in range(1, STEPS+1):
        u, v = step_overlapped(ctx, u, v, k, source, halos, dt, dx*dx, dy*dy,
                               boundary_stream, interior_stream, overlapped)
        if step % 100 == 0:
            energy, residual = global_diagnostics(ctx, u, v, dx, dy)
            trace.append({'step': step, 'energy': energy, 'l2_residual': residual})
    boundary_stream.synchronize()
    interior_stream.synchronize()
    end.record(boundary_stream)
    end.synchronize()
    wall_local_ms = (MPI.Wtime()-start)*1000.0
    wall_ms = cart.allreduce(wall_local_ms, op=MPI.MAX)
    event_local_ms = cuda.event_elapsed_time(begin, end)
    event_ms = cart.allreduce(event_local_ms, op=MPI.MAX)
    return u.copy_to_host(), trace, wall_ms, event_ms


def ptx_audit():
    signatures = tiled_stencil_kernel.signatures
    if not signatures:
        raise RuntimeError('Kernel did not compile')
    ptx = tiled_stencil_kernel.inspect_asm(signatures[0])
    counts = {
        'st_shared_f64': len(re.findall(r'\bst\.shared\.f64\b', ptx)),
        'ld_shared_f64': len(re.findall(r'\bld\.shared\.f64\b', ptx)),
        'bar_sync': len(re.findall(r'\bbar\.sync\b', ptx)),
        'st_shared_total': len(re.findall(r'\bst\.shared\.', ptx)),
        'ld_shared_total': len(re.findall(r'\bld\.shared\.', ptx)),
    }
    counts['passed'] = bool(counts['st_shared_total'] and
                            counts['ld_shared_total'] and counts['bar_sync'])
    return counts


def run_solver(github_user):
    ctx = init_distributed_context(github_user)
    rank = ctx['rank']
    initial, local_kappa, local_source = initialize_subdomains(ctx)
    kmax = ctx['cart'].allreduce(float(np.max(local_kappa)), op=MPI.MAX)
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    dt_max = dx*dx / (4.0*kmax)
    dt = 0.35*dt_max
    k = cuda.to_device(local_kappa)
    source = cuda.to_device(local_source)
    if rank == 0:
        print('MPI ranks: 4; Cartesian grid: 2x2', flush=True)
        print(f'GitHub username: {github_user}', flush=True)
        print(f'GPU UUID: {ctx["gpu_uuid"]}', flush=True)
        print(f'INSTANCE_SEED: {ctx["seed"]}', flush=True)
        print(f'kappa_max={kmax:.12g} dx={dx:.12g} dt_max={dt_max:.12g} dt={dt:.12g}', flush=True)
    for label, overlap in (('serialized', False), ('overlapped', True)):
        if rank == 0:
            print(f'Running {label} mode: {STEPS} steps', flush=True)
        output, trace, wall_ms, cuda_ms = run_mode(ctx, initial, k, source, dt, dx, dy, overlap)
        if rank == 0:
            print(f'{label}_ms={wall_ms:.3f} cuda_event_ms={cuda_ms:.3f}', flush=True)
            for entry in trace:
                print(f'{label} step={entry["step"]} energy={entry["energy"]:.12g} '
                      f'l2_residual={entry["l2_residual"]:.12g}', flush=True)
        if not overlap:
            serialized = output
            serialized_trace = trace
            serialized_ms = wall_ms
            serialized_event_ms = cuda_ms
        else:
            overlapped_output = output
            overlapped_trace = trace
            overlapped_ms = wall_ms
            overlapped_event_ms = cuda_ms
    local_diff = float(np.max(np.abs(serialized[1:-1,1:-1] - overlapped_output[1:-1,1:-1])))
    parity = ctx['cart'].allreduce(local_diff, op=MPI.MAX)
    audit = ptx_audit()
    if rank == 0:
        print('PTX audit:', json.dumps(audit), flush=True)
        print(f'parity_max_diff={parity:.16g}', flush=True)
    manifest = {
        'github_username': github_user,
        'gpu_uuid': ctx['gpu_uuid'],
        'instance_seed': ctx['seed'],
        'mpi_ranks': ctx['size'],
        'cartesian_dims': [2, 2],
        'global_shape': [GLOBAL_NY, GLOBAL_NX],
        'local_padded_shape': [ctx['ny']+2, ctx['nx']+2],
        'steps': STEPS,
        'kappa_max': kmax,
        'dx': dx, 'dt': dt, 'dt_max': dt_max,
        'serialized_ms': serialized_ms,
        'overlapped_ms': overlapped_ms,
        'serialized_cuda_event_ms': serialized_event_ms,
        'overlapped_cuda_event_ms': overlapped_event_ms,
        'serialized_energy_trace': serialized_trace,
        'overlapped_energy_trace': overlapped_trace,
        'parity_max_diff': parity,
        'ptx_counts': audit,
        'ptx_audit_passed': audit['passed'],
        'verification_passed': audit['passed'] and parity < 1e-12,
    }
    if rank == 0:
        Path('submission_manifest.json').write_text(json.dumps(manifest, indent=2))
        print('Saved submission_manifest.json', flush=True)
        if not manifest['verification_passed']:
            raise RuntimeError('PTX or numerical parity verification failed')
    ctx['cart'].Barrier()


if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        raise SystemExit('Usage: python distributed_cuda_solver.py YOUR_GITHUB_USERNAME')
    run_solver(sys.argv[1].strip())
