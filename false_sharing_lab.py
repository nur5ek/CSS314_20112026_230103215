import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import ctypes
import re
from pathlib import Path

import numpy as np
from numba import njit, uint64, types
from numba.extending import intrinsic
from llvmlite import ir

try:
    import resource
except ImportError:
    resource = None
try:
    import psutil
except ImportError:
    psutil = None

GIT_USERNAME = 'nur5ek'
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)


def detect_cache_line_size():
    path = '/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size'
    if os.path.exists(path):
        return int(Path(path).read_text().strip())
    if sys.platform == 'darwin':
        return int(subprocess.check_output(['sysctl', '-n', 'hw.cachelinesize'], text=True).strip())
    return 64


def cpu_name():
    if sys.platform.startswith('linux'):
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def init_system_context(git_user):
    if not git_user.strip() or git_user == 'REPLACE_WITH_YOUR_GIT_USERNAME':
        raise ValueError('Set a valid Git username')
    logical = os.cpu_count() or 1
    if logical < NUM_THREADS:
        raise RuntimeError('At least 4 logical processors are required')
    model = cpu_name()
    signature = f'{platform.machine()}::{model}::{logical}'
    seed = int(hashlib.sha256(f'{git_user.strip()}::{signature}'.encode()).hexdigest()[:16], 16)
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
    elif psutil is not None:
        try:
            allowed = sorted(psutil.Process().cpu_affinity())
        except (AttributeError, OSError):
            allowed = list(range(logical))
    else:
        allowed = list(range(logical))
    return {
        'git_username': git_user.strip(),
        'cpu_model': model,
        'cpu_signature': signature,
        'logical_cores': logical,
        'cache_line_bytes': detect_cache_line_size(),
        'instance_seed': seed,
        'cpu_affinity_candidates': allowed,
        'system': platform.platform(),
        'python_version': platform.python_version(),
    }


@intrinsic
def compiler_memory_barrier(typingctx):
    signature = types.void()

    def codegen(context, builder, signature, args):
        asm = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), '', '~{memory}', side_effect=True)
        builder.call(asm, [])
        return context.get_dummy_value()

    return signature, codegen


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    add = uint64(1 + (thread_seed & uint64(3)))
    for _ in range(iters):
        buffer[slot] = buffer[slot] + add
        compiler_memory_barrier()


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    add = uint64(1 + (thread_seed & uint64(3)))
    value = uint64(0)
    for _ in range(iters):
        value += add
    buffer[slot] = value


def aligned_uint64(size, alignment=64):
    backing = np.zeros(size + alignment // 8, dtype=np.uint64)
    padding = (-backing.ctypes.data) % alignment
    result = backing[padding // 8: padding // 8 + size]
    assert result.ctypes.data % alignment == 0
    return result, backing


def pin_this_thread(cpu_id):
    if sys.platform.startswith('linux') and hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(threading.get_native_id(), {cpu_id})
            return True
        except OSError:
            return False
    if sys.platform == 'win32' and cpu_id < 64:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentThread.restype = ctypes.c_void_p
        kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
        return bool(kernel.SetThreadAffinityMask(kernel.GetCurrentThread(), 1 << cpu_id))
    return False


def context_switches():
    if resource is not None:
        x = resource.getrusage(resource.RUSAGE_SELF)
        return {'voluntary': int(x.ru_nvcsw), 'involuntary': int(x.ru_nivcsw), 'source': 'resource.getrusage'}
    if psutil is not None:
        x = psutil.Process().num_ctx_switches()
        return {'voluntary': int(x.voluntary), 'involuntary': int(x.involuntary), 'source': 'psutil'}
    return {'voluntary': None, 'involuntary': None, 'source': 'unavailable'}


def run_stride_benchmark(ctx, stride, private=False, iters=ITERATIONS_PER_THREAD):
    buffer, backing = aligned_uint64(NUM_THREADS * stride + 1)
    slots = [t * stride for t in range(NUM_THREADS)]
    seeds = [uint64((ctx['instance_seed'] >> (8 * t)) & 0xff) for t in range(NUM_THREADS)]
    allowed = ctx['cpu_affinity_candidates']
    assigned = [allowed[t % len(allowed)] for t in range(NUM_THREADS)]
    pins = [False] * NUM_THREADS
    errors = []
    worker = worker_private_accumulate if private else worker_strided_accumulate
    def task(i):
        try:
            pins[i] = pin_this_thread(assigned[i])
            worker(buffer, slots[i], iters, seeds[i])
        except Exception as exc:
            errors.append(str(exc))
    before = context_switches()
    start = time.perf_counter_ns()
    threads = [threading.Thread(target=task, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed_ns = time.perf_counter_ns() - start
    after = context_switches()
    if errors:
        raise RuntimeError('Thread error: ' + '; '.join(errors))
    expected = [int(iters * (1 + (int(seed) & 3))) & ((1 << 64) - 1) for seed in seeds]
    actual = [int(buffer[k]) for k in slots]
    result = {
        'layout': 'private' if private else f'stride_{stride}',
        'stride_elements': stride,
        'stride_bytes': 8 * stride,
        'elapsed_ns': elapsed_ns,
        'ns_per_update': elapsed_ns / (iters * NUM_THREADS),
        'iterations_per_thread': iters,
        'thread_count': NUM_THREADS,
        'expected': expected,
        'actual': actual,
        'parity_passed': actual == expected,
        'affinity_cpu_ids': assigned,
        'affinity_set': pins,
        'context_switches': {
            'voluntary': None if before['voluntary'] is None or after['voluntary'] is None else after['voluntary'] - before['voluntary'],
            'involuntary': None if before['involuntary'] is None or after['involuntary'] is None else after['involuntary'] - before['involuntary'],
            'source': after['source'],
        },
    }
    return result


def audit_llvm_and_parity(results):
    sig = worker_strided_accumulate.signatures[0]
    llvm = worker_strided_accumulate.inspect_llvm(sig)
    pattern_store = r'\bstore\s+i64\b'
    stores = len(re.findall(pattern_store, llvm))
    barriers = len(re.findall(r'asm sideeffect', llvm))
    native = bool(stores > 0 and barriers > 0 and worker_strided_accumulate.nopython_signatures)
    return {
        'native_i64_store_count_in_llvm': stores,
        'compiler_memory_barrier_count_in_llvm': barriers,
        'python_object_fallback_detected': not bool(worker_strided_accumulate.nopython_signatures),
        'native_llvm_verified': native,
        'all_counter_parity_passed': all(x['parity_passed'] for x in results),
        'llvm_excerpt': '\n'.join(line.strip() for line in llvm.splitlines() if re.search(r'\bstore\s+i64\b|asm sideeffect', line))[:1400],
    }


def make_analysis(manifest):
    runs = {x['layout']: x for x in manifest['measurements']}
    a, b = runs['stride_1'], runs['stride_8']
    r = runs['stride_16']
    ratio = a['elapsed_ns'] / b['elapsed_ns']
    ratio16 = b['elapsed_ns'] / r['elapsed_ns']
    def fmt(x):
        return f"{x['elapsed_ns']:,} ns ({x['ns_per_update']:.4f} ns/update)"
    text = f'''# Cache Line and False Sharing — Analysis

GitHub: {manifest['git_username']}  
CPU: {manifest['cpu_model']}  
Cache line: {manifest['cache_line_bytes']} bytes  
Iterations per thread: {manifest['iterations_per_thread']:,}

## Q1 — MESI

With stride 1, buffer[0] and buffer[1] are in the same cache line. When Core 0 writes, it needs the line in Modified state and other copies become Invalid. When Core 1 wants to write buffer[1], it also needs ownership, so it sends an RFO request. Core 0 loses ownership and Core 1 gets Modified state. This repeats and makes both cores wait even if they update different numbers.

## Q2 — Measurements

- Stride 1: {fmt(runs['stride_1'])}
- Stride 4: {fmt(runs['stride_4'])}
- Stride 8: {fmt(runs['stride_8'])}
- Stride 16: {fmt(runs['stride_16'])}
- Private: {fmt(runs['private'])}

Stride 1 / stride 8 time ratio = {ratio:.3f}x. I calculate ns per update as total elapsed ns divided by four threads times {manifest['iterations_per_thread']:,} updates. Results can change a little because of Windows scheduling and CPU frequency.

## Q3 — Padding and prefetching

Stride 8 uses 64 bytes between thread counters, while stride 16 uses 128 bytes. My stride 8 / stride 16 time ratio is {ratio16:.3f}x. Both should avoid normal false sharing if counters are aligned. Some processors may also prefetch neighbor lines and this can affect cache traffic, but the result here does not prove that prefetching caused any slowdown.

## Q4 — LLVM

The Numba function compiled to native code. The LLVM inspection finds {manifest['llvm_audit']['native_i64_store_count_in_llvm']} i64 store instructions and {manifest['llvm_audit']['compiler_memory_barrier_count_in_llvm']} compiler memory barriers in the generated IR. Without a compiler barrier, the optimizer can keep the value in a register and only write at the end, even if the Python source has an update inside a loop. The barrier is used here so the memory accesses are not moved out of the loop.

## Q5 — GIL vs Numba

Normal Python threads still share the GIL, so the CPU work is mostly run one thread at a time. This makes it hard to see real cache line ping-pong. Numba with nogil=True runs the compiled loop on different cores together. Then threads with stride 1 can fight for the same cache line and become slower.
'''
    Path('ANALYSIS.md').write_text(text, encoding='utf-8')


def main():
    iters = int(os.environ.get('CACHE_LAB_ITERS', ITERATIONS_PER_THREAD))
    if iters <= 0:
        raise ValueError('Iterations must be positive')
    ctx = init_system_context(GIT_USERNAME)
    warm, _ = aligned_uint64(80)
    worker_strided_accumulate(warm, 0, 2, uint64(0))
    worker_private_accumulate(warm, 8, 2, uint64(0))
    print(f"CPU: {ctx['cpu_model']}", flush=True)
    print(f"GitHub: {GIT_USERNAME} | logical cores: {ctx['logical_cores']} | cache line: {ctx['cache_line_bytes']} bytes", flush=True)
    print(f'Iterations per thread: {iters:,}', flush=True)
    runs = []
    for stride in STRIDES:
        result = run_stride_benchmark(ctx, stride, iters=iters)
        runs.append(result)
        print(f"{result['layout']}: {result['elapsed_ns']:,} ns | {result['ns_per_update']:.4f} ns/update | parity={result['parity_passed']}", flush=True)
    result = run_stride_benchmark(ctx, 8, private=True, iters=iters)
    runs.append(result)
    print(f"private: {result['elapsed_ns']:,} ns | {result['ns_per_update']:.4f} ns/update | parity={result['parity_passed']}", flush=True)
    audit = audit_llvm_and_parity(runs)
    verified = bool(audit['native_llvm_verified'] and audit['all_counter_parity_passed'] and iters == ITERATIONS_PER_THREAD and ctx['cache_line_bytes'] == 64)
    manifest = dict(ctx)
    manifest.update({
        'num_threads': NUM_THREADS,
        'iterations_per_thread': iters,
        'measurements': runs,
        'llvm_audit': audit,
        'verification_passed': verified,
        'execution_timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    Path('cache_lab_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    make_analysis(manifest)
    print('Saved cache_lab_manifest.json and ANALYSIS.md', flush=True)
    print('verification_passed:', verified, flush=True)
    if not verified:
        print('Check iteration count, cache line size, counter parity and LLVM audit.', flush=True)


if __name__ == '__main__':
    main()
