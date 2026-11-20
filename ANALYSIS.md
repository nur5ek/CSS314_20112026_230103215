# Cache Line and False Sharing — Analysis

GitHub: nur5ek  
CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel  
Cache line: 64 bytes  
Iterations per thread: 60,000,000

## Q1 — MESI

With stride 1, buffer[0] and buffer[1] are in the same cache line. When Core 0 writes, it needs the line in Modified state and other copies become Invalid. When Core 1 wants to write buffer[1], it also needs ownership, so it sends an RFO request. Core 0 loses ownership and Core 1 gets Modified state. This repeats and makes both cores wait even if they update different numbers.

## Q2 — Measurements

- Stride 1: 275,455,300 ns (1.1477 ns/update)
- Stride 4: 208,425,600 ns (0.8684 ns/update)
- Stride 8: 99,348,400 ns (0.4140 ns/update)
- Stride 16: 103,476,400 ns (0.4312 ns/update)
- Private: 1,498,100 ns (0.0062 ns/update)

Stride 1 / stride 8 time ratio = 2.773x. I calculate ns per update as total elapsed ns divided by four threads times 60,000,000 updates. Results can change a little because of Windows scheduling and CPU frequency.

## Q3 — Padding and prefetching

Stride 8 uses 64 bytes between thread counters, while stride 16 uses 128 bytes. My stride 8 / stride 16 time ratio is 0.960x. Both should avoid normal false sharing if counters are aligned. Some processors may also prefetch neighbor lines and this can affect cache traffic, but the result here does not prove that prefetching caused any slowdown.

## Q4 — LLVM

The Numba function compiled to native code. The LLVM inspection finds 1 i64 store instructions and 1 compiler memory barriers in the generated IR. Without a compiler barrier, the optimizer can keep the value in a register and only write at the end, even if the Python source has an update inside a loop. The barrier is used here so the memory accesses are not moved out of the loop.

## Q5 — GIL vs Numba

Normal Python threads still share the GIL, so the CPU work is mostly run one thread at a time. This makes it hard to see real cache line ping-pong. Numba with nogil=True runs the compiled loop on different cores together. Then threads with stride 1 can fight for the same cache line and become slower.
