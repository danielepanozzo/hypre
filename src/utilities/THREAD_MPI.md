# hypre without mpirun: MPI ranks as threads

Runs hypre's ordinary domain-decomposed MPI algorithms inside a **single
process**, so hypre can be called from a library without launching under
`mpirun`. Ranks become threads; `hypre_MPI_*` is implemented over pthreads.

This is not OpenMP. hypre keeps its normal decomposition — each rank still owns
a subdomain and still runs the rank-local coarsening that makes the MPI build
fast. That is the whole point: hypre's OpenMP backend is capped around 2.5×
because BoomerAMG's setup phase is largely sequential, while the MPI backend
gets ~22× on the same machine. This gives you the MPI algorithm without the
MPI launcher.

## What's in the patch

| file | change |
|---|---|
| `utilities/mpithreads.c` | **new** — the 54-function `hypre_MPI_*` surface over pthreads |
| `utilities/mpithreads.h` | **new** — `hypre_tmpi_run()` entry point |
| `utilities/mpistubs.h`, `_hypre_utilities.h` | `Aint` → `intptr_t`; `status.COUNT` field; real `ANY_SOURCE`/`ANY_TAG` sentinels |
| `utilities/{general,error,state,timing,random,printf}.c` + headers | 6 globals → `_Thread_local` (re-entrancy) |
| `utilities/CMakeLists.txt` | builds `mpithreads.c` instead of `mpistubs.c` |

Total: **+25 / −21 lines** in existing files, plus the two new ones.

## Building

```sh
cmake -S src -B build-tmpi -DCMAKE_BUILD_TYPE=Release \
      -DHYPRE_ENABLE_MPI=OFF -DHYPRE_ENABLE_OPENMP=OFF
cmake --build build-tmpi -j
```

`HYPRE_ENABLE_MPI=OFF` selects hypre's `HYPRE_SEQUENTIAL` path, which is what
remaps every `MPI_*` name to `hypre_MPI_*`. The patch swaps the *implementation*
behind those names; hypre's numerical code is untouched.

## Using it

Your rank function runs on every thread. Everything inside it is ordinary hypre
code using `hypre_MPI_COMM_WORLD`.

```c
#include "mpithreads.h"

static int solve_rank(int argc, char **argv, void *user)
{
    MPI_Comm  comm = hypre_MPI_COMM_WORLD;
    HYPRE_Int myid, nprocs;

    HYPRE_Initialize();
    hypre_MPI_Comm_rank(comm, &myid);
    hypre_MPI_Comm_size(comm, &nprocs);

    /* build your slice of the matrix, solve -- exactly as under mpirun */

    HYPRE_Finalize();
    return 0;
}

int main(void)
{
    return hypre_tmpi_run(64, solve_rank, 0, NULL, &my_context);
}
```

`hypre_tmpi_run(nranks, fn, argc, argv, user)` spawns `nranks` threads, gives
each one a rank of `COMM_WORLD`, and joins them. `user` is passed through
untouched. Also available: `hypre_tmpi_rank()`, `hypre_tmpi_nranks()`.

### Choosing the number of threads

Pass a positive count and it is used as given:

```c
hypre_tmpi_run(64, solve_rank, argc, argv, ctx);
```

Pass `0` (or any value `<= 0`) to take the default, which is
**`HYPRE_TMPI_NUM_THREADS`** if set to a positive integer, otherwise the number
of cores online:

```c
hypre_tmpi_run(0, solve_rank, argc, argv, ctx);   /* honours the environment */
```

```sh
HYPRE_TMPI_NUM_THREADS=32 ./your_program
```

An explicit positive argument always wins over the environment, the same way
`omp_set_num_threads()` overrides `OMP_NUM_THREADS`. So a caller that wants the
variable to take effect should ask for the default rather than hard-coding a
count. `hypre_tmpi_num_threads()` returns the same value if you want to inspect
or reuse it. A malformed setting is reported on stderr and ignored rather than
silently treated as 1.

| backend | how the count is set |
|---|---|
| MPI | `mpirun -np N` |
| OpenMP | `OMP_NUM_THREADS` |
| this backend | `hypre_tmpi_run(N, ...)`, or `HYPRE_TMPI_NUM_THREADS` with `N <= 0` |

Pin the threads yourself if it matters; this library does not set affinity.

### Running hypre's own drivers

hypre's stock test drivers (`ij`, `struct`, `sstruct`) can run unmodified under
this backend by renaming their `main()` at compile time and supplying a
launcher. This is how the suite comparisons above were produced:

```c
/* tmpi_launch.c */
#include <stdlib.h>
#include "mpithreads.h"

int hypre_driver_main(int argc, char *argv[]);          /* the driver's main() */
static int adapter(int argc, char **argv, void *user)
{ (void) user; return hypre_driver_main(argc, argv); }

int main(int argc, char **argv)
{
   const char *e = getenv("TMPI_NP");
   int np = e ? atoi(e) : 1;
   return hypre_tmpi_run(np < 1 ? 1 : np, adapter, argc, argv, NULL);
}
```

```sh
cc -c ../test/ij.c -Dmain=hypre_driver_main $INCLUDES -o ij_body.o
cc -o ij_tmpi tmpi_launch.c ij_body.o $INCLUDES -lHYPRE -lm -lpthread
TMPI_NP=8 ./ij_tmpi -n 60 60 60 -P 2 2 2 -solver 1
```

`TMPI_NP` is a convention of that launcher, not of the library.

## Debug switches

| build flag | effect |
|---|---|
| `-DTMPI_DISABLE_FASTSEND` | never copy straight into a posted receive; always buffer |
| `-DTMPI_LOCK_BARRIER` | use a mutex/condvar barrier instead of the atomic spin barrier |

Both are for bisecting a suspected bug down to the message layer. A fatal message
truncation also dumps the posted-receive and unexpected-message queues of the
receiving rank.

## Debugging a hang

```sh
TMPI_WATCHDOG=20 ./your_program
```

If no rank makes progress for 20 s, every rank reports what it is blocked on:

```
=== TMPI WATCHDOG: no progress for 20s, 64 ranks ===
  rank   0: barrier  comm=0 a=0 b=0
  ...
  rank  37: barrier  comm=2 a=0 b=0     <- diverged onto another communicator
```

This found the sub-communicator bug during development in one run. Worth
keeping: `perf` and `ptrace` are often restricted on shared clusters.

## Correctness

**hypre's own test suites, 978 cases**, run under real Open MPI and under
threads-as-ranks, comparing numerical output case by case. The drivers are
unmodified — their `main()` is renamed at compile time.

| suite | cases | identical |
|---|---|---|
| `TEST_struct` | 196 | **196** |
| `TEST_sstruct` | 376 | **376** |
| `TEST_ij` | 406 | 404 (2 = the flexamg issue below) |

All 971 runnable cases are byte-identical to Open MPI.

Apart from that one issue, every case that runs produces **byte-identical**
output, including every digit of every residual.

Getting there required matching the reduction association. Floating-point
addition is not associative, so the order contributions are combined decides
the last bits; summing linearly gave residuals that differed in the last digits
and drifted over long solves (~0.3% of the final residual after 202 iterations,
though never a different iteration count). `reduce_all` now uses recursive
doubling, which is the standard algorithm, is more accurate than a linear sum
(error grows as log n rather than n), and reproduces Open MPI exactly --
verified over 400 random trials at every rank count from 2 to 16, including
non-powers of two.

Two caveats on that bit-identity: it holds for Open MPI's small-message
allreduce path, which is what hypre's dot products use -- large reductions
switch to ring or segmented algorithms with a different association. And it is
specific to Open MPI's algorithm selection; MPICH may associate differently.
Neither affects correctness, only reproducibility against a particular MPI.

Two real bugs were found by that suite and fixed: `Scatterv`/`Scatter`/`Gather`
dereferencing root-only arguments on every rank (segfault under `-seq_th`), and
missing persistent communication (ParaSails aborted). Both now match MPI exactly.

Also validated on a fixed 80³ Laplacian at 1/2/4/8/16/32/64 ranks: identical
iteration counts and residuals to every printed digit.

## Performance

Apple M3 Max (12 performance cores), 160³ Laplacian, AMG-PCG, best of 3:

| units | threads | real MPI | ratio | where the difference is |
|---|---|---|---|---|
| 1 | 7.29 s | 7.46 s | 0.98× | — |
| 2 | 4.92 | 4.97 | 0.99× | — |
| 4 | 3.37 | 2.82 | 1.20× | setup 1.30×, solve 1.01× |
| 8 | 2.48 | 1.63 | 1.52× | setup 1.86×, solve 1.02× |
| 12 | 2.90 | 1.70 | 1.71× | setup 2.13×, solve 1.05× |

**The messaging layer is not the bottleneck.** The solve phase — which is where
essentially all the communication happens — matches real MPI to within 1–5% at
every rank count. Confirmed independently with diagonal-scaled PCG (`-solver 2`,
communication-heavy, negligible setup): 3.69 s vs 3.62 s at 12 ranks, a 2% gap.

The entire difference is BoomerAMG's **setup** phase, which builds the multigrid
hierarchy with heavy `malloc`/`free` of large arrays. As separate processes each
rank has its own heap and page table; as threads they share one of each, so large
allocations serialise in `mmap`/`munmap` and each `munmap` triggers a TLB
shootdown across every core in the process. No change to the MPI layer can fix
this. If it matters for your workload, link a thread-caching allocator
(jemalloc, tcmalloc, mimalloc) — that targets the actual cause.

## Known issue

`-flexamg_cycle_struct` (BoomerAMG's flexible cycle) fails on 2 or more ranks
with a message-truncation abort; it is correct on 1 rank, and correct under real
MPI at every rank count. Minimal reproducer:

```sh
# built as described under "Running hypre's own drivers" above
TMPI_NP=2 ./ij_tmpi -n 40 40 40 -P 2 1 1 -solver 1 \
    -flexamg_cycle_struct 0,0,-1,-1,-1,-1,1,1,1,1,-2
```

The two ranks end up disagreeing about the message sequence (a receive posted for
2 doubles is offered a 1600-double message on the same communicator and tag),
which points at a collective returning results that differ across ranks and then
diverging the control flow. Ruled out so far: the single-copy send fast path and
the spin barrier — building with `-DTMPI_DISABLE_FASTSEND` or
`-DTMPI_LOCK_BARRIER` reproduces it identically. This affects 2 of hypre's 976
`TEST_ij`/`TEST_struct`/`TEST_sstruct` cases.

## GPU (CUDA) builds

Builds and runs with CUDA. Both configurations compile clean on CUDA 13.2:

```sh
cmake -S src -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DHYPRE_ENABLE_MPI=OFF -DHYPRE_ENABLE_THREAD_MPI=ON \
      -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_UMPIRE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=<arch>
```

`-DHYPRE_ENABLE_UMPIRE=OFF` is hypre's own requirement for GPU builds without
Umpire, not something this backend adds.

**One rank per GPU works and matches real MPI exactly.** On an RTX 3080 Ti with
a 240³ problem, threads-as-ranks and MPI both converge in 20 iterations to the
same residual, and take essentially the same time (5.16 s vs 5.25 s).

**More than one rank per GPU does not.** The solve itself is correct — the
iteration count and residual are right, and match MPI — but the process then
segfaults during `HYPRE_Finalize`:

```
HYPRE_Finalize -> hypre_DeviceDataDestroy -> cudaStreamDestroy -> libcuda: SIGSEGV
```

Under real MPI each rank is a separate process with its own CUDA context, so
several ranks can share a device safely. Here every rank is a thread of one
process sharing a single primary context, and hypre's device teardown assumes
it owns that context. Serialising the vendor-handle destruction with a mutex
does not help: the first thread to tear down already fails, so the context is
unusable before teardown starts. Making hypre's GPU layer safe for several
ranks per process is a larger change than this backend.

In practice this matters little — the normal GPU configuration is one rank per
GPU, which works. Use MPI if you need several ranks sharing one device.

## Limits

- **Exercised on the IJ/ParCSR path** (BoomerAMG, PCG, ParaSails, FSAI, ILU, MGR
  — everything `TEST_ij` covers). The struct/sstruct interfaces use the
  `Type_vector`/`Type_hvector` code paths, which are implemented but not covered
  by a test here.
- **Sends are eager.** When a matching receive is already posted (hypre's normal
  pattern) the data goes straight into the receiver's buffer with one copy and no
  allocation; otherwise it is buffered. A rendezvous path for very large messages
  would remove the remaining copy.
- `Get_count` reports byte-based counts; adequate for hypre's usage.
- Not for multi-node. This deliberately replaces distributed MPI with
  shared-memory threads on one machine.

## Design notes

Two things about MPI semantics that are easy to get wrong here, both of which
hypre actually depends on:

1. **Messages match receives in the order the receives were *posted*, not the
   order they are waited on.** `hypre_exchange_interp_data` creates comm handle
   A then S, and completes S first. The implementation keeps a posted-receive
   queue and an unexpected-message queue per rank and matches at post/send time.

2. **`MPI_ANY_SOURCE` and `MPI_ANY_TAG` must be out-of-band values.** hypre's
   serial stubs define both as `1`, which is harmless only because the stubs
   never match anything. With real matching they collide with rank 1 and tag 1.
