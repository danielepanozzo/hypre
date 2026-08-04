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

Apart from that one issue, the only differences are

* **floating-point last digit** — this implementation reduces in rank order,
  Open MPI uses a tree. Addition is not associative and MPI guarantees no
  bitwise reproducibility across implementations. This backend is deterministic
  and gives identical results on every rank, which is the property that matters.
  On a 202-iteration run the drift reaches ~0.3% of the final residual, with an
  identical iteration count.
* **tests whose input data is absent from the repo** (`elast`), which fail
  identically under both backends.

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
TMPI_NP=2 ./ij -n 40 40 40 -P 2 1 1 -solver 1 \
    -flexamg_cycle_struct 0,0,-1,-1,-1,-1,1,1,1,1,-2
```

The two ranks end up disagreeing about the message sequence (a receive posted for
2 doubles is offered a 1600-double message on the same communicator and tag),
which points at a collective returning results that differ across ranks and then
diverging the control flow. Ruled out so far: the single-copy send fast path and
the spin barrier — building with `-DTMPI_DISABLE_FASTSEND` or
`-DTMPI_LOCK_BARRIER` reproduces it identically. This affects 2 of hypre's 976
`TEST_ij`/`TEST_struct`/`TEST_sstruct` cases.

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
