# Running hypre without mpirun

hypre's MPI backend is the fast one. On a 64-core machine BoomerAMG gets ~22x
under `mpirun`, while the OpenMP backend caps out around 2.5x because the setup
phase is largely sequential. So a library that wants hypre's speed wants the MPI
build -- and then inherits the requirement that its caller launch under
`mpirun`, which for a library embedded in an application is often impossible.

[nano-mpi](https://github.com/danielepanozzo/nano-mpi) resolves that. It is an
implementation of a subset of MPI in which **every rank is a thread of one
process**: no launcher, no daemon, `MPI_Send` between ranks is a `memcpy`. hypre
keeps its normal domain decomposition and its normal rank-local coarsening --
the MPI algorithm, without the MPI launcher.

hypre needs no special build mode for this. nano-mpi installs a header named
`mpi.h` and a compiler wrapper, so it is found by `find_package(MPI)` the same
way Open MPI is.

## Building

```sh
# 1. nano-mpi
git clone https://github.com/danielepanozzo/nano-mpi
cmake -S nano-mpi -B nano-mpi/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/nanompi
cmake --build nano-mpi/build -j && cmake --install nano-mpi/build

# 2. hypre, as an ordinary MPI build
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release \
      -DHYPRE_ENABLE_MPI=ON \
      -DMPI_C_COMPILER=$HOME/nanompi/bin/nanompicc
cmake --build build -j
```

That is the whole integration. `HYPRE_ENABLE_MPI=ON` selects hypre's ordinary
MPI path, `mpistubs.c` narrows `HYPRE_Int` to `int` exactly as it does for Open
MPI, and nothing in hypre's numerical code knows the difference.

## Creating the ranks

There is no `mpirun`, so the embedding application creates the rank threads.
nano-mpi offers three ways; which one fits depends on who calls whom. The one a
solver library usually wants keeps the ranks alive across many calls:

```c
#include <nanompi.h>

nanompi_team *team;
nanompi_team_create(0, &team);              /* 0 = NANOMPI_NUM_RANKS, else cores */
nanompi_team_invoke(team, build_and_setup, ctx);
nanompi_team_invoke(team, solve, ctx);      /* many times */
nanompi_team_destroy(team);
```

Everything inside those callbacks is ordinary hypre code against
`hypre_MPI_COMM_WORLD`. See nano-mpi's README for `nanompi_run` (you own `main`)
and `nanompi_team_start` (you *are* rank 0, for SPMD-style callers).

With no team started at all, the calling thread is a one-rank world, so a
program that happens to be single-threaded needs no special case.

## The one requirement on hypre

Ranks are threads, so anything that was process-wide state under `mpirun` is now
shared between ranks. hypre's own globals -- the error record, the handle, the
state machine, the timing table, the random seed, the printf buffer, and
Euclid's globals -- are therefore marked `HYPRE_THREAD_LOCAL`, defined in
`HYPRE_utilities.h`. That is the only hypre change this needs, and it is worth
having on its own: it makes hypre re-entrant, so two threads can drive it
independently whether or not nano-mpi is involved.

The same rule applies to **your** code and to any library you call from inside a
rank. A file-scope `static` that a rank writes is one variable shared by every
rank, and the failure is a silent data race rather than a crash. nano-mpi's
`SCOPE.md` section 5 covers this properly; read it before porting anything.

## Two things to know

**A rank that calls `exit()` takes the whole process down**, including every
other rank, without unwinding them. Return from the rank function instead.

**Reductions are bit-identical to a real MPI run.** nano-mpi combines
contributions by recursive doubling, matching what Open MPI and MPICH do rather
than summing linearly -- which is the difference between reproducing a reference
residual and drifting from it over a long solve.

## Number of ranks

| backend | how the count is set |
|---|---|
| MPI | `mpirun -np N` |
| OpenMP | `OMP_NUM_THREADS` |
| nano-mpi | the `nranks` argument, or `NANOMPI_NUM_RANKS` when that argument is `<= 0` |

## Debugging a hang

```sh
NANOMPI_WATCHDOG=10 ./your_program
```

If no rank makes progress for ten seconds, every rank prints what it is blocked
on -- barrier, receive or probe, with communicator, peer and tag -- and the
process aborts. Considerably cheaper than attaching a debugger to sixty-four
threads.
