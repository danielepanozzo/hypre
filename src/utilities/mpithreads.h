/******************************************************************************
 * Public entry points for hypre's threads-as-MPI-ranks backend.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *****************************************************************************/

#ifndef hypre_MPITHREADS_HEADER
#define hypre_MPITHREADS_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/* Run 'fn' on 'nranks' threads; each thread is one MPI rank of COMM_WORLD.
 * Returns the first nonzero return value from any rank, else 0.
 * 'user' is passed through to every rank untouched.
 *
 * Pass nranks <= 0 to take the default: the environment variable
 * HYPRE_TMPI_NUM_THREADS if it is set to a positive integer, otherwise the
 * number of cores online. A positive nranks is used as given, so an explicit
 * choice by the caller always wins (as omp_set_num_threads does over
 * OMP_NUM_THREADS). */
int hypre_tmpi_run(int nranks, int (*fn)(int argc, char **argv, void *user),
                   int argc, char **argv, void *user);

/* The default described above, so callers can inspect or reuse it. */
int hypre_tmpi_num_threads(void);

/* Calling thread's world rank, and the total number of ranks. */
int hypre_tmpi_rank(void);
int hypre_tmpi_nranks(void);

#ifdef __cplusplus
}
#endif

#endif
