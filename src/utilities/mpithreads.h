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
 * 'user' is passed through to every rank untouched. */
int hypre_tmpi_run(int nranks, int (*fn)(int argc, char **argv, void *user),
                   int argc, char **argv, void *user);

/* Calling thread's world rank, and the total number of ranks. */
int hypre_tmpi_rank(void);
int hypre_tmpi_nranks(void);

#ifdef __cplusplus
}
#endif

#endif
