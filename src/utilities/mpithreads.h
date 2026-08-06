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

/*--------------------------------------------------------------------------
 * Persistent rank team.
 *
 * hypre_tmpi_run() spawns, runs one function and joins. A library embedding
 * hypre usually needs the ranks to outlive individual calls instead, so that a
 * solver object can be constructed once and then have factorize() and solve()
 * invoked on it collectively, many times.
 *
 *   hypre_tmpi_team *team;
 *   hypre_tmpi_team_create(8, &team);          // spawn 8 rank-threads, parked
 *   hypre_tmpi_team_invoke(team, factorize, ctx);   // all 8 run it, caller waits
 *   hypre_tmpi_team_invoke(team, solve, ctx);       // ... and again
 *   hypre_tmpi_team_destroy(team);
 *
 * The calling thread is NOT a rank: it blocks inside invoke while the team
 * runs. Only one team (or one hypre_tmpi_run) may exist per process, since the
 * rank universe is process-wide. Pass nranks <= 0 to take the default described
 * for hypre_tmpi_run above.
 *--------------------------------------------------------------------------*/
typedef struct hypre_tmpi_team_struct hypre_tmpi_team;

int hypre_tmpi_team_create(int nranks, hypre_tmpi_team **team);
int hypre_tmpi_team_invoke(hypre_tmpi_team *team, int (*fn)(void *user), void *user);
int hypre_tmpi_team_size(hypre_tmpi_team *team);
int hypre_tmpi_team_destroy(hypre_tmpi_team *team);

/* Calling thread's world rank, and the total number of ranks. */
int hypre_tmpi_rank(void);
int hypre_tmpi_nranks(void);

#ifdef __cplusplus
}
#endif

#endif
