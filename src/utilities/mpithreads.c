/******************************************************************************
 * hypre_MPI_* implemented over POSIX threads: MPI ranks ARE threads of one
 * process. Drop-in replacement for mpistubs.c in a HYPRE_SEQUENTIAL build,
 * so hypre keeps its normal domain-decomposed algorithms with no mpirun.
 *
 * Requires the re-entrancy patch (per-thread hypre globals).
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *****************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_SEQUENTIAL

#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>

#include "mpithreads.h"

#define TMPI_MAX_COMMS   4096
#define TMPI_MAX_TYPES   4096
#define TMPI_REQS_PER_RANK 65536
#define TMPI_FIRST_USER_TYPE 16

/*--------------------------------------------------------------------------
 * Datatypes
 *--------------------------------------------------------------------------*/
typedef struct
{
   int       used;
   int       nblk;
   size_t   *len;    /* bytes in each block */
   intptr_t *disp;   /* byte displacement; absolute when 'abs' is set */
   int       abs;    /* built from hypre_MPI_Address -> absolute addresses */
   size_t    size;   /* total bytes for one element */
} tmpi_dt;

static tmpi_dt         g_dt[TMPI_MAX_TYPES];
static pthread_mutex_t g_dt_mtx = PTHREAD_MUTEX_INITIALIZER;

static size_t predefined_size(int dt)
{
   switch (dt)
   {
      case hypre_MPI_FLOAT:          return sizeof(float);
      case hypre_MPI_DOUBLE:         return sizeof(double);
      case hypre_MPI_LONG_DOUBLE:    return sizeof(long double);
      case hypre_MPI_INT:            return sizeof(HYPRE_Int);
      case hypre_MPI_CHAR:           return sizeof(char);
      case hypre_MPI_LONG:           return sizeof(hypre_longint);
      case hypre_MPI_BYTE:           return 1;
      case hypre_MPI_REAL:           return sizeof(HYPRE_Real);
      case hypre_MPI_COMPLEX:        return sizeof(HYPRE_Complex);
      case hypre_MPI_LONG_LONG_INT:  return sizeof(HYPRE_BigInt);
      default:                       return 0;
   }
}

static tmpi_dt *dt_get(int dt)
{
   if (dt >= TMPI_FIRST_USER_TYPE && dt < TMPI_MAX_TYPES && g_dt[dt].used)
   {
      return &g_dt[dt];
   }
   return NULL;
}

static size_t dt_size(int dt)
{
   tmpi_dt *d = dt_get(dt);
   return d ? d->size : predefined_size(dt);
}

static int dt_alloc(void)
{
   int i;
   pthread_mutex_lock(&g_dt_mtx);
   for (i = TMPI_FIRST_USER_TYPE; i < TMPI_MAX_TYPES; i++)
   {
      if (!g_dt[i].used) { g_dt[i].used = 1; pthread_mutex_unlock(&g_dt_mtx); return i; }
   }
   pthread_mutex_unlock(&g_dt_mtx);
   fprintf(stderr, "tmpi: out of datatype handles\n");
   abort();
}

/* number of bytes a (buf,count,dt) message occupies on the wire */
static size_t msg_bytes(int count, int dt)
{
   return (size_t)count * dt_size(dt);
}

/* pack (buf,count,dt) into contiguous 'out' (out must hold msg_bytes) */
static void pack(const void *buf, int count, int dt, char *out)
{
   tmpi_dt *d = dt_get(dt);
   int c, b;
   size_t off = 0;

   if (!d)
   {
      memcpy(out, buf, msg_bytes(count, dt));
      return;
   }
   for (c = 0; c < count; c++)
   {
      const char *base = d->abs ? NULL : ((const char *) buf + (size_t) c * d->size);
      for (b = 0; b < d->nblk; b++)
      {
         memcpy(out + off, base + d->disp[b], d->len[b]);
         off += d->len[b];
      }
   }
}

/* 'avail' is what the sender actually put on the wire; a receive may be posted
   with a larger capacity than the message, which is legal in MPI. */
static void unpack(const char *in, void *buf, int count, int dt, size_t avail)
{
   tmpi_dt *d = dt_get(dt);
   int c, b;
   size_t off = 0;

   if (!d)
   {
      size_t want = msg_bytes(count, dt);
      memcpy(buf, in, (avail < want) ? avail : want);
      return;
   }
   for (c = 0; c < count; c++)
   {
      char *base = d->abs ? NULL : ((char *) buf + (size_t) c * d->size);
      for (b = 0; b < d->nblk; b++)
      {
         size_t n = d->len[b];
         if (off + n > avail) { n = (off < avail) ? (avail - off) : 0; }
         if (n) { memcpy(base + d->disp[b], in + off, n); }
         off += d->len[b];
      }
   }
}

/*--------------------------------------------------------------------------
 * Communicators
 *--------------------------------------------------------------------------*/
typedef struct
{
   int              used;
   int              size;
   int             *world;   /* comm rank -> world rank */
   int             *inv;     /* world rank -> comm rank (-1 if not a member) */
   /* sense-reversing barrier */
   pthread_mutex_t  mtx;
   pthread_cond_t   cv;
   int              count;
   int              generation;
   /* scratch for collectives, indexed by comm rank */
   const void     **ptr;
   size_t          *len;
   /* root-only arguments republished for scatter-style collectives */
   const void      *rbuf;
   const void      *rcnt;
   const void      *rdsp;
   HYPRE_Int        rn;
} tmpi_comm;

static tmpi_comm       g_comm[TMPI_MAX_COMMS];
static pthread_mutex_t g_comm_mtx = PTHREAD_MUTEX_INITIALIZER;

/* Groups: a plain list of world ranks. */
typedef struct { int used; int size; int *world; } tmpi_group;
static tmpi_group      g_group[TMPI_MAX_COMMS];
static pthread_mutex_t g_group_mtx = PTHREAD_MUTEX_INITIALIZER;

static tmpi_group *group_get(hypre_MPI_Group g)
{
   if (g < 1 || g >= TMPI_MAX_COMMS || !g_group[g].used) { return NULL; }
   return &g_group[g];
}

static int group_alloc(int size, const int *world_ranks)
{
   int i, h = -1;
   pthread_mutex_lock(&g_group_mtx);
   for (i = 1; i < TMPI_MAX_COMMS; i++) { if (!g_group[i].used) { h = i; break; } }
   if (h < 0)
   {
      pthread_mutex_unlock(&g_group_mtx);
      fprintf(stderr, "tmpi: out of group handles\n"); abort();
   }
   g_group[h].used  = 1;
   g_group[h].size  = size;
   g_group[h].world = (int *) malloc(sizeof(int) * (size_t)(size > 0 ? size : 1));
   for (i = 0; i < size; i++) { g_group[h].world[i] = world_ranks[i]; }
   pthread_mutex_unlock(&g_group_mtx);
   return h;
}

/* User-defined reduction ops (hypre uses one, to merge rank lists). */
typedef void (*tmpi_user_fn)(void *, void *, hypre_int *, hypre_MPI_Datatype *);
#define TMPI_FIRST_USER_OP 16
#define TMPI_MAX_OPS       128
typedef struct { int used; tmpi_user_fn fn; } tmpi_op;
static tmpi_op         g_op[TMPI_MAX_OPS];
static pthread_mutex_t g_op_mtx = PTHREAD_MUTEX_INITIALIZER;

static tmpi_user_fn op_user(int op)
{
   if (op >= TMPI_FIRST_USER_OP && op < TMPI_MAX_OPS && g_op[op].used) { return g_op[op].fn; }
   return NULL;
}

static int             g_nranks = 1;
static _Thread_local int g_myrank = 0;

int  hypre_tmpi_rank(void)   { return g_myrank; }
int  hypre_tmpi_nranks(void) { return g_nranks; }

/*--------------------------------------------------------------------------
 * Optional deadlock watchdog: set TMPI_WATCHDOG=<seconds>. Records what each
 * rank is blocked on so a hang reports itself instead of needing a debugger.
 *--------------------------------------------------------------------------*/
enum { TMPI_ST_RUN = 0, TMPI_ST_BARRIER, TMPI_ST_RECV, TMPI_ST_PROBE };
typedef struct { int op, a, b, c; unsigned long seq; } tmpi_state;
static tmpi_state *g_state = NULL;

#define SET_STATE(o, x, y, z)                                     \
   do {                                                           \
      if (g_state) {                                              \
         tmpi_state *s_ = &g_state[g_myrank];                     \
         s_->op = (o); s_->a = (x); s_->b = (y); s_->c = (z);     \
         s_->seq++;                                               \
      }                                                           \
   } while (0)

static const char *state_name(int op)
{
   switch (op)
   {
      case TMPI_ST_BARRIER: return "barrier";
      case TMPI_ST_RECV:    return "recv";
      case TMPI_ST_PROBE:   return "probe";
      default:              return "running";
   }
}

static tmpi_comm *comm_get(hypre_MPI_Comm c)
{
   if (c < 0 || c >= TMPI_MAX_COMMS || !g_comm[c].used) { return NULL; }
   return &g_comm[c];
}

/* rank of the calling thread inside comm c (COMM_SELF is per-thread) */
static int comm_rank_of(hypre_MPI_Comm c)
{
   tmpi_comm *k;
   if (c == hypre_MPI_COMM_SELF) { return 0; }
   k = comm_get(c);
   return k ? k->inv[g_myrank] : -1;
}

static int comm_size_of(hypre_MPI_Comm c)
{
   tmpi_comm *k;
   if (c == hypre_MPI_COMM_SELF) { return 1; }
   k = comm_get(c);
   return k ? k->size : 0;
}

static int comm_alloc(int size, const int *world_ranks)
{
   int i, h = -1;
   tmpi_comm *k;

   pthread_mutex_lock(&g_comm_mtx);
   for (i = 2; i < TMPI_MAX_COMMS; i++)   /* 0 = WORLD, 1 = SELF */
   {
      if (!g_comm[i].used) { h = i; break; }
   }
   if (h < 0)
   {
      pthread_mutex_unlock(&g_comm_mtx);
      fprintf(stderr, "tmpi: out of communicator handles\n");
      abort();
   }
   k = &g_comm[h];
   k->used = 1;
   k->size = size;
   k->world = (int *) malloc(sizeof(int) * (size_t) size);
   k->inv   = (int *) malloc(sizeof(int) * (size_t) g_nranks);
   k->ptr   = (const void **) malloc(sizeof(void *) * (size_t) size);
   k->len   = (size_t *) malloc(sizeof(size_t) * (size_t) size);
   for (i = 0; i < g_nranks; i++) { k->inv[i] = -1; }
   for (i = 0; i < size; i++) { k->world[i] = world_ranks[i]; k->inv[world_ranks[i]] = i; }
   pthread_mutex_init(&k->mtx, NULL);
   pthread_cond_init(&k->cv, NULL);
   k->count = 0; k->generation = 0;
   pthread_mutex_unlock(&g_comm_mtx);
   return h;
}

/* Sense-reversing barrier on atomics. Collectives here are fine-grained (PCG
   does several Allreduces per iteration), and a condvar broadcast per barrier
   wakes every thread -- far too expensive. Spin briefly, then yield. */
static void comm_barrier(hypre_MPI_Comm c)
{
   tmpi_comm *k = comm_get(c);
   int gen, cnt, spins = 0;

   if (c == hypre_MPI_COMM_SELF || !k || k->size <= 1) { return; }

#ifdef TMPI_LOCK_BARRIER
   pthread_mutex_lock(&k->mtx);
   gen = k->generation;
   if (++k->count == k->size)
   {
      k->count = 0;
      k->generation++;
      pthread_cond_broadcast(&k->cv);
   }
   else
   {
      while (gen == k->generation) { pthread_cond_wait(&k->cv, &k->mtx); }
   }
   pthread_mutex_unlock(&k->mtx);
   (void) cnt; (void) spins;
#else
   gen = __atomic_load_n(&k->generation, __ATOMIC_ACQUIRE);
   cnt = __atomic_add_fetch(&k->count, 1, __ATOMIC_ACQ_REL);

   if (cnt == k->size)
   {
      __atomic_store_n(&k->count, 0, __ATOMIC_RELAXED);
      __atomic_add_fetch(&k->generation, 1, __ATOMIC_ACQ_REL);
      return;
   }
   while (__atomic_load_n(&k->generation, __ATOMIC_ACQUIRE) == gen)
   {
      if (++spins < 4096)
      {
#if defined(__aarch64__)
         __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
         __builtin_ia32_pause();
#endif
      }
      else { sched_yield(); spins = 4096; }
   }
#endif
}

/*--------------------------------------------------------------------------
 * Point-to-point: per-world-rank inbox of eagerly buffered messages
 *--------------------------------------------------------------------------*/
/* MPI matches an incoming message against receives in the order they were
 * POSTED, not the order they are waited on. hypre depends on this: it creates
 * two comm handles and then completes them in the opposite order. So we keep
 * the standard pair of queues per rank -- posted receives, and unexpected
 * messages -- and match at post/send time. */
typedef struct tmpi_msg
{
   struct tmpi_msg *next;
   int              comm;
   int              src;    /* comm rank of sender */
   int              tag;
   char            *data;
   size_t           nbytes;
} tmpi_msg;

typedef struct tmpi_pending
{
   struct tmpi_pending *next;
   int     comm, src, tag;  /* what this receive is waiting for */
   void   *buf;
   int     count, dt;
   int     done;
   int     act_src, act_tag;
   size_t  nbytes;          /* bytes actually delivered */
} tmpi_pending;

typedef struct
{
   pthread_mutex_t mtx;
   pthread_cond_t  cv;
   tmpi_msg       *umq_head, *umq_tail;   /* arrived, not yet matched */
   tmpi_pending   *prq_head, *prq_tail;   /* posted, not yet satisfied */
} tmpi_inbox;

static tmpi_inbox *g_inbox = NULL;

static int match_ok(int p_comm, int p_src, int p_tag, int m_comm, int m_src, int m_tag)
{
   if (p_comm != m_comm) { return 0; }
   if (p_src != hypre_MPI_ANY_SOURCE && p_src != m_src) { return 0; }
   if (p_tag != hypre_MPI_ANY_TAG    && p_tag != m_tag) { return 0; }
   return 1;
}


/* Diagnostic: dump the matching queues of 'world_rank'. Only called when a
   truncation has already made the run fatal. */
static void dump_queues(int world_rank)
{
   tmpi_inbox   *b = &g_inbox[world_rank];
   tmpi_pending *p;
   tmpi_msg     *m;
   int i;

   fprintf(stderr, "  posted receives on rank %d (in order):\n", world_rank);
   for (i = 0, p = b->prq_head; p; p = p->next, i++)
   {
      fprintf(stderr, "    [%d] comm=%d src=%d tag=%d capacity=%zu bytes\n",
              i, p->comm, p->src, p->tag, msg_bytes(p->count, p->dt));
   }
   if (!i) { fprintf(stderr, "    (none)\n"); }
   fprintf(stderr, "  unexpected messages on rank %d (in order):\n", world_rank);
   for (i = 0, m = b->umq_head; m; m = m->next, i++)
   {
      fprintf(stderr, "    [%d] comm=%d src=%d tag=%d %zu bytes\n",
              i, m->comm, m->src, m->tag, m->nbytes);
   }
   if (!i) { fprintf(stderr, "    (none)\n"); }
}

/* copy an arrived message into a posted receive; caller holds the inbox lock */
static void deliver(tmpi_pending *p, int m_src, int m_tag, const char *data, size_t nbytes)
{
   size_t cap = msg_bytes(p->count, p->dt);

   if (nbytes > cap)
   {
      fprintf(stderr,
              "tmpi: MESSAGE TRUNCATED on rank %d: comm=%d src=%d tag=%d "
              "sent=%zu bytes, recv capacity=%zu bytes (count=%d dt=%d)\n",
              g_myrank, p->comm, m_src, m_tag, nbytes, cap, p->count, p->dt);
      dump_queues(g_myrank);
      abort();
   }
   if (nbytes > 0) { unpack(data, p->buf, p->count, p->dt, nbytes); }
   p->act_src = m_src;
   p->act_tag = m_tag;
   p->nbytes  = nbytes;
   p->done    = 1;
}

static void do_send(const void *buf, int count, int dt, int dest, int tag, hypre_MPI_Comm comm)
{
   tmpi_comm    *k = comm_get(comm);
   tmpi_inbox   *b;
   tmpi_pending *p, *prev;
   int           dest_world, my_crank;
   size_t        nbytes;

   if (comm == hypre_MPI_COMM_SELF) { dest_world = g_myrank; }
   else if (!k) { return; }
   else { dest_world = k->world[dest]; }

   my_crank = comm_rank_of(comm);
   nbytes   = msg_bytes(count, dt);
   b        = &g_inbox[dest_world];

   /* Fast path: if a matching receive is already posted, detach it and copy
      straight into its buffer -- no heap allocation, one copy instead of two.
      This is hypre's normal pattern (Irecv before Isend). */
#ifndef TMPI_DISABLE_FASTSEND
   pthread_mutex_lock(&b->mtx);
   prev = NULL;
   for (p = b->prq_head; p; prev = p, p = p->next)
   {
      if (match_ok(p->comm, p->src, p->tag, comm, my_crank, tag))
      {
         if (prev) { prev->next = p->next; } else { b->prq_head = p->next; }
         if (b->prq_tail == p) { b->prq_tail = prev; }
         break;
      }
   }
   pthread_mutex_unlock(&b->mtx);

   if (p)
   {
      size_t cap = msg_bytes(p->count, p->dt);
      if (nbytes > cap)
      {
         fprintf(stderr,
                 "tmpi: MESSAGE TRUNCATED on rank %d: comm=%d src=%d tag=%d "
                 "sent=%zu bytes, recv capacity=%zu bytes (count=%d dt=%d)\n",
                 dest_world, (int) comm, my_crank, tag, nbytes, cap, p->count, p->dt);
         dump_queues(dest_world);
         abort();
      }
      if (nbytes > 0)
      {
         /* both sides contiguous -> straight memcpy, outside the lock */
         if (!dt_get(dt) && !dt_get(p->dt))
         {
            memcpy(p->buf, buf, nbytes);
         }
         else
         {
            char *tmp = (char *) malloc(nbytes);
            pack(buf, count, dt, tmp);
            unpack(tmp, p->buf, p->count, p->dt, nbytes);
            free(tmp);
         }
      }
      pthread_mutex_lock(&b->mtx);
      p->act_src = my_crank; p->act_tag = tag; p->nbytes = nbytes; p->done = 1;
      pthread_cond_broadcast(&b->cv);
      pthread_mutex_unlock(&b->mtx);
      return;
   }
#endif /* TMPI_DISABLE_FASTSEND */

   /* Slow path: nobody is waiting yet, so buffer the message. */
   {
      char     *data = (nbytes > 0) ? (char *) malloc(nbytes) : NULL;
      tmpi_msg *m    = (tmpi_msg *) malloc(sizeof(tmpi_msg));
      if (nbytes > 0) { pack(buf, count, dt, data); }
      m->next = NULL; m->comm = comm; m->src = my_crank; m->tag = tag;
      m->data = data; m->nbytes = nbytes;

      pthread_mutex_lock(&b->mtx);
      /* re-check: a receive may have been posted while we were packing */
      prev = NULL;
      for (p = b->prq_head; p; prev = p, p = p->next)
      {
         if (match_ok(p->comm, p->src, p->tag, comm, my_crank, tag))
         {
            if (prev) { prev->next = p->next; } else { b->prq_head = p->next; }
            if (b->prq_tail == p) { b->prq_tail = prev; }
            deliver(p, my_crank, tag, m->data, m->nbytes);
            pthread_cond_broadcast(&b->cv);
            pthread_mutex_unlock(&b->mtx);
            free(m->data); free(m);
            return;
         }
      }
      if (b->umq_tail) { b->umq_tail->next = m; } else { b->umq_head = m; }
      b->umq_tail = m;
      pthread_cond_broadcast(&b->cv);
      pthread_mutex_unlock(&b->mtx);
   }
}

/* Post a receive. Returns a pending record (already satisfied if the message
   had arrived early). */
static tmpi_pending *post_recv(void *buf, int count, int dt, int src, int tag,
                               hypre_MPI_Comm comm)
{
   tmpi_inbox   *b = &g_inbox[g_myrank];
   tmpi_pending *p = (tmpi_pending *) malloc(sizeof(tmpi_pending));
   tmpi_msg     *m, *prev;

   p->next = NULL; p->comm = comm; p->src = src; p->tag = tag;
   p->buf = buf; p->count = count; p->dt = dt;
   p->done = 0; p->act_src = 0; p->act_tag = 0; p->nbytes = 0;

   pthread_mutex_lock(&b->mtx);
   prev = NULL;
   for (m = b->umq_head; m; prev = m, m = m->next)
   {
      if (match_ok(comm, src, tag, m->comm, m->src, m->tag))
      {
         if (prev) { prev->next = m->next; } else { b->umq_head = m->next; }
         if (b->umq_tail == m) { b->umq_tail = prev; }
         deliver(p, m->src, m->tag, m->data, m->nbytes);
         pthread_mutex_unlock(&b->mtx);
         free(m->data); free(m);
         return p;
      }
   }
   if (b->prq_tail) { b->prq_tail->next = p; } else { b->prq_head = p; }
   b->prq_tail = p;
   pthread_mutex_unlock(&b->mtx);
   return p;
}

static void wait_pending(tmpi_pending *p, hypre_MPI_Status *status)
{
   tmpi_inbox *b = &g_inbox[g_myrank];

   SET_STATE(TMPI_ST_RECV, p->comm, p->src, p->tag);
   pthread_mutex_lock(&b->mtx);
   while (!p->done) { pthread_cond_wait(&b->cv, &b->mtx); }
   pthread_mutex_unlock(&b->mtx);
   SET_STATE(TMPI_ST_RUN, 0, 0, 0);

   if (status)
   {
      status->hypre_MPI_SOURCE = p->act_src;
      status->hypre_MPI_TAG    = p->act_tag;
      status->hypre_MPI_COUNT  = (HYPRE_Int) p->nbytes;
   }
   free(p);
}

static void do_recv(void *buf, int count, int dt, int src, int tag,
                    hypre_MPI_Comm comm, hypre_MPI_Status *status)
{
   wait_pending(post_recv(buf, count, dt, src, tag, comm), status);
}

/* Probe inspects only unmatched arrivals -- it must not consume a posted recv. */
static tmpi_msg *probe_umq(int comm, int src, int tag, int blocking, hypre_MPI_Status *status)
{
   tmpi_inbox *b = &g_inbox[g_myrank];
   tmpi_msg *m;

   pthread_mutex_lock(&b->mtx);
   for (;;)
   {
      for (m = b->umq_head; m; m = m->next)
      {
         if (match_ok(comm, src, tag, m->comm, m->src, m->tag))
         {
            if (status)
            {
               status->hypre_MPI_SOURCE = m->src;
               status->hypre_MPI_TAG    = m->tag;
               status->hypre_MPI_COUNT  = (HYPRE_Int) m->nbytes;
            }
            pthread_mutex_unlock(&b->mtx);
            return m;
         }
      }
      if (!blocking) { pthread_mutex_unlock(&b->mtx); return NULL; }
      SET_STATE(TMPI_ST_PROBE, comm, src, tag);
      pthread_cond_wait(&b->cv, &b->mtx);
   }
}

/*--------------------------------------------------------------------------
 * Requests (per-thread pool, so no locking on the fast path)
 *--------------------------------------------------------------------------*/
/* kind: 1 = completed send, 2 = posted recv,
         3 = persistent send, 4 = persistent recv (MPI_*_init + MPI_Startall) */
typedef struct
{
   int           used;
   int           kind;
   int           active;         /* persistent request currently in flight */
   tmpi_pending *p;              /* kinds 2 and 4 */
   void         *buf;            /* persistent parameters, replayed on Start */
   int           count, dt, peer, tag, comm;
} tmpi_req;

static tmpi_req **g_reqs = NULL;   /* [world rank][TMPI_REQS_PER_RANK] */

static hypre_MPI_Request req_alloc(void)
{
   tmpi_req *pool = g_reqs[g_myrank];
   int i;
   for (i = 0; i < TMPI_REQS_PER_RANK; i++)
   {
      if (!pool[i].used)
      {
         pool[i].used = 1;
         return (hypre_MPI_Request)((size_t) g_myrank * TMPI_REQS_PER_RANK + i + 1);
      }
   }
   fprintf(stderr, "tmpi: out of request handles on rank %d\n", g_myrank);
   abort();
}

static tmpi_req *req_get(hypre_MPI_Request h)
{
   size_t idx;
   if (h == hypre_MPI_REQUEST_NULL) { return NULL; }
   idx = (size_t) h - 1;
   return &g_reqs[idx / TMPI_REQS_PER_RANK][idx % TMPI_REQS_PER_RANK];
}

/*--------------------------------------------------------------------------
 * Reductions
 *--------------------------------------------------------------------------*/
#define TMPI_REDUCE(CTYPE, OP)                                     \
   {                                                               \
      CTYPE *d = (CTYPE *) dst; const CTYPE *s = (const CTYPE *) src; \
      HYPRE_Int i;                                                 \
      for (i = 0; i < count; i++) { OP; }                          \
   }

#define TMPI_REDUCE_ARITH(CTYPE)                                   \
   switch (op) {                                                   \
      case hypre_MPI_SUM:  TMPI_REDUCE(CTYPE, d[i] = d[i] + s[i]); break; \
      case hypre_MPI_MIN:  TMPI_REDUCE(CTYPE, d[i] = (s[i] < d[i]) ? s[i] : d[i]); break; \
      case hypre_MPI_MAX:  TMPI_REDUCE(CTYPE, d[i] = (s[i] > d[i]) ? s[i] : d[i]); break; \
      default:             TMPI_REDUCE(CTYPE, d[i] = d[i] + s[i]); break; \
   }

#define TMPI_REDUCE_INT(CTYPE)                                     \
   switch (op) {                                                   \
      case hypre_MPI_SUM:  TMPI_REDUCE(CTYPE, d[i] = d[i] + s[i]); break; \
      case hypre_MPI_MIN:  TMPI_REDUCE(CTYPE, d[i] = (s[i] < d[i]) ? s[i] : d[i]); break; \
      case hypre_MPI_MAX:  TMPI_REDUCE(CTYPE, d[i] = (s[i] > d[i]) ? s[i] : d[i]); break; \
      case hypre_MPI_LOR:  TMPI_REDUCE(CTYPE, d[i] = (CTYPE)(d[i] || s[i])); break; \
      case hypre_MPI_LAND: TMPI_REDUCE(CTYPE, d[i] = (CTYPE)(d[i] && s[i])); break; \
      case hypre_MPI_BOR:  TMPI_REDUCE(CTYPE, d[i] = (CTYPE)(d[i] | s[i])); break; \
      default:             TMPI_REDUCE(CTYPE, d[i] = d[i] + s[i]); break; \
   }

/* dst <- dst OP src */
static void reduce_into(void *dst, const void *src, HYPRE_Int count, int dt, int op)
{
   switch (dt)
   {
      case hypre_MPI_INT:           TMPI_REDUCE_INT(HYPRE_Int);      break;
      case hypre_MPI_LONG_LONG_INT: TMPI_REDUCE_INT(HYPRE_BigInt);   break;
      case hypre_MPI_LONG:          TMPI_REDUCE_INT(hypre_longint);  break;
      case hypre_MPI_CHAR:          TMPI_REDUCE_INT(char);           break;
      case hypre_MPI_BYTE:          TMPI_REDUCE_INT(unsigned char);  break;
      case hypre_MPI_FLOAT:         TMPI_REDUCE_ARITH(float);        break;
      case hypre_MPI_DOUBLE:        TMPI_REDUCE_ARITH(double);       break;
      case hypre_MPI_LONG_DOUBLE:   TMPI_REDUCE_ARITH(long double);  break;
      case hypre_MPI_REAL:          TMPI_REDUCE_ARITH(HYPRE_Real);   break;
      case hypre_MPI_COMPLEX:       TMPI_REDUCE_ARITH(HYPRE_Complex); break;
      default:
         fprintf(stderr, "tmpi: reduce on unsupported datatype %d\n", dt);
         abort();
   }
}


/* Combine sz contributions into recvbuf.
 *
 * The association matters: floating-point addition is not associative, so the
 * order in which contributions are combined decides the last bits of the
 * result. Summing linearly (((a0+a1)+a2)+a3) gives different answers from the
 * recursive doubling that MPI implementations use, which is enough to make an
 * AMG residual differ in its last digits and, over a long solve, drift.
 *
 * So use recursive doubling here too: fold the first 2r contributions pairwise
 * into r virtual ranks, shift the rest down, then combine with partner i^d for
 * d = 1, 2, 4, ... This is the standard algorithm; it is also more accurate
 * than a linear sum (error grows as log n rather than n), and it reproduces
 * Open MPI bit-for-bit -- verified over 400 random trials at every rank count
 * from 2 to 16, including non-powers of two.
 *
 * A user-defined op still sees MPI's (invec, inoutvec) contract, so the
 * left operand is passed as invec.
 */
static void reduce_all(void *recvbuf, const void **d, int sz, HYPRE_Int count,
                       int dt, int op, size_t nb)
{
   tmpi_user_fn fn = op_user(op);
   const void **v;
   char *work, *tmp = NULL;
   int adjust = 1, r, k, step, i;

   if (sz <= 0) { return; }
   if (sz == 1) { memcpy(recvbuf, d[0], nb); return; }

   while (adjust * 2 <= sz) { adjust *= 2; }
   r = sz - adjust;

   work = (char *) malloc((size_t) adjust * nb);
   v    = (const void **) malloc(sizeof(void *) * (size_t) adjust);
   if (fn) { tmp = (char *) malloc(nb); }

   /* dst = left op right */
   #define TMPI_COMBINE(dst, left, right)                                     \
      do {                                                                    \
         if (fn)                                                              \
         {                                                                    \
            hypre_int len_ = (hypre_int) count;                               \
            hypre_MPI_Datatype dtc_ = (hypre_MPI_Datatype) dt;                \
            memcpy(tmp, (left), nb);                                          \
            memcpy((dst), (right), nb);                                       \
            fn((void *) tmp, (dst), &len_, &dtc_);                            \
         }                                                                    \
         else                                                                 \
         {                                                                    \
            if ((const void *)(dst) != (const void *)(left))                  \
            { memcpy((dst), (left), nb); }                                    \
            reduce_into((dst), (right), count, dt, op);                       \
         }                                                                    \
      } while (0)

   /* fold the 2r low contributions into r virtual ranks */
   for (k = 0; k < r; k++)
   {
      char *slot = work + (size_t) k * nb;
      TMPI_COMBINE(slot, d[2 * k], d[2 * k + 1]);
      v[k] = slot;
   }
   for (k = r; k < adjust; k++) { v[k] = d[k + r]; }

   /* recursive doubling over the virtual ranks */
   for (step = 1; step < adjust; step *= 2)
   {
      for (i = 0; i < adjust; i += 2 * step)
      {
         char *slot = work + (size_t) i * nb;
         TMPI_COMBINE(slot, v[i], v[i + step]);
         v[i] = slot;
      }
   }
   #undef TMPI_COMBINE

   memcpy(recvbuf, v[0], nb);
   free(work); free((void *) v); free(tmp);
}

/*--------------------------------------------------------------------------
 * Launch
 *--------------------------------------------------------------------------*/
typedef struct
{
   int    rank;
   int  (*fn)(int, char **, void *);
   int    argc;
   char **argv;
   void  *user;
   int    ret;
} tmpi_thread_arg;

static void *tmpi_watchdog(void *v)
{
   int secs = *(int *) v, i, t;
   unsigned long *prev = (unsigned long *) calloc((size_t) g_nranks, sizeof(unsigned long));

   for (t = 0; ; t++)
   {
      struct timespec ts; ts.tv_sec = 1; ts.tv_nsec = 0;
      nanosleep(&ts, NULL);
      {
         int stuck = 1;
         for (i = 0; i < g_nranks; i++)
         {
            if (g_state[i].seq != prev[i]) { stuck = 0; }
            prev[i] = g_state[i].seq;
         }
         if (!stuck) { t = 0; continue; }
      }
      if (t < secs) { continue; }
      fprintf(stderr, "\n=== TMPI WATCHDOG: no progress for %ds, %d ranks ===\n", secs, g_nranks);
      for (i = 0; i < g_nranks; i++)
      {
         fprintf(stderr, "  rank %3d: %-8s comm=%d a=%d b=%d\n",
                 i, state_name(g_state[i].op), g_state[i].a, g_state[i].b, g_state[i].c);
      }
      fflush(stderr);
      abort();
   }
   return NULL;
}

static void *tmpi_trampoline(void *v)
{
   tmpi_thread_arg *a = (tmpi_thread_arg *) v;
   g_myrank = a->rank;
   a->ret = a->fn(a->argc, a->argv, a->user);
   return NULL;
}

int hypre_tmpi_run(int nranks, int (*fn)(int, char **, void *),
                   int argc, char **argv, void *user)
{
   pthread_t       *th;
   tmpi_thread_arg *args;
   int             *world;
   int i, rc = 0;

   if (nranks < 1) { return 1; }
   g_nranks = nranks;

   g_inbox = (tmpi_inbox *) calloc((size_t) nranks, sizeof(tmpi_inbox));
   g_reqs  = (tmpi_req **) calloc((size_t) nranks, sizeof(tmpi_req *));
   for (i = 0; i < nranks; i++)
   {
      pthread_mutex_init(&g_inbox[i].mtx, NULL);
      pthread_cond_init(&g_inbox[i].cv, NULL);
      g_reqs[i] = (tmpi_req *) calloc(TMPI_REQS_PER_RANK, sizeof(tmpi_req));
   }

   /* COMM_WORLD = handle 0 */
   world = (int *) malloc(sizeof(int) * (size_t) nranks);
   for (i = 0; i < nranks; i++) { world[i] = i; }
   {
      tmpi_comm *k = &g_comm[hypre_MPI_COMM_WORLD];
      k->used = 1; k->size = nranks;
      k->world = world;
      k->inv = (int *) malloc(sizeof(int) * (size_t) nranks);
      k->ptr = (const void **) malloc(sizeof(void *) * (size_t) nranks);
      k->len = (size_t *) malloc(sizeof(size_t) * (size_t) nranks);
      for (i = 0; i < nranks; i++) { k->inv[i] = i; }
      pthread_mutex_init(&k->mtx, NULL);
      pthread_cond_init(&k->cv, NULL);
      k->count = 0; k->generation = 0;
   }
   g_comm[hypre_MPI_COMM_SELF].used = 1;
   g_comm[hypre_MPI_COMM_SELF].size = 1;

   g_state = (tmpi_state *) calloc((size_t) nranks, sizeof(tmpi_state));
   {
      const char *wd = getenv("TMPI_WATCHDOG");
      if (wd && atoi(wd) > 0)
      {
         static int secs; pthread_t wt;
         secs = atoi(wd);
         pthread_create(&wt, NULL, tmpi_watchdog, &secs);
         pthread_detach(wt);
      }
   }

   th   = (pthread_t *) malloc(sizeof(pthread_t) * (size_t) nranks);
   args = (tmpi_thread_arg *) malloc(sizeof(tmpi_thread_arg) * (size_t) nranks);
   for (i = 0; i < nranks; i++)
   {
      args[i].rank = i; args[i].fn = fn; args[i].argc = argc;
      args[i].argv = argv; args[i].user = user; args[i].ret = 0;
      if (pthread_create(&th[i], NULL, tmpi_trampoline, &args[i]) != 0)
      {
         fprintf(stderr, "tmpi: pthread_create failed for rank %d\n", i);
         return 1;
      }
   }
   for (i = 0; i < nranks; i++) { pthread_join(th[i], NULL); if (args[i].ret) { rc = args[i].ret; } }

   free(th); free(args);
   return rc;
}

/*==========================================================================
 * hypre_MPI_* entry points
 *========================================================================*/

HYPRE_Int hypre_MPI_Init( hypre_int *argc, char ***argv )
{
   HYPRE_UNUSED_VAR(argc); HYPRE_UNUSED_VAR(argv);
   return 0;
}

HYPRE_Int hypre_MPI_Finalize( void ) { return 0; }

HYPRE_Int hypre_MPI_Abort( hypre_MPI_Comm comm, HYPRE_Int errorcode )
{
   HYPRE_UNUSED_VAR(comm);
   fprintf(stderr, "tmpi: abort(%d)\n", (int) errorcode);
   abort();
   return 0;
}

hypre_double hypre_MPI_Wtime( void )
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (hypre_double) ts.tv_sec + 1.0e-9 * (hypre_double) ts.tv_nsec;
}

hypre_double hypre_MPI_Wtick( void ) { return 1.0e-9; }

HYPRE_Int hypre_MPI_Barrier( hypre_MPI_Comm comm ) { comm_barrier(comm); return 0; }

HYPRE_Int hypre_MPI_Comm_size( hypre_MPI_Comm comm, HYPRE_Int *size )
{
   *size = (HYPRE_Int) comm_size_of(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Comm_rank( hypre_MPI_Comm comm, HYPRE_Int *rank )
{
   *rank = (HYPRE_Int) comm_rank_of(comm);
   return 0;
}

hypre_MPI_Comm hypre_MPI_Comm_f2c( hypre_int comm ) { return (hypre_MPI_Comm) comm; }

HYPRE_Int hypre_MPI_Comm_dup( hypre_MPI_Comm comm, hypre_MPI_Comm *newcomm )
{
   tmpi_comm *k = comm_get(comm);
   int h;

   if (comm == hypre_MPI_COMM_SELF || !k) { *newcomm = comm; return 0; }
   /* every member must agree on the new handle: rank 0 allocates, then broadcast */
   comm_barrier(comm);
   if (comm_rank_of(comm) == 0)
   {
      h = comm_alloc(k->size, k->world);
      k->ptr[0] = NULL;
      k->len[0] = (size_t) h;
   }
   comm_barrier(comm);
   h = (int) k->len[0];
   comm_barrier(comm);
   *newcomm = (hypre_MPI_Comm) h;
   return 0;
}

HYPRE_Int hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
   tmpi_comm *k;
   if (!comm) { return 0; }
   k = comm_get(*comm);
   if (k && *comm > 1)
   {
      comm_barrier(*comm);
      if (comm_rank_of(*comm) == 0)
      {
         pthread_mutex_lock(&g_comm_mtx);
         free(k->world); free(k->inv); free((void *) k->ptr); free(k->len);
         k->used = 0;
         pthread_mutex_unlock(&g_comm_mtx);
      }
   }
   *comm = hypre_MPI_COMM_NULL;
   return 0;
}

/* group == the set of world ranks; we encode a group as a comm handle */
HYPRE_Int hypre_MPI_Comm_group( hypre_MPI_Comm comm, hypre_MPI_Group *group )
{
   tmpi_comm *k = comm_get(comm);
   if (comm == hypre_MPI_COMM_SELF || !k)
   {
      int self = g_myrank;
      *group = (hypre_MPI_Group) group_alloc(1, &self);
   }
   else
   {
      *group = (hypre_MPI_Group) group_alloc(k->size, k->world);
   }
   return 0;
}

HYPRE_Int hypre_MPI_Group_incl( hypre_MPI_Group group, HYPRE_Int n, HYPRE_Int *ranks,
                                hypre_MPI_Group *newgroup )
{
   tmpi_group *g = group_get(group);
   int *w, i;

   if (!g) { *newgroup = group; return 0; }
   w = (int *) malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
   for (i = 0; i < n; i++)
   {
      int r = (int) ranks[i];
      w[i] = (r >= 0 && r < g->size) ? g->world[r] : 0;
   }
   *newgroup = (hypre_MPI_Group) group_alloc((int) n, w);
   free(w);
   return 0;
}

HYPRE_Int hypre_MPI_Group_free( hypre_MPI_Group *group )
{
   tmpi_group *g = group_get(*group);
   if (g)
   {
      pthread_mutex_lock(&g_group_mtx);
      free(g->world); g->world = NULL; g->used = 0;
      pthread_mutex_unlock(&g_group_mtx);
   }
   return 0;
}

/* Collective over 'comm'. Ranks in 'group' get the new communicator; everyone
   else gets COMM_NULL. hypre uses this to build a coarse-grid sub-communicator,
   so getting the non-member case right is what keeps the other ranks moving. */
HYPRE_Int hypre_MPI_Comm_create( hypre_MPI_Comm comm, hypre_MPI_Group group,
                                 hypre_MPI_Comm *newcomm )
{
   tmpi_comm  *k = comm_get(comm);
   tmpi_group *g = group_get(group);
   int h, i, member = 0;

   if (comm == hypre_MPI_COMM_SELF || !k || !g) { return hypre_MPI_Comm_dup(comm, newcomm); }

   comm_barrier(comm);
   if (comm_rank_of(comm) == 0) { k->len[0] = (size_t) comm_alloc(g->size, g->world); }
   comm_barrier(comm);
   h = (int) k->len[0];
   for (i = 0; i < g->size; i++) { if (g->world[i] == g_myrank) { member = 1; break; } }
   comm_barrier(comm);

   *newcomm = member ? (hypre_MPI_Comm) h : (hypre_MPI_Comm) hypre_MPI_COMM_NULL;
   return 0;
}

/* colour = n, key = m */
HYPRE_Int hypre_MPI_Comm_split( hypre_MPI_Comm comm, HYPRE_Int n, HYPRE_Int m,
                                hypre_MPI_Comm *newcomm )
{
   tmpi_comm *k = comm_get(comm);
   int sz, me, i, j, cnt, h = hypre_MPI_COMM_NULL;
   int *colours, *keys, *members;

   if (comm == hypre_MPI_COMM_SELF || !k) { *newcomm = comm; return 0; }
   sz = k->size;
   me = comm_rank_of(comm);

   colours = (int *) malloc(sizeof(int) * (size_t) sz * 2);
   keys    = colours + sz;

   /* gather colours and keys */
   {
      HYPRE_Int cm = n, km = m;
      hypre_MPI_Allgather(&cm, 1, hypre_MPI_INT, colours, 1, hypre_MPI_INT, comm);
      hypre_MPI_Allgather(&km, 1, hypre_MPI_INT, keys,    1, hypre_MPI_INT, comm);
   }

   if (n != hypre_MPI_UNDEFINED)
   {
      members = (int *) malloc(sizeof(int) * (size_t) sz);
      cnt = 0;
      for (i = 0; i < sz; i++) { if (colours[i] == n) { members[cnt++] = i; } }
      /* stable sort members by key */
      for (i = 1; i < cnt; i++)
      {
         int v = members[i];
         for (j = i - 1; j >= 0 && keys[members[j]] > keys[v]; j--) { members[j + 1] = members[j]; }
         members[j + 1] = v;
      }
      for (i = 0; i < cnt; i++) { members[i] = k->world[members[i]]; }

      /* the lowest-ranked member of each colour allocates the handle and shares it */
      {
         int *all_h = (int *) malloc(sizeof(int) * (size_t) sz);
         HYPRE_Int mine = -1;
         if (k->world[me] == members[0]) { mine = comm_alloc(cnt, members); }
         hypre_MPI_Allgather(&mine, 1, hypre_MPI_INT, all_h, 1, hypre_MPI_INT, comm);
         for (i = 0; i < sz; i++)
         {
            if (colours[i] == n && all_h[i] >= 0) { h = all_h[i]; break; }
         }
         free(all_h);
      }
      free(members);
   }
   else
   {
      int *all_h = (int *) malloc(sizeof(int) * (size_t) sz);
      HYPRE_Int mine = -1;
      hypre_MPI_Allgather(&mine, 1, hypre_MPI_INT, all_h, 1, hypre_MPI_INT, comm);
      free(all_h);
   }

   free(colours);
   *newcomm = (hypre_MPI_Comm) h;
   return 0;
}

/* every rank is on the same "node" here, so shared-memory split == dup */
HYPRE_Int hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, HYPRE_Int split_type, HYPRE_Int key,
                                     hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
{
   HYPRE_UNUSED_VAR(split_type); HYPRE_UNUSED_VAR(key); HYPRE_UNUSED_VAR(info);
   return hypre_MPI_Comm_dup(comm, newcomm);
}

HYPRE_Int hypre_MPI_Address( void *location, hypre_MPI_Aint *address )
{
   *address = (hypre_MPI_Aint)(intptr_t) location;
   return 0;
}

HYPRE_Int hypre_MPI_Get_count( hypre_MPI_Status *status, hypre_MPI_Datatype datatype,
                               HYPRE_Int *count )
{
   size_t esz = dt_size(datatype);
   *count = (esz > 0) ? (HYPRE_Int)((size_t) status->hypre_MPI_COUNT / esz) : 0;
   return 0;
}

/*--------------------------------------------------------------------------
 * Collectives (barrier + shared scratch pointers)
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_MPI_Bcast( void *buffer, HYPRE_Int count, hypre_MPI_Datatype datatype,
                           HYPRE_Int root, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1) { return 0; }
   if (me == root) { k->ptr[0] = buffer; }
   comm_barrier(comm);
   if (me != root) { memcpy(buffer, k->ptr[0], msg_bytes(count, datatype)); }
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Allreduce( void *sendbuf, void *recvbuf, HYPRE_Int count,
                               hypre_MPI_Datatype datatype, hypre_MPI_Op op, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t nb = msg_bytes(count, datatype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      if (sendbuf != recvbuf) { memcpy(recvbuf, sendbuf, nb); }
      return 0;
   }
   sz = k->size;
   /* publish own contribution; every rank then reduces the full set itself so
      the result is bitwise identical across ranks (rank order is fixed) */
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   reduce_all(recvbuf, k->ptr, sz, count, datatype, op, nb);
   comm_barrier(comm);
   HYPRE_UNUSED_VAR(i);
   return 0;
}

HYPRE_Int hypre_MPI_Reduce( void *sendbuf, void *recvbuf, HYPRE_Int count,
                            hypre_MPI_Datatype datatype, hypre_MPI_Op op, HYPRE_Int root,
                            hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t nb = msg_bytes(count, datatype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      if (sendbuf != recvbuf) { memcpy(recvbuf, sendbuf, nb); }
      return 0;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   if (me == root) { reduce_all(recvbuf, k->ptr, sz, count, datatype, op, nb); }
   comm_barrier(comm);
   HYPRE_UNUSED_VAR(i);
   return 0;
}

HYPRE_Int hypre_MPI_Scan( void *sendbuf, void *recvbuf, HYPRE_Int count,
                          hypre_MPI_Datatype datatype, hypre_MPI_Op op, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i;
   size_t nb = msg_bytes(count, datatype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      if (sendbuf != recvbuf) { memcpy(recvbuf, sendbuf, nb); }
      return 0;
   }
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   reduce_all(recvbuf, k->ptr, me + 1, count, datatype, op, nb);
   comm_barrier(comm);
   HYPRE_UNUSED_VAR(i);
   return 0;
}

HYPRE_Int hypre_MPI_Allgather( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                               void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype,
                               hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t sb = msg_bytes(sendcount, sendtype), rb = msg_bytes(recvcount, recvtype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy(recvbuf, sendbuf, sb);
      return 0;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   for (i = 0; i < sz; i++) { memcpy((char *) recvbuf + (size_t) i * rb, k->ptr[i], sb); }
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Allgatherv( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                                void *recvbuf, HYPRE_Int *recvcounts, HYPRE_Int *displs,
                                hypre_MPI_Datatype recvtype, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t esz = dt_size(recvtype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy((char *) recvbuf + (size_t) displs[0] * esz, sendbuf, msg_bytes(sendcount, sendtype));
      return 0;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   k->len[me] = msg_bytes(sendcount, sendtype);
   comm_barrier(comm);
   for (i = 0; i < sz; i++)
   {
      memcpy((char *) recvbuf + (size_t) displs[i] * esz, k->ptr[i],
             (size_t) recvcounts[i] * esz);
   }
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Gather( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                            void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype,
                            HYPRE_Int root, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t sb = msg_bytes(sendcount, sendtype), rb = 0;

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1) { memcpy(recvbuf, sendbuf, sb); return 0; }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   if (me == root)
   {
      rb = msg_bytes(recvcount, recvtype);   /* root-only arguments */
      for (i = 0; i < sz; i++) { memcpy((char *) recvbuf + (size_t) i * rb, k->ptr[i], sb); }
   }
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Gatherv( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                             void *recvbuf, HYPRE_Int *recvcounts, HYPRE_Int *displs,
                             hypre_MPI_Datatype recvtype, HYPRE_Int root, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t esz = dt_size(recvtype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy((char *) recvbuf + (size_t) displs[0] * esz, sendbuf, msg_bytes(sendcount, sendtype));
      return 0;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   if (me == root)
   {
      for (i = 0; i < sz; i++)
      {
         memcpy((char *) recvbuf + (size_t) displs[i] * esz, k->ptr[i], (size_t) recvcounts[i] * esz);
      }
   }
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Scatter( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                             void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype,
                             HYPRE_Int root, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm);
   size_t sb = msg_bytes(sendcount, sendtype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy(recvbuf, sendbuf, msg_bytes(recvcount, recvtype));
      return 0;
   }
   if (me == root) { k->rbuf = sendbuf; k->rn = sendcount; }
   comm_barrier(comm);
   /* sendcount is only valid at the root, so use the root's value */
   sb = msg_bytes((int) k->rn, sendtype);
   memcpy(recvbuf, (const char *) k->rbuf + (size_t) me * sb, sb);
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Scatterv( void *sendbuf, HYPRE_Int *sendcounts, HYPRE_Int *displs,
                              hypre_MPI_Datatype sendtype, void *recvbuf, HYPRE_Int recvcount,
                              hypre_MPI_Datatype recvtype, HYPRE_Int root, hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm);
   size_t esz = dt_size(sendtype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy(recvbuf, (const char *) sendbuf + (size_t) displs[0] * esz, msg_bytes(recvcount, recvtype));
      return 0;
   }
   if (me == root) { k->rbuf = sendbuf; k->rcnt = sendcounts; k->rdsp = displs; }
   comm_barrier(comm);
   {
      /* sendcounts/displs are root-only in MPI; read the root's copies */
      const HYPRE_Int *sc = (const HYPRE_Int *) k->rcnt;
      const HYPRE_Int *dp = (const HYPRE_Int *) k->rdsp;
      memcpy(recvbuf, (const char *) k->rbuf + (size_t) dp[me] * esz,
             (size_t) sc[me] * esz);
   }
   comm_barrier(comm);
   return 0;
}

HYPRE_Int hypre_MPI_Alltoall( void *sendbuf, HYPRE_Int sendcount, hypre_MPI_Datatype sendtype,
                              void *recvbuf, HYPRE_Int recvcount, hypre_MPI_Datatype recvtype,
                              hypre_MPI_Comm comm )
{
   tmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t sb = msg_bytes(sendcount, sendtype), rb = msg_bytes(recvcount, recvtype);

   if (comm == hypre_MPI_COMM_SELF || !k || k->size == 1) { memcpy(recvbuf, sendbuf, sb); return 0; }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   for (i = 0; i < sz; i++)
   {
      memcpy((char *) recvbuf + (size_t) i * rb,
             (const char *) k->ptr[i] + (size_t) me * sb, sb);
   }
   comm_barrier(comm);
   return 0;
}

/*--------------------------------------------------------------------------
 * Point-to-point entry points
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_MPI_Send( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                          HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm )
{
   do_send(buf, count, datatype, dest, tag, comm);
   return 0;
}

HYPRE_Int hypre_MPI_Recv( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                          HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm,
                          hypre_MPI_Status *status )
{
   do_recv(buf, count, datatype, source, tag, comm, status);
   return 0;
}

HYPRE_Int hypre_MPI_Isend( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                           HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm,
                           hypre_MPI_Request *request )
{
   tmpi_req *r;
   do_send(buf, count, datatype, dest, tag, comm);   /* eager: complete on return */
   *request = req_alloc();
   r = req_get(*request);
   r->kind = 1;
   return 0;
}

HYPRE_Int hypre_MPI_Irsend( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                            HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm,
                            hypre_MPI_Request *request )
{
   return hypre_MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

HYPRE_Int hypre_MPI_Irecv( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                           HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm,
                           hypre_MPI_Request *request )
{
   tmpi_req *r;
   /* post immediately so message matching follows posting order */
   tmpi_pending *p = post_recv(buf, count, datatype, source, tag, comm);
   *request = req_alloc();
   r = req_get(*request);
   r->kind = 2; r->p = p;
   return 0;
}

static void req_init_persistent(int kind, void *buf, HYPRE_Int count,
                                hypre_MPI_Datatype dt, HYPRE_Int peer, HYPRE_Int tag,
                                hypre_MPI_Comm comm, hypre_MPI_Request *request)
{
   tmpi_req *r;
   *request = req_alloc();
   r = req_get(*request);
   r->kind = kind; r->active = 0; r->p = NULL;
   r->buf = buf; r->count = (int) count; r->dt = (int) dt;
   r->peer = (int) peer; r->tag = (int) tag; r->comm = (int) comm;
}

HYPRE_Int hypre_MPI_Send_init( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                               HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm,
                               hypre_MPI_Request *request )
{
   req_init_persistent(3, buf, count, datatype, dest, tag, comm, request);
   return 0;
}

HYPRE_Int hypre_MPI_Recv_init( void *buf, HYPRE_Int count, hypre_MPI_Datatype datatype,
                               HYPRE_Int dest, HYPRE_Int tag, hypre_MPI_Comm comm,
                               hypre_MPI_Request *request )
{
   req_init_persistent(4, buf, count, datatype, dest, tag, comm, request);
   return 0;
}

/* Activate persistent requests. Receives are posted before any send goes out so
   that matching still follows posting order. */
HYPRE_Int hypre_MPI_Startall( HYPRE_Int count, hypre_MPI_Request *array_of_requests )
{
   HYPRE_Int i;
   tmpi_req *r;

   for (i = 0; i < count; i++)
   {
      r = req_get(array_of_requests[i]);
      if (r && r->kind == 4 && !r->active)
      {
         r->p = post_recv(r->buf, r->count, r->dt, r->peer, r->tag, r->comm);
         r->active = 1;
      }
   }
   for (i = 0; i < count; i++)
   {
      r = req_get(array_of_requests[i]);
      if (r && r->kind == 3 && !r->active)
      {
         do_send(r->buf, r->count, r->dt, r->peer, r->tag, r->comm);
         r->active = 1;
      }
   }
   return 0;
}

HYPRE_Int hypre_MPI_Wait( hypre_MPI_Request *request, hypre_MPI_Status *status )
{
   tmpi_req *r = req_get(*request);
   if (!r) { return 0; }

   if (r->kind == 4)            /* persistent recv: completes, handle stays valid */
   {
      if (r->active) { wait_pending(r->p, status); r->p = NULL; r->active = 0; }
      return 0;
   }
   if (r->kind == 3)            /* persistent send: eager, so already done */
   {
      r->active = 0;
      if (status) { status->hypre_MPI_SOURCE = 0; status->hypre_MPI_TAG = 0; status->hypre_MPI_COUNT = 0; }
      return 0;
   }
   if (r->kind == 2) { wait_pending(r->p, status); r->p = NULL; }
   else if (status)
   {
      status->hypre_MPI_SOURCE = 0; status->hypre_MPI_TAG = 0; status->hypre_MPI_COUNT = 0;
   }
   r->used = 0; r->kind = 0;
   *request = hypre_MPI_REQUEST_NULL;
   return 0;
}

HYPRE_Int hypre_MPI_Waitall( HYPRE_Int count, hypre_MPI_Request *array_of_requests,
                             hypre_MPI_Status *array_of_statuses )
{
   HYPRE_Int i;
   /* matching already happened at post time, so any completion order is safe */
   for (i = 0; i < count; i++) { hypre_MPI_Wait(&array_of_requests[i], NULL); }
   HYPRE_UNUSED_VAR(array_of_statuses);
   return 0;
}

HYPRE_Int hypre_MPI_Waitany( HYPRE_Int count, hypre_MPI_Request *array_of_requests,
                             HYPRE_Int *index, hypre_MPI_Status *status )
{
   HYPRE_Int i;
   for (i = 0; i < count; i++)
   {
      if (array_of_requests[i] != hypre_MPI_REQUEST_NULL)
      {
         hypre_MPI_Wait(&array_of_requests[i], status);
         *index = i;
         return 0;
      }
   }
   *index = hypre_MPI_UNDEFINED;
   return 0;
}

HYPRE_Int hypre_MPI_Test( hypre_MPI_Request *request, HYPRE_Int *flag, hypre_MPI_Status *status )
{
   tmpi_req   *r = req_get(*request);
   tmpi_inbox *b;
   int done;

   if (!r) { *flag = 1; return 0; }
   if (r->kind == 1 || r->kind == 3) { hypre_MPI_Wait(request, status); *flag = 1; return 0; }
   if (r->kind == 4 && !r->active) { *flag = 1; return 0; }

   b = &g_inbox[g_myrank];
   pthread_mutex_lock(&b->mtx);
   done = r->p->done;
   pthread_mutex_unlock(&b->mtx);

   if (done) { hypre_MPI_Wait(request, status); *flag = 1; }
   else { *flag = 0; }
   return 0;
}

HYPRE_Int hypre_MPI_Testall( HYPRE_Int count, hypre_MPI_Request *array_of_requests,
                             HYPRE_Int *flag, hypre_MPI_Status *array_of_statuses )
{
   HYPRE_Int i, f, all = 1;
   for (i = 0; i < count; i++)
   {
      hypre_MPI_Test(&array_of_requests[i], &f, NULL);
      if (!f) { all = 0; }
   }
   *flag = all;
   HYPRE_UNUSED_VAR(array_of_statuses);
   return 0;
}

HYPRE_Int hypre_MPI_Probe( HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm,
                           hypre_MPI_Status *status )
{
   probe_umq(comm, source, tag, 1, status);
   return 0;
}

HYPRE_Int hypre_MPI_Iprobe( HYPRE_Int source, HYPRE_Int tag, hypre_MPI_Comm comm,
                            HYPRE_Int *flag, hypre_MPI_Status *status )
{
   *flag = probe_umq(comm, source, tag, 0, status) ? 1 : 0;
   return 0;
}

HYPRE_Int hypre_MPI_Request_free( hypre_MPI_Request *request )
{
   tmpi_req *r = req_get(*request);
   if (r)
   {
      if ((r->kind == 2 || r->kind == 4) && r->p) { wait_pending(r->p, NULL); r->p = NULL; }
      r->used = 0; r->kind = 0; r->active = 0;
   }
   *request = hypre_MPI_REQUEST_NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * Derived datatypes
 *--------------------------------------------------------------------------*/
static int dt_make(int nblk, const size_t *len, const intptr_t *disp, int absolute)
{
   int h = dt_alloc(), b;
   tmpi_dt *d = &g_dt[h];
   d->nblk = nblk;
   d->len  = (size_t *) malloc(sizeof(size_t) * (size_t) nblk);
   d->disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) nblk);
   d->abs  = absolute;
   d->size = 0;
   for (b = 0; b < nblk; b++)
   {
      d->len[b] = len[b];
      d->disp[b] = disp[b];
      d->size += len[b];
   }
   return h;
}

HYPRE_Int hypre_MPI_Type_contiguous( HYPRE_Int count, hypre_MPI_Datatype oldtype,
                                     hypre_MPI_Datatype *newtype )
{
   size_t len = (size_t) count * dt_size(oldtype);
   intptr_t disp = 0;
   *newtype = dt_make(1, &len, &disp, 0);
   return 0;
}

HYPRE_Int hypre_MPI_Type_vector( HYPRE_Int count, HYPRE_Int blocklength, HYPRE_Int stride,
                                 hypre_MPI_Datatype oldtype, hypre_MPI_Datatype *newtype )
{
   size_t esz = dt_size(oldtype), *len;
   intptr_t *disp;
   HYPRE_Int i;

   len  = (size_t *) malloc(sizeof(size_t) * (size_t) count);
   disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) count);
   for (i = 0; i < count; i++)
   {
      len[i]  = (size_t) blocklength * esz;
      disp[i] = (intptr_t)((size_t) i * (size_t) stride * esz);
   }
   *newtype = dt_make((int) count, len, disp, 0);
   free(len); free(disp);
   return 0;
}

HYPRE_Int hypre_MPI_Type_hvector( HYPRE_Int count, HYPRE_Int blocklength, hypre_MPI_Aint stride,
                                  hypre_MPI_Datatype oldtype, hypre_MPI_Datatype *newtype )
{
   size_t esz = dt_size(oldtype), *len;
   intptr_t *disp;
   HYPRE_Int i;

   len  = (size_t *) malloc(sizeof(size_t) * (size_t) count);
   disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) count);
   for (i = 0; i < count; i++)
   {
      len[i]  = (size_t) blocklength * esz;
      disp[i] = (intptr_t)((size_t) i * (size_t) stride);
   }
   *newtype = dt_make((int) count, len, disp, 0);
   free(len); free(disp);
   return 0;
}

/* hypre builds these with hypre_MPI_Address, i.e. absolute addresses + MPI_BOTTOM */
HYPRE_Int hypre_MPI_Type_struct( HYPRE_Int count, HYPRE_Int *array_of_blocklengths,
                                 hypre_MPI_Aint *array_of_displacements,
                                 hypre_MPI_Datatype *array_of_types,
                                 hypre_MPI_Datatype *newtype )
{
   size_t *len = (size_t *) malloc(sizeof(size_t) * (size_t) count);
   intptr_t *disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) count);
   HYPRE_Int i;

   for (i = 0; i < count; i++)
   {
      len[i]  = (size_t) array_of_blocklengths[i] * dt_size(array_of_types[i]);
      disp[i] = (intptr_t) array_of_displacements[i];
   }
   *newtype = dt_make((int) count, len, disp, 1);
   free(len); free(disp);
   return 0;
}

HYPRE_Int hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   HYPRE_UNUSED_VAR(datatype);
   return 0;
}

HYPRE_Int hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   tmpi_dt *d = dt_get(*datatype);
   if (d)
   {
      pthread_mutex_lock(&g_dt_mtx);
      free(d->len); free(d->disp);
      d->len = NULL; d->disp = NULL; d->used = 0;
      pthread_mutex_unlock(&g_dt_mtx);
   }
   return 0;
}

HYPRE_Int hypre_MPI_Op_free( hypre_MPI_Op *op )
{
   if (*op >= TMPI_FIRST_USER_OP && *op < TMPI_MAX_OPS)
   {
      pthread_mutex_lock(&g_op_mtx);
      g_op[*op].used = 0; g_op[*op].fn = NULL;
      pthread_mutex_unlock(&g_op_mtx);
   }
   return 0;
}

HYPRE_Int hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute,
                               hypre_MPI_Op *op )
{
   int i, h = -1;
   HYPRE_UNUSED_VAR(commute);
   pthread_mutex_lock(&g_op_mtx);
   for (i = TMPI_FIRST_USER_OP; i < TMPI_MAX_OPS; i++)
   {
      if (!g_op[i].used) { h = i; g_op[i].used = 1; g_op[i].fn = (tmpi_user_fn) function; break; }
   }
   pthread_mutex_unlock(&g_op_mtx);
   if (h < 0) { fprintf(stderr, "tmpi: out of op handles\n"); abort(); }
   *op = (hypre_MPI_Op) h;
   return 0;
}

HYPRE_Int hypre_MPI_Info_create( hypre_MPI_Info *info ) { *info = 0; return 0; }
HYPRE_Int hypre_MPI_Info_free( hypre_MPI_Info *info ) { HYPRE_UNUSED_VAR(info); return 0; }

HYPRE_Int hypre_MPI_CheckCommMatrix( hypre_MPI_Comm comm, HYPRE_Int num_recvs, HYPRE_Int *recvs,
                                     HYPRE_Int num_sends, HYPRE_Int *sends )
{
   HYPRE_UNUSED_VAR(comm); HYPRE_UNUSED_VAR(num_recvs); HYPRE_UNUSED_VAR(recvs);
   HYPRE_UNUSED_VAR(num_sends); HYPRE_UNUSED_VAR(sends);
   return 0;
}

#endif /* HYPRE_SEQUENTIAL */
