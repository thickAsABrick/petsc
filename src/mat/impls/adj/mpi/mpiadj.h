/* $Id: mpiadj.h,v 1.2 1997/09/26 02:19:39 bsmith Exp bsmith $ */

#include "src/mat/matimpl.h"

#if !defined(__ADJ_H)
#define __ADJ_H

/*  
  MATMPIADJ format - Compressed row storage for storing adjacency lists, but no 
                     matrix values. This is for grid reorderings (to reduce bandwidth)
                     grid partitionings, etc. This is NOT currently a dynamic data-structure.
                     
*/

typedef struct {
  int              *rowners;         /* ranges owned by each processor */
  int              rstart, rend;     /* start and end of local rows */
  int              m;                /* local rows */
  int              nz;
  int              *diag;            /* pointers to diagonal elements, if they exist */
  int              *i;               /* pointer to beginning of each row */
  int              *j;               /* column values: j + i[k] - 1 is start of row k */
  PetscTruth       symmetric;        /* user indicates the nonzero structure is symmetric */
} Mat_MPIAdj;

#endif
