/* "$Id: general.h,v 1.6 2000/05/05 22:14:40 balay Exp bsmith $"; */

#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "src/vec/is/isimpl.h"
#include "petscsys.h"

typedef struct {
  int        N;         /* number of indices */ 
  int        n;         /* local number of indices */ 
  PetscTruth sorted;    /* indicates the indices are sorted */ 
  int        *idx;
} IS_General;

#endif
