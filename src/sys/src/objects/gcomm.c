/*$Id: gcomm.c,v 1.21 2000/04/12 04:21:29 bsmith Exp bsmith $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectGetComm"
/*@C
   PetscObjectGetComm - Gets the MPI communicator for any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example, 
         PetscObjectGetComm((PetscObject)mat,&comm);

   Output Parameter:
.  comm - the MPI communicator

   Level: advanced

   Concepts: communicator^getting from object
   Concepts: MPI communicator^getting from object

@*/
int PetscObjectGetComm(PetscObject obj,MPI_Comm *comm)
{
  int ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  if (obj->bops->getcomm) {
    ierr = obj->bops->getcomm(obj,comm);CHKERRQ(ierr);
  } else {
    *comm = obj->comm;
  }
  PetscFunctionReturn(0);
}


