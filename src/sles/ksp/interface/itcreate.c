#ifndef lint
static char vcid[] = "$Id: itcreate.c,v 1.69 1996/01/03 16:11:18 curfman Exp curfman $";
#endif
/*
     The basic KSP routines, Create, View etc. are here.
*/
#include "petsc.h"
#include "kspimpl.h"      /*I "ksp.h" I*/
#include <stdio.h>
#include "sys/nreg.h"     /*I "sys/nreg.h" I*/
#include "sys.h"
#include "viewer.h"       /*I "viewer.h" I*/
#include "pinclude/pviewer.h"

/*@ 
   KSPView - Prints the KSP data structure.

   Input Parameters:
.  ksp - the Krylov space context
.  viewer - visualization context

   Note:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: KSP, view

.seealso: PCView(), ViewerFileOpenASCII()
@*/
int KSPView(KSP ksp,Viewer viewer)
{
  PetscObject vobj = (PetscObject) viewer;
  FILE        *fd;
  char        *method;
  int         ierr;

  if (vobj->cookie == VIEWER_COOKIE && (vobj->type == ASCII_FILE_VIEWER ||
                                        vobj->type == ASCII_FILES_VIEWER)) {
    ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
    MPIU_fprintf(ksp->comm,fd,"KSP Object:\n");
    KSPGetType(ksp,PETSC_NULL,&method);
    MPIU_fprintf(ksp->comm,fd,"  method: %s\n",method);
    if (ksp->view) (*ksp->view)((PetscObject)ksp,viewer);
    if (ksp->guess_zero) MPIU_fprintf(ksp->comm,fd,
      "  maximum iterations=%d, initial guess is zero\n",ksp->max_it);
    else MPIU_fprintf(ksp->comm,fd,"  maximum iterations=%d\n", ksp->max_it);
    MPIU_fprintf(ksp->comm,fd,
      "  tolerances:  relative=%g, absolute=%g, divergence=%g\n",
      ksp->rtol, ksp->atol, ksp->divtol);
    if (ksp->pc_side == KSP_RIGHT_PC) MPIU_fprintf(ksp->comm,fd,"  right preconditioning\n");
    else if (ksp->pc_side == KSP_SYMMETRIC_PC) MPIU_fprintf(ksp->comm,fd,"  symmetric preconditioning\n");
    else MPIU_fprintf(ksp->comm,fd,"  left preconditioning\n");
  }
  return 0;
}

static NRList *__KSPList = 0;
/*@C
   KSPCreate - Creates the default KSP context.

   Output Parameter:
.  ksp - location to put the KSP context
.  comm - MPI communicator

   Notes:
   The default KSP type is GMRES with a restart of 10.

.keywords: KSP, create, context

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy()
@*/
int KSPCreate(MPI_Comm comm,KSP *ksp)
{
  KSP ctx;

  *ksp = 0;
  PetscHeaderCreate(ctx,_KSP,KSP_COOKIE,KSPGMRES,comm);
  PLogObjectCreate(ctx);
  *ksp               = ctx;
  ctx->view          = 0;
  ctx->prefix        = 0;

  ctx->type          = (KSPType) -1;
  ctx->max_it        = 10000;
  ctx->pc_side       = KSP_LEFT_PC;
  ctx->use_pres      = 0;
  ctx->rtol          = 1.e-5;
  ctx->atol          = 1.e-50;
  ctx->divtol        = 1.e4;

  ctx->guess_zero          = 1;
  ctx->calc_eigs           = 0;
  ctx->calc_res            = 0;
  ctx->residual_history    = 0;
  ctx->res_hist_size       = 0;
  ctx->res_act_size        = 0;
  ctx->monitor             = 0;
  ctx->adjust_work_vectors = 0;
  ctx->converged           = KSPDefaultConverged;
  ctx->buildsolution       = KSPDefaultBuildSolution;
  ctx->buildresidual       = KSPDefaultBuildResidual;

  ctx->vec_sol   = 0;
  ctx->vec_rhs   = 0;
  ctx->B         = 0;

  ctx->solver    = 0;
  ctx->setup     = 0;
  ctx->destroy   = 0;
  ctx->adjustwork= 0;

  ctx->data          = 0;
  ctx->nwork         = 0;
  ctx->work          = 0;

  ctx->monP          = 0;
  ctx->cnvP          = 0;

  ctx->setupcalled   = 0;
  /* this violates our rule about seperating abstract from implementations*/
  return KSPSetType(*ksp,KSPGMRES);
}

/*@
   KSPSetType - Builds KSP for a particular solver. 

   Input Parameter:
.  ctx      - the Krylov space context
.  itmethod - a known method

   Options Database Command:
$  -ksp_type  <method>
$      Use -help for a list of available methods
$      (for instance, cg or gmres)

   Notes:  
   See "petsc/include/ksp.h" for available methods (for instance KSPCG 
   or KSPGMRES).

.keywords: KSP, set, method
@*/
int KSPSetType(KSP ctx,KSPType itmethod)
{
  int ierr,(*r)(KSP);

  PETSCVALIDHEADERSPECIFIC(ctx,KSP_COOKIE);
  if (ctx->setupcalled) {
    /* destroy the old private KSP context */
    ierr = (*(ctx)->destroy)((PetscObject)ctx); CHKERRQ(ierr);
    ctx->data = 0;
  }
  /* Get the function pointers for the iterative method requested */
  if (!__KSPList) {KSPRegisterAll();}
  if (!__KSPList) SETERRQ(1,"KSPSetType:Could not get list of KSP types"); 
  r =  (int (*)(KSP))NRFindRoutine( __KSPList, (int)itmethod, (char *)0 );
  if (!r) {SETERRQ(1,"KSPSetType:Unknown method");}
  if (ctx->data) PetscFree(ctx->data);
  ctx->data = 0;
  return (*r)(ctx);
}

/*@C
   KSPRegister - Adds the iterative method to the KSP package,  given
   an iterative name (KSPType) and a function pointer.

   Input Parameters:
.  name   - for instance KSPCG, KSPGMRES, ...
.  sname  - corresponding string for name
.  create - routine to create method context

.keywords: KSP, register

.seealso: KSPRegisterAll(), KSPRegisterDestroy()
@*/
int  KSPRegister(KSPType name, char *sname, int  (*create)(KSP))
{
  int ierr;
  int (*dummy)(void *) = (int (*)(void *)) create;
  if (!__KSPList) {ierr = NRCreate(&__KSPList); CHKERRQ(ierr);}
  return NRRegister( __KSPList, (int) name, sname, dummy );
}

/*@C
   KSPRegisterDestroy - Frees the list of iterative solvers that were
   registered by KSPRegister().

.keywords: KSP, register, destroy

.seealso: KSPRegister(), KSPRegisterAll()
@*/
int KSPRegisterDestroy()
{
  if (__KSPList) {
    NRDestroy( __KSPList );
    __KSPList = 0;
  }
  return 0;
}

/*
   KSPGetTypeFromOptions_Private - Sets the selected KSP type from 
   the options database.

   Input Parameter:
.  ctx - the KSP context

   Output Parameter:
.  itmethod - iterative method

   Returns:
   Returns 1 if the method is found; 0 otherwise.
*/
int KSPGetTypeFromOptions_Private(KSP ctx,KSPType *itmethod)
{
  char sbuf[50];
  if (OptionsGetString(ctx->prefix,"-ksp_type", sbuf, 50 )) {
    if (!__KSPList) KSPRegisterAll();
    *itmethod = (KSPType)NRFindID( __KSPList, sbuf );
    return 1;
  }
  return 0;
}

/*@C
   KSPGetType - Gets the KSP type and method name (as a string) from 
   the method type.

   Input Parameter:
.  ksp - Krylov context 

   Output Parameters:
.  itmeth - KSP method (or use PETSC_NULL)
.  name - name of KSP method (or use PETSC_NULL)

.keywords: KSP, get, method, name
@*/
int KSPGetType(KSP ksp,KSPType *itmeth,char **name)
{
  int ierr;
  if (!__KSPList) {ierr = KSPRegisterAll(); CHKERRQ(ierr);}
  if (itmeth) *itmeth = (KSPType) ksp->type;
  if (name)  *name = NRFindName( __KSPList, (int) ksp->type);
  return 0;
}

#include <stdio.h>
/*
   KSPPrintTypes_Private - Prints the KSP methods available from the options 
   database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "ksp_type") 
*/
int KSPPrintTypes_Private(char* prefix,char *name)
{
  FuncList *entry;
  if (!__KSPList) {KSPRegisterAll();}
  entry = __KSPList->head;
  fprintf(stdout," %s%s (one of)",prefix,name);
  while (entry) {
    fprintf(stdout," %s",entry->name);
    entry = entry->next;
  }
  fprintf(stdout,"\n");
  return 1;
}
