#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sor.c,v 1.74 1998/10/19 22:17:22 bsmith Exp bsmith $";
#endif

/*
   Defines a  (S)SOR  preconditioner for any Mat implementation
*/
#include "src/pc/pcimpl.h"               /*I "pc.h" I*/

typedef struct {
  int        its;        /* inner iterations, number of sweeps */
  MatSORType sym;        /* forward, reverse, symmetric etc. */
  double     omega;
} PC_SOR;

#undef __FUNC__  
#define __FUNC__ "PCApply_SOR"
static int PCApply_SOR(PC pc,Vec x,Vec y)
{
  PC_SOR *jac = (PC_SOR *) pc->data;
  int    ierr, flag = jac->sym | SOR_ZERO_INITIAL_GUESS;

  PetscFunctionBegin;
  ierr = MatRelax(pc->pmat,x,jac->omega,(MatSORType)flag,0.0,jac->its,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApplyRichardson_SOR"
static int PCApplyRichardson_SOR(PC pc,Vec b,Vec y,Vec w,int its)
{
  PC_SOR *jac = (PC_SOR *) pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatRelax(pc->mat,b,jac->omega,(MatSORType)jac->sym,0.0,its,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_SOR"
static int PCSetFromOptions_SOR(PC pc)
{
  int    its,ierr,flg;
  double omega;

  PetscFunctionBegin;
  ierr = OptionsGetDouble(pc->prefix,"-pc_sor_omega",&omega,&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetOmega(pc,omega);CHKERRQ(ierr);} 
  ierr = OptionsGetInt(pc->prefix,"-pc_sor_its",&its,&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetIterations(pc,its);CHKERRQ(ierr);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_symmetric",&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetSymmetric(pc,SOR_SYMMETRIC_SWEEP);CHKERRQ(ierr);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_backward",&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetSymmetric(pc,SOR_BACKWARD_SWEEP);CHKERRQ(ierr);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_local_symmetric",&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetSymmetric(pc,SOR_LOCAL_SYMMETRIC_SWEEP);CHKERRQ(ierr);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_local_backward",&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetSymmetric(pc,SOR_LOCAL_BACKWARD_SWEEP);CHKERRQ(ierr);}
  ierr = OptionsHasName(pc->prefix,"-pc_sor_local_forward",&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCSORSetSymmetric(pc,SOR_LOCAL_FORWARD_SWEEP);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_SOR" 
static int PCPrintHelp_SOR(PC pc,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(pc->comm," Options for PCSOR preconditioner:\n");
  (*PetscHelpPrintf)(pc->comm," %spc_sor_omega <omega>: relaxation factor (0 < omega < 2)\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_sor_symmetric: use SSOR\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_sor_backward: use backward sweep instead of forward\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_sor_local_symmetric: use SSOR on each processor\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_sor_local_backward: use backward sweep locally\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_sor_local_forward: use forward sweep locally\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_sor_its <its>: number of inner SOR iterations to use\n",p);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_SOR"
static int PCView_SOR(PC pc,Viewer viewer)
{
  PC_SOR     *jac = (PC_SOR *) pc->data;
  FILE       *fd;
  MatSORType sym = jac->sym;
  char       *sortype;
  int        ierr;
  ViewerType vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (!PetscStrcmp(vtype,ASCII_VIEWER)) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (sym & SOR_ZERO_INITIAL_GUESS) PetscFPrintf(pc->comm,fd,"    SOR:  zero initial guess\n");
    if (sym == SOR_APPLY_UPPER)              sortype = "apply_upper";
    else if (sym == SOR_APPLY_LOWER)         sortype = "apply_lower";
    else if (sym & SOR_EISENSTAT)            sortype = "Eisenstat";
    else if ((sym & SOR_SYMMETRIC_SWEEP) == SOR_SYMMETRIC_SWEEP)
                                             sortype = "symmetric";
    else if (sym & SOR_BACKWARD_SWEEP)       sortype = "backward";
    else if (sym & SOR_FORWARD_SWEEP)        sortype = "forward";
    else if ((sym & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP)
                                             sortype = "local_symmetric";
    else if (sym & SOR_LOCAL_FORWARD_SWEEP)  sortype = "local_forward";
    else if (sym & SOR_LOCAL_BACKWARD_SWEEP) sortype = "local_backward"; 
    else                                     sortype = "unknown";
    PetscFPrintf(pc->comm,fd,"    SOR: type = %s, iterations = %d, omega = %g\n",
                 sortype,jac->its,jac->omega);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCSORSetSymmetric_SOR"
int PCSORSetSymmetric_SOR(PC pc, MatSORType flag)
{
  PC_SOR *jac;

  PetscFunctionBegin;
  jac = (PC_SOR *) pc->data; 
  jac->sym = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCSORSetOmega_SOR"
int PCSORSetOmega_SOR(PC pc, double omega)
{
  PC_SOR *jac;

  PetscFunctionBegin;
  if (omega >= 2.0 || omega <= 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Relaxation out of range");
  jac        = (PC_SOR *) pc->data; 
  jac->omega = omega;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCSORSetIterations_SOR"
int PCSORSetIterations_SOR(PC pc, int its)
{
  PC_SOR *jac;

  PetscFunctionBegin;
  jac      = (PC_SOR *) pc->data; 
  jac->its = its;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PCSORSetSymmetric"
/*@
   PCSORSetSymmetric - Sets the SOR preconditioner to use symmetric (SSOR), 
   backward, or forward relaxation.  The local variants perform SOR on
   each processor.  By default forward relaxation is used.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - one of the following
.vb
    SOR_FORWARD_SWEEP
    SOR_BACKWARD_SWEEP
    SOR_SYMMETRIC_SWEEP
    SOR_LOCAL_FORWARD_SWEEP
    SOR_LOCAL_BACKWARD_SWEEP
    SOR_LOCAL_SYMMETRIC_SWEEP
.ve

   Options Database Keys:
.  -pc_sor_symmetric - Activates symmetric version
.  -pc_sor_backward - Activates backward version
.  -pc_sor_local_forward - Activates local forward version
.  -pc_sor_local_symmetric - Activates local symmetric version
.  -pc_sor_local_backward - Activates local backward version

   Notes: 
   To use the Eisenstat trick with SSOR, employ the PCEISENSTAT preconditioner,
   which can be chosen with the option 
.  -pc_type eisenstat - Activates Eisenstat trick

.keywords: PC, SOR, SSOR, set, relaxation, sweep, forward, backward, symmetric

.seealso: PCEisenstatSetOmega(), PCSORSetIterations(), PCSORSetOmega()
@*/
int PCSORSetSymmetric(PC pc, MatSORType flag)
{
  int ierr, (*f)(PC,MatSORType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSORSetSymmetric_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSORSetOmega"
/*@
   PCSORSetOmega - Sets the SOR relaxation coefficient, omega
   (where omega = 1.0 by default).

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  omega - relaxation coefficient (0 < omega < 2). 

   Options Database Key:
.  -pc_sor_omega <omega> - Sets omega

.keywords: PC, SOR, SSOR, set, relaxation, omega

.seealso: PCSORSetSymmetric(), PCSORSetIterations(), PCEisenstatSetOmega()
@*/
int PCSORSetOmega(PC pc, double omega)
{
  int ierr, (*f)(PC,double);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSORSetOmega_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,omega);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSORSetIterations"
/*@
   PCSORSetIterations - Sets the number of inner iterations to 
   be used by the SOR preconditioner. The default is 1.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  its - number of iterations to use

   Options Database Key:
.  -pc_sor_its <its> - Sets number of iterations

.keywords: PC, SOR, SSOR, set, iterations

.seealso: PCSORSetOmega(), PCSORSetSymmetric()
@*/
int PCSORSetIterations(PC pc, int its)
{
  int ierr, (*f)(PC,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSORSetIterations_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,its);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_SOR"
int PCCreate_SOR(PC pc)
{
  int    ierr;
  PC_SOR *jac   = PetscNew(PC_SOR); CHKPTRQ(jac);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_SOR));

  pc->apply          = PCApply_SOR;
  pc->applyrich      = PCApplyRichardson_SOR;
  pc->setfromoptions = PCSetFromOptions_SOR;
  pc->printhelp      = PCPrintHelp_SOR;
  pc->setup          = 0;
  pc->data           = (void *) jac;
  pc->view           = PCView_SOR;
  jac->sym           = SOR_FORWARD_SWEEP;
  jac->omega         = 1.0;
  jac->its           = 1;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSORSetSymmetric_C","PCSORSetSymmetric_SOR",
                    (void*)PCSORSetSymmetric_SOR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSORSetOmega_C","PCSORSetOmega_SOR",
                    (void*)PCSORSetOmega_SOR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSORSetIterations_C","PCSORSetIterations_SOR",
                    (void*)PCSORSetIterations_SOR);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


