!
!  "$Id: petscksp.h,v 1.26 2000/08/18 20:02:55 balay Exp bsmith $";
!
!  Include file for Fortran use of the KSP package in PETSc
!
#if !defined (__PETSCKSP_H)
#define __PETSCKSP_H

#define KSP                PetscFortranAddr
#define KSPCGType          integer
#define KSPType            character*(80)
#define KSPConvergedReason integer 
!
!  Various Krylov subspace methods
!
#define KSPRICHARDSON 'richardson'
#define KSPCHEBYCHEV  'chebychev'
#define KSPCG         'cg'
#define KSPGMRES      'gmres'
#define KSPTCQMR      'tcqmr'
#define KSPBCGS       'bcgs'
#define KSPCGS        'cgs'
#define KSPTFQMR      'tfqmr'
#define KSPCR         'cr'
#define KSPLSQR       'lsqr'
#define KSPPREONLY    'preonly'
#define KSPQCG        'qcg'
#define KSPBICG       'bicg'
#define KSPMINRES     'minres'
#define KSPSYMMLQ     'symmlq'
#endif


#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  CG Types
!
      integer KSP_CG_SYMMETRIC,KSP_CG_HERMITIAN

      parameter (KSP_CG_SYMMETRIC=1,KSP_CG_HERMITIAN=2)

      integer KSP_CONVERGED_RTOL,KSP_CONVERGED_ATOL
      integer KSP_DIVERGED_ITS,KSP_DIVERGED_DTOL
      integer KSP_DIVERGED_BREAKDOWN,KSP_CONVERGED_ITERATING
      integer KSP_CONVERGED_QCG_NEG_CURVE
      integer KSP_CONVERGED_QCG_CONSTRAINED
      integer KSP_CONVERGED_STEP_LENGTH
      integer KSP_DIVERGED_BREAKDOWN_BICG
      integer KSP_DIVERGED_NONSYMMETRIC
      integer KSP_DIVERGED_INDEFINITE_PC

      parameter (KSP_CONVERGED_RTOL      = 2)
      parameter (KSP_CONVERGED_ATOL      = 3)
      parameter (KSP_CONVERGED_QCG_NEG_CURVE = 5)
      parameter (KSP_CONVERGED_QCG_CONSTRAINED = 6)
      parameter (KSP_CONVERGED_STEP_LENGTH = 7)

      parameter (KSP_DIVERGED_ITS        = -3)
      parameter (KSP_DIVERGED_DTOL       = -4)
      parameter (KSP_DIVERGED_BREAKDOWN  = -5)
      parameter (KSP_DIVERGED_BREAKDOWN_BICG = -6)
      parameter (KSP_DIVERGED_NONSYMMETRIC = -7)
      parameter (KSP_DIVERGED_INDEFINITE_PC = -8)

      parameter (KSP_CONVERGED_ITERATING = 0)
!
!
!   Possible arguments to KSPSetMonitor()
!
      external KSPDEFAULTCONVERGED

      external KSPDEFAULTMONITOR
      external KSPTRUEMONITOR
      external KSPLGMONITOR
      external KSPLGTRUEMONITOR
      external KSPVECVIEWMONITOR
      external KSPSINGULARVALUEMONITOR

!  End of Fortran include file for the KSP package in PETSc

#endif







