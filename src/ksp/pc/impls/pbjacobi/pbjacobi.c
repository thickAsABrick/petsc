
/*
   Include files needed for the PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

/*
   Private context (data structure) for the PBJacobi preconditioner.
*/
typedef struct {
  const MatScalar *diag;
  PetscInt        bs,mbs;
} PC_PBJacobi;


static PetscErrorCode PCApply_PBJacobi_1(PC pc,Vec x,Vec y)
{
  PC_PBJacobi       *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          i,m = jac->mbs;
  const MatScalar   *diag = jac->diag;
  const PetscScalar *xx;
  PetscScalar       *yy;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) yy[i] = diag[i]*xx[i];
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void MatMult_2(const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  y[0] = A[0]*x[0] + A[2]*x[1];
  y[1] = A[1]*x[0] + A[3]*x[1];
}

PETSC_STATIC_INLINE void MatMult_3(const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  y[0] = A[0]*x[0] + A[3]*x[1] + A[6]*x[2];
  y[1] = A[1]*x[0] + A[4]*x[1] + A[7]*x[2];
  y[2] = A[2]*x[0] + A[5]*x[1] + A[8]*x[2];
}

PETSC_STATIC_INLINE void MatMult_4(const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  y[0] = A[0]*x[0] + A[4]*x[1] + A[8]*x[2]  + A[12]*x[3];
  y[1] = A[1]*x[0] + A[5]*x[1] + A[9]*x[2]  + A[13]*x[3];
  y[2] = A[2]*x[0] + A[6]*x[1] + A[10]*x[2] + A[14]*x[3];
  y[3] = A[3]*x[0] + A[7]*x[1] + A[11]*x[2] + A[15]*x[3];
}

PETSC_STATIC_INLINE void MatMult_5(const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  y[0] = A[0]*x[0] + A[5]*x[1] + A[10]*x[2] + A[15]*x[3] + A[20]*x[4];
  y[1] = A[1]*x[0] + A[6]*x[1] + A[11]*x[2] + A[16]*x[3] + A[21]*x[4];
  y[2] = A[2]*x[0] + A[7]*x[1] + A[12]*x[2] + A[17]*x[3] + A[22]*x[4];
  y[3] = A[3]*x[0] + A[8]*x[1] + A[13]*x[2] + A[18]*x[3] + A[23]*x[4];
  y[4] = A[4]*x[0] + A[9]*x[1] + A[14]*x[2] + A[19]*x[3] + A[24]*x[4];
}

PETSC_STATIC_INLINE void MatMult_6(const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  y[0] = A[0]*x[0] + A[6]*x[1]  + A[12]*x[2]  + A[18]*x[3] + A[24]*x[4] + A[30]*x[5];
  y[1] = A[1]*x[0] + A[7]*x[1]  + A[13]*x[2]  + A[19]*x[3] + A[25]*x[4] + A[31]*x[5];
  y[2] = A[2]*x[0] + A[8]*x[1]  + A[14]*x[2]  + A[20]*x[3] + A[26]*x[4] + A[32]*x[5];
  y[3] = A[3]*x[0] + A[9]*x[1]  + A[15]*x[2]  + A[21]*x[3] + A[27]*x[4] + A[33]*x[5];
  y[4] = A[4]*x[0] + A[10]*x[1] + A[16]*x[2]  + A[22]*x[3] + A[28]*x[4] + A[34]*x[5];
  y[5] = A[5]*x[0] + A[11]*x[1] + A[17]*x[2]  + A[23]*x[3] + A[29]*x[4] + A[35]*x[5];
}

PETSC_STATIC_INLINE void MatMult_7(const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  y[0] = A[0]*x[0] + A[7]*x[1]  + A[14]*x[2]  + A[21]*x[3] + A[28]*x[4] + A[35]*x[5] + A[42]*x[6];
  y[1] = A[1]*x[0] + A[8]*x[1]  + A[15]*x[2]  + A[22]*x[3] + A[29]*x[4] + A[36]*x[5] + A[43]*x[6];
  y[2] = A[2]*x[0] + A[9]*x[1]  + A[16]*x[2]  + A[23]*x[3] + A[30]*x[4] + A[37]*x[5] + A[44]*x[6];
  y[3] = A[3]*x[0] + A[10]*x[1] + A[17]*x[2]  + A[24]*x[3] + A[31]*x[4] + A[38]*x[5] + A[45]*x[6];
  y[4] = A[4]*x[0] + A[11]*x[1] + A[18]*x[2]  + A[25]*x[3] + A[32]*x[4] + A[39]*x[5] + A[46]*x[6];
  y[5] = A[5]*x[0] + A[12]*x[1] + A[19]*x[2]  + A[26]*x[3] + A[33]*x[4] + A[40]*x[5] + A[47]*x[6];
  y[6] = A[6]*x[0] + A[13]*x[1] + A[20]*x[2]  + A[27]*x[3] + A[34]*x[4] + A[41]*x[5] + A[48]*x[6];
}

PETSC_STATIC_INLINE void MatMult_N(PetscInt N, const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  PetscInt i, j;

  for (i = 0; i < N; ++i) {
    PetscScalar rowsum = 0.;
    for (j = 0; j < N; ++j) {
      rowsum += A[i+j*N] * x[j];
    }
    y[i] = rowsum;
  }
}

static PetscErrorCode PCApply_PBJacobi_2(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_2(diag, &xx[2*i], &yy[2*i]);
    diag     += 4;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(6.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_3(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_3(diag, &xx[3*i], &yy[3*i]);
    diag += 9;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(15.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_4(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_4(diag, &xx[4*i], &yy[4*i]);
    diag += 16;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(28.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_5(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_5(diag, &xx[5*i], &yy[5*i]);
    diag += 25;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(45.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_6(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_6(diag, &xx[6*i], &yy[6*i]);
    diag += 36;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(66.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_7(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_7(diag, &xx[7*i], &yy[7*i]);
    diag += 49;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(91*m);CHKERRQ(ierr); /* 2*bs2 - bs */
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_N(PC pc,Vec x,Vec y)
{
  PC_PBJacobi       *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          i;
  const PetscInt    m = jac->mbs;
  const PetscInt    bs = jac->bs;
  const MatScalar   *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    MatMult_N(bs, diag, &xx[bs*i], &yy[bs*i]);
    diag += bs*bs;
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*bs*bs-bs);CHKERRQ(ierr); /* 2*bs2 - bs */
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCSetUp_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  Mat            A = pc->pmat;
  MatFactorError err;
  PetscInt       nlocal;
  
  PetscFunctionBegin;
  ierr = MatInvertBlockDiagonal(A,&jac->diag);CHKERRQ(ierr);
  ierr = MatFactorGetError(A,&err);CHKERRQ(ierr);
  if (err) pc->failedreason = (PCFailedReason)err;
 
  ierr = MatGetBlockSize(A,&jac->bs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&nlocal,NULL);CHKERRQ(ierr);
  jac->mbs = nlocal/jac->bs;
  switch (jac->bs) {
  case 1:
    pc->ops->apply = PCApply_PBJacobi_1;
    break;
  case 2:
    pc->ops->apply = PCApply_PBJacobi_2;
    break;
  case 3:
    pc->ops->apply = PCApply_PBJacobi_3;
    break;
  case 4:
    pc->ops->apply = PCApply_PBJacobi_4;
    break;
  case 5:
    pc->ops->apply = PCApply_PBJacobi_5;
    break;
  case 6:
    pc->ops->apply = PCApply_PBJacobi_6;
    break;
  case 7:
    pc->ops->apply = PCApply_PBJacobi_7;
    break;
  default:
    pc->ops->apply = PCApply_PBJacobi_N;
    break;
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCDestroy_PBJacobi(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_PBJacobi(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  point-block size %D\n",jac->bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCPBJACOBI - Point block Jacobi preconditioner


   Notes: See PCJACOBI for point Jacobi preconditioning

   This works for AIJ and BAIJ matrices and uses the blocksize provided to the matrix

   Uses dense LU factorization with partial pivoting to invert the blocks; if a zero pivot
   is detected a PETSc error is generated.

   Developer Notes: This should support the PCSetErrorIfFailure() flag set to PETSC_TRUE to allow
   the factorization to continue even after a zero pivot is found resulting in a Nan and hence
   terminating KSP with a KSP_DIVERGED_NANORIF allowing
   a nonlinear solver/ODE integrator to recover without stopping the program as currently happens.

   Developer Note: Perhaps should provide an option that allows generation of a valid preconditioner
   even if a block is singular as the PCJACOBI does.

   Level: beginner

  Concepts: point block Jacobi


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCJACOBI

M*/

PETSC_EXTERN PetscErrorCode PCCreate_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr     = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  pc->data = (void*)jac;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag = 0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = 0; /*set depending on the block size */
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_PBJacobi;
  pc->ops->destroy             = PCDestroy_PBJacobi;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = PCView_PBJacobi;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}


