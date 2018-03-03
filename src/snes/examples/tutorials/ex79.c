static char help[] = "The variable-viscosity Stokes Problem in 2d and 3d with finite elements.\n\
We solve the regional mantle convection problem for a subducting slab\n\
modeled by the variable-viscosity Stokes problem by Margarete Jadamec.\n\
This is meant to take in the same input as CitcomCU from Shijie Zhong.\n\n";

/* We discretize the variable-viscosity Stokes problem using the finite element method on an unstructured mesh. The weak form equations are
\begin{align*}
  (\nabla v, \mu (\nabla u + {\nabla u}^T)) - (\nabla\cdot v, p) + (v, f) &= 0 \\
  (q, \nabla\cdot u)                                                      &= 0
\end{align*}
Free slip conditions for velocity are enforced on every wall. The pressure is
constrained to have zero integral over the domain.

To produce nice output, use

  -dm_view hdf5:mantle.h5 -sol_vec_view hdf5:mantle.h5::append -initial_vec_view hdf5:mantle.h5::append -temp_fine_view hdf5:mantle.h5::append -viscosity_vec_view hdf5:mantle.h5::append

and to get fields at each solver iterate

  -dmsnes_solution_vec_view hdf5:mantle.h5::append -dmsnes_residual_vec_view hdf5:mantle.h5::append

and to get temperature on different levels (averaged temp on level 0 and injected temp on level 1)

  -dm_s_l0_view hdf5:mantle0.h5 -temp_s_l0_view hdf5:mantle0.h5::append -dm_r_l1_view hdf5:mantle1.h5 -temp_r_l1_view hdf5:mantle1.h5::append

Testing Solver:

./ex69  -sol_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf4 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -aux_0_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_velocity_ksp_monitor_no -fieldsplit_velocity_ksp_converged_reason_no -fieldsplit_pressure_ksp_monitor -fieldsplit_pressure_ksp_converged_reason -ksp_rtol 1e-8 -fieldsplit_pressure_ksp_rtol 1e-3 -fieldsplit_pressure_pc_type lu -snes_max_it 1 -snes_error_if_not_converged 0 -snes_view -petscds_jac_pre 1

Testing Jacobian:

./ex69  -sol_type test3 -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -aux_0_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason -fieldsplit_velocity_ksp_monitor -fieldsplit_velocity_ksp_converged_reason -fieldsplit_velocity_ksp_view_pmat -fieldsplit_pressure_ksp_monitor -fieldsplit_pressure_ksp_converged_reason -fieldsplit_pressure_ksp_view_pmat -snes_type test -petscds_jac_pre 0 -snes_test_display 1

Citcom performance:
 1250 x 850 on 900 steps (3.5h per 100 timeteps)

Citcom input:
 1249 x 481 vertices
 Domain: x: 0 - 45deg z: 0 - 2500km

 *_vects.ascii
 Nx Ny Nz
 <Nx values in co-latitude = 90 - degrees latitude>
 <Ny values in degrees longitude>
 <Nz values in non-dimensionalized by the radius of the Earth R = 6.371137 10^6 m, and these are ordered bottom to top>

 *_therm.bin
  Temperature is non-dimensionalized [0 (top), 1 (bottom)]
  The ordering is Y, X, Z where Z is the fastest dimension
  X is lat, counts N to S
  Y is long, counts W to E
  Z is depth, count bottom to top
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscsf.h>
#include <petscbag.h>
#include <petsc/private/petscimpl.h> /* For PetscObjectComposedDataRegister() */

typedef enum {NONE, BASIC, ANALYTIC_0, NUM_SOLUTION_TYPES} SolutionType;
const char *solutionTypes[NUM_SOLUTION_TYPES+1] = {"none", "basic", "analytic_0", "unknown"};
typedef enum {CONSTANT, LINEAR_T, EXP_T, EXP_INVT, TEST, TEST2, TEST3, TEST4, TEST5, TEST6, TEST7, DIFFUSION, DISLOCATION, COMPOSITE, NUM_RHEOLOGY_TYPES} RheologyType;
/*
  constant:    mu = 1
  linear_t:    mu = 2 - T
  exp_t:       mu = 2 - exp(T)
  exp_invt:    mu = 1 + exp(1/(1 + T))
  test:        mu = 1/2 |u|^2
  test2:       mu = 1/2 |u_x|^2_F
  test3:       mu = eps_II
  test4:       mu = eps^{(1-n)/n}_II
  test5:       mu = mu_ds*mu_ds/mu0
  test6:       mu = mu_df/(mu_ds + mu_df)
  test7:       mu = broken
  diffusion:   mu = mu_df
  dislocation: mu = mu_ds
  composite:   mu = 2.0*mu_df*mu_ds/(mu_df + mu_ds)
 */
const char *rheologyTypes[NUM_RHEOLOGY_TYPES+1] = {"constant", "linear_t", "exp_t", "exp_invt", "test", "test2", "test3", "test4", "test5", "test6", "test7", "diffusion", "dislocation", "composite", "unknown"};
typedef enum {FREE_SLIP, DIRICHLET, NUM_BC_TYPES} BCType;
const char *bcTypes[NUM_BC_TYPES+1] = {"free_slip", "dirichlet", "unknown"};

static PetscInt MEAN_EXP_TAG;

typedef struct {
  PetscInt      debug;             /* The debugging level */
  PetscBool     byte_swap;         /* Flag to byte swap on temperature input */
  PetscBool     showError;
  /* Domain and mesh definition */
  DM            cdm;               /* The original serial coarse mesh */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     simplex;           /* Use simplices or tensor product cells */
  PetscInt      refine;            /* Number of parallel refinements after the mesh gets distributed */
  PetscInt      coarsen;           /* Number of parallel coarsenings after the mesh gets distributed */
  char          mantleBasename[PETSC_MAX_PATH_LEN];
  int           verts[3];          /* The number of vertices in each dimension for mantle problems */
  int           perm[3] ;          /* The permutation of axes for mantle problems */
  /* Input distribution and interpolation */
  PetscSF       pointSF;           /* The SF describing mesh distribution */
  Vec           Tinit;             /* The initial, serial non-dimensional temperature distribution */
  PetscReal     meanExp;           /* The exponent to use for the power mean */
  /* Problem definition */
  BCType        bcType;            /* The type of boundary conditions */
  RheologyType  muTypePre;         /* The type of rheology for the main problem */
  SolutionType  solTypePre;        /* The type of solution for the presolve */
  RheologyType  muType;            /* The type of rheology for the main problem */
  SolutionType  solType;           /* The type of solution for the main solve */
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);
  PetscErrorCode (**initialGuess)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx);
} AppCtx;

static PetscErrorCode zero_scalar(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}
static PetscErrorCode one_scalar(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return 0;
}
static PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

/*
  Our first 2D exact solution is quadratic:

    u = x^2 + y^2
    v = 2 x^2 - 2xy
    p = x + y - 1 over [0, 1/2] x [1/2, 1]

  so that

    \nabla \cdot u = 2x - 2x = 0

  For constant viscosity, we have

    -\nabla\cdot mu/2 (\nabla u + \nabla u^T) + \nabla p + f = 0
    -\nabla\cdot mu / / 2x  4x-2y \ + /   2x   2y \  \ + <1 , 1> + f = 0
                 2  \ \ 2y   -2x  /   \ 4x-2y -2x /  /
    mu / -2 \ + <1, 1> + f = 0
       \ -2 /

  so that fx = 2mu - 1 and f_y = 2mu - 1.

  If the viscosity is depth-dependent, we have

    -\nabla\cdot / 2 mu x  2 mu x \ + <1, 1> + f = 0
                 \ 2 mu x -2 mu x /

    < -2 mu - 2 dmu/dy x,  -2 mu + 2 dmu/dy x > + <1, 1> + f = 0

  so that f = <2 mu + 2 dmu/dy x - 1, 2 mu - 2 dmu/dy x - 1>

  For Arrenhius behavior, we have to assume a temperature field. We will let the temperature be

    T(y) = 1 - gamma y
    dT/y = -gamma

  For linear rheology, we will have

    mu(y)  = 2 - T = 2 - (1 - gamma y) = 1 + gamma y
    dmu/dy = gamma

  For diffusion creep, the viscosity will be

   mu(y) = A exp((E + Pl(y) V)/(nR (T(y) + T_ad(y))))
  dmu/dy = mu ( (dPl/dy V)/(nR (T + T_ad)) - (dT/dy + dT_ad/dy) (E + Pl V)/(nR (T + T_ad))^2 )

  and be careful to keep the domain entirely in the lower mantle.

*/
PetscErrorCode analytic_2d_0_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
  return 0;
}

PetscErrorCode analytic_2d_0_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0] + x[1] - 1.0;
  return 0;
}

PetscErrorCode analytic_2d_0_T(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *T, void *ctx)
{
#ifdef SIMPLIFY
  *T = 1. - 0.0*x[dim-1];
#else
  *T = 1. - 0.3*x[dim-1];
#endif
  return 0;
}

PetscErrorCode analytic_2d_0_dT_dr(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *dT, void *ctx)
{
#ifdef SIMPLIFY
  *dT = -0.0;
#else
  *dT = -0.3;
#endif
  return 0;
}

/* Returns the lithostatic pressure in Pa */
static PetscScalar LithostaticPressure(PetscInt dim, const PetscReal x[], PetscReal R_E, PetscReal rho0, PetscReal beta)
{
  const PetscReal  g  = 9.8;          /* Acceleration due to gravity m/s^2 */
#ifdef SPHERICAL
  const PetscReal  r   = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional radius */
#else
  const PetscReal  r   = x[dim-1];                                                     /* Nondimensional radius */
#endif
  const PetscReal  z   = R_E*(1. - r); /* Depth m */
  const PetscReal  P_l = (-1./(1000.*beta))*log(1.-rho0*g*beta*z);   /* Lithostatic pressure kPa = kJ/m^3 */

#ifdef SIMPLIFY
  return (-1./(1000.*beta))*log(1.-rho0*g*beta*R_E*(1. - 0.7));
#else
  return P_l;
#endif
}

/* We get the derivative with respect to the dimensional coordinate z, not the non-dimensional one */
static PetscScalar LithostaticPressureDerivativeR(PetscInt dim, const PetscReal x[], PetscReal R_E, PetscReal rho0, PetscReal beta)
{
  const PetscReal  g  = 9.8;          /* Acceleration due to gravity m/s^2 */
#ifdef SPHERICAL
  const PetscReal  r   = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional radius */
#else
  const PetscReal  r   = x[dim-1];                                                     /* Nondimensional radius */
#endif
  const PetscReal  z   = R_E*(1. - r); /* Depth m */

#ifdef SIMPLIFY
  return 0.0;
#else
  return -rho0*g*beta/(1000.*beta*(1.-rho0*g*beta*z));
#endif
}

/* Assume 10 Pa/m gradient */
static PetscErrorCode dynamic_pressure(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscScalar *constants = (const PetscScalar *) ctx;
  const PetscReal    mu0       = constants[0]; /* Mantle viscosity kg/(m s)  : Mass Scale */
  const PetscReal    R_E       = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal    kappa     = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
#ifdef SPHERICAL
  const PetscReal    r         = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional radius */
#else
  const PetscReal    r         = x[dim-1];                                                     /* Nondimensional radius */
#endif
  const PetscReal    z         = R_E*r;        /* Height m */
  const PetscReal    gradP     = 905.0;        /* Pressure gradient in Pa/m, should be rho0*g*alpha*DeltaT, I get 905 */
  const PetscReal    P_d       = constants[8] * gradP*z;

  /*
   Pa/m = kg /m^2 s^2
   rho0 * g * alpha = kg/m^2 s^2 1/K
   */

  u[0] = P_d * (R_E*R_E / (mu0*kappa));
  return 0;
}

/* The second invariant of a tensor, according to mathematicians:

   A_{II} = 1/2 (Tr(A)^2 - Tr(A^2))
   A_{ij} = 1/2 (u_{x,ij} + u_{x,ji})
2D: It is the determinant
  A       = [[a, b], [c, d]]
  A^2     = [[a^2 + bc, ab + bd], [ac + bd, bc + d^2]]
  Tr(A)^2 = a^2 + 2ad + d^2
  Tr(A^2) = a^2 + 2bc + d^2
  A_{II}  = ad - bc
3D:
  A       = [[a, b, c], [d, e, f], [g, h, i]]
  A^2     = [[a^2 + bd + cg, ?, ?], [?, bd + e^2 + fh, ?], [?, ?, cg + fh + i^2]]
  Tr(A)^2 = a^2 + e^2 + i^2 + 2ae + 2ai + 2ei
  Tr(A^2) = a^2 + e^2 + i^2 + 2bd + 2cg + 2fh
  A_{II}  = ae + ai + ei - bd + cg + fh
*/
#if 0
static PetscReal SecondInvariantSymmetricMath(PetscInt dim, const PetscScalar u_x[])
{
  switch (dim) {
  case 2:
    return PetscRealPart(u_x[0]*u_x[3] - u_x[1]*u_x[2]);
  case 3:
    return PetscRealPart(u_x[0]*u_x[4] + u_x[0]*u_x[8] + u_x[4]*u_x[8] - u_x[1]*u_x[3] - u_x[2]*u_x[6] - u_x[5]*u_x[7]);
  }
  return 0.0;
}
#endif

/* The second invariant of a tensor, according to geophysicists:

   A_{II} = sqrt{1/2 Tr(A^2)}
*/
static PetscReal SecondInvariantSymmetric(PetscInt dim, const PetscScalar A[])
{
  switch (dim) {
  case 2:
    return PetscSqrtReal(0.5*PetscRealPart(A[0]*A[0] + A[1]*A[1] + A[2]*A[2] + A[3]*A[3]));
  case 3:
    return PetscSqrtReal(0.5*PetscRealPart(A[0]*A[0] + A[4]*A[4] + A[8]*A[8] + 2.0*A[1]*A[3] + 2.0*A[2]*A[6] + 2.0*A[5]*A[7]));
  }
  return 0.0;
}

static PetscReal SecondInvariantStress(PetscInt dim, const PetscScalar u_x[])
{
  PetscScalar epsilon[9];
  PetscInt    i, j;

  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      epsilon[i*dim+j] = 0.5*(u_x[i*dim+j] + u_x[j*dim+i]);
    }
  }
  return SecondInvariantSymmetric(dim, epsilon);
}

/* The division between upper and lower mantle, depth in m */
#ifdef SIMPLIFY
#define MOHO 0.0*1000.0
#else
#define MOHO 670.0*1000.0
#endif

/* Assumes that u_x[], x[], and T are dimensionless */
static void MantleViscosity(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], PetscReal R_E, PetscReal kappa, PetscReal DeltaT, PetscReal rho0, PetscReal beta, PetscReal dT_ad_dr, PetscReal T_nondim, PetscReal *epsilon_II, PetscReal *mu_df, PetscReal *mu_ds)
{
#ifdef SPHERICAL
  const PetscReal r       = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional radius */
#else
  const PetscReal r       = x[dim-1];                            /* Nondimensional radius */
#endif
  const PetscReal z       = R_E*(1. - r);                        /* Depth m */
  const PetscBool isUpper = z < MOHO ? PETSC_TRUE : PETSC_FALSE; /* Are we in the upper mantle? */
  const PetscReal T       = DeltaT*T_nondim + 273.0;             /* Temperature K */
  const PetscReal R       = 8.314459848e-3;                      /* Gas constant kJ/K mol */
  const PetscReal d_df_um = 1e4;                                 /* Grain size micrometers in the upper mantle (mum) */
  const PetscReal d_df_lm = 4e4;                                 /* Grain size micrometers in the lower mantle (mum) */
  const PetscReal n_df    = 1.0;                                 /* Stress exponent, dimensionless */
  const PetscReal n_ds    = 3.5;                                 /* Stress exponent, dimensionless */
  const PetscReal C_OH    = 1000.0;                              /* OH concentration, H/10^6 Si, dimensionless */
  const PetscReal E_df    = 335.0;                               /* Activation energy, kJ/mol */
  const PetscReal E_ds    = 480.0;                               /* Activation energy, kJ/mol */
  const PetscReal V_df_um = 4e-6;                                /* Activation volume in the upper mantle, m^3/mol */
  const PetscReal V_df_lm = 1.5e-6;                              /* Activation volume in the upper mantle, m^3/mol */
  const PetscReal V_ds    = 11e-6;                               /* Activation volume, m^3/mol */
  const PetscReal P_l     = LithostaticPressure(dim, x, R_E, rho0, beta); /* Lithostatic pressure, Pa */
  const PetscReal T_ad    = -dT_ad_dr*z;                         /* Adiabatic temperature, K */
  const PetscReal eps_II  = SecondInvariantStress(dim, u_x)*(kappa/PetscSqr(R_E)); /* Second invariant of strain rate, 1/s */
  const PetscReal A_df    = 1.0;                                 /* mum^3 / Pa^n s */
  const PetscReal A_ds    = 9.0e-20;                             /* 1 / Pa^n s */
  const PetscReal pre_df  = isUpper ? PetscPowReal(PetscPowRealInt(d_df_um, 3) / (A_df * C_OH), 1.0/n_df)
                                    : PetscPowReal(PetscPowRealInt(d_df_lm, 3) / (A_df * C_OH), 1.0/n_df); /* Pa s^{1/n} */
  const PetscReal pre_ds  = PetscPowReal(1.0 / (A_ds * PetscPowReal(C_OH, 1.2)), 1.0/n_ds);   /* Pa s^{1/n} : 25886.5 */
  const PetscReal mid_df  = PetscPowReal(eps_II, (1.0 - n_df)/n_df); /* s^{(n-1)/n} */
  const PetscReal mid_ds  = PetscPowReal(eps_II, (1.0 - n_ds)/n_ds); /* s^{(n-1)/n} */
  const PetscReal post_df = isUpper ? PetscExpReal((E_df + P_l * V_df_um)/(n_df * R * (T + T_ad)))
                                    : PetscExpReal((E_df + P_l * V_df_lm)/(n_df * R * (T + T_ad))); /* Dimensionless, (kJ/mol) / (kJ/mol) */
  const PetscReal post_ds = PetscExpReal((E_ds + P_l * V_ds)/(n_ds * R * (T + T_ad))); /* Dimensionless, (kJ/mol) / (kJ/mol) */

  *epsilon_II = eps_II;
  *mu_df = pre_df * mid_df * post_df;   /* Pa s^{1/n} s^{(n-1)/n} = Pa s */
  *mu_ds = eps_II <= 0.0 ? 5e24 : pre_ds * mid_ds * post_ds;   /* Pa s^{1/n} s^{(n-1)/n} = Pa s */
  //PetscPrintf(PETSC_COMM_SELF, "Diffusion mu %g  %s pre: %g mid: %g post: %g T: %g Depth: %g Num: %g Denom: %g\n", *mu_df, isUpper ? "Upper": "Lower",
  //            pre_df, mid_df, post_df, T, z, isUpper ? E_df + P_l * V_df_um : E_df + P_l * V_df_lm, n_df * R * (T + T_ad));
#if 0
  if (*mu_df < 1.e9) PetscPrintf(PETSC_COMM_SELF, "Diffusion   pre: %g mid: %g post: %g T: %g Num: %g Denom: %g\n", pre_df, mid_df, post_df, T, E_df + P_l * V_df, n_df * R * (T + T_ad));
  if (*mu_ds < 1.e9) PetscPrintf(PETSC_COMM_SELF, "Dislocation pre: %g mid: %g post: %g T: %g z: %g\n", pre_ds, mid_ds, post_ds, T, z);
#endif
  return;
}

static PetscReal DiffusionCreepViscosity(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], const PetscReal constants[], PetscReal T)
{
  const PetscReal mu_max   = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min   = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal R_E      = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa    = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT   = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0     = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta     = constants[5]; /* Adiabatic compressibility, 1/Pa */
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  PetscReal       eps_II;                  /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu;        /* Pa s = kg/(m s) */

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = mu_df;
  //if (mu < mu_min) PetscPrintf(PETSC_COMM_SELF, "MIN VIOLATION: %g < %g (%g, %g)\n", mu, mu_min, x[0], x[1]);
  //if (mu > mu_max) PetscPrintf(PETSC_COMM_SELF, "MAX VIOLATION: %g > %g (%g, %g)\n", mu, mu_max, x[0], x[1]);
#ifdef SIMPLIFY
  return mu;
#else
  return PetscMin(mu_max, PetscMax(mu_min, mu));
#endif
}
static void DiffusionCreepViscosityf0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = DiffusionCreepViscosity(dim, u_x, x, constants, PetscRealPart(a[0]));
}

static PetscReal DislocationCreepViscosity(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], const PetscReal constants[], PetscReal T)
{
  const PetscReal mu_max   = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min   = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal R_E      = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa    = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT   = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0     = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta     = constants[5]; /* Adiabatic compressibility, 1/Pa */
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  PetscReal       eps_II;                  /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu;        /* Pa s = kg/(m s) */

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = mu_ds;
  //if (eps_II <= 0.0) PetscPrintf(PETSC_COMM_SELF, "EPS VIOLATION: (%g, %g)\n", x[0], x[1]);
  if (mu < mu_min) PetscPrintf(PETSC_COMM_SELF, "MIN VIOLATION: %g < %g (%g, %g)\n", mu, mu_min, x[0], x[1]);
  if (mu > mu_max) PetscPrintf(PETSC_COMM_SELF, "MAX VIOLATION: %g > %g (%g, %g)\n", mu, mu_max, x[0], x[1]);
  return PetscMin(mu_max, PetscMax(mu_min, mu));
}
static void DislocationCreepViscosityf0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = DislocationCreepViscosity(dim, u_x, x, constants, PetscRealPart(a[0]));
}

static PetscReal CompositeViscosity(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], const PetscReal constants[], PetscReal T)
{
  const PetscReal mu_max   = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min   = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal R_E      = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa    = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT   = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0     = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta     = constants[5]; /* Adiabatic compressibility, 1/Pa */
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  PetscReal       eps_II;                  /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu;        /* Pa s = kg/(m s) */

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = 2.0*mu_ds*mu_df/(mu_ds + mu_df);
  //PetscPrintf(PETSC_COMM_SELF, "Composite mu %g mu_df %g mu_ds %g, eps_II %g T %g\n", mu, mu_df, mu_ds, eps_II, T);
  //if (eps_II <= 0.0) PetscPrintf(PETSC_COMM_SELF, "EPS VIOLATION: (%g, %g)\n", x[0], x[1]);
  if (mu < mu_min) PetscPrintf(PETSC_COMM_SELF, "MIN VIOLATION: %g < %g (%g, %g)\n", mu, mu_min, x[0], x[1]);
  //if (mu > mu_max) PetscPrintf(PETSC_COMM_SELF, "MAX VIOLATION: %g > %g (%g, %g)\n", mu, mu_max, x[0], x[1]);
  return PetscMin(mu_max, PetscMax(mu_min, mu));
}
static void CompositeViscosityf0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = CompositeViscosity(dim, u_x, x, constants, PetscRealPart(a[0]));
}

/* Assumes that u_x[], x[], and T are dimensionless */
static void MantleViscosityDerivativeR(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], PetscReal R_E, PetscReal kappa, PetscReal DeltaT, PetscReal rho0, PetscReal beta, PetscReal dT_ad_dr, PetscReal T_nondim, PetscReal dT_dr_nondim, PetscReal *epsilon_II, PetscReal *mu_df, PetscReal *mu_ds, PetscReal *dmu_df_dr)
{
#ifdef SPHERICAL
  const PetscReal r       = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional radius */
#else
  const PetscReal r       = x[dim-1];                            /* Nondimensional radius */
#endif
  const PetscReal z       = R_E*(1. - r);                        /* Depth m */
  const PetscBool isUpper = z < MOHO ? PETSC_TRUE : PETSC_FALSE; /* Are we in the upper mantle? */
  const PetscReal T       = DeltaT*T_nondim + 273.0;             /* Temperature K */
  const PetscReal dT_dr   = DeltaT*dT_dr_nondim/R_E;             /* Temperature gradient K/m */
  const PetscReal R       = 8.314459848e-3;                      /* Gas constant kJ/K mol */
  const PetscReal d_df_um = 1e4;                                 /* Grain size micrometers in the upper mantle (mum) */
  const PetscReal d_df_lm = 4e4;                                 /* Grain size micrometers in the lower mantle (mum) */
  const PetscReal n_df    = 1.0;                                 /* Stress exponent, dimensionless */
  const PetscReal n_ds    = 3.5;                                 /* Stress exponent, dimensionless */
  const PetscReal C_OH    = 1000.0;                              /* OH concentration, H/10^6 Si, dimensionless */
  const PetscReal E_df    = 335.0;                               /* Activation energy, kJ/mol */
  const PetscReal E_ds    = 480.0;                               /* Activation energy, kJ/mol */
  const PetscReal V_df_um = 4e-6;                                /* Activation volume in the upper mantle, m^3/mol */
  const PetscReal V_df_lm = 1.5e-6;                              /* Activation volume in the upper mantle, m^3/mol */
  const PetscReal V_ds    = 11e-6;                               /* Activation volume, m^3/mol */
  const PetscReal P_l     = LithostaticPressure(dim, x, R_E, rho0, beta); /* Lithostatic pressure, Pa */
  const PetscReal dP_l_dr = LithostaticPressureDerivativeR(dim, x, R_E, rho0, beta); /* Lithostatic pressure derivative, Pa/m */
  const PetscReal T_ad    = -dT_ad_dr*z;                         /* Adiabatic temperature, K */
  const PetscReal eps_II  = SecondInvariantStress(dim, u_x)*(kappa/PetscSqr(R_E)); /* Second invariant of strain rate, 1/s */
  const PetscReal A_df    = 1.0;                                 /* mum^3 / Pa^n s */
  const PetscReal A_ds    = 9.0e-20;                             /* 1 / Pa^n s */
  const PetscReal pre_df  = isUpper ? PetscPowReal(PetscPowRealInt(d_df_um, 3) / (A_df * C_OH), 1.0/n_df)
                                    : PetscPowReal(PetscPowRealInt(d_df_lm, 3) / (A_df * C_OH), 1.0/n_df); /* Pa s^{1/n} */
  const PetscReal pre_ds  = PetscPowReal(1.0 / (A_ds * PetscPowReal(C_OH, 1.2)), 1.0/n_ds);   /* Pa s^{1/n} : 25886.5 */
  const PetscReal mid_df  = 1.0; /* s^{(n-1)/n} */
  const PetscReal mid_ds  = PetscPowReal(eps_II, (1.0 - n_ds)/n_ds); /* s^{(n-1)/n} */
  const PetscReal post_df = isUpper ? PetscExpReal((E_df + P_l * V_df_um)/(n_df * R * (T + T_ad)))
                                    : PetscExpReal((E_df + P_l * V_df_lm)/(n_df * R * (T + T_ad))); /* Dimensionless, (kJ/mol) / (kJ/mol) */
  const PetscReal post_ds = PetscExpReal((E_ds + P_l * V_ds)/(n_ds * R * (T + T_ad))); /* Dimensionless, (kJ/mol) / (kJ/mol) */

  *epsilon_II = eps_II;
  *mu_df = pre_df * mid_df * post_df;   /* Pa s^{1/n} s^{(n-1)/n} = Pa s */
  *mu_ds = eps_II <= 0.0 ? 5e24 : pre_ds * mid_ds * post_ds;   /* Pa s^{1/n} s^{(n-1)/n} = Pa s */
  if (isUpper) {
    *dmu_df_dr  = 0.0;
    *dmu_df_dr += *mu_df * ((dP_l_dr * V_df_um)/(n_df * R * (T + T_ad)));
    *dmu_df_dr -= *mu_df * ((dT_dr + dT_ad_dr)*(E_df + P_l * V_df_um)/PetscSqr(n_df * R * (T + T_ad)));
  } else {
    *dmu_df_dr  = 0.0;
    *dmu_df_dr += *mu_df * ((dP_l_dr * V_df_lm)/(n_df * R * (T + T_ad)));
    *dmu_df_dr -= *mu_df * ((dT_dr + dT_ad_dr)*(E_df + P_l * V_df_lm)/PetscSqr(n_df * R * (T + T_ad)));
  }
#if 0
  if (*mu_df < 1.e9) PetscPrintf(PETSC_COMM_SELF, "Diffusion   pre: %g mid: %g post: %g T: %g Num: %g Denom: %g\n", pre_df, mid_df, post_df, T, E_df + P_l * V_df, n_df * R * (T + T_ad));
  if (*mu_ds < 1.e9) PetscPrintf(PETSC_COMM_SELF, "Dislocation pre: %g mid: %g post: %g T: %g z: %g\n", pre_ds, mid_ds, post_ds, T, z);
#endif
  return;
}

static void analytic_2d_0_constant(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal mu = 1.0;

  f0[0] = 2.*mu - 1.;
  f0[1] = 2.*mu - 1.;
}

static void analytic_2d_0_linear_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal T, dT_dr, mu, dmu_dr;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  analytic_2d_0_dT_dr(dim, t, x, Nf, &dT_dr, NULL);
  mu     = 2.0 - T;
  dmu_dr = - dT_dr;
  f0[0]  = 2.*mu + 2.*dmu_dr*x[0] - 1.;
  f0[1]  = 2.*mu - 2.*dmu_dr*x[0] - 1.;
}

static void analytic_2d_0_exp_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal T, dT_dr, mu, dmu_dr;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  analytic_2d_0_dT_dr(dim, t, x, Nf, &dT_dr, NULL);
  mu     = 2.0 - PetscExpReal(T);
  dmu_dr = -dT_dr*PetscExpReal(T);
  f0[0]  = 2.*mu + 2.*dmu_dr*x[0] - 1.;
  f0[1]  = 2.*mu - 2.*dmu_dr*x[0] - 1.;
}

static void analytic_2d_0_exp_invt(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal T, dT_dr, mu, dmu_dr;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  analytic_2d_0_dT_dr(dim, t, x, Nf, &dT_dr, NULL);
  mu     = 1.0 + PetscExpReal(1.0/(1.0 + T));
  dmu_dr = -(dT_dr/(1.0 + T))*PetscExpReal(1.0/(1.0 + T));
  f0[0]  = 2.*mu + 2.*dmu_dr*x[0] - 1.;
  f0[1]  = 2.*mu - 2.*dmu_dr*x[0] - 1.;
}

static void analytic_2d_0_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal mu0      = constants[0]; /* Mantle viscosity kg/(m s)  : Mass Scale */
  const PetscReal R_E      = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa    = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT   = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0     = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta     = constants[5]; /* Adiabatic compressibility, 1/Pa */
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  PetscReal       T, dT_dr, eps_II, mu_df, mu_ds, dmu_df_dr;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  analytic_2d_0_dT_dr(dim, t, x, Nf, &dT_dr, NULL);
  MantleViscosityDerivativeR(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, dT_dr, &eps_II, &mu_df, &mu_ds, &dmu_df_dr);
  f0[0] = 2.*mu_df/mu0 + 2.*dmu_df_dr*x[0]*(R_E/mu0) - 1.;
  f0[1] = 2.*mu_df/mu0 - 2.*dmu_df_dr*x[0]*(R_E/mu0) - 1.;
}

static void stokes_momentum_constant(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*(u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_linear_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal T, mu;
  PetscInt  c, d;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 2.0 - T;
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_exp_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal T, mu;
  PetscInt  c, d;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 2.0 - PetscExpReal(T);
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_exp_invt(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal T, mu;
  PetscInt  c, d;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 1.0 + PetscExpReal(1.0/(1.0 + T));
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal mu = 0.5*(u[0]*u[0] + u[1]*u[1]);
  PetscInt        c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal mu = 0.5*(u_x[0]*u_x[0] + u_x[1]*u_x[1] + u_x[2]*u_x[2] + u_x[3]*u_x[3]);
  PetscInt        c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test3(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = eps_II*(PetscSqr(R_E)/kappa);

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test4(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = eps_II <= 0.0 ? 0.0 : PetscPowReal(eps_II*(PetscSqr(R_E)/kappa), (1.0 - n_ds)/n_ds);
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test5(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal mu0      = constants[0];
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = eps_II <= 0.0 ? 0.0 : (mu_ds*mu_ds/mu0)/mu0;
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test6(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = eps_II <= 0.0 ? 0.0 : (mu_df/(mu_ds + mu_df));
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_test7(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal mu0      = constants[0];
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = eps_II <= 0.0 ? 0.0 : (2.0*mu_ds*mu_df/(mu_ds + mu_df))/(1e17*mu0);
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = DiffusionCreepViscosity(dim, u_x, x, constants, T)/constants[0];
  PetscInt        c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_analytic_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal T, mu;
  PetscInt  c, d;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = DiffusionCreepViscosity(dim, u_x, x, constants, T)/constants[0];
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_dislocation(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = DislocationCreepViscosity(dim, u_x, x, constants, T)/constants[0];
  PetscInt        c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}
static void stokes_momentum_composite(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = CompositeViscosity(dim, u_x, x, constants, T)/constants[0];
  PetscInt        c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = mu * (u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[dim];
  }
}

static void stokes_mass(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

static void f0_bouyancy(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
#ifdef SPHERICAL
  const PetscReal r      = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional readius */
#endif
  const PetscReal mu0    = constants[0]; /* Mantle viscosity kg/(m s)  : Mass Scale */
  const PetscReal R_E    = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa  = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0   = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal alpha  = constants[6]; /* Coefficient of thermal expansivity K^{-1} */
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal g      = 9.8;    /* Acceleration due to gravity m/s^2 */
  const PetscReal Ra     = (rho0*g*alpha*DeltaT*PetscPowRealInt(R_E, 3))/(mu0*kappa);
  const PetscReal f      = -constants[8] * Ra * T; /* Nondimensional body force */
  PetscInt        d;

#ifdef SPHERICAL
  for (d = 0; d < dim; ++d) f0[d] = f*(x[d]/r); /* f \hat r */
#else
  for (d = 0; d < dim-1; ++d) f0[d] = 0.0;
  f0[dim-1] = f;
#endif
}

static void f1_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim*dim; ++d) f1[d] = 0.0;
}

/* < q, \nabla\cdot u >, J_{pu} */
static void stokes_mass_J(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}


/* -< \nabla\cdot v, p >, J_{up} */
static void stokes_momentum_pres_J(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

static void stokes_momentum_vel_J_constant(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt fc, df;

  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5; /*g3[fc, fc, df, df]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5; /*g3[fc, df, df, fc]*/
    }
  }
}
static void stokes_momentum_vel_J_linear_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal T, mu;
  PetscInt  fc, df;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 2.0 - T;
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[fc, fc, df, df]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[fc, df, df, fc]*/
    }
  }
}
static void stokes_momentum_vel_J_exp_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal T, mu;
  PetscInt  fc, df;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 2.0 - PetscExpReal(T);
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[fc, fc, df, df]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[fc, df, df, fc]*/
    }
  }
}
static void stokes_momentum_vel_J_exp_invt(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal T, mu;
  PetscInt  fc, df;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 1.0 + PetscExpReal(1.0/(1.0 + T));
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[fc, fc, df, df]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[fc, df, df, fc]*/
    }
  }
}
static void stokes_momentum_vel_J_mu_test(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt fc, gc, df;

  for (fc = 0; fc < dim; ++fc) {
    for (gc = 0; gc < dim; ++gc) {
      for (df = 0; df < dim; ++df) {
        g2[(fc*dim+gc)*dim+df] += 0.5*(u_x[fc*dim+df] + u_x[df*dim+fc])*u[gc];
      }
    }
  }
}
static void stokes_momentum_vel_J_test(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu  = 0.5*(u[0]*u[0] + u[1]*u[1]);
  PetscInt        fc, df;

  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
    }
  }
}
static void stokes_momentum_vel_J_test2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu = 0.5*(u_x[0]*u_x[0] + u_x[1]*u_x[1] + u_x[2]*u_x[2] + u_x[3]*u_x[3]);
  PetscInt        fc, df, gc, dg;

  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += 0.5*(u_x[fc*dim+df] + u_x[df*dim+fc])*u_x[gc*dim+dg];
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_test3(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  mu = eps_II*(PetscSqr(R_E)/kappa);
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : ((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*mu);
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_test4(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  eps_II *= PetscSqr(R_E)/kappa;
  mu = eps_II <= 0.0 ? 0.0 : PetscPowReal(eps_II, (1.0 - n_ds)/n_ds);
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : mu*((1.0 - n_ds)/n_ds)*((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*PetscSqr(eps_II));
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_test5(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu0      = constants[0];
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  eps_II *= PetscSqr(R_E)/kappa;
  mu = eps_II <= 0.0 ? 0.0 : (mu_ds*mu_ds/mu0)/mu0;
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : 2.0 * mu*((1.0 - n_ds)/n_ds)*((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*PetscSqr(eps_II));
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_test6(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  eps_II *= PetscSqr(R_E)/kappa;
  mu = eps_II <= 0.0 ? 0.0 : (mu_df/(mu_ds + mu_df));
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : -mu*(mu_ds/mu_df) * mu*((1.0 - n_ds)/n_ds)*((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*PetscSqr(eps_II));
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_test7(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu0      = constants[0];
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  eps_II *= PetscSqr(R_E)/kappa;
  mu = eps_II <= 0.0 ? 0.0 : (2.0*mu_ds*mu_df/(mu_ds + mu_df))/(1e17*mu0);
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : (mu/(2.0*PetscSqr(mu_ds))) * mu*((1.0 - n_ds)/n_ds)*((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*PetscSqr(eps_II));
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                            PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = DiffusionCreepViscosity(dim, u_x, x, constants, T)/constants[0];
  PetscInt        cI, d;

  for (cI = 0; cI < dim; ++cI) {
    for (d = 0; d < dim; ++d) {
      g3[((cI*dim+cI)*dim+d)*dim+d] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((cI*dim+d)*dim+d)*dim+cI] += 0.5*mu; /*g3[cI, d, d, cI]*/
    }
  }
}
static void stokes_momentum_vel_J_analytic_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal T, mu;
  PetscInt  cI, d;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = DiffusionCreepViscosity(dim, u_x, x, constants, T)/constants[0];
  for (cI = 0; cI < dim; ++cI) {
    for (d = 0; d < dim; ++d) {
      g3[((cI*dim+cI)*dim+d)*dim+d] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((cI*dim+d)*dim+d)*dim+cI] += 0.5*mu; /*g3[cI, d, d, cI]*/
    }
  }
}
static void stokes_momentum_vel_J_dislocation(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu_max   = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min   = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal mu0      = constants[0];
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  eps_II *= PetscSqr(R_E)/kappa;
  mu = PetscMin(mu_max, PetscMax(mu_min, mu_ds))/mu0;
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : mu*((1.0 - n_ds)/n_ds)*((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*PetscSqr(eps_II));
        }
      }
    }
  }
}
static void stokes_momentum_vel_J_composite(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                            PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu_max   = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min   = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal mu0      = constants[0];
  const PetscReal R_E      = constants[1];
  const PetscReal kappa    = constants[2];
  const PetscReal DeltaT   = constants[3];
  const PetscReal rho0     = constants[4];
  const PetscReal beta     = constants[5];
  const PetscReal dT_ad_dr = constants[7]; /* Adiabatic temperature gradient, K/m */
  const PetscReal T        = PetscRealPart(a[0]);
  const PetscReal n_ds     = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, dT_ad_dr, T, &eps_II, &mu_df, &mu_ds);
  eps_II *= PetscSqr(R_E)/kappa;
  mu = (2.0*mu_ds*mu_df/(mu_ds + mu_df));
  mu = PetscMin(mu_max, PetscMax(mu_min, mu))/mu0;
  for (fc = 0; fc < dim; ++fc) {
    for (df = 0; df < dim; ++df) {
      g3[((fc*dim+fc)*dim+df)*dim+df] += 0.5*mu; /*g3[cI, cI, d, d]*/
      g3[((fc*dim+df)*dim+df)*dim+fc] += 0.5*mu; /*g3[cI, d, d, cI]*/
      for (gc = 0; gc < dim; ++gc) {
        for (dg = 0; dg < dim; ++dg) {
          g3[((fc*dim+gc)*dim+df)*dim+dg] += eps_II <= 0.0 ? 0.0 : (mu/(2.0*PetscSqr(mu_ds))) * mu*((1.0 - n_ds)/n_ds)*((u_x[fc*dim+df] + u_x[df*dim+fc])*(u_x[gc*dim+dg] + u_x[dg*dim+gc]))/(8.0*PetscSqr(eps_II));
        }
      }
    }
  }
}

/* 1/mu < q, I q >, Jp_{pp} */
static void stokes_identity_J_constant(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
static void stokes_identity_J_linear_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscReal T, mu;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 2.0 - T;
  g0[0] = 1.0/mu;
}
static void stokes_identity_J_exp_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscReal T, mu;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 2.0 - PetscExpReal(T);
  g0[0] = 1.0/mu;
}
static void stokes_identity_J_exp_invt(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscReal T, mu;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = 1.0 + PetscExpReal(1.0/(1.0 + T));
  g0[0] = 1.0/mu;
}
static void stokes_identity_J_test(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                   PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
static void stokes_identity_J_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = DiffusionCreepViscosity(dim, u_x, x, constants, T)/constants[0];

  g0[0] = 1.0/mu;
}
static void stokes_identity_J_analytic_diffusion(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscReal T, mu;

  analytic_2d_0_T(dim, t, x, Nf, &T, NULL);
  mu = DiffusionCreepViscosity(dim, u_x, x, constants, T)/constants[0];
  g0[0] = 1.0/mu;
}
static void stokes_identity_J_dislocation(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = DislocationCreepViscosity(dim, u_x, x, constants, T)/constants[0];

  g0[0] = 1.0/mu;
}
static void stokes_identity_J_composite(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal T  = PetscRealPart(a[0]);
  const PetscReal mu = CompositeViscosity(dim, u_x, x, constants, T)/constants[0];

  g0[0] = 1.0/mu;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       bc, sol, mu;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug           = 0;
  options->showError       = PETSC_FALSE;
  options->byte_swap       = PETSC_TRUE;
  options->dim             = 2;
  options->refine          = 0;
  options->coarsen         = 0;
  options->simplex         = PETSC_TRUE;
  options->bcType          = FREE_SLIP;
  options->muTypePre       = CONSTANT;
  options->solTypePre      = NONE;
  options->muType          = CONSTANT;
  options->solType         = BASIC;
  options->mantleBasename[0] = '\0';
  options->verts[0]        = 0;
  options->verts[1]        = 0;
  options->verts[2]        = 0;
  options->cdm             = NULL;
  options->pointSF         = NULL;
  options->Tinit           = NULL;
  options->meanExp         = 1.0;

  ierr = PetscObjectComposedDataRegister(&MEAN_EXP_TAG);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Variable-Viscosity Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex79.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_error", "Output the error for verification", "ex79.c", options->showError, &options->showError, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-byte_swap", "Flag to swap bytes on input", "ex79.c", options->byte_swap, &options->byte_swap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex79.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex79.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-refine", "Number of parallel uniform refinement steps", "ex79.c", options->refine, &options->refine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-coarsen", "Number of parallel uniform coarsening steps", "ex79.c", options->coarsen, &options->coarsen, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type", "Type of boundary conditions", "ex79.c", bcTypes, NUM_BC_TYPES, bcTypes[options->bcType], &bc, NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  mu   = options->muTypePre;
  ierr = PetscOptionsEList("-mu_type_pre", "Type of rheology for the presolve", "ex79.c", rheologyTypes, NUM_RHEOLOGY_TYPES, rheologyTypes[options->muTypePre], &mu, NULL);CHKERRQ(ierr);
  options->muTypePre = (RheologyType) mu;
  mu   = options->muType;
  ierr = PetscOptionsEList("-mu_type", "Type of rheology", "ex79.c", rheologyTypes, NUM_RHEOLOGY_TYPES, rheologyTypes[options->muType], &mu, NULL);CHKERRQ(ierr);
  options->muType = (RheologyType) mu;
  sol  = options->solTypePre;
  ierr = PetscOptionsEList("-sol_type_pre", "Type of problem for presolve", "ex79.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solTypePre], &sol, NULL);CHKERRQ(ierr);
  options->solTypePre = (SolutionType) sol;
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of problem", "ex79.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsString("-mantle_basename", "The basename for mantle files", "ex79.c", options->mantleBasename, options->mantleBasename, sizeof(options->mantleBasename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mean_exp", "The exponent to use for the power mean for the coarse temperature (Arith = 1, Geom -> 0, Harmonic = -1)", "ex79.c", options->meanExp, &options->meanExp, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/* View vertex temperatures */
static PetscErrorCode TempViewFromOptions(DM dm, const char tempName[], const char optbase[], ...)
{
  DM             tdm;
  Vec            T, Tg;
  char           optmid[PETSC_MAX_PATH_LEN];
  char           opt[PETSC_MAX_PATH_LEN];
  va_list        Argp;
  size_t         fullLength;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  va_start(Argp, optbase);
  ierr = PetscVSNPrintf(optmid, PETSC_MAX_PATH_LEN, optbase ? optbase : "", &fullLength, Argp);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A",     (PetscObject *) &T);CHKERRQ(ierr);
  ierr = PetscSNPrintf(opt, PETSC_MAX_PATH_LEN, !optbase ? "-dm_view" : "-dm_%s_view", optmid);CHKERRQ(ierr);
  ierr = DMViewFromOptions(tdm, NULL, opt);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(tdm, &Tg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(tdm, T, INSERT_VALUES, Tg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(tdm,   T, INSERT_VALUES, Tg);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Tg, tempName ? tempName : "Temperature");CHKERRQ(ierr);
  ierr = PetscSNPrintf(opt, PETSC_MAX_PATH_LEN, !optbase ? "-temp_view" : "-temp_%s_view", optmid);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Tg, NULL, opt);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(tdm, &Tg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CellTempViewFromOptions(DM dm, const char tempName[], const char optbase[], ...)
{
  DM             tdm;
  Vec            T, Tg;
  char           optmid[PETSC_MAX_PATH_LEN];
  char           opt[PETSC_MAX_PATH_LEN];
  va_list        Argp;
  size_t         fullLength;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  va_start(Argp, optbase);
  ierr = PetscVSNPrintf(optmid, PETSC_MAX_PATH_LEN, optbase ? optbase : "", &fullLength, Argp);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "cdmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "cA",     (PetscObject *) &T);CHKERRQ(ierr);
  ierr = PetscSNPrintf(opt, PETSC_MAX_PATH_LEN, !optbase ? "-dm_view" : "-dm_%s_view", optmid);CHKERRQ(ierr);
  ierr = DMViewFromOptions(tdm, NULL, opt);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(tdm, &Tg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(tdm, T, INSERT_VALUES, Tg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(tdm,   T, INSERT_VALUES, Tg);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Tg, tempName ? tempName : "Temperature");CHKERRQ(ierr);
  ierr = PetscSNPrintf(opt, PETSC_MAX_PATH_LEN, !optbase ? "-temp_view" : "-temp_%s_view", optmid);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Tg, NULL, opt);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(tdm, &Tg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create non-dimensional local temperature vector, stored in cell, named "temperature" in the DM
     numRef - The number of refinements between the coarse grid and temperature grid
*/
static PetscErrorCode CreateCellTemperatureVector(DM dm, PetscInt numRef, AppCtx *user)
{
  DM              tdm;
  Vec             temp;
  PetscDS         tprob;
  PetscFE         tfe;
  PetscInt        dim;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMClone(dm, &tdm);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, tdm);CHKERRQ(ierr);
  ierr = DMGetDS(tdm, &tprob);CHKERRQ(ierr);
  {
    PetscSpace     P;
    PetscDualSpace Q;
    DM             K;
    const PetscInt order = PetscPowInt(2, numRef)-1;

    /* Create space */
    ierr = PetscSpaceCreate(comm, &P);CHKERRQ(ierr);
    ierr = PetscSpaceSetOrder(P, order);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetTensor(P, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumComponents(P, 1);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
    /* Create dual space */
    ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(Q, dim, PETSC_FALSE, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetNumComponents(Q, 1);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetTensor(Q, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetContinuity(Q, PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
    /* Create element */
    ierr = PetscFECreate(comm, &tfe);CHKERRQ(ierr);
    ierr = PetscFESetType(tfe, PETSCFEBASIC);CHKERRQ(ierr);
    ierr = PetscFESetBasisSpace(tfe, P);CHKERRQ(ierr);
    ierr = PetscFESetDualSpace(tfe, Q);CHKERRQ(ierr);
    ierr = PetscFESetNumComponents(tfe, 1);CHKERRQ(ierr);
    ierr = PetscFESetUp(tfe);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
    ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) tfe, "cell temperature");CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(tprob, 0, (PetscObject) tfe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&tfe);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) tprob, "ctemp_");CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(tprob);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(tdm, &temp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "cdmAux", (PetscObject) tdm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "cA",     (PetscObject) temp);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  ierr = DMDestroy(&tdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create non-dimensional local temperature vector, stored on vertices, named "temperature" in the DM */
static PetscErrorCode CreateTemperatureVector(DM dm, AppCtx *user)
{
  DM              tdm;
  Vec             temp;
  PetscFE         tfe, vfe;
  PetscDS         tprob, prob;
  PetscSpace      sp;
  PetscQuadrature q;
  PetscInt        dim, order;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  /* Need to match velocity quadrature */
  ierr = PetscFECreateDefault(comm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &vfe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(vfe, &q);CHKERRQ(ierr);

  ierr = DMClone(dm, &tdm);CHKERRQ(ierr);
  ierr = DMPlexSetRegularRefinement(tdm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, tdm);CHKERRQ(ierr);
  ierr = DMGetDS(tdm, &tprob);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, user->simplex, "temp_", PETSC_DEFAULT, &tfe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) tfe, "temperature");CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(tfe, q);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(tprob, 0, (PetscObject) tfe);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(tfe, &sp);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  if (order != 1) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Temperature element must be linear, not order %D", order);
  ierr = PetscDSSetFromOptions(tprob);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&tfe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&vfe);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(tdm, &temp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) tdm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A",     (PetscObject) temp);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  ierr = DMDestroy(&tdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateInitialTemperature(DM dm, AppCtx *user)
{
  DM             tdm;
  Vec            Ts;
  PetscSection   tsec;
  PetscViewer    viewer;
  PetscScalar   *T;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = CreateTemperatureVector(dm, user);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A",     (PetscObject *) &Ts);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(tdm, &tsec);CHKERRQ(ierr);
  ierr = VecGetArray(Ts, &T);CHKERRQ(ierr);
  if (!rank) {
    PetscInt Nx = user->verts[user->perm[0]], Ny = user->verts[user->perm[1]], Nz = user->verts[user->perm[2]];
    PetscInt vStart, vx, vy, vz, count;
    char     filename[PETSC_MAX_PATH_LEN];
    float   *temp;

    ierr = PetscStrcpy(filename, user->mantleBasename);CHKERRQ(ierr);
    ierr = PetscStrcat(filename, "_therm.bin");CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nz, &temp);CHKERRQ(ierr);
    /* The ordering is Y, X, Z where Z is the fastest dimension */
    ierr = DMPlexGetDepthStratum(tdm, 0, &vStart, NULL);CHKERRQ(ierr);
    for (vy = 0; vy < Ny; ++vy) {
      for (vx = 0; vx < Nx; ++vx) {
        ierr = PetscViewerRead(viewer, temp, Nz, &count, PETSC_FLOAT);CHKERRQ(ierr);
        if (count != Nz) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Mantle temperature file %s had incorrect length", filename);
        if (user->byte_swap) {ierr = PetscByteSwap(temp, PETSC_FLOAT, count);CHKERRQ(ierr);}
        for (vz = 0; vz < Nz; ++vz) {
          PetscInt off;

          ierr = PetscSectionGetOffset(tsec, (vz*Ny + vy)*Nx + vx + vStart, &off);CHKERRQ(ierr);
          if ((temp[vz] < 0.0) || (temp[vz] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) temp[vz]);
          T[off] = temp[vz];
        }
      }
    }
    ierr = PetscFree(temp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(Ts, &T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TransferCellTemperature(DM cdm, DM rdm)
{
  DM                 ctdm, rtdm;
  Vec                cT,   rT;
  PetscSection       cs;
  const PetscScalar *ca;
  PetscScalar       *ra;
  PetscInt           dim, cStart, cEnd, c, dof;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject) cdm, "cdmAux", (PetscObject *) &ctdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) cdm, "cA",     (PetscObject *) &cT);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) rdm, "cdmAux", (PetscObject *) &rtdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) rdm, "cA",     (PetscObject *) &rT);CHKERRQ(ierr);
  ierr = DMGetDimension(cdm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(ctdm, &cs);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(ctdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cT, &ca);CHKERRQ(ierr);
  ierr = VecGetArray(rT, &ra);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    /*
     3---------2---------2
     |         |         |
     |    D    2    C    |
     |         |         |
     3----3----0----1----1
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
    ierr = PetscSectionGetDof(cs, cStart, &dof);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscScalar *ct;
      PetscScalar       *rt;
      const PetscInt     ioff[2][2] = {{0, 1}, {1, 0}};
      const PetscInt     joff[2][2] = {{0, 0}, {1, 1}};
      const PetscInt     num        = (PetscInt) PetscPowReal(dof, 1./dim);
      const PetscInt     half       = num/2;
      PetscInt           i, j, k, l;

      ierr = DMPlexPointLocalRead(ctdm, c, ca, &ct);CHKERRQ(ierr);
      for (j = 0; j < 2; ++j) {
        for (i = 0; i < 2; ++i) {
          ierr = DMPlexPointLocalRef(rtdm, c*4 + j*2+i, ra, &rt);CHKERRQ(ierr);
          for (l = 0; l < half; ++l) {
            for (k = 0; k < half; ++k) {
              rt[l*half+k] = ct[(joff[j][i]*half+l)*num + ioff[j][i]*half+k];
            }
          }
        }
      }
    }
    break;
  default: SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No refinement for dimension %d", dim);
  }
  ierr = VecRestoreArrayRead(cT, &ca);CHKERRQ(ierr);
  ierr = VecRestoreArray(rT, &ra);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int IsVeryCloseReal(PetscReal x, PetscReal y)
{
  return PetscAbsReal(x - y) < PETSC_SMALL;
}

static PetscErrorCode TransferCellToVertexTemperature(DM dm)
{
  DM                 ctdm, vtdm;
  Vec                cT,   vT;
  PetscSection       cs,   vs;
  const PetscScalar *ca;
  PetscScalar       *va;
  DMLabel            lright, ltop;
  DM                 coorddm;
  Vec                coordinates, Tg;
  const PetscScalar *coords;
  PetscInt           dim, cStart, cEnd, c, vStart, vEnd, dof;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject) dm, "cdmAux", (PetscObject *) &ctdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "cA",     (PetscObject *) &cT);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux",  (PetscObject *) &vtdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A",      (PetscObject *) &vT);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "markerRight", &lright);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "markerTop"  , &ltop);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(ctdm, &cs);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(vtdm, &vs);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(ctdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(ctdm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &coorddm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cT, &ca);CHKERRQ(ierr);
  ierr = VecGetArray(vT, &va);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (cEnd > cStart) {ierr = PetscSectionGetDof(cs, cStart, &dof);CHKERRQ(ierr);}
    for (c = cStart; c < cEnd; ++c) {
      const PetscScalar *ct, *vx;
      PetscScalar       *vt;
      PetscReal          xmin = PETSC_MAX_REAL, xmax = PETSC_MIN_REAL, ymin = PETSC_MAX_REAL, ymax = PETSC_MIN_REAL;
      const PetscInt     num  = (PetscInt) PetscPowReal(dof, 1./dim);
      PetscReal          ccoords[8];
      PetscInt           cone[4], cp = 0, rval, tval;
      PetscInt          *closure = NULL;
      PetscInt           closureSize, cl;
      PetscBool          found = PETSC_FALSE;

      ierr = DMPlexPointLocalRead(ctdm, c, ca, &ct);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(ctdm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if (closureSize != 9) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Closure size should be 9, not %D", closureSize);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        PetscInt point = closure[cl];
        if ((point < vStart) || (point >= vEnd)) continue;
        ierr = DMPlexPointLocalRead(coorddm, point, coords, &vx);CHKERRQ(ierr);
        xmin = PetscMin(xmin, vx[0]); ymin = PetscMin(ymin, vx[1]);
        xmax = PetscMax(xmax, vx[0]); ymax = PetscMax(ymax, vx[1]);
        ccoords[cp*2+0] = vx[0]; ccoords[cp*2+1] = vx[1];
        cone[cp++] = point;
      }
      ierr = DMPlexRestoreTransitiveClosure(ctdm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if (cp != 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cone size should be 4, not %D", cp);
      for (cp = 0; cp < 4; ++cp) {
        /* Set lower left vertex */
        if (IsVeryCloseReal(ccoords[cp*2+0], xmin) && IsVeryCloseReal(ccoords[cp*2+1], ymin)) {
          ierr = DMPlexPointLocalRef(vtdm, cone[cp], va, &vt);CHKERRQ(ierr);
          vt[0] = ct[0];
          if ((vt[0] < 0.0) || (vt[0] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) vt[0]);
          found = PETSC_TRUE;
        } else {
          /* Set right and top side vertices */
          ierr = DMLabelGetValue(lright, cone[cp], &rval);CHKERRQ(ierr);
          ierr = DMLabelGetValue(ltop, cone[cp], &tval);CHKERRQ(ierr);
          if ((rval == 1) && (tval == 1)) {
            ierr = DMPlexPointLocalRef(vtdm, cone[cp], va, &vt);CHKERRQ(ierr);
            vt[0] = ct[dof-1];
            if ((vt[0] < 0.0) || (vt[0] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) vt[0]);
          } else if (rval == 1) {
            ierr = DMPlexPointLocalRef(vtdm, cone[cp], va, &vt);CHKERRQ(ierr);
            vt[0] = ct[num-1];
            if ((vt[0] < 0.0) || (vt[0] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) vt[0]);
          } else if (tval == 1) {
            ierr = DMPlexPointLocalRef(vtdm, cone[cp], va, &vt);CHKERRQ(ierr);
            vt[0] = ct[dof-num];
            if ((vt[0] < 0.0) || (vt[0] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) vt[0]);
          }
        }
      }
      if (!found) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Lower left vertex of cell %D not found", c);
    }
    break;
  default: SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No cell-to-vertex for dimension %d", dim);
  }
  ierr = VecRestoreArrayRead(cT, &ca);CHKERRQ(ierr);
  ierr = VecRestoreArray(vT, &va);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  /* Since we only set the left/bottom edges, we need to sum the contributions across procs */
  ierr = DMGetGlobalVector(vtdm, &Tg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(vtdm, vT, ADD_VALUES, Tg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(vtdm,   vT, ADD_VALUES, Tg);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(vtdm, Tg, INSERT_VALUES, vT);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(vtdm,   Tg, INSERT_VALUES, vT);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(vtdm, &Tg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateInitialCoarseTemperature(DM dm, AppCtx *user)
{
  DM             tdm;
  Vec            Ts;
  PetscSection   tsec;
  PetscViewer    viewer;
  PetscScalar   *T;
  PetscMPIInt    rank;
  const PetscInt div = PetscPowInt(2, user->coarsen); /* The divisor in each direction */
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!dm) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = CreateCellTemperatureVector(dm, user->coarsen, user);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "cdmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "cA",     (PetscObject *) &Ts);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(tdm, &tsec);CHKERRQ(ierr);
  ierr = VecGetArray(Ts, &T);CHKERRQ(ierr);
  if (!rank) {
    PetscInt Nx = user->verts[user->perm[0]], Ny = user->verts[user->perm[1]], Nz = user->verts[user->perm[2]];
    PetscInt Ncx = PetscMax(1, (Nx-1)/div), Ncy = (Ny-1)/div;
    PetscInt Nlx = PetscMin(Ncx, div), Nly = PetscMin(Ncy, div);
    PetscInt cStart, vx, vy, vz, count;
    char     filename[PETSC_MAX_PATH_LEN];
    float   *temp;

    ierr = PetscStrcpy(filename, user->mantleBasename);CHKERRQ(ierr);
    ierr = PetscStrcat(filename, "_therm.bin");CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nz, &temp);CHKERRQ(ierr);
    /* The ordering is Y, X, Z where Z is the fastest dimension */
    ierr = DMPlexGetHeightStratum(tdm, 0, &cStart, NULL);CHKERRQ(ierr);
    for (vy = 0; vy < Ny; ++vy) {
      for (vx = 0; vx < Nx; ++vx) {
        ierr = PetscViewerRead(viewer, temp, Nz, &count, PETSC_FLOAT);CHKERRQ(ierr);
        if (count != Nz) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Mantle temperature file %s had incorrect length", filename);
        /* Ignore right and top edges, and just use the same temperature as the last vertex */
        if ((vy == Ny-1) || (vx && vx == Nx-1)) continue;
        if (user->byte_swap) {ierr = PetscByteSwap(temp, PETSC_FLOAT, count);CHKERRQ(ierr);}
        for (vz = 0; vz < Nz-1; ++vz) {
          /* fine cell is (vz*(Ny-1) + vy)*(Nx-1) + vx */
          const PetscInt ccell = (vz/div*Ncy + vy/div)*Ncx + vx/div;
          PetscInt       coff  = (vz%div*Nly + vy%div)*Nlx + vx%div;
          PetscInt       off;

          ierr = PetscSectionGetOffset(tsec, ccell, &off);CHKERRQ(ierr);
          if ((temp[vz] < 0.0) || (temp[vz] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) temp[vz]);
          T[off + coff] = temp[vz];
        }
      }
    }
    ierr = PetscFree(temp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(Ts, &T);CHKERRQ(ierr);
  ierr = CreateTemperatureVector(dm, user);CHKERRQ(ierr);
  ierr = TransferCellToVertexTemperature(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Distribute initial serial temperature to the new distributed mesh
     cell: Flag for temperatures on cells or vertices
*/
static PetscErrorCode DistributeTemperature(DM dm, PetscBool cell, AppCtx *user)
{
  DM             tdm;
  Vec            T;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->pointSF) {
    DM             dms, dmp;
    PetscSection   secs, secp;
    Vec            tmp;

    if (cell) {
      ierr = CreateCellTemperatureVector(dm, user->coarsen, user);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) dm, "cdmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) dm, "cA",     (PetscObject *) &T);CHKERRQ(ierr);
    } else {
      ierr = CreateTemperatureVector(dm, user);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) dm, "A",     (PetscObject *) &T);CHKERRQ(ierr);
    }

    ierr = VecGetDM(user->Tinit, &dms);CHKERRQ(ierr);
    ierr = VecGetDM(T, &dmp);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dms, &secs);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) T), &secp);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &tmp);CHKERRQ(ierr);
    ierr = DMPlexDistributeField(dms, user->pointSF, secs, user->Tinit, secp, tmp);CHKERRQ(ierr);
    ierr = VecCopy(tmp, T);CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);

    ierr = PetscSFDestroy(&user->pointSF);CHKERRQ(ierr);
    ierr = VecDestroy(&user->Tinit);CHKERRQ(ierr);
    if (cell) {
      ierr = CreateTemperatureVector(dm, user);CHKERRQ(ierr);
      ierr = TransferCellToVertexTemperature(dm);CHKERRQ(ierr);
    }
  }
  ierr = CellTempViewFromOptions(dm, NULL, "cdist");CHKERRQ(ierr);
  ierr = TempViewFromOptions(dm, NULL, "dist");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Make split labels so that we can have corners in multiple labels */
static PetscErrorCode MeshSplitLabels(DM dm)
{
  const char    *names[4] = {"markerBottom", "markerRight", "markerTop", "markerLeft"};
  PetscInt       ids[4]   = {1, 2, 3, 4};
  DMLabel        label;
  IS             is;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!dm) PetscFunctionReturn(0);
  for (f = 0; f < 4; ++f) {
    ierr = DMGetStratumIS(dm, "marker", ids[f],  &is);CHKERRQ(ierr);
    if (!is) continue;
    ierr = DMCreateLabel(dm, names[f]);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, names[f], &label);CHKERRQ(ierr);
    if (is) {ierr = DMLabelInsertIS(label, is, 1);CHKERRQ(ierr);}
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             dmDist = NULL;
  PetscViewer    viewer;
  PetscInt       count;
  char           filename[PETSC_MAX_PATH_LEN];
  char           line[PETSC_MAX_PATH_LEN];
  double        *axes[3];
  int           *verts = user->verts;
  int           *perm  = user->perm;
  int            snum, d;
  PetscInt       dim   = user->dim;
  PetscInt       div   = PetscPowInt(2, user->coarsen); /* The divisor in each direction */
  PetscInt       cells[3];
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (dim > 3) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"dim %D is too big, must be <= 3",dim);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  cells[0] = cells[1] = cells[2] = user->simplex ? dim : 3;
  if (dim == 2) {perm[0] = 2; perm[1] = 0; perm[2] = 1;}
  else          {perm[0] = 0; perm[1] = 1; perm[2] = 2;}
  if (!rank) {
    if (user->simplex) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Citom grids do not use simplices. Use -simplex 0");
    ierr = PetscStrcpy(filename, user->mantleBasename);CHKERRQ(ierr);
    ierr = PetscStrcat(filename, "_vects.ascii");CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d %d %d", &verts[perm[0]], &verts[perm[1]], &verts[perm[2]]);
    if (snum != 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Citcom vertex file header: %s", line);
    for (d = 0; d < 3; ++d) {
      ierr = PetscMalloc1(verts[perm[d]], &axes[perm[d]]);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, axes[perm[d]], verts[perm[d]], &count, PETSC_DOUBLE);CHKERRQ(ierr);
      if (count != verts[perm[d]]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Citcom vertex file dimension %d: %D %= %d", d, count, verts[perm[d]]);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    if (dim == 2) {verts[perm[0]] = 1;}
    for (d = 0; d < 3; ++d) cells[d] = verts[d]-1;
  }
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* Handle coarsening */
  if (user->coarsen) {
    PetscInt ccells[3];
    /* Create coarse mesh */
    for (d = 0; d < dim; ++d) {
      if (verts[d] && (verts[d]-1) % div) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Cannot divide Cells_%c = %D by %D evenly", 'x'+((char) d), verts[d]-1, div);
      ccells[d] = (verts[d]-1)/div;
    }
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, ccells, NULL, NULL, NULL, PETSC_TRUE, &user->cdm);CHKERRQ(ierr);
  }
  /* Remap coordinates to unit ball */
  {
    Vec           coordinates;
    PetscSection  coordSection;
    PetscScalar  *coords;
    PetscInt      vStart, vEnd, v;

    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  vert[3] = {0, 0, 0};
      PetscReal theta, phi, r;
      PetscInt  off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) vert[d] = PetscRoundReal(PetscRealPart(coords[off+d])*cells[d]);
      theta = axes[perm[0]][vert[perm[0]]]*2.0*PETSC_PI/360;
      phi   = axes[perm[1]][vert[perm[1]]]*2.0*PETSC_PI/360;
      r     = axes[perm[2]][vert[perm[2]]];
      if (dim > 2) {
        coords[off+0] = r*PetscSinReal(theta)*PetscSinReal(phi);
        coords[off+1] = r*PetscSinReal(theta)*PetscCosReal(phi);
        coords[off+2] = r*PetscCosReal(theta);
      } else {
#if 0
        coords[off+0] = r*PetscSinReal(theta)*PetscSinReal(phi);
        coords[off+1] = r*PetscSinReal(theta)*PetscCosReal(phi);
#else
        /* Goddamn it. I can't enforce no slip on curved surfaces */
        coords[off+0] = PetscSinReal(theta)*PetscSinReal(phi);
        coords[off+1] = r;
#endif
      }
    }
    /* Create coarse coordinates */
    if (!rank && user->coarsen) {
      PetscInt      Nx = user->verts[user->perm[0]], Ny = user->verts[user->perm[1]], Nz = user->verts[user->perm[2]];
      PetscInt      Ncx = (Nx-1)/div + 1, Ncy = (Ny-1)/div + 1, Ncz = (Nz-1)/div + 1;
      Vec           coordinatesC;
      PetscSection  coordSectionC;
      PetscScalar  *coordsC;
      PetscInt      vStartC, vx, vy, vz;

      ierr = DMPlexGetDepthStratum(user->cdm, 0, &vStartC, NULL);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(user->cdm, &coordSectionC);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(user->cdm, &coordinatesC);CHKERRQ(ierr);
      ierr = VecGetArray(coordinatesC, &coordsC);CHKERRQ(ierr);
      for (vz = 0; vz < Ncz; ++vz) {
        for (vy = 0; vy < Ncy; ++vy) {
          for (vx = 0; vx < Ncx; ++vx) {
            const PetscInt vc = (vz*Ncy    + vy)*Ncx    + vx;
            const PetscInt vf = (vz*div*Ny + vy*div)*Nx + vx*div;
            PetscInt       offF, offC;

            ierr = PetscSectionGetOffset(coordSectionC, vc + vStartC, &offC);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(coordSection,  vf + vStart,  &offF);CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) coordsC[offC+d] = coords[offF+d];
          }
        }
      }
      ierr = VecRestoreArray(coordinatesC, &coordsC);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  }
  if (!rank) {for (d = 0; d < 3; ++d) {ierr = PetscFree(axes[d]);CHKERRQ(ierr);}}
  ierr = PetscObjectSetName((PetscObject)(*dm),"Mesh");CHKERRQ(ierr);
  if (user->bcType == FREE_SLIP) {
    ierr = MeshSplitLabels(*dm);CHKERRQ(ierr);
    ierr = MeshSplitLabels(user->cdm);CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(*dm, NULL, "-rdm_view");CHKERRQ(ierr);
  ierr = DMViewFromOptions(user->cdm, NULL, "-cdm_view");CHKERRQ(ierr);
#if 0
  if (user->coarsen) {ierr = CreateInitialCoarseTemperature(user->cdm, user);CHKERRQ(ierr);}
  else               {ierr = CreateInitialTemperature(*dm, user);CHKERRQ(ierr);}
#else
  ierr = CreateInitialCoarseTemperature(user->cdm, user);CHKERRQ(ierr);
  ierr = CreateInitialTemperature(*dm, user);CHKERRQ(ierr);
#endif
  /*
   Steps for coarsening:
   * Create coarse mesh matching temp mesh
   *   NO Create coarse mesh coordinates
       Actually, handle coordinate input similar to temperature (P1 field defined on the fine mesh)
   * Read initial temperaure onto coarse mesh
   *   cells get values, left/bottom sides except cells along right and top
   * Distribute coarse mesh and temperature
   * In steps,
   *   spread out coarse temperature
       for next finer mesh split temp into fine cells
       also make the vertex temp field at the same time
     Destroy all cell temp fields
  */
  if (user->coarsen) {
    /* Distribute coarse mesh over processes */
    ierr = DMPlexDistribute(user->cdm, 0, &user->pointSF, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      DM  tdm;
      Vec T;

      ierr = PetscObjectSetName((PetscObject) dmDist,"Distributed Mesh");CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) user->cdm, "cdmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) user->cdm, "cA",     (PetscObject *) &T);CHKERRQ(ierr);
      user->Tinit = T;
      ierr = PetscObjectReference((PetscObject) user->Tinit);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) user->cdm, "cdmAux", NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) user->cdm, "cA", NULL);CHKERRQ(ierr);
      ierr = DMDestroy(&user->cdm);CHKERRQ(ierr);
      user->cdm = dmDist;
    }
  } else {
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, &user->pointSF, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      DM  tdm;
      Vec T;

      ierr = PetscObjectSetName((PetscObject) dmDist,"Distributed Mesh");CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) *dm, "dmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) *dm, "A",     (PetscObject *) &T);CHKERRQ(ierr);
      user->Tinit = T;
      ierr = PetscObjectReference((PetscObject) user->Tinit);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *dm, "dmAux", NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *dm, "A", NULL);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = dmDist;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEquations(PetscDS prob, PetscInt dim, RheologyType muType, SolutionType solType)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  switch (solType) {
  case BASIC:
    switch (muType) {
    case CONSTANT:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_constant);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_constant);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_constant);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_constant, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, stokes_momentum_vel_J_mu_test,  stokes_momentum_vel_J_test);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST2:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test2);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_test2);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test2);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST3:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test3);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_test3);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test3);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST4:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test4);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_test4);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test4);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST5:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test5);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_test5);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test5);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST6:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test6);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_test6);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test6);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case TEST7:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_test7);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_test7);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_test7);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_test, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case DIFFUSION:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_diffusion);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL,  stokes_momentum_vel_J_diffusion);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_diffusion);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_diffusion, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case DISLOCATION:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_dislocation);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_dislocation);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_dislocation);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_dislocation, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    case COMPOSITE:
      ierr = PetscDSSetResidual(prob, 0, f0_bouyancy, stokes_momentum_composite);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_composite);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_composite);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_composite, NULL, NULL, NULL);CHKERRQ(ierr);
      break;
    default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rheology type %d (%s)", (PetscInt) muType, rheologyTypes[PetscMin(muType, NUM_RHEOLOGY_TYPES)]);
    }
    break;
  case ANALYTIC_0:
    switch (dim) {
    case 2:
      switch (muType) {
      case CONSTANT:
        ierr = PetscDSSetResidual(prob, 0, analytic_2d_0_constant, stokes_momentum_constant);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_constant);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_constant);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_constant, NULL, NULL, NULL);CHKERRQ(ierr);
        break;
      case LINEAR_T:
        ierr = PetscDSSetResidual(prob, 0, analytic_2d_0_linear_t, stokes_momentum_linear_t);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_linear_t);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_linear_t);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_linear_t, NULL, NULL, NULL);CHKERRQ(ierr);
        break;
      case EXP_T:
        ierr = PetscDSSetResidual(prob, 0, analytic_2d_0_exp_t, stokes_momentum_exp_t);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_exp_t);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_exp_t);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_exp_t, NULL, NULL, NULL);CHKERRQ(ierr);
        break;
      case EXP_INVT:
        ierr = PetscDSSetResidual(prob, 0, analytic_2d_0_exp_invt, stokes_momentum_exp_invt);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_exp_invt);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_exp_invt);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_exp_invt, NULL, NULL, NULL);CHKERRQ(ierr);
        break;
      case DIFFUSION:
        ierr = PetscDSSetResidual(prob, 0, analytic_2d_0_diffusion, stokes_momentum_analytic_diffusion);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 1, stokes_mass, f1_zero);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_analytic_diffusion);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobian(prob, 1, 0, NULL, stokes_mass_J, NULL,  NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL, NULL, NULL, stokes_momentum_vel_J_analytic_diffusion);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL, NULL, stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL, stokes_mass_J, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, stokes_identity_J_analytic_diffusion, NULL, NULL, NULL);CHKERRQ(ierr);
        break;
      default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rheology type %d (%s)", (PetscInt) muType, rheologyTypes[PetscMin(muType, NUM_RHEOLOGY_TYPES)]);
      }
      break;
    default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No solution %s for dimension %d", solutionTypes[solType], dim);
    }
    break;
  default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid solution type %d (%s)", (PetscInt) solType, solutionTypes[PetscMin(solType, NUM_SOLUTION_TYPES)]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(PetscDS prob, PetscInt dim, AppCtx *user)
{
  const PetscInt id  = 1;
  PetscInt       comp;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SetupEquations(prob, dim, user->muType, user->solType);CHKERRQ(ierr);
  /* Exact solutions */
  switch (user->solType) {
  case NONE:
  case BASIC:
    user->exactFuncs[0]   = zero_vector;
    user->exactFuncs[1]   = zero_scalar;
    user->initialGuess[0] = zero_vector;
    user->initialGuess[1] = dynamic_pressure;
    break;
  case ANALYTIC_0:
    switch (user->dim) {
    case 2:
      user->exactFuncs[0]   = analytic_2d_0_u;
      user->exactFuncs[1]   = analytic_2d_0_p;
      user->initialGuess[0] = zero_vector;
      user->initialGuess[1] = zero_scalar;
      break;
    default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No solution %s for dimension %d", solutionTypes[user->solType], user->dim);
    }
    break;
  default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid solution type %d (%s)", (PetscInt) user->solType, solutionTypes[PetscMin(user->solType, NUM_SOLUTION_TYPES)]);
  }
  ierr = PetscDSSetExactSolution(prob, 0, user->exactFuncs[0]);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 1, user->exactFuncs[1]);CHKERRQ(ierr);
  /* Boundary conditions */
  switch (user->bcType) {
  case FREE_SLIP:
    switch (dim) {
    case 2:
      comp = 1;
      ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallB", "markerBottom", 0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
      comp = 0;
      ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallR", "markerRight",  0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
      comp = 1;
      ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallT", "markerTop",    0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
      comp = 0;
      ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallL", "markerLeft",   0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
      break;
    default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No boundary condition %s for dimension %d", bcTypes[user->bcType], user->dim);
    }
    break;
  case DIRICHLET:
  {
    const PetscInt ids[4] = {1, 2, 3, 4};

    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) user->exactFuncs[0], 4, ids, user);CHKERRQ(ierr);
  }
  break;
  default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary condition type %d (%s)", (PetscInt) user->bcType, bcTypes[PetscMin(user->bcType, NUM_BC_TYPES)]);
  }
#define NUM_CONSTANTS 9
  /* Units */
  {
    /* TODO: Make these a bag */
    const PetscReal rho0     = 3300.;      /* Mantle density kg/m^3 */
    const PetscReal g        = 9.8;        /* Acceleration due to gravity m/s^2 */
    const PetscReal alpha    = 2.0e-5;     /* Coefficient of thermal expansivity K^{-1} */
    const PetscReal beta     = 4.3e-12;    /* Adiabatic compressibility, 1/Pa */
    PetscReal       mu0      = 1.0e20;     /* Mantle viscosity kg/(m s)  : Mass Scale */
    PetscReal       R_E      = 6.371137e6; /* Radius of the Earth m      : Length Scale */
    PetscReal       kappa    = 1.0e-6;     /* Thermal diffusivity m^2/s  : Time Scale */
    PetscReal       DeltaT   = 1400;       /* Mantle temperature range K : Temperature Scale */
    PetscReal       dT_ad_dr = -3.e-4;     /* Adiabatic temperature gradient, K/m */
    PetscReal       Ra;                    /* Rayleigh number */
    PetscScalar     constants[NUM_CONSTANTS];

    constants[0] = mu0;
    constants[1] = R_E;
    constants[2] = kappa;
    constants[3] = DeltaT;
    constants[4] = rho0;
    constants[5] = beta;
    constants[6] = alpha;
#ifdef SIMPLIFY
    constants[7] = 0.0;
#else
    constants[7] = dT_ad_dr;
#endif
    constants[8] = 1.0;

    Ra   = (rho0*g*alpha*DeltaT*PetscPowRealInt(R_E, 3))/(mu0*kappa);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-Ra_mult", &constants[8], NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ra: %g\n", Ra * constants[8]);CHKERRQ(ierr);
    ierr = PetscDSSetConstants(prob, NUM_CONSTANTS, constants);CHKERRQ(ierr);
    /* ierr = DMPlexSetScale(dm, PETSC_UNIT_LENGTH, R_E);CHKERRQ(ierr);
     ierr = DMPlexSetScale(dm, PETSC_UNIT_TIME,   R_E*R_E/kappa);CHKERRQ(ierr); */
  }
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(MPI_Comm comm, PetscDS *newprob, AppCtx *user)
{
  const PetscInt  dim = user->dim;
  PetscFE         fe[2];
  PetscQuadrature q;
  PetscDS         prob;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create discretization of solution fields */
  ierr = PetscFECreateDefault(comm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = PetscDSCreate(comm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = SetupProblem(prob, dim, user);CHKERRQ(ierr);
  *newprob = prob;
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/aij.h>
/* https://en.wikipedia.org/wiki/Generalized_mean */
PetscErrorCode MatMultTransposePowerMean_SeqAIJ(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJ        *a        = (Mat_SeqAIJ *) A->data;
  Mat_CompressedRow  cprow    = a->compressedrow;
  PetscBool          usecprow = cprow.use;
  PetscInt           m        = A->rmap->n;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *v;
  const PetscInt    *ii, *ridx = NULL;
  PetscInt           i, j;
  PetscReal          p;
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposedDataGetReal((PetscObject) A, MEAN_EXP_TAG, p, flg);CHKERRQ(ierr);
  ierr = VecSet(yy, 0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx, &x);CHKERRQ(ierr);
  ierr = VecGetArray(yy, &y);CHKERRQ(ierr);

  if (usecprow) {
    m    = cprow.nrows;
    ii   = cprow.i;
    ridx = cprow.rindex;
  } else {
    ii = a->i;
  }
  for (i = 0; i < m; ++i) {
    const PetscInt *idx;
    PetscScalar     xi;
    PetscInt        n;

    idx = a->j + ii[i];
    v   = a->a + ii[i];
    n   = ii[i+1] - ii[i];
    if (usecprow) {
      xi = x[ridx[i]];
    } else {
      xi = x[i];
    }
    for (j = 0; j < n; ++j) {
      y[idx[j]] += v[j]*PetscPowScalarReal(xi, p);
      PetscPrintf(PETSC_COMM_SELF, "1/y[%D]: 1/%g = %g\n", idx[j], PetscPowScalarReal(xi, -p)/v[j], v[j]*PetscPowScalarReal(xi, p));
    }
  }
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  for (i = 0; i < A->cmap->n; ++i) {
    PetscPrintf(PETSC_COMM_SELF, "y[%D]: %g\n", i, PetscPowScalarReal(y[i], 1.0/p));
  }
  ierr = VecRestoreArrayRead(xx, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy, &y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateHierarchy(DM dm, PetscDS prob, DM *newdm, AppCtx *user)
{
  DM             rdm, cdm, tdm, ctdm;
  PetscInt       dim, c, r;
  char           tempName[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  *newdm = dm;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDS(dm, prob);CHKERRQ(ierr);
  if (user->coarsen) {
    cdm = user->cdm;
    user->cdm = NULL;
    for (c = 0; c < user->coarsen; ++c) {
      ierr = DMPlexSetRefinementUniform(cdm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMRefine(cdm, PetscObjectComm((PetscObject) cdm), &rdm);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(rdm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMSetDS(rdm, prob);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(rdm, cdm);CHKERRQ(ierr);
      /* SetupMaterial */
      ierr = CreateCellTemperatureVector(rdm, user->coarsen-c-1, user);CHKERRQ(ierr);
      ierr = TransferCellTemperature(cdm, rdm);CHKERRQ(ierr);
      ierr = CreateTemperatureVector(rdm, user);CHKERRQ(ierr);
      ierr = TransferCellToVertexTemperature(rdm);CHKERRQ(ierr);
      ierr = PetscSNPrintf(tempName, PETSC_MAX_PATH_LEN, "Temperature (Injection) L %D", c+1);CHKERRQ(ierr);
      ierr = CellTempViewFromOptions(rdm, tempName, "rc_l%D", c+1);CHKERRQ(ierr);
      ierr = TempViewFromOptions(rdm, tempName, "r_l%D", c+1);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) cdm, "cdmAux", NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) cdm, "cA", NULL);CHKERRQ(ierr);
      ierr = DMDestroy(&cdm);CHKERRQ(ierr);
      cdm  = rdm;
    }
    ierr = PetscObjectCompose((PetscObject) rdm, "cdmAux", NULL);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) rdm, "cA", NULL);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm     = rdm;
    *newdm = dm;
    /* The previous loop used injection on the temperature, but we probably want smoothing */
    for (c = 0; c < user->coarsen; ++c) {
      DM  rtdm, ctdm;
      Vec rT,   cT, Rscale;
      Vec rTg,  cTg;
      Mat In;

      ierr = DMGetCoarseDM(rdm, &cdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) rdm, "dmAux", (PetscObject *) &rtdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) rdm, "A",     (PetscObject *) &rT);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) cdm, "dmAux", (PetscObject *) &ctdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) cdm, "A",     (PetscObject *) &cT);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(rtdm, ctdm);CHKERRQ(ierr);
      ierr = DMCreateInterpolation(ctdm, rtdm, &In, &Rscale);CHKERRQ(ierr);
      ierr = PetscObjectComposedDataSetReal((PetscObject) In, MEAN_EXP_TAG, user->meanExp);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(ctdm, &cTg);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(rtdm, &rTg);CHKERRQ(ierr);
      ierr = DMLocalToGlobalBegin(rtdm, rT, INSERT_VALUES, rTg);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(rtdm,   rT, INSERT_VALUES, rTg);CHKERRQ(ierr);
      if (PetscEqualReal(user->meanExp, 1.0)) {
        ierr = PetscSNPrintf(tempName, PETSC_MAX_PATH_LEN, "Temperature (Linear Average) L %D", c);CHKERRQ(ierr);
        ierr = MatMultTranspose(In, rTg, cTg);CHKERRQ(ierr);
        ierr = VecPointwiseMult(cTg, cTg, Rscale);CHKERRQ(ierr);
      } else {
        ierr = PetscSNPrintf(tempName, PETSC_MAX_PATH_LEN, "Temperature (Power Law p = %.1f) L %D", user->meanExp, c);CHKERRQ(ierr);
        ierr = MatShellSetOperation(In, MATOP_MULT_TRANSPOSE, (void (*)(void)) MatMultTransposePowerMean_SeqAIJ);CHKERRQ(ierr);
        ierr = MatMultTranspose(In, rTg, cTg);CHKERRQ(ierr);
        ierr = VecPointwiseMult(cTg, cTg, Rscale);CHKERRQ(ierr);
        ierr = VecPow(cTg, user->meanExp);CHKERRQ(ierr);
      }
      ierr = DMGlobalToLocalBegin(ctdm, cTg, INSERT_VALUES, cT);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(ctdm,   cTg, INSERT_VALUES, cT);CHKERRQ(ierr);
      ierr = MatDestroy(&In);CHKERRQ(ierr);
      ierr = VecDestroy(&Rscale);CHKERRQ(ierr);
      ierr = TempViewFromOptions(cdm, tempName, "s_l%D", c);CHKERRQ(ierr);
      rdm  = cdm;
    }
  }
  if (user->refine) {
    cdm = dm;
    for (r = 0; r < user->refine; ++r) {
      Mat In;
      Vec Rscale, T, cT;

      ierr = DMPlexSetRefinementUniform(cdm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMRefine(cdm, PetscObjectComm((PetscObject) cdm), &rdm);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(rdm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMSetDS(rdm, prob);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(rdm, cdm);CHKERRQ(ierr);
      /* SetupMaterial */
      ierr = CreateTemperatureVector(rdm, user);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) cdm, "dmAux", (PetscObject *) &ctdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) cdm, "A", (PetscObject *) &cT);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) rdm, "dmAux", (PetscObject *) &tdm);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) rdm, "A", (PetscObject *) &T);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(tdm, ctdm);CHKERRQ(ierr);
      ierr = DMCreateInterpolation(ctdm, tdm, &In, &Rscale);CHKERRQ(ierr);
      ierr = MatMult(In, cT, T);CHKERRQ(ierr);
      ierr = MatDestroy(&In);CHKERRQ(ierr);
      ierr = VecDestroy(&Rscale);CHKERRQ(ierr);
      ierr = DMDestroy(&cdm);CHKERRQ(ierr);
      cdm = rdm;
    }
    *newdm = rdm;
  }
  ierr = DMSetFromOptions(*newdm);CHKERRQ(ierr);
  ierr = TempViewFromOptions(*newdm, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateNullSpaces(DM dm, AppCtx *user)
{
  PetscObject    pressure,      velocity;
  MatNullSpace   nullSpacePres, nullSpaceVel;
  Vec            coordinates;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetField(dm, 1, &pressure);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres);CHKERRQ(ierr);
  ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);

  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, &velocity);CHKERRQ(ierr);
  ierr = MatNullSpaceCreateRigidBody(coordinates, &nullSpaceVel);CHKERRQ(ierr);
  ierr = PetscObjectCompose(velocity, "nearnullspace", (PetscObject) nullSpaceVel);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullSpaceVel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, Vec *v, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, one_scalar};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec, NULL, "-null_space_vec_view");CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  if (v) {*v = vec;}
  else   {ierr = VecDestroy(&vec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode OutputViscosity(Vec u, AppCtx *user)
{
  DM             dm,   dmVisc;
  PetscDS        prob, probVisc;
  PetscFE        feVisc;
  PetscObject    dmAux, auxVec;
  Vec            mu;
  void         (*funcs[1])(PetscInt, PetscInt, PetscInt,
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  PetscScalar   *constants;
  PetscInt       dim, numConstants;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsHasName(NULL, NULL, "-viscosity_vec_view", &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = VecGetDM(u, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMClone(dm, &dmVisc);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, dmVisc);CHKERRQ(ierr);
  ierr = DMGetDS(dmVisc, &probVisc);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, "visc_", PETSC_DEFAULT, &feVisc);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feVisc, "viscosity");CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(probVisc, 0, (PetscObject) feVisc);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feVisc);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", &auxVec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dmVisc, "dmAux", dmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dmVisc, "A", auxVec);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, (const PetscScalar **) &constants);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(probVisc, numConstants, constants);CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(probVisc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmVisc, &mu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) mu, "Viscosity");CHKERRQ(ierr);
  switch (user->muType) {
  case DIFFUSION:   funcs[0] = DiffusionCreepViscosityf0;break;
  case DISLOCATION: funcs[0] = DislocationCreepViscosityf0;break;
  case COMPOSITE:   funcs[0] = CompositeViscosityf0;break;
  default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rheology type %d (%s)", (PetscInt) user->muType, rheologyTypes[PetscMin(user->muType, NUM_RHEOLOGY_TYPES)]);
  }
  ierr = DMProjectField(dmVisc, 0.0, u, funcs, INSERT_VALUES, mu);CHKERRQ(ierr);
  ierr = VecViewFromOptions(mu, NULL, "-viscosity_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmVisc, &mu);CHKERRQ(ierr);
  ierr = DMDestroy(&dmVisc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  DM             dm;                   /* mesh and discretization */
  PetscDS        prob;                 /* problem definition */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            J, M;                 /* Jacobian matrix */
  MatNullSpace   nullSpace;            /* May be necessary for pressure */
  Vec            nullVec;
  PetscScalar    pint;
  AppCtx         user;                 /* user-defined work context */
  PetscReal      error = 0.0;          /* L_2 error in the solution */
  PetscReal      ferrors[2];
  void          *ctxs[2] = {NULL, NULL};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = PetscMalloc2(2, &user.exactFuncs, 2, &user.initialGuess);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(PETSC_COMM_WORLD, &prob, &user);CHKERRQ(ierr);
  if (user.coarsen) {
    ierr = DistributeTemperature(user.cdm, PETSC_TRUE, &user);CHKERRQ(ierr);
  } else {
    ierr = DistributeTemperature(dm, PETSC_FALSE, &user);CHKERRQ(ierr);
  }
  ierr = CreateHierarchy(dm, prob, &dm, &user);CHKERRQ(ierr);
  ierr = CreateNullSpaces(dm, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = CreatePressureNullSpace(dm, &user, &nullVec, &nullSpace);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* There should be a way to express this using the DM */
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes, &J, &M, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);

  /* Make exact solution */
  ierr = PetscDSGetConstants(prob, NULL, (const PetscScalar **) &ctxs[0]);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, NULL, (const PetscScalar **) &ctxs[1]);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, ctxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Exact Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-exact_vec_view");CHKERRQ(ierr);
  ierr = VecDot(nullVec, u, &pint);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Integral of pressure for exact solution: %g\n",(double) (PetscAbsScalar(pint) < 1.0e-14 ? 0.0 : PetscRealPart(pint)));CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, user.exactFuncs, ctxs);CHKERRQ(ierr);
  /* Make initial guess */
  ierr = DMProjectFunction(dm, 0.0, user.initialGuess, ctxs, INSERT_VALUES, u);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullSpace, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Initial Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-initial_vec_view");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  if (user.solTypePre != NONE) {
    PetscInt dim;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = SetupEquations(prob, dim, user.muTypePre, user.solTypePre);CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SetupEquations(prob, dim, user.muType, user.solType);CHKERRQ(ierr);
  }
  ierr = PetscDSDestroy(&prob);CHKERRQ(ierr);
  /* Solve problem */
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  /* Compute error and pressure integral for computed solution */
  ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, ctxs, u, &error);CHKERRQ(ierr);
  ierr = DMComputeL2FieldDiff(dm, 0.0, user.exactFuncs, ctxs, u, ferrors);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %.3g [%.3g, %.3g]\n", (double)error, (double)ferrors[0], (double)ferrors[1]);CHKERRQ(ierr);
  ierr = VecDot(nullVec, u, &pint);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Integral of pressure for computed solution: %g\n", (double) (PetscAbsScalar(pint) < 1.0e-14 ? 0.0 : PetscRealPart(pint)));CHKERRQ(ierr);
  if (user.showError) {
    Vec r;

    ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, ctxs, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
    ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) r, "Solution Error");CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-error_vec_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  }
  ierr = OutputViscosity(u, &user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Computed Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&nullVec);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree2(user.exactFuncs, user.initialGuess);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # 2D serial mantle tests
  test:
    suffix: small_q1p0_constant_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -mu_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9 -snes_err_if_not_converged 0

  test:
    suffix: small_q1p0_diffusion_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -mu_type diffusion -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: small_q1p0_composite_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -mu_type composite -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: small_q1p0_constant
    args: -mu_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: small_q1p0_diffusion
    args: -mu_type diffusion -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: small_q1p0_composite
    args: -mu_type composite -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: small_ref_q1p0_composite
    requires: broken
    args: -mu_type composite -simplex 0 -refine 1 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: small_q1p0_analytic_0
    args: -sol_type analytic_0 -mu_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -bc_type dirichlet -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8 -refine 1 -snes_convergence_estimate

  test:
    suffix: small_q2q1_analytic_0
    args: -sol_type analytic_0 -mu_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -bc_type dirichlet -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8 -refine 1 -snes_convergence_estimate

  test:
    suffix: small_q1p0_linear_t_analytic_0
    args: -sol_type analytic_0 -mu_type linear_t -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -bc_type dirichlet -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8 -refine 1 -snes_convergence_estimate

  test:
    suffix: small_q1p0_exp_t_analytic_0
    args: -sol_type analytic_0 -mu_type exp_t -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -bc_type dirichlet -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8 -refine 1 -snes_convergence_estimate

  test:
    suffix: small_q1p0_exp_invt_analytic_0
    args: -sol_type analytic_0 -mu_type exp_invt -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -bc_type dirichlet -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8 -refine 1 -snes_convergence_estimate

  test:
    suffix: small_q1p0_diffusion_analytic_0
    args: -sol_type analytic_0 -mu_type diffusion -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -bc_type dirichlet -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8 -refine 1 -snes_convergence_estimate

  test:
    suffix: small_q2q1_constant
    args: -mu_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_constant_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: uf16_q1p0_diffusion_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: uf16_q1p0_composite_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type composite -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: uf16_q1p0_constant
    args: -sol_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_diffusion
    args: -sol_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_diffusion_gamg
    nsize: 2
    args: -mu_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -snes_atol 1e-12 -snes_rtol 1e-7 -ksp_rtol 1e-8 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_ksp_rtol 1e-8 -fieldsplit_velocity_pc_type gamg -fieldsplit_velocity_pc_gamg_bs 2 -fieldsplit_velocity_pc_gamg_threshold 0.05 -fieldsplit_velocity_mg_levels_pc_type bjacobi -fieldsplit_velocity_mg_levels_sub_pc_type sor -fieldsplit_velocity_mg_levels_sub_pc_sor_lits 4 -fieldsplit_velocity_ksp_converged_reason -fieldsplit_pressure_ksp_rtol 1e-4 -fieldsplit_pressure_pc_type gamg -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason

  test:
    suffix: uf16_q1p0_composite
    requires: broken
    args: -sol_type composite -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_constant_lev_2
    args: -mu_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -coarsen 1 -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf4_q1p0_constant_lev_4
    args: -mu_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -coarsen 3 -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_constant_lev_2_p2
    nsize: 2
    args: -mu_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -coarsen 1 -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q2q1_constant_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 2 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: uf16_q2q1_diffusion_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type diffusion -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: uf16_q2q1_composite_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type composite -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: uf16_q2q1_constant
    args: -sol_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q2q1_diffusion
    args: -sol_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q2q1_composite
    requires: broken
    args: -sol_type composite -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -snes_linesearch_monitor -snes_linesearch_maxstep 1e20 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

TEST*/
