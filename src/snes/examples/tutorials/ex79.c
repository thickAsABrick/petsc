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

  -dm_view hdf5:mantle.h5 -sol_vec_view hdf5:mantle.h5::append -initial_vec_view hdf5:mantle.h5::append -temp_vec_view hdf5:mantle.h5::append -viscosity_vec_view hdf5:mantle.h5::append

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

typedef enum {CONSTANT, TEST, TEST2, TEST3, TEST4, TEST5, TEST6, TEST7, DIFFUSION, DISLOCATION, COMPOSITE, NONE, NUM_SOL_TYPES} SolutionType;
const char *solTypes[NUM_SOL_TYPES+1] = {"constant", "test", "test2", "test3", "test4", "test5", "test6", "test7", "diffusion", "dislocation", "composite", "none", "unknown"};

typedef struct {
  PetscInt      debug;             /* The debugging level */
  PetscBool     showError;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     simplex;           /* Use simplices or tensor product cells */
  PetscInt      serRef;            /* Number of serial refinements before the mesh gets distributed */
  char          mantleBasename[PETSC_MAX_PATH_LEN];
  int           verts[3];          /* The number of vertices in each dimension for mantle problems */
  int           perm[3] ;          /* The permutation of axes for mantle problems */
  /* Parallel temperature input */
  Vec           T;                 /* The non-dimensional temperature field */
  PetscSF       pointSF;           /* The SF describing mesh distribution */
  /* Problem definition */
  SolutionType  preType;           /* The type of problem for the presolve */
  SolutionType  solType;           /* The type of problem */
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);
} AppCtx;

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

  return P_l;
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
  const PetscReal    P_d       = constants[6] * gradP*z;

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

/* Assumes that u_x[], x[], and T are dimensionless */
static void MantleViscosity(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], PetscReal R_E, PetscReal kappa, PetscReal DeltaT, PetscReal rho0, PetscReal beta, PetscReal T_nondim, PetscReal *epsilon_II, PetscReal *mu_df, PetscReal *mu_ds)
{
#ifdef SPHERICAL
  const PetscReal r       = PetscSqrtReal(x[0]*x[0]+x[1]*x[1]+(dim>2 ? x[2]*x[2] : 0.0)); /* Nondimensional radius */
#else
  const PetscReal r       = x[dim-1];                            /* Nondimensional radius */
#endif
  const PetscReal z       = R_E*(1. - r);                        /* Depth m */
  const PetscBool isUpper = z < 670.0*1000.0 ? PETSC_TRUE : PETSC_FALSE; /* Are we in the upper mantle? */
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
  const PetscReal T_ad    = (3.e-4)*z;                           /* Adiabatic temperature, K with gradient of 3e-4 K/m */
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
#if 0
  if (*mu_df < 1.e9) PetscPrintf(PETSC_COMM_SELF, "Diffusion   pre: %g mid: %g post: %g T: %g Num: %g Denom: %g\n", pre_df, mid_df, post_df, T, E_df + P_l * V_df, n_df * R * (T + T_ad));
  if (*mu_ds < 1.e9) PetscPrintf(PETSC_COMM_SELF, "Dislocation pre: %g mid: %g post: %g T: %g z: %g\n", pre_ds, mid_ds, post_ds, T, z);
#endif
  return;
}

static PetscReal DiffusionCreepViscosity(PetscInt dim, const PetscScalar u_x[], const PetscReal x[], const PetscReal constants[], PetscReal T)
{
  const PetscReal mu_max  = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min  = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal R_E     = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa   = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT  = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0    = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta    = constants[5]; /* Adiabatic compressibility, 1/Pa */
  PetscReal       eps_II;                 /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu;       /* Pa s = kg/(m s) */

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
  mu = mu_df;
  //if (mu < mu_min) PetscPrintf(PETSC_COMM_SELF, "MIN VIOLATION: %g < %g (%g, %g)\n", mu, mu_min, x[0], x[1]);
  //if (mu > mu_max) PetscPrintf(PETSC_COMM_SELF, "MAX VIOLATION: %g > %g (%g, %g)\n", mu, mu_max, x[0], x[1]);
  return PetscMin(mu_max, PetscMax(mu_min, mu));
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
  const PetscReal mu_max  = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min  = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal R_E     = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa   = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT  = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0    = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta    = constants[5]; /* Adiabatic compressibility, 1/Pa */
  PetscReal       eps_II;                 /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu;       /* Pa s = kg/(m s) */

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu_max  = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min  = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal R_E     = constants[1]; /* Radius of the Earth m      : Length Scale */
  const PetscReal kappa   = constants[2]; /* Thermal diffusivity m^2/s  : Time Scale */
  const PetscReal DeltaT  = constants[3]; /* Mantle temperature range K : Temperature Scale */
  const PetscReal rho0    = constants[4]; /* Mantle density kg/m^3 */
  const PetscReal beta    = constants[5]; /* Adiabatic compressibility, 1/Pa */
  PetscReal       eps_II;                 /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu;       /* Pa s = kg/(m s) */

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        c, d;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal g      = 9.8;    /* Acceleration due to gravity m/s^2 */
  const PetscReal alpha  = 2.0e-5; /* Coefficient of thermal expansivity K^{-1} */
  const PetscReal Ra     = (rho0*g*alpha*DeltaT*PetscPowRealInt(R_E, 3))/(mu0*kappa);
  const PetscReal f      = -constants[6] * Ra * T; /* Nondimensional body force */
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
  PetscInt cI, d;

  for (cI = 0; cI < dim; ++cI) {
    for (d = 0; d < dim; ++d) {
      g3[((cI*dim+cI)*dim+d)*dim+d] += 0.5; /*g3[cI, cI, d, d]*/
      g3[((cI*dim+d)*dim+d)*dim+cI] += 0.5; /*g3[cI, d, d, cI]*/
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
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds   = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds   = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds   = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds   = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
static void stokes_momentum_vel_J_dislocation(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal mu_max = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds   = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  const PetscReal mu_max = 5e24;         /* Maximum viscosity kg/(m s) : The model no longer applies in colder lithosphere since it ignores brittle fracture */
  const PetscReal mu_min = 1e17;         /* Minimum viscosity kg/(m s) : The model no longer applies in hotter mantle since it ignores melting */
  const PetscReal mu0    = constants[0];
  const PetscReal R_E    = constants[1];
  const PetscReal kappa  = constants[2];
  const PetscReal DeltaT = constants[3];
  const PetscReal rho0   = constants[4];
  const PetscReal beta   = constants[5];
  const PetscReal T      = PetscRealPart(a[0]);
  const PetscReal n_ds   = 3.5;
  PetscReal       eps_II;           /* Second invariant of strain rate, 1/s */
  PetscReal       mu_df, mu_ds, mu; /* Pa s = kg/(m s) */
  PetscInt        fc, df, gc, dg;

  MantleViscosity(dim, u_x, x, R_E, kappa, DeltaT, rho0, beta, T, &eps_II, &mu_df, &mu_ds);
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
  g0[0] = 1.0/100.0;
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

static PetscErrorCode CompositeSolutionVelocity(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar v[], void *ctx)
{
  PetscFunctionBegin;
  v[0] = v[1] = 0.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode CompositeSolutionPressure(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar p[], void *ctx)
{
  PetscFunctionBegin;
  p[0] = 0.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug           = 0;
  options->dim             = 2;
  options->serRef          = 0;
  options->simplex         = PETSC_TRUE;
  options->showError       = PETSC_FALSE;
  options->preType         = NONE;
  options->solType         = CONSTANT;
  options->mantleBasename[0] = '\0';
  options->pointSF         = NULL;

  ierr = PetscOptionsBegin(comm, "", "Variable-Viscosity Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex69.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex69.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex69.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-serial_refinements", "Number of serial uniform refinements steps", "ex69.c", options->serRef, &options->serRef, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_error", "Output the error for verification", "ex69.c", options->showError, &options->showError, NULL);CHKERRQ(ierr);
  sol  = options->preType;
  ierr = PetscOptionsEList("-pre_type", "Type of problem for presolve", "ex69.c", solTypes, NUM_SOL_TYPES, solTypes[options->preType], &sol, NULL);CHKERRQ(ierr);
  options->preType = (SolutionType) sol;
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of problem", "ex69.c", solTypes, NUM_SOL_TYPES, solTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsString("-mantle_basename", "The basename for mantle files", "ex69.c", options->mantleBasename, options->mantleBasename, sizeof(options->mantleBasename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateTemperatureVector(DM dm, Vec *T, AppCtx *user)
{
  DM              tdm;
  PetscFE         tfe, vfe;
  PetscDS         tprob, prob;
  PetscSpace      sp;
  PetscQuadrature q;
  PetscInt        dim, order;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  /* Need to match velocity quadrature */
  ierr = PetscFECreateDefault(dm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &vfe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(vfe, &q);CHKERRQ(ierr);

  ierr = DMClone(dm, &tdm);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, tdm);CHKERRQ(ierr);
  ierr = DMGetDS(tdm, &tprob);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, user->simplex, "temp_", PETSC_DEFAULT, &tfe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) tfe, "temperature");CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(tfe, q);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(tprob, 0, (PetscObject) tfe);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(tfe, &sp);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  if (order != 1) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Temperature element must be linear, not order %D", order);
  ierr = PetscDSSetFromOptions(tprob);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&tfe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&vfe);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(tdm, T);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) tdm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A",     (PetscObject) *T);CHKERRQ(ierr);
  ierr = DMDestroy(&tdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateInitialTemperature(DM dm, Vec *Ts, AppCtx *user)
{
  DM             tdm;
  PetscSection   tsec;
  PetscViewer    viewer;
  PetscScalar   *T;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = CreateTemperatureVector(dm, Ts, user);CHKERRQ(ierr);
  ierr = VecGetDM(*Ts, &tdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(tdm, &tsec);CHKERRQ(ierr);
  ierr = VecGetArray(*Ts, &T);CHKERRQ(ierr);
  if (!rank) {
    PetscInt  Nx = user->verts[user->perm[0]], Ny = user->verts[user->perm[1]], Nz = user->verts[user->perm[2]], vStart, vx, vy, vz, count;
    PetscBool byteswap = PETSC_TRUE;
    char      filename[PETSC_MAX_PATH_LEN];
    float    *temp;

    ierr = PetscOptionsGetBool(NULL, NULL, "-byte_swap", &byteswap, NULL);CHKERRQ(ierr);
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
        /* They are written little endian, so I need to swap back (mostly) */
        if (byteswap) {ierr = PetscByteSwap(temp, PETSC_FLOAT, count);CHKERRQ(ierr);}
        for (vz = 0; vz < Nz; ++vz) {
          PetscInt off;

          ierr = PetscSectionGetOffset(tsec, (vz*Ny + vy)*Nx + vx + vStart, &off);CHKERRQ(ierr);
          //#define CHECKING 1
#if CHECKING
          p[off] = 0.0*temp[vz] + 0.7;
#else
          if ((temp[vz] < 0.0) || (temp[vz] > 1.0)) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Temperature %g not in [0.0, 1.0]", (double) temp[vz]);
          T[off] = temp[vz];
#endif
        }
      }
    }
    ierr = PetscFree(temp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecRestoreArray(*Ts, &T);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DistributeTemperature(Vec Ts, PetscSF pointSF, Vec Tp)
{
  DM             dms, dmp;
  PetscSection   secs, secp;
  Vec            tmp;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(Ts, &dms);CHKERRQ(ierr);
  ierr = VecGetDM(Tp, &dmp);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dms, &secs);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) Tp), &secp);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &tmp);CHKERRQ(ierr);
  ierr = DMPlexDistributeField(dms, pointSF, secs, Ts, secp, tmp);CHKERRQ(ierr);
  ierr = VecCopy(tmp, Tp);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
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
  PetscInt       dim    = user->dim;
  PetscInt       cells[3];
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (dim > 3) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"dim %D is too big, must be <= 3",dim);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  cells[0] = cells[1] = cells[2] = user->simplex ? dim : 3;
  if (!rank) {
    if (user->simplex) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Citom grids do not use simplices. Use -simplex 0");
    ierr = PetscStrcpy(filename, user->mantleBasename);CHKERRQ(ierr);
    ierr = PetscStrcat(filename, "_vects.ascii");CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
    if (dim == 2) {perm[0] = 2; perm[1] = 0; perm[2] = 1;}
    else          {perm[0] = 0; perm[1] = 1; perm[2] = 2;}
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
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  }
  if (!rank) {for (d = 0; d < 3; ++d) {ierr = PetscFree(axes[d]);CHKERRQ(ierr);}}
  /* Make split labels so that we can have corners in multiple labels */
  {
    const char *names[4] = {"markerBottom", "markerRight", "markerTop", "markerLeft"};
    PetscInt    ids[4]   = {1, 2, 3, 4};
    DMLabel     label;
    IS          is;
    PetscInt    f;

    for (f = 0; f < 4; ++f) {
      ierr = DMGetStratumIS(*dm, "marker", ids[f],  &is);CHKERRQ(ierr);
      if (!is) continue;
      ierr = DMCreateLabel(*dm, names[f]);CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, names[f], &label);CHKERRQ(ierr);
      if (is) {
        ierr = DMLabelInsertIS(label, is, 1);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&is);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectSetName((PetscObject)(*dm),"Mesh");CHKERRQ(ierr);
  {
    PetscInt i;
    for (i = 0; i < user->serRef; ++i) {
      DM dmRefined;

      ierr = DMPlexSetRefinementUniform(*dm,PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMRefine(*dm,PetscObjectComm((PetscObject)*dm),&dmRefined);CHKERRQ(ierr);
      if (dmRefined) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmRefined;
      }
    }
  }
  {
    ierr = CreateInitialTemperature(*dm, &user->T, user);CHKERRQ(ierr);
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, &user->pointSF, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = PetscObjectSetName((PetscObject) dmDist,"Distributed Mesh");CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = dmDist;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEquations(PetscDS prob, SolutionType solType)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  switch (solType) {
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
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  stokes_momentum_vel_J_diffusion);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  stokes_momentum_pres_J, NULL);CHKERRQ(ierr);
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
  default:
    SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid solution type %d (%s)", (PetscInt) solType, solTypes[PetscMin(solType, NUM_SOL_TYPES)]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id  = 1;
  PetscInt       comp;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SetupEquations(prob, user->solType);CHKERRQ(ierr);
  switch (user->dim) {
  case 2:
    switch (user->solType) {
    case CONSTANT:
    case TEST:
    case TEST2:
    case TEST3:
    case TEST4:
    case TEST5:
    case TEST6:
    case TEST7:
    case DIFFUSION:
    case DISLOCATION:
    case COMPOSITE:
      user->exactFuncs[0] = CompositeSolutionVelocity;
      user->exactFuncs[1] = CompositeSolutionPressure;
      break;
    default:
      SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid solution type %d (%s)", (PetscInt) user->solType, solTypes[PetscMin(user->solType, NUM_SOL_TYPES)]);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %D", user->dim);
  }
  comp = 1;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallB", "markerBottom", 0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
  comp = 0;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallR", "markerRight",  0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
  comp = 1;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallT", "markerTop",    0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
  comp = 0;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wallL", "markerLeft",   0, 1, &comp, (void (*)(void)) user->exactFuncs[0], 1, &id, NULL);CHKERRQ(ierr);
  /* Units */
  {
    /* TODO: Make these a bag */
    const PetscReal rho0   = 3300.;      /* Mantle density kg/m^3 */
    const PetscReal g      = 9.8;        /* Acceleration due to gravity m/s^2 */
    const PetscReal alpha  = 2.0e-5;     /* Coefficient of thermal expansivity K^{-1} */
    const PetscReal beta   = 4.3e-12;    /* Adiabatic compressibility, 1/Pa */
    PetscReal       mu0    = 1.0e20;     /* Mantle viscosity kg/(m s)  : Mass Scale */
    PetscReal       R_E    = 6.371137e6; /* Radius of the Earth m      : Length Scale */
    PetscReal       kappa  = 1.0e-6;     /* Thermal diffusivity m^2/s  : Time Scale */
    PetscReal       DeltaT = 1400;       /* Mantle temperature range K : Temperature Scale */
    PetscReal       Ra;                  /* Rayleigh number */
    PetscScalar     constants[7];

    constants[0] = mu0;
    constants[1] = R_E;
    constants[2] = kappa;
    constants[3] = DeltaT;
    constants[4] = rho0;
    constants[5] = beta;
    constants[6] = 1.0;

    Ra   = (rho0*g*alpha*DeltaT*PetscPowRealInt(R_E, 3))/(mu0*kappa);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-Ra_mult", &constants[6], NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ra: %g\n", Ra * constants[6]);CHKERRQ(ierr);
    ierr = PetscDSSetConstants(prob, 7, constants);CHKERRQ(ierr);
    /* ierr = DMPlexSetScale(dm, PETSC_UNIT_LENGTH, R_E);CHKERRQ(ierr);
     ierr = DMPlexSetScale(dm, PETSC_UNIT_TIME,   R_E*R_E/kappa);CHKERRQ(ierr); */
  }
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  const PetscInt  dim = user->dim;
  PetscFE         fe[2];
  PetscQuadrature q;
  PetscDS         prob;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create discretization of solution fields */
  ierr = PetscFECreateDefault(dm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = SetupProblem(prob, user);CHKERRQ(ierr);
  /* Handle temperature */
  {
    DM  tdm;
    Vec Tg;

    if (user->pointSF) {
      Vec Tnew;

      ierr = CreateTemperatureVector(dm, &Tnew, user);CHKERRQ(ierr);
      ierr = DistributeTemperature(user->T, user->pointSF, Tnew);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&user->pointSF);CHKERRQ(ierr);
      ierr = VecDestroy(&user->T);CHKERRQ(ierr);
      user->T = Tnew;
    }
    ierr = VecGetDM(user->T, &tdm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(tdm, NULL, "-dm_aux_view");CHKERRQ(ierr);
    ierr = DMGetGlobalVector(tdm, &Tg);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(tdm, user->T, INSERT_VALUES, Tg);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(tdm,   user->T, INSERT_VALUES, Tg);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) Tg, "Temperature");CHKERRQ(ierr);
    ierr = VecViewFromOptions(Tg, NULL, "-temp_vec_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(tdm, &Tg);CHKERRQ(ierr);
    ierr = VecDestroy(&user->T);CHKERRQ(ierr);
  }
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
#if 0
    DM dmAux;

    ierr = DMClone(cdm, &dmAux);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(cdm, dmAux);CHKERRQ(ierr);
    ierr = DMSetDS(dmAux, probAux);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
    ierr = SetupMaterial(cdm, dmAux, user);CHKERRQ(ierr);
    ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
#endif
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    ierr = DMGetField(dm, 1, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);
  }
  {
    PetscObject  velocity;
    Vec          coordinates;
    MatNullSpace nullSpaceVel;

    ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetField(dm, 0, &velocity);CHKERRQ(ierr);
    ierr = MatNullSpaceCreateRigidBody(coordinates, &nullSpaceVel);CHKERRQ(ierr);
    ierr = PetscObjectCompose(velocity, "nearnullspace", (PetscObject) nullSpaceVel);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpaceVel);CHKERRQ(ierr);
  }
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
  ierr = PetscFECreateDefault(dm, dim, 1, user->simplex, "visc_", PETSC_DEFAULT, &feVisc);CHKERRQ(ierr);
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
  switch (user->solType) {
  case DIFFUSION:   funcs[0] = DiffusionCreepViscosityf0;break;
  case DISLOCATION: funcs[0] = DislocationCreepViscosityf0;break;
  case COMPOSITE:   funcs[0] = CompositeViscosityf0;break;
  default: SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No viscosity function for solution type %d (%s)", (PetscInt) user->solType, solTypes[PetscMin(user->solType, NUM_SOL_TYPES)]);
  }
  ierr = DMProjectField(dmVisc, 0.0, u, funcs, INSERT_VALUES, mu);CHKERRQ(ierr);
  ierr = VecViewFromOptions(mu, NULL, "-viscosity_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmVisc, &mu);CHKERRQ(ierr);
  ierr = DMDestroy(&dmVisc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES            snes;                 /* nonlinear solver */
  DM              dm;                   /* mesh and discretization */
  PetscDS         prob;                 /* problem definition */
  Vec             u,r;                  /* solution, residual vectors */
  Mat             J, M;                 /* Jacobian matrix */
  MatNullSpace    nullSpace;            /* May be necessary for pressure */
  Vec             nullVec;
  PetscScalar     pint;
  AppCtx          user;                 /* user-defined work context */
  PetscReal       error = 0.0;          /* L_2 error in the solution */
  PetscReal       ferrors[2];
  PetscErrorCode  (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, dynamic_pressure};
  void            *ctxs[2] = {NULL, NULL};
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  /* Setup problem */
  ierr = PetscMalloc(2 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, NULL, (const PetscScalar **) &ctxs[0]);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, NULL, (const PetscScalar **) &ctxs[1]);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = CreatePressureNullSpace(dm, &user, &nullVec, &nullSpace);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* There should be a way to express this using the DM */
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes, &J, &M, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);

  /* Make exact solution */
  ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, ctxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Exact Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-exact_vec_view");CHKERRQ(ierr);
  ierr = VecDot(nullVec, u, &pint);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Integral of pressure for exact solution: %g\n",(double) (PetscAbsScalar(pint) < 1.0e-14 ? 0.0 : PetscRealPart(pint)));CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, user.exactFuncs, ctxs);CHKERRQ(ierr);
  /* Make initial guess */
  ierr = DMProjectFunction(dm, 0.0, initialGuess, ctxs, INSERT_VALUES, u);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullSpace, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Initial Solution");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-initial_vec_view");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  if (user.preType != NONE) {
    ierr = SetupEquations(prob, user.preType);CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SetupEquations(prob, user.solType);CHKERRQ(ierr);
  }
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
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # 2D serial mantle tests
  test:
    suffix: small_q1p0_constant_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9 -snes_err_if_not_converged 0

  test:
    suffix: small_q1p0_diffusion_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type diffusion -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: small_q1p0_composite_jac_check
    filter: Error: egrep "Norm of matrix"
    args: -sol_type composite -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -dm_view -snes_type test -petscds_jac_pre 0 -Ra_mult 1e-9

  test:
    suffix: small_q1p0_constant
    args: -sol_type constant -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: small_q1p0_diffusion
    args: -sol_type diffusion -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: small_q1p0_composite
    args: -sol_type composite -simplex 0 -mantle_basename $PETSC_DIR/share/petsc/datafiles/mantle/small -byte_swap 0 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

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
    args: -sol_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_diffusion
    args: -sol_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q1p0_composite
    requires: broken
    args: -sol_type composite -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 1 -pres_petscspace_order 0 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

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
    args: -sol_type constant -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q2q1_diffusion
    args: -sol_type diffusion -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

  test:
    suffix: uf16_q2q1_composite
    requires: broken
    args: -sol_type composite -simplex 0 -mantle_basename /PETSc3/geophysics/MM/input_data/TwoDimSlab45cg1deguf16 -dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -temp_petscspace_order 1 -pc_fieldsplit_diag_use_amat -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition a11 -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type lu -snes_error_if_not_converged -snes_view -ksp_error_if_not_converged -dm_view -snes_monitor -snes_converged_reason -ksp_monitor_true_residual -ksp_converged_reason -fieldsplit_pressure_ksp_monitor_no -fieldsplit_pressure_ksp_converged_reason -snes_atol 1e-12 -ksp_rtol 1e-10 -fieldsplit_pressure_ksp_rtol 1e-8

TEST*/
