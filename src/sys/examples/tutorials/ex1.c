#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.4 1998/04/15 02:48:44 curfman Exp curfman $";
#endif

static char help[] = "This is an introductory PETSc example that illustrates printing.\n\n";

/*T
   Concepts: Introduction to PETSc;
   Routines: PetscInitialize(); PetscPrintf(); PetscFinalize();
   Processors: n
T*/
 
#include "petsc.h"
int main(int argc,char **argv)
{
  int ierr, rank, size;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help); CHKERRA(ierr);

  /* 
     The following MPI calls return the number of processes
     being used and the rank of this process in the group.
   */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRA(ierr);

  /* 
     Here we would like to print only one message that represents
     all the processes in the group.  We use PetscPrintf() with the 
     communicator PETSC_COMM_WORLD.  Thus, only one message is
     printed representng PETSC_COMM_WORLD, i.e., all the processors.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"No of processors = %d rank = %d\n",size,rank);CHKERRA(ierr);

  /*
    Here a barrier is used to separate the two program states.
  */
  ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRA(ierr);

  /*
    Here we simply use PetscPrintf() with the communicator PETSC_COMM_SELF,
    where each process is considered separately and prints independently
    to the screen.  Thus, the output from different processes does not
    appear in any particular order.
  */

  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Jumbled Hello World\n",rank); CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).  See PetscFinalize()
     manpage for more information.
  */
  ierr = PetscFinalize(); CHKERRA(ierr);
  return 0;
}
