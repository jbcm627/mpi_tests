#include "CosmoArray.h"

using namespace utils;

// # OpenMP threads (each grid can utilize up to this many)
#define OMP_THREADS 1

#define NX 40
#define NY 40
#define NZ 40
#define BW 1

#define MX (NX + 2*BW)
#define MY (NY + 2*BW)
#define MZ (NZ + 2*BW)

#define N_INDEX(i, j, k, r) (r*MX*MY*MZ + (i + BW)*MY*MZ + ((j + BW)*NZ) + (k + BW))
#define M_INDEX(i, j, k, r) ((r)*MX*MY*MZ + (i)*MY*MZ + ((j)*MZ) + (k))

#define h 0.1

/**
 * @brief Create a grid (each MPI process), solve the diffusion equation.
 * 
 * @details 
 *  Compile using, eg: $ mpic++ main.cc --std=c++11 -Wall -fopenmp -O3
 *  Run using, eg:     $ mpirun -n 12 ./a.out
 */
int main(int argc, char **argv)
{
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  omp_set_num_threads(OMP_THREADS);

  // each MPI process creates a grid:
  CosmoArray grid (
      NX, NY, NZ, // x/y/z-dimensions
      BW, // boundary size
      2, // number of "registers" to keep
         // only boundaries of the first register are produced
      COSMO_IO_FLAG_OFF // produce output
    );
  
  // directly manipulate array from grid
  // Set grid values to a constant number
  #pragma omp parallel for
  for(int i=0; i<MX; i++)
    for(int j=0; j<MY; j++)
      for(int k=0; k<MZ; k++)
      {
        grid[M_INDEX(i,j,k,0)] = 1.0;
        grid[M_INDEX(i,j,k,1)] = 1.0;
      }
  // end for

  // create a "hot" point
  // as a check, the computation should not depend on where this point is,
  // how many nodes, processes, etc.
  int hot_idx = M_INDEX(5,5,5,0);
  if(grid.num() == 0)
  {
    grid[hot_idx] = 2.0;
  }

  for(int s=0; s<10; s++)
  {
    // exchange boundaries between 0-register grids
    // (having this first ensures correct initialization too)
    grid.exchangeBoundaries();

    if(grid.num() == 0)
    {
      std::cout << "Value at hot point: " << grid[hot_idx] << "\n";
    }

    // solve the diffision equation
    // eulerian integration (2 registers)
    #pragma omp parallel for
    for(int i=BW; i<MX-BW; i++)
      for(int j=BW; j<MY-BW; j++)
        for(int k=BW; k<MZ-BW; k++)
        {
          grid[M_INDEX(i,j,k,1)] = grid[M_INDEX(i,j,k,0)] + h*(
            grid[M_INDEX(i+1,j,k,0)] + grid[M_INDEX(i,j+1,k,0)] + grid[M_INDEX(i,j,k+1,0)]
            + grid[M_INDEX(i-1,j,k,0)] + grid[M_INDEX(i,j-1,k,0)] + grid[M_INDEX(i,j,k-1,0)]
            - 6.0*grid[M_INDEX(i,j,k,0)]
          );
        }
    #pragma omp parallel for
    for(int i=BW; i<MX-BW; i++)
      for(int j=BW; j<MY-BW; j++)
        for(int k=BW; k<MZ-BW; k++)
        {
          grid[M_INDEX(i,j,k,0)] = grid[M_INDEX(i,j,k,1)];
        }
  }

  // Finalize the MPI environment
  MPI_Finalize();
  return 0;
}
