#include "CosmoArray.h"

#include <chrono>
#include <ctime>


using namespace utils;

#define WX 128
#define WY 128
#define WZ 128
#define BW 1

#define M_INDEX(i, j, k, r, mx, my, mz) ((r)*mx*my*mz + (i)*mx*my + ((j)*mz) + (k))

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
  // omp_set_num_threads(OMP_THREADS);

  int i,j,k;

  // each MPI process creates a grid:
  CosmoArray grid (
      WX, WY, WZ, // x/y/z-dimensions
      BW, // boundary size
      2, // number of "registers" to keep
         // only boundaries of the first register are produced
      COSMO_IO_FLAG_ON // produce output
    );

  int grid_dims[3];
  grid.getGridDims(grid_dims);
  int n_x = grid_dims[0];
  int n_y = grid_dims[1];
  int n_z = grid_dims[2];

  // directly manipulate array from grid
  // Set grid values to a constant number
  #pragma omp parallel for
  for(i=0; i<n_x; i++)
    for(j=0; j<n_y; j++)
      for(k=0; k<n_z; k++)
      {
        int idx = grid.ijkr2idx(i,j,k,0);
        grid[idx] = 1.0;
      }
  // end for

  // create a "hot" point
  // as a check, the computation should not depend on where this point is,
  // how many nodes, processes, etc.
  int hot_idx = grid.ijkr2idx(i,j,k,0);
  if(grid.num() == 0)
  {
    grid[hot_idx] = 2.0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto t_start = std::chrono::high_resolution_clock::now();

  for(int s=0; s<100; s++)
  {

    // exchange boundaries between 0-register grids
    // (having this first ensures correct initialization too)
    grid.exchangeBoundaries();
    
    if(grid.num() == 0)
    {
      // std::cout << "Value at hot point: " << grid[hot_idx] << "\n";
    }


    // solve the diffision equation
    // eulerian integration (2 registers)
    #pragma omp parallel for default(shared) private(i,j,k)
    for(i=0; i<n_x; i++)
      for(j=0; j<n_y; j++)
        for(k=0; k<n_z; k++)
        {
          grid[grid.ijkr2idx(i,j,k,1)] = grid[grid.ijkr2idx(i,j,k,0)] + h*(
            grid[grid.ijkr2idx(i+1,j,k,0)] + grid[grid.ijkr2idx(i,j+1,k,0)] + grid[grid.ijkr2idx(i,j,k+1,0)]
            + grid[grid.ijkr2idx(i-1,j,k,0)] + grid[grid.ijkr2idx(i,j-1,k,0)] + grid[grid.ijkr2idx(i,j,k-1,0)]
            - 6.0*grid[grid.ijkr2idx(i,j,k,0)]
          );
        }

    #pragma omp parallel for default(shared) private(i,j,k)
    for(i=0; i<n_x; i++)
      for(j=0; j<n_y; j++)
        for(k=0; k<n_z; k++)
        {
          grid[grid.ijkr2idx(i,j,k,0)] = grid[grid.ijkr2idx(i,j,k,1)];
        }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  auto t_end = std::chrono::high_resolution_clock::now();
  if(grid.num() == 0)
  {
    std::cout << "Wall clock time passed: "
      + std::to_string( std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1000.0 )
      + "\n";
  }

  // Finalize the MPI environment
  MPI_Finalize();
  return 0;
}
