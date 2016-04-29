#ifndef COSMO_ARRAY_H
#define COSMO_ARRAY_H

#define COSMO_IO_FLAG_ON 1
#define COSMO_IO_FLAG_VERBOSE 2
#define COSMO_IO_FLAG_OFF 0

#include <string>
#include <iostream>
#include <cmath>

#include <stdlib.h>

#include <mpi.h>
#include <omp.h>

namespace utils
{

/**
 * @brief         CosmoArray: A generic array utilizing OpenMP and MPI
 * @description   exchange information between boundaries on an array.
 * 
 *  Use as, eg:
 *    1) CosmoArray grid (WORLD_NX, WORLD_NY, WORLD_NZ, 1, 1, false);
 *    2) initialize grid
 *    3) loop:
 *    3a) grid.exchangeBoundaries()
 *    3b) perform computation
 */
class CosmoArray
{

private:

  // Output information?
  bool IOflag;
  bool boundaries_sent;

  // Particular MPI process number / array number
  int grid_number;
  // # of total MPI processes
  int total_grids;

  // grid dimensions along _?-axis of partricular grid
  // excluding boundaries:
  int n_x, n_y, n_z;
  // including boundaries:
  int m_x, m_y, m_z;
  // dimensions (# grid points in direction) of world
  int w_x, w_y, w_z;
  // dimensions (# *grids* in direction) of world
  int d_x, d_y, d_z;
  // number of registers
  float * array;

  // arrays for storing values to exchange at boundaries
  int bnd_width;
  float *incoming_bnd_values[3][3][3];
  float *outgoing_bnd_values[3][3][3];

  MPI_Request incoming_requests[3][3][3];
  MPI_Request outgoing_requests[3][3][3];
  MPI_Status status;

  // process numbers (ranks) of adjacent MPI processes / grids
  int adjacent_grid_numbers[3][3][3];

  /**
   * @brief Compute integer divisor of a dividend nearest a desired divisor
   * 
   * @param dividend  number to find divisor of
   * @param num       number to find divisor near
   * @return divisor
   */
  int _getDivisorNearestToNum(int dividend, double num)
  {
    // search for integer divisors above and below desired number
    int divisor_above = (int) std::llround(num)+1;
    int divisor_below = (int) std::llround(num);

    while( true )
    {
      if( dividend % divisor_above == 0 )
        return divisor_above;
      if( dividend % divisor_below == 0 )
        return divisor_below;
      
      divisor_above++;
      divisor_below--;
    }
  }

  /**
   * @brief      Send an array containing boundary values (nonblocking send)
   */
  void _sendBoundary(float * outgoing_bnd_array, int bnd_x,
    int bnd_y, int bnd_z)
  {
    int bnd_x_width = (bnd_x == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z == 0 ? n_z : bnd_width);
    int receiver_bnd_id = (-1*bnd_x+1)*100 + (-1*bnd_y+1)*10 + (-1*bnd_z+1);

    // size of (# of elements in) boundary array
    int bnd_size = bnd_x_width*bnd_y_width*bnd_z_width;
    int receiver_grid_number =
      adjacent_grid_numbers[bnd_x+1][bnd_y+1][bnd_z+1];

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Sending (" + std::to_string(bnd_x)
        + ", " + std::to_string(bnd_y)
        + ", " + std::to_string(bnd_z)
        + ")-boundary on grid " + std::to_string(grid_number)
        + " from (" + std::to_string(-1*bnd_x)
        + ", " + std::to_string(-1*bnd_y)
        + ", " + std::to_string(-1*bnd_z)
        + ")-boundary on grid " + std::to_string(receiver_grid_number)
        + "\n";

    MPI_Isend(&outgoing_bnd_array[0],
      bnd_size, MPI_FLOAT, 
      receiver_grid_number, receiver_bnd_id,
      MPI_COMM_WORLD, &outgoing_requests[bnd_x+1][bnd_y+1][bnd_z+1]);

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Completed sending (" + std::to_string(bnd_x)
        + ", " + std::to_string(bnd_y)
        + ", " + std::to_string(bnd_z)
        + ")-boundary on grid " + std::to_string(grid_number) + ".\n";
  }

  /**
   * @brief      Receive an array containing boundary values
   */
  void _nonBlockingReceiveBoundary(float * incoming_bnd_array, int bnd_x,
    int bnd_y, int bnd_z)
  {
    int bnd_x_width = (bnd_x == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z == 0 ? n_z : bnd_width);
    int bnd_id = (bnd_x+1)*100 + (bnd_y+1)*10 + (bnd_z+1);

    // size of (# of elements in) boundary array
    int bnd_size = bnd_x_width*bnd_y_width*bnd_z_width;
    int sender_grid_number =
      adjacent_grid_numbers[bnd_x+1][bnd_y+1][bnd_z+1];

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Receiving (" + std::to_string(bnd_x)
        + ", " + std::to_string(bnd_y)
        + ", " + std::to_string(bnd_z)
        + ")-boundary on grid " + std::to_string(grid_number)
        + " from (" + std::to_string(-1*bnd_x)
        + ", " + std::to_string(-1*bnd_y)
        + ", " + std::to_string(-1*bnd_z)
        + ")-boundary on grid " + std::to_string(sender_grid_number)
        + "\n";

    MPI_Irecv(&incoming_bnd_array[0],
      bnd_size, MPI_FLOAT, 
      sender_grid_number, bnd_id,
      MPI_COMM_WORLD, &incoming_requests[bnd_x+1][bnd_y+1][bnd_z+1]);

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Completed receiving (" + std::to_string(bnd_x)
        + ", " + std::to_string(bnd_y)
        + ", " + std::to_string(bnd_z)
        + ")-boundary on grid " + std::to_string(grid_number) + ".\n";
  }

  /**
   * @brief      Store boundary values from grid in an array
   */
  void _setOutgoingBoundaryArray(float * outgoing_bnd_array,
    int bnd_x, int bnd_y, int bnd_z)
  {
    int bnd_x_width = (bnd_x == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z == 0 ? n_z : bnd_width);

    int grid_offset_x = bnd_width,
        grid_offset_y = bnd_width,
        grid_offset_z = bnd_width;

    if(bnd_x == 1)
      grid_offset_x = n_x;
    if(bnd_y == 1)
      grid_offset_y = n_y;
    if(bnd_z == 1)
      grid_offset_z = n_z;

    for(int i = 0; i < bnd_x_width; ++i)
      for(int j = 0; j < bnd_y_width; ++j)
        for(int k = 0; k < bnd_z_width; ++k)
          outgoing_bnd_array[
            i*bnd_y_width*bnd_z_width
            + j*bnd_z_width + k
          ] = array[
            (i+grid_offset_x)*m_y*m_z
            + (j+grid_offset_y)*m_z + (k+grid_offset_z)
          ];
  }

  /**
   * @brief      Store boundary values from array to grid
   */
  void _storeIncomingBoundaryArray(float * incoming_bnd_array,
    int bnd_x, int bnd_y, int bnd_z)
  {
    int bnd_x_width = (bnd_x == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z == 0 ? n_z : bnd_width);

    int grid_offset_x = 0,
        grid_offset_y = 0,
        grid_offset_z = 0;

    if(bnd_x == 0)
      grid_offset_x = bnd_width;
    if(bnd_y == 0)
      grid_offset_y = bnd_width;
    if(bnd_z == 0)
      grid_offset_z = bnd_width;

    if(bnd_x == 1)
      grid_offset_x = n_x + bnd_width;
    if(bnd_y == 1)
      grid_offset_y = n_y + bnd_width;
    if(bnd_z == 1)
      grid_offset_z = n_z + bnd_width;


    for(int i = 0; i < bnd_x_width; ++i)
      for(int j = 0; j < bnd_y_width; ++j)
        for(int k = 0; k < bnd_z_width; ++k)
          array[
            (i+grid_offset_x)*m_y*m_z
            + (j+grid_offset_y)*m_z + (k+grid_offset_z)
          ] = incoming_bnd_array[
            i*bnd_y_width*bnd_z_width
            + j*bnd_z_width + k
          ];
  }

public:

  /**
   * @brief      Initialize CosmoArray: basic MPI array class
   *  World volume is w_x * w_y * w_z
   *
   * @param[in]  w_x_in              World num. grid points in x-direction
   * @param[in]  w_y_in              World num. grid points in y-direction
   * @param[in]  w_z_in              World num. grid points in z-direction
   * @param[in]  bnd_width_in        width of boundaries to exchange
   * @param[in]  n_m_in              num. of registers (at least 1)
   * @param[in]  IOflat_in           print output?
   */
  CosmoArray(int w_x_in, int w_y_in, int w_z_in, int bnd_width_in,
    int n_r_in, bool IOflag_in)
  {
    // iterators
    int bnd_x, bnd_y, bnd_z;
    int i, j, k;

    IOflag = IOflag_in;
    w_x = w_x_in;
    w_y = w_y_in;
    w_z = w_z_in;
    bnd_width = bnd_width_in;

    // initialize MPI communication variables; grid #'s
    MPI_Comm_rank(MPI_COMM_WORLD, &grid_number);
    MPI_Comm_size(MPI_COMM_WORLD, &total_grids);
    if(IOflag)
    {
      system ("hostname");
      std::cout << "Hello from MPI grid number " + std::to_string(grid_number)
        + " of " + std::to_string(total_grids) + "! I'll be trying to use "
        + std::to_string(omp_get_max_threads()) + " threads.\n" << std::flush;
    }
 
    // initialize request data somehow?
    for(i=0; i<3; ++i)
      for(j=0; j<3; ++j)
        for(k=0; k<3; ++k)
        {
          // initialize incoming_requests[i][j][k];
          // initialize outgoing_requests[i][j][k];
        }

    // determine dimensions of the world: find the divisor nearest to
    // the side-length-weighted cube root of the total number of grids (ranks)
    d_x = _getDivisorNearestToNum(total_grids,
      std::cbrt((double) total_grids / (double) (w_x*w_y*w_z))*w_x);
    if(IOflag && grid_number == 0)
      std::cout << "d_x is " + std::to_string(d_x) + "\n";
    int remaining_grids = total_grids / d_x;
    d_y = _getDivisorNearestToNum(remaining_grids,
      std::sqrt((double) remaining_grids / (double) (w_y*w_z))*w_y);
    if(IOflag && grid_number == 0)
      std::cout << "d_y is " + std::to_string(d_y) + "\n";
    d_z = remaining_grids / d_y;
    if(IOflag && grid_number == 0)
      std::cout << "d_z is " + std::to_string(d_z) + "\n";

    // let MPI determine coordinates of each grid in world (d_* variables)
    MPI_Comm comm;
    int dimensions[3] = {d_x, d_y, d_z};
    int is_dim_periodic[3] = {1, 1, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions,
      is_dim_periodic, reorder, &comm);

    // determine coordinates & dimensions of this grid/rank:
    int this_grid_coords[3];
    MPI_Cart_coords(comm, grid_number, 3, this_grid_coords);
    n_x = w_x/d_x;  int remainders_x = w_x % d_x;
    n_y = w_y/d_y;  int remainders_y = w_y % d_y;
    n_z = w_z/d_z;  int remainders_z = w_z % d_z;
    // determine grids to pick up extra points when d_* doesn't evenly divide w_*
    if(this_grid_coords[0] < remainders_x) n_x++;
    if(this_grid_coords[1] < remainders_y) n_y++;
    if(this_grid_coords[2] < remainders_z) n_z++;
    m_x = n_x + 2*bnd_width;
    m_y = n_y + 2*bnd_width;
    m_z = n_z + 2*bnd_width;

    // determine adjacent grid cells assuming a periodic domain
    // and allocate memory for incoming/outgoing buffer arrays
    for(bnd_x=-1; bnd_x<=1; bnd_x++)
      for(bnd_y=-1; bnd_y<=1; bnd_y++)
        for(bnd_z=-1; bnd_z<=1; bnd_z++)
        {
          int adjacent_grid_coords[3];
          std::copy(this_grid_coords, this_grid_coords+3, adjacent_grid_coords);
          adjacent_grid_coords[0] += bnd_x;
          adjacent_grid_coords[1] += bnd_y;
          adjacent_grid_coords[2] += bnd_z;
          
          int adjacent_grid_num;
          MPI_Cart_rank(comm, adjacent_grid_coords, &adjacent_grid_num);
          adjacent_grid_numbers[bnd_x+1][bnd_y+1][bnd_z+1]
            = adjacent_grid_num;

          // only create grids for boundaries
          if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
          {
            int bnd_x_width = (bnd_x == 0 ? n_x : bnd_width);
            int bnd_y_width = (bnd_y == 0 ? n_y : bnd_width);
            int bnd_z_width = (bnd_z == 0 ? n_z : bnd_width);
            incoming_bnd_values[bnd_x+1][bnd_y+1][bnd_z+1] = new float[
              bnd_x_width*bnd_y_width*bnd_z_width ];
            outgoing_bnd_values[bnd_x+1][bnd_y+1][bnd_z+1] = new float[
              bnd_x_width*bnd_y_width*bnd_z_width ];
          }
        }
    // end for

    // allocate memory for grid.
    array = new float[ m_x*m_y*m_z*n_r_in ];
    boundaries_sent = false;
  }

  /**
   * @brief      Exchange boundary information between adjacent grids
   * @details    Perform the following:
   *  - MPI_Irecv
   *  - wait on previous Isend**
   *  - copy data to send buffer
   *  - MPI_Isend
   *  - wait on MPI_Irecv
   *  - copy data from recv buffer
   *  
   *  - once complete, compute can proceed
   *  
   *  **not needed on first step
   *    achieved by calling MPI_Request_free(MPI_Request *request) for all
   *    requests during initialization
   */
  void exchangeBoundaries()
  {
    int bnd_x, bnd_y, bnd_z;

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Listening for boundary arrays...\n";
    for(bnd_x=-1; bnd_x<=1; bnd_x++)
      for(bnd_y=-1; bnd_y<=1; bnd_y++)
        for(bnd_z=-1; bnd_z<=1; bnd_z++)
          if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
            _nonBlockingReceiveBoundary(
              incoming_bnd_values[bnd_x+1][bnd_y+1][bnd_z+1],
              bnd_x, bnd_y, bnd_z);

    // only wait for boundaries to finish being sent if they have been sent before
    // would be better: initialize outgoing_requests somehow?
    if(boundaries_sent)
    {
      if(IOflag == COSMO_IO_FLAG_VERBOSE)
        std::cout << "Waiting boundary data to be sent...\n";
      for(bnd_x=-1; bnd_x<=1; bnd_x++)
        for(bnd_y=-1; bnd_y<=1; bnd_y++)
          for(bnd_z=-1; bnd_z<=1; bnd_z++)
            if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
              MPI_Wait(&outgoing_requests[bnd_x+1][bnd_y+1][bnd_z+1], &status);
    }

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Storing boundary data in outgoing buffers...\n";
    #pragma omp parallel for default(shared) private(bnd_x,bnd_y,bnd_z)
    for(bnd_x=-1; bnd_x<=1; bnd_x++)
      for(bnd_y=-1; bnd_y<=1; bnd_y++)
        for(bnd_z=-1; bnd_z<=1; bnd_z++)
          if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
            _setOutgoingBoundaryArray(
              outgoing_bnd_values[bnd_x+1][bnd_y+1][bnd_z+1],
              bnd_x, bnd_y, bnd_z);

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Sending boundary data...\n";
    for(bnd_x=-1; bnd_x<=1; bnd_x++)
      for(bnd_y=-1; bnd_y<=1; bnd_y++)
        for(bnd_z=-1; bnd_z<=1; bnd_z++)
          if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
            _sendBoundary(
              outgoing_bnd_values[bnd_x+1][bnd_y+1][bnd_z+1],
                bnd_x, bnd_y, bnd_z);
    boundaries_sent = true;

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Waiting for incoming boundary data to be received...\n";
    for(bnd_x=-1; bnd_x<=1; bnd_x++)
      for(bnd_y=-1; bnd_y<=1; bnd_y++)
        for(bnd_z=-1; bnd_z<=1; bnd_z++)
          if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
            MPI_Wait(&incoming_requests[bnd_x+1][bnd_y+1][bnd_z+1], &status);

    if(IOflag == COSMO_IO_FLAG_VERBOSE)
      std::cout << "Storing received boundary data...\n";
    #pragma omp parallel for default(shared) private(bnd_x,bnd_y,bnd_z)
    for(bnd_x=-1; bnd_x<=1; bnd_x++)
      for(bnd_y=-1; bnd_y<=1; bnd_y++)
        for(bnd_z=-1; bnd_z<=1; bnd_z++)
          if(bnd_x != 0 || bnd_y != 0 || bnd_z != 0)
            _storeIncomingBoundaryArray(
              incoming_bnd_values[bnd_x+1][bnd_y+1][bnd_z+1],
              bnd_x, bnd_y, bnd_z);

  }

  float * getArray()
  {
    return array;
  }

  float& operator[](int idx)
  {
    return array[idx];
  }

  int num()
  {
    return grid_number;
  }

  int ijkr2idx(int i, int j, int k, int r)
  {
    return r*m_x*m_y*m_z
      + (i+bnd_width)*m_y*m_z + (j+bnd_width)*m_z + k+bnd_width;
  }

  int laplacian(int i, int j, int k, int r)
  {

  }

  void getGridDims(int dims[3])
  {
    dims[0] = n_x;
    dims[1] = n_y;
    dims[2] = n_z;
  }

};

}

#endif
