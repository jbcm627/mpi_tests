#ifndef COSMO_ARRAY_H
#define COSMO_ARRAY_H

#define COSMO_IO_FLAG_ON true
#define COSMO_IO_FLAG_OFF false

#include <string>
#include <iostream>
#include <cmath>

#include <mpi.h>
#include <omp.h>

namespace utils
{

/**
 * @brief         CosmoArray: A generic array utilizing OpenMP and MPI
 * @description   exchange information between boundaries on an array.
 * 
 *  Use as:
 *    1) CosmoArray grid (10, 10, 10, 1);
 *    2) do stuff to grid
 *    3) grid.exchangeBoundaries()
 *  
 */
class CosmoArray
{

public:

  /**
   * @brief      Initialize CosmoArray: basic MPI array class
   *  Volume is n_x * n_y * n_z
   *
   * @param[in]  n_x_in                   num. grid points in x-direction
   * @param[in]  n_y_in                   num. grid points in y-direction
   * @param[in]  n_z_in                   num. grid points in z-direction
   * @param[in]  bnd_width_in        width of boundaries to exchange
   * @param[in]  n_m_in                   num. of registers (at least 1)
   * @param[in]  IOflat_in                print output?
   */
  CosmoArray(int n_x_in, int n_y_in, int n_z_in, int bnd_width_in,
    int n_m_in, bool IOflag_in)
  {
    IOflag = IOflag_in;

    MPI_Comm_rank(MPI_COMM_WORLD, &grid_number);
    MPI_Comm_size(MPI_COMM_WORLD, &total_grids);
    _setAdjacentGridNumbers(); // requires grid_number and total_grids be set

    n_x = n_x_in;
    n_y = n_y_in;
    n_z = n_z_in;
    bnd_width = bnd_width_in;

    M_x = n_x + 2*bnd_width;
    M_y = n_y + 2*bnd_width;
    M_z = n_z + 2*bnd_width;

    array = new float[ M_x*M_y*M_z*n_m_in ];

    // determine adjacent grid cells assuming a periodic domain
    int bnd_x_idx, bnd_y_idx, bnd_z_idx;
    for(bnd_x_idx=-1; bnd_x_idx<=1; bnd_x_idx++)
      for(bnd_y_idx=-1; bnd_y_idx<=1; bnd_y_idx++)
        for(bnd_z_idx=-1; bnd_z_idx<=1; bnd_z_idx++)
        {
          // only create grids for boundaries
          if(bnd_x_idx != 0 || bnd_y_idx != 0 || bnd_z_idx != 0)
          {
            int bnd_x_width = (bnd_x_idx == 0 ? n_x : bnd_width);
            int bnd_y_width = (bnd_y_idx == 0 ? n_y : bnd_width);
            int bnd_z_width = (bnd_z_idx == 0 ? n_z : bnd_width);
            incoming_bnd_values[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1] = new float[
              bnd_x_width*bnd_y_width*bnd_z_width ];
            outgoing_bnd_values[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1] = new float[
              bnd_x_width*bnd_y_width*bnd_z_width ];
          }
        }
    /* end for */
  }

  /**
   * @brief      Exchange boundary information between adjacent grids
   */
  void exchangeBoundaries()
  {
    int bnd_x_idx, bnd_y_idx, bnd_z_idx;

    if(IOflag)
      std::cout << "Storing values in boundary arrays...\n";
    // determine adjacent grid cells assuming a periodic domain
    #pragma omp parallel for default(shared) private(bnd_x_idx,bnd_y_idx,bnd_z_idx)
    for(bnd_x_idx=-1; bnd_x_idx<=1; bnd_x_idx++)
      for(bnd_y_idx=-1; bnd_y_idx<=1; bnd_y_idx++)
        for(bnd_z_idx=-1; bnd_z_idx<=1; bnd_z_idx++)
          if(bnd_x_idx != 0 || bnd_y_idx != 0 || bnd_z_idx != 0)
            _setOutgoingBoundaryArray(
              outgoing_bnd_values[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1],
              bnd_x_idx, bnd_y_idx, bnd_z_idx);

// sending/receiving in parallel breaks things. 
// segfault happens - unsure why.
    // perform exchange - send & receive boundary values (nonblocking sends)
    if(IOflag)
      std::cout << "Sending boundary arrays...\n";
    for(bnd_x_idx=-1; bnd_x_idx<=1; bnd_x_idx++)
      for(bnd_y_idx=-1; bnd_y_idx<=1; bnd_y_idx++)
        for(bnd_z_idx=-1; bnd_z_idx<=1; bnd_z_idx++)
          if(bnd_x_idx != 0 || bnd_y_idx != 0 || bnd_z_idx != 0)
            _sendBoundary(
              outgoing_bnd_values[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1],
              bnd_x_idx, bnd_y_idx, bnd_z_idx);
    if(IOflag)
      std::cout << "Receiving boundary arrays...\n";
    for(bnd_x_idx=-1; bnd_x_idx<=1; bnd_x_idx++)
      for(bnd_y_idx=-1; bnd_y_idx<=1; bnd_y_idx++)
        for(bnd_z_idx=-1; bnd_z_idx<=1; bnd_z_idx++)
          if(bnd_x_idx != 0 || bnd_y_idx != 0 || bnd_z_idx != 0)
            _receiveBoundary(
              incoming_bnd_values[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1],
              bnd_x_idx, bnd_y_idx, bnd_z_idx);

    // set values in boundary arrays for exchange
    if(IOflag)
      std::cout << "Storing boundary arrays...\n";
    #pragma omp parallel for default(shared) private(bnd_x_idx,bnd_y_idx,bnd_z_idx)
    for(bnd_x_idx=-1; bnd_x_idx<=1; bnd_x_idx++)
      for(bnd_y_idx=-1; bnd_y_idx<=1; bnd_y_idx++)
        for(bnd_z_idx=-1; bnd_z_idx<=1; bnd_z_idx++)
          if(bnd_x_idx != 0 || bnd_y_idx != 0 || bnd_z_idx != 0)
            _storeIncomingBoundaryArray(
              incoming_bnd_values[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1],
              bnd_x_idx, bnd_y_idx, bnd_z_idx);

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

private:

  // Output information?
  bool IOflag;

  // Particular MPI process number / array number
  int grid_number;
  // # of total MPI processes
  int total_grids;

  // grid dimensions along _?-axis
  // excluding boundaries:
  int n_x, n_y, n_z;
  // including boundaries:
  int M_x, M_y, M_z;
  // number of registers
  float * array;

  // boundaries for exchanging
  int bnd_width;
  float *incoming_bnd_values[3][3][3];
  float *outgoing_bnd_values[3][3][3];

  // process numbers of adjacent MPI processes / grids
  int adjacent_grid_numbers[3][3][3];

  int _getDivisorNearestCubeRoot(int num)
  {
    // search for integer divisors above and below cube root
    int divisor_above = (int) std::cbrt((double) num);
    int divisor_below = (int) std::cbrt((double) num);

    while( true )
    {
      if( num % divisor_above == 0 )
        return divisor_above;
      if( num % divisor_below == 0 )
        return divisor_below;
      
      divisor_above++;
      divisor_below++;
    }
  }

  int _getDivisorNearestSquareRoot(int num)
  {
    // search for integer divisors above and below cube root
    int divisor_above = (int) std::sqrt((double) num);
    int divisor_below = (int) std::sqrt((double) num);

    while( true )
    {
      if( num % divisor_above == 0 )
        return divisor_above;
      if( num % divisor_below == 0 )
        return divisor_below;
      
      divisor_above++;
      divisor_below++;
    }
  }

  /**
   * @brief      Store adjacent grid numbers / grid IDs for future exchanges
   * @description
   *  sets an array of grid numbers of adjacent grids
   *  requires grid_number and total_grids be set.
   */
  void _setAdjacentGridNumbers()
  {
    // get dimensions of grid domain
    int d_x, d_y, d_z;
    d_x = _getDivisorNearestCubeRoot(total_grids);
    int remaining_grids = total_grids / d_x;
    d_y = _getDivisorNearestSquareRoot(remaining_grids);
    d_z = remaining_grids / d_y;

    // let MPI set up coordinates of each grid
    MPI_Comm comm;
    int dimensions[3] = {d_x, d_y, d_z};
    int is_dim_periodic[3] = {1, 1, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions,
      is_dim_periodic, reorder, &comm);

    // grid coordinates of this grid
    int grid_coords[3];
    MPI_Cart_coords(comm, grid_number, 3, grid_coords);

    // determine adjacent grid cells assuming a periodic domain
    int bnd_x_idx, bnd_y_idx, bnd_z_idx;
    for(bnd_x_idx=-1; bnd_x_idx<=1; bnd_x_idx++)
      for(bnd_y_idx=-1; bnd_y_idx<=1; bnd_y_idx++)
        for(bnd_z_idx=-1; bnd_z_idx<=1; bnd_z_idx++)
        {
          int adjacent_grid_coords[3];
          std::copy(grid_coords, grid_coords+3, adjacent_grid_coords);
          adjacent_grid_coords[0] += bnd_x_idx;
          adjacent_grid_coords[1] += bnd_y_idx;
          adjacent_grid_coords[2] += bnd_z_idx;
          
          int adjacent_grid_num;
          MPI_Cart_rank(comm, adjacent_grid_coords, &adjacent_grid_num);
          adjacent_grid_numbers[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1]
            = adjacent_grid_num;
        }
  }

  /**
   * @brief      Send an array containing boundary values (nonblocking send)
   */
  void _sendBoundary(float * outgoing_bnd_array, int bnd_x_idx,
    int bnd_y_idx, int bnd_z_idx)
  {
    int bnd_x_width = (bnd_x_idx == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y_idx == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z_idx == 0 ? n_z : bnd_width);
    int receiver_bnd_id = (-1*bnd_x_idx+1)*100 + (-1*bnd_y_idx+1)*10 + (-1*bnd_z_idx+1);

    // size of (# of elements in) boundary array
    int bnd_size = bnd_x_width*bnd_y_width*bnd_z_width;
    int receiver_grid_number =
      adjacent_grid_numbers[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1];

    if(IOflag)
      std::cout << "Sending (" + std::to_string(bnd_x_idx)
        + ", " + std::to_string(bnd_y_idx)
        + ", " + std::to_string(bnd_z_idx)
        + ")-boundary on grid " + std::to_string(grid_number)
        + " from (" + std::to_string(-1*bnd_x_idx)
        + ", " + std::to_string(-1*bnd_y_idx)
        + ", " + std::to_string(-1*bnd_z_idx)
        + ")-boundary on grid " + std::to_string(receiver_grid_number)
        + "\n";

    MPI_Request request;
    MPI_Isend(&outgoing_bnd_array[0],
      bnd_size, MPI_FLOAT, 
      receiver_grid_number, receiver_bnd_id,
      MPI_COMM_WORLD, &request);

    if(IOflag)
      std::cout << "Completed sending (" + std::to_string(bnd_x_idx)
        + ", " + std::to_string(bnd_y_idx)
        + ", " + std::to_string(bnd_z_idx)
        + ")-boundary on grid " + std::to_string(grid_number) + ".\n";
  }

  /**
   * @brief      Receive an array containing boundary values
   */
  void _receiveBoundary(float * incoming_bnd_array, int bnd_x_idx,
    int bnd_y_idx, int bnd_z_idx)
  {
    int bnd_x_width = (bnd_x_idx == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y_idx == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z_idx == 0 ? n_z : bnd_width);
    int bnd_id = (bnd_x_idx+1)*100 + (bnd_y_idx+1)*10 + (bnd_z_idx+1);

    // size of (# of elements in) boundary array
    int bnd_size = bnd_x_width*bnd_y_width*bnd_z_width;
    int sender_grid_number =
      adjacent_grid_numbers[bnd_x_idx+1][bnd_y_idx+1][bnd_z_idx+1];

    if(IOflag)
      std::cout << "Receiving (" + std::to_string(bnd_x_idx)
        + ", " + std::to_string(bnd_y_idx)
        + ", " + std::to_string(bnd_z_idx)
        + ")-boundary on grid " + std::to_string(grid_number)
        + " from (" + std::to_string(-1*bnd_x_idx)
        + ", " + std::to_string(-1*bnd_y_idx)
        + ", " + std::to_string(-1*bnd_z_idx)
        + ")-boundary on grid " + std::to_string(sender_grid_number)
        + "\n";

    MPI_Recv(&incoming_bnd_array[0],
      bnd_size, MPI_FLOAT, 
      sender_grid_number, bnd_id,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(IOflag)
      std::cout << "Completed receiving (" + std::to_string(bnd_x_idx)
        + ", " + std::to_string(bnd_y_idx)
        + ", " + std::to_string(bnd_z_idx)
        + ")-boundary on grid " + std::to_string(grid_number) + ".\n";
  }

  /**
   * @brief      Store boundary values from grid in an array
   */
  void _setOutgoingBoundaryArray(float * outgoing_bnd_array,
    int bnd_x_idx, int bnd_y_idx, int bnd_z_idx)
  {
    int bnd_x_width = (bnd_x_idx == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y_idx == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z_idx == 0 ? n_z : bnd_width);

    int grid_offset_x = bnd_width,
        grid_offset_y = bnd_width,
        grid_offset_z = bnd_width;

    if(bnd_x_idx == 1)
      grid_offset_x = n_x;
    if(bnd_y_idx == 1)
      grid_offset_y = n_y;
    if(bnd_z_idx == 1)
      grid_offset_z = n_z;

    for(int i = 0; i < bnd_x_width; ++i)
      for(int j = 0; j < bnd_y_width; ++j)
        for(int k = 0; k < bnd_z_width; ++k)
          outgoing_bnd_array[
            i*bnd_y_width*bnd_z_width
            + j*bnd_z_width + k
          ] = array[
            (i+grid_offset_x)*M_y*M_z
            + (j+grid_offset_y)*M_z + (k+grid_offset_z)
          ];
  }

  /**
   * @brief      Store boundary values from array to grid
   */
  void _storeIncomingBoundaryArray(float * incoming_bnd_array,
    int bnd_x_idx, int bnd_y_idx, int bnd_z_idx)
  {
    int bnd_x_width = (bnd_x_idx == 0 ? n_x : bnd_width);
    int bnd_y_width = (bnd_y_idx == 0 ? n_y : bnd_width);
    int bnd_z_width = (bnd_z_idx == 0 ? n_z : bnd_width);

    int grid_offset_x = 0,
        grid_offset_y = 0,
        grid_offset_z = 0;

    if(bnd_x_idx == 0)
      grid_offset_x = bnd_width;
    if(bnd_y_idx == 0)
      grid_offset_y = bnd_width;
    if(bnd_z_idx == 0)
      grid_offset_z = bnd_width;

    if(bnd_x_idx == 1)
      grid_offset_x = n_x + bnd_width;
    if(bnd_y_idx == 1)
      grid_offset_y = n_y + bnd_width;
    if(bnd_z_idx == 1)
      grid_offset_z = n_z + bnd_width;


    for(int i = 0; i < bnd_x_width; ++i)
      for(int j = 0; j < bnd_y_width; ++j)
        for(int k = 0; k < bnd_z_width; ++k)
          array[
            (i+grid_offset_x)*M_y*M_z
            + (j+grid_offset_y)*M_z + (k+grid_offset_z)
          ] = incoming_bnd_array[
            i*bnd_y_width*bnd_z_width
            + j*bnd_z_width + k
          ];
  }

};

}

#endif
