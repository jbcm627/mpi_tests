#!/bin/bash

#SBATCH --ntasks=64
#SBATCH -t 00:30:00
#SBATCH --nodes=4-4

export OMP_NUM_THREADS=1

module load openmpi/1.8.5
module load intel/2015
mpic++ main.cc -std=c++11 -Wall -openmp -O3

# try running using different # ranks

COUNTER=1
while [ $COUNTER -lt 65 ]; do
  echo "Running with rank=" $COUNTER
  mpirun -n $COUNTER --report-bindings ./a.out
  let COUNTER=COUNTER+1
  echo ""
  echo ""
  sleep 1
done
