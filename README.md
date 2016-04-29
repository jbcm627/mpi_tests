# mpi_tests

## On CWRU's cluster:

### Request interactive session, load modules
```{r, engine='bash', loadmodules}
srun -N 8 -c 16 --time=1:00:00 --pty /bin/bash
module load openmpi-gnu/1.8.5
```

### Compile code, run code
```{r, engine='bash', compile}
mpic++ main.cc --std=c++11 -Wall -fopenmp -O3
time mpirun -n 8 ./a.out
```

## Results


