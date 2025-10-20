#!/bin/bash

module load gcc/9.3.0
module load cmake/3.26.3
module load cuda/11.8.0
module load hdf5/1.10.1
module load netcdf/4.6.0


cmake -B build -DENABLE_CUDA=ON
make -C build

compute-sanitizer ./build/benchmarks
