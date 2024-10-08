# DSA5208PJ1

## Environment Setup
```
python -m pip install mpi4py
```

## Kernel Ridge Regression
```
python prediction.py --data_root /path/to/your/data --name KernelName --standard
```
We provide the realization of three kernels: Gaussian Kernel, Linear Kernel and  Polynommial Kernel

We conduct conjugate gradient descent and you will see the visualization of the error curve and the final result after running the prediction.py

## Kernel Ridge Regression with MPI

```
mpirun -n 4 prediction.py --data_root /path/to/your/data --name KernelName --standard --do_parallel
```
you can change '4' into the cpu number you want to use. Setting do_parrallel to True enabling the mpi engagement.

In this part, we parallelize the kernel computation and the conjugate gradient descent by splitting the data storage and matrix computation into multiple processes, which greatly reduce the memory usage in each process, thus improve the calculation speed. Notice that in this project we only test the algorithm with around 14000 data. So the advantage of using MPI upon speed may not that notable because it also adds the communication overhead between processes. 

## Grid Search
We conduct grid search to find the optimal parameters combination which yeilds best results. You can add '--do_grid_search' to do it.
