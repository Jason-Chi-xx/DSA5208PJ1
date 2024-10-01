# DSA5208PJ1
Project 1 of DSA5208

## Environment Setup
```
python -m pip install mpi4py
```

## Kernel Ridge Regression
```
python prediction.py --data_root /path/to/your/data --name KernelName
```
We provide the realization of three kernels: Gaussian Kernel, Linear Kernel and  Polynommial Kernel

We conduct conjugate gradient descent and you will see the visualization of the error curve and the final result after running the prediction.py

## Kernel Ridge Regression with MPI

```
mpirun -n 4 prediction.py --data_root /path/to/your/data --name KernelName --do_parallel True
```
you can change '4' into the cpu number you want to use. Setting do_parrallel to True enabling the mpi engagement.

