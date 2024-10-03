"""
parallel training:
version 1.
    split the data
    split the weight computation
version 2.
    Additionally split the matrix computation
"""

import argparse
from mpi4py import MPI
# import matplotlib.pyplot as plt
from dataset import SplitedDataset
from utils import Kernel, KernelRidgeRegression
import numpy as np
# from sklearn.kernel_ridge import KernelRidge
import math


# def visualization(error_list):
#     x = list(range(len(error_list)))
#     # print(error_list)
#     log_error_list = list(map(lambda k: math.log(k,10), error_list))
#     plt.plot(x,log_error_list,)
#     plt.ylabel("residual (log10)")
#     plt.legend()
#     plt.savefig("residual_fig.png")


def arg_parse():
    parser = argparse.ArgumentParser(description="Parameters for House value prediction")
    
    parser.add_argument("--name",type=str, default="Gaussian", help="kernel name")
    parser.add_argument("--lambd",type=float, default=1.0, help="path of dataset")
    parser.add_argument("--data_root",type=str, default=None, required=True, help="path of dataset")
    parser.add_argument("--standard", type=bool, default=True, help="whether the dataset is standardized")
    parser.add_argument("--sigma", type=float, default=1.0, help="parameter for Gaussian")
    parser.add_argument("--c", type=float, default=0.0, help="parameter for Polynomial")
    parser.add_argument("--degree", type=float, default=2.0, help="parameter for Polynomial")
    parser.add_argument("--do_parallel", type=bool, default=False, help="whether use MPI")
    args = parser.parse_args()
    return args

import time

if __name__ == "__main__":
    stime = time.time()
    print('Start time', stime)
    args = arg_parse()
    Dataset = SplitedDataset(args.data_root,standard=args.standard)
    
    train_data, train_label, test_data, test_label = Dataset.get_data(shuffle=False)
    if args.do_parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        size = 1

    print("dataset size:", len(train_data))
    max_len = (len(train_data) // size ) * size
    max_len = min(max_len, 20000)
    train_data = train_data[:max_len]
    train_label = train_label[:max_len]
    
    if args.do_parallel:
        train_data = np.array_split(train_data, size, axis=0)[rank]
        train_label = np.array_split(train_label, size, axis=0)[rank]
        print("local dataset size:", len(train_data))

    kernel = Kernel(args, comm)
    Krr = KernelRidgeRegression(kernel, args, comm)
    parameter_dict = {
        "sigma": args.sigma,
        "c": args.c,
        "degree": args.degree
    }
    error_list = Krr.train(train_data, train_label, **parameter_dict)
    
    test_mse, predicted_label = Krr.test(train_data, test_data, test_label, **parameter_dict)
    if not args.do_parallel or (args.do_parallel and rank == 0):
        
        print(f"the test mse is : {test_mse}")
        error = (test_label - predicted_label) / test_label
        mean_error = np.mean(error)
        print(f"the average test error is : {mean_error}")
        std_error = np.mean((error - mean_error) ** 2)
        print(f"the std of error is {std_error}")
        with open("result.txt", 'w') as f:

            # f.write(f"train_mse is : {train_mse} \n")
            f.write(f"test_mse is : {test_mse} \n")
            f.write(f"the average error is {mean_error} \n")
            f.write(f"the std of error is {std_error} \n")
        etime = time.time()
        print('End time', etime)
        print(f'Programe takes: {etime - stime} s')