import numpy as np
from mpi4py import MPI


class Kernel:
    def __init__(self,args, comm=None):
        if args.do_parallel:
            self.comm = comm  # Use the provided global communicator
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
    def get_kernel(self, name, X1, X2, **parameters):
        if name == "Gaussian":
            kernel = self.get_gaussian_kernel(X1,X2, **parameters)
        # The linear and Sigmoid can not converge during conjugate gradient descent, still exists some problems 
        if name == "Linear":
            kernel = self.get_linear_kernel(X1,X2, **parameters)
        if name == "Polynomial":
            kernel = self.get_poly_kernel(X1, X2, **parameters)
        return kernel

    def get_kernel_parallel(self, name, train_data, **parameters):
        # Step 2: Compute local kernel matrix
        def compute_local_kernel(X_local, X):
            return self.get_kernel(name, X_local, X, **parameters)

        # Step 3: Gather kernel matrix from all processes
        def gather_kernel_matrix(K_local, rank):
            # Gather the local parts of the kernel matrix
            K_full = self.comm.gather(K_local, root=0)
            if rank == 0:
                # Combine the gathered parts into the full kernel matrix
                K_full = np.vstack([np.hstack(K_local) for K_local in K_full])
                print('Dim of K {} x {}'.format(K_full.shape[0],K_full.shape[1]))
                return K_full
            else:
                return None

        X1_split = np.array_split(train_data, self.size, axis=0)
        X2_split = np.array_split(train_data, self.size, axis=0)
        X1_local = X1_split[self.rank]
        local_kernel = []
    
        for round in range(self.size):
            # send_data = X1_local.copy()
            # recv_data = np.empty_like(X1_local)
            # self.comm.Sendrecv(send_data, dest=(self.rank + round) % size, recvbuf=recv_data, source=(rank - round) % size)
            X2_local = X2_split[round]
            kernel_block = compute_local_kernel(X1_local, X2_local)
            local_kernel.append(kernel_block)

        K_full = gather_kernel_matrix(local_kernel, self.rank)
        return K_full
        # rows_per_process = None
        # if self.rank == 0:
        #     rows_per_process = np.array_split(train_data, self.size, axis=0)
        # else:
        #     rows_per_process = None
        # rows_per_process = self.comm.scatter(rows_per_process, root=0)
        # local_kernel = compute_local_kernel(rows_per_process, rows_per_process)
        # print(f"Rank {self.rank} computed local_result:")
        # print(local_kernel)
        # print(train_data.shape[0])
        # if self.rank == 0:
        #     full_kernel = np.zeros((train_data.shape[0],train_data.shape[0]))
        # else:
        #     full_kernel = None
        # self.comm.Gather(local_kernel, full_kernel)
        # if self.rank == 0:
        #     return full_kernel
        # else:
        #     return None

    def get_gaussian_kernel(self,X1, X2, **paramters):
        sigma = paramters.pop("sigma")
        dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-dists / (2 * sigma ** 2))
    
    def get_linear_kernel(self, X1, X2, **paramters):
        return np.dot(X1, X2.T)
    
    def get_poly_kernel(self, X1, X2, **paramters):
        c = paramters.pop("c")
        degree = paramters.pop("degree")
        return (np.dot(X1, X2.T) + c) ** degree
    
class KernelRidgeRegression:
    def __init__(self,kernel,args, comm):
        self.kernel = kernel
        self.kernel_name = args.name
        self.lambd = args.lambd
        self.do_parallel=args.do_parallel
        if args.do_parallel:
            self.comm = comm
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
    def conjugate_gradient(self, A, y, threshold=1e-10):
        iter = 0
        n = y.shape[0]
        # Initialization
        alpha = np.ones(n)
        r = y - np.dot(A,alpha)
        p = r.copy()
        se = np.dot(r, r)
        error_list = [se]
        while True: 
            print(f"step {iter+1} : {se}")
            w = np.dot(A,p)
            s=se / np.dot(p, w)
            alpha = alpha + s * p
            r = r - s * w
            # import pdb; pdb.set_trace()
            new_se = np.dot(r, r)
            error = y - np.dot(A, alpha)
            error_list.append(np.sqrt(np.dot(error, error)))
            if np.sqrt(new_se) < threshold:
                print(f"Conjugate Gradient Descent converges at {iter+1} step")
                break
            beta = new_se / se 
            p = r + beta * p
            se = new_se
            iter += 1
        return alpha, error_list       
    def conjugate_gradient_parallel(self, A, y, tol=1e-10, max_iter=1000):
        A = self.comm.bcast(A, root=0)
        y = self.comm.bcast(y, root=0)

        self.tol = tol
        self.max_iter = max_iter
        # Get matrix and vector dimensions
        self.N = A.shape[0]  # Assume A is square, N x N

        # Determine how many rows each process should handle
        self.local_rows = self.N // self.size
        self.extra_rows = self.N % self.size

        # Determine the slice of A each process will handle
        if self.rank < self.extra_rows:
            self.row_start = self.rank * (self.local_rows + 1)
            self.row_end = self.row_start + self.local_rows + 1
        else:
            self.row_start = self.rank * self.local_rows + self.extra_rows
            self.row_end = self.row_start + self.local_rows

        # Each process gets its own slice of the matrix and the corresponding part of y
        self.A_local = A[self.row_start:self.row_end, :]
        self.y_local = y[self.row_start:self.row_end]

        # Initialize global solution vector alpha
        alpha = np.zeros(self.N)

        # Compute the initial residual: r = y - A * alpha
        r_local = self.y_local - self.A_local.dot(alpha)
        p_local = r_local.copy()

        # Compute the local dot product r^T r
        local_rs_old = np.dot(r_local, r_local)

        # Perform a global reduction to sum r^T r across all processes
        rs_old = self.comm.allreduce(local_rs_old, op=MPI.SUM)

        error_list = [rs_old]

        for i in range(self.max_iter):
            if self.rank == 0:
                print(f"step {i+1} : {rs_old}")
            # Gather the local p into global_p across all processes
            p_global = np.zeros(self.N)
            self.comm.Allgather(p_local, p_global)

            # Local matrix-vector product A_local * global_p
            local_Ap = self.A_local.dot(p_global)

            # Compute the local dot product p^T A p
            local_pAp = np.dot(p_local, local_Ap)

            # Perform a global reduction to sum p^T A p across all processes
            global_pAp = self.comm.allreduce(local_pAp, op=MPI.SUM)

            # Compute the global step size alpha_step
            alpha_step = rs_old / global_pAp

            # Update the global solution vector alpha using the global step size
            alpha += alpha_step * p_global

            # Update the residual: r = r - alpha_step * A * p
            r_local -= alpha_step * local_Ap

            # Compute the new local dot product r^T r
            local_rs_new = np.dot(r_local, r_local)

            # Perform a global reduction to sum the new r^T r
            rs_new = self.comm.allreduce(local_rs_new, op=MPI.SUM)
            if self.rank == 0:
                error = y - np.dot(A, alpha)
                
                error_list.append(np.sqrt(np.dot(error, error)))

            # Check for convergence
            if np.sqrt(rs_new) < self.tol:
                if self.rank == 0:
                    print(f"Converged in {i + 1} iterations.")
                break

            # Update the conjugate direction p
            p_local = r_local + (rs_new / rs_old) * p_local

            # Update rs_old
            rs_old = rs_new

        # Return the solution only on the root process
        if self.rank == 0:
            return alpha, error_list
        else:
            return None, None
    def train(self, train_data, train_label, **parameters):
        if not self.do_parallel:
            n_samples = train_data.shape[0]
            K = self.kernel.get_kernel(self.kernel_name, train_data, train_data, **parameters)
            K += self.lambd * np.eye(n_samples)
            alpha, error_list = self.conjugate_gradient(K, train_label)
            self.alpha = alpha
            predicted_label = np.dot(K, alpha)

            train_mse = np.mean((predicted_label - train_label) ** 2)
            return train_mse, error_list
        else:
            print(f"rank{self.rank} calculating kernel now")
            K = self.kernel.get_kernel_parallel(self.kernel_name, train_data, **parameters)
            # train_label = self.comm.gather(train_label, root=0)
            K = self.comm.bcast(K, root=0)
            n_samples = K.shape[0]
            K += self.lambd * np.eye(n_samples)
            # train_label = np.vstack(train_label)
            alpha, error_list = self.conjugate_gradient_parallel(K, train_label)
            if self.rank==0:
                self.alpha = alpha
                predicted_label = np.dot(K, alpha)

                train_mse = np.mean((predicted_label - train_label) ** 2)
                return train_mse, error_list
            else:
                return None, None

    def test(self, train_data, test_data, test_label,**parameters):
        K_test = self.kernel.get_kernel(self.kernel_name, test_data, train_data, **parameters)
        predicted_label = np.dot(K_test, self.alpha)
        print(predicted_label)
        print(test_label)
        # print(predicted_label - test_label)
        test_mse = np.mean((predicted_label - test_label) ** 2)
        return test_mse, predicted_label
    
    def grid_search(self,train_data, train_label, test_data, test_label, **parameters):
        optimal_parameter = {self.kernel:{}}
        if self.kernel_name == "Gaussian":
            sigma_space = np.linspace(-5, 5, 0.1)
            min_mse = 0.
            for sigma in sigma_space:
                train_mse, _ = self.train(train_data, train_label, sigma=sigma)
                test_mse = self.test(train_data, test_data, test_label, sigma=sigma)
                if test_mse < min_mse:
                     min_mse = test_mse
                     optimal_parameter[self.kernel_name]["sigma"] = sigma
        if self.kernel_name == "Linear":
            pass
        if self.kernel_name == "Polynomial":
            pass
        
        return optimal_parameter
