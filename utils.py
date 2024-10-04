import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
LOCAL = True

class Kernel:
    def __init__(self,args, comm=None):
        if args.do_parallel:
            self.comm = comm  # Use the provided global communicator
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
    def get_kernel(self, name, X1, X2, **parameters):
        if name == "Gaussian":
            kernel = self.get_gaussian_kernel(X1,X2, **parameters)
        if name == "Linear":
            kernel = self.get_linear_kernel(X1,X2, **parameters)
        if name == "Polynomial":
            kernel = self.get_poly_kernel(X1, X2, **parameters)
        return kernel
    def get_kernel_parallel(self, name, train_data, **parameters):
        def compute_local_kernel(X_local, X):
            return self.get_kernel(name, X_local, X, **parameters)

        def gather_kernel_matrix(K_local, rank):
            recvbuf = self.comm.gather(K_local, root=0)
            if rank == 0:
                print(recvbuf)
                return np.vstack([np.hstack(local_data) for local_data in recvbuf])
            else:
                return None

        if not LOCAL:
            X_split = np.array_split(train_data, self.size, axis=0)
            local_data = X_split[self.rank]
        else:
            local_data = train_data
        local_kernel = [0] * self.size
    
        for round in range(self.size):
            if LOCAL:
                send_data = local_data.copy()
                recv_data = np.empty_like(send_data)
                self.comm.Sendrecv(send_data, dest=(self.rank + round) % self.size, recvbuf=recv_data, source=(self.rank - round) % self.size)
            else:
                recv_data = X_split[round]
            
            kernel_block = compute_local_kernel(local_data, recv_data)
            local_kernel[(self.rank - round) % self.size] = kernel_block
        K_full = gather_kernel_matrix(local_kernel, self.rank)
        return K_full
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
    def conjugate_gradient_parallel(self, A, y, tol=1e-5, max_iter=1000):
        A = self.comm.bcast(A, root=0)
        y = self.comm.bcast(y, root=0)

        self.tol = tol
        self.max_iter = max_iter
        self.N = A.shape[0]

        self.local_rows = self.N // self.size
        self.extra_rows = self.N % self.size

        if self.rank < self.extra_rows:
            self.row_start = self.rank * (self.local_rows + 1)
            self.row_end = self.row_start + self.local_rows + 1
        else:
            self.row_start = self.rank * self.local_rows + self.extra_rows
            self.row_end = self.row_start + self.local_rows

        self.A_local = A[self.row_start:self.row_end, :]
        self.y_local = y[self.row_start:self.row_end]

        alpha = np.zeros(self.N)
        r_local = self.y_local - self.A_local.dot(alpha)
        p_local = r_local.copy()
        local_rs_old = np.dot(r_local, r_local)
        rs_old = self.comm.allreduce(local_rs_old, op=MPI.SUM)
        error_list = [rs_old]

        for i in range(self.max_iter):
            if self.rank == 0:
                print(f"step {i+1} : {rs_old}")
            p_global = np.zeros(self.N)
            self.comm.Allgather(p_local, p_global)

            local_Ap = self.A_local.dot(p_global)

            local_pAp = np.dot(p_local, local_Ap)

            global_pAp = self.comm.allreduce(local_pAp, op=MPI.SUM)

            alpha_step = rs_old / global_pAp

            alpha += alpha_step * p_global

            r_local -= alpha_step * local_Ap

            local_rs_new = np.dot(r_local, r_local)

            rs_new = self.comm.allreduce(local_rs_new, op=MPI.SUM)
            if self.rank == 0:
                error = y - np.dot(A, alpha)
                
                error_list.append(np.sqrt(np.dot(error, error)))

            if np.sqrt(rs_new) < self.tol:
                if self.rank == 0:
                    print(f"Converged in {i + 1} iterations.")
                break

            p_local = r_local + (rs_new / rs_old) * p_local

            rs_old = rs_new

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
    
    def grid_search(self, train_data, train_label, validation_data, validation_label, **parameters):
        optimal_parameter = {self.kernel_name: {}}
        lambd_space = [10**i for i in range(-5, 4)]

        if self.kernel_name == "Gaussian":
            sigma_space = np.logspace(-2, 2, 10)  # 10 values between 0.01 and 100 on a logarithmic scale
            min_mse = float('inf')

            # Initialize lists to store values for plotting
            sigma_values = []
            lambd_values = []
            validation_mses = []

            with open(f'{self.kernel_name}_girdsearch_results.txt', 'w') as f:
                for sigma in sigma_space:
                    for lambd in lambd_space:
                        parameters['sigma'] = sigma
                        parameters['lambd'] = lambd
                        train_mse, _ = self.train(train_data, train_label, **parameters)
                        validation_mse, _ = self.test(train_data, validation_data, validation_label, **parameters)

                        # Store values for plotting
                        sigma_values.append(sigma)
                        lambd_values.append(lambd)
                        validation_mses.append(validation_mse)

                        f.write(f"Parameters: sigma={sigma}, lambd={lambd}\n")
                        f.write(f"Train MSE: {train_mse}\n")
                        f.write(f"Validation MSE: {validation_mse}\n\n")
                        f.write("---------------------------------------\n")

                        if validation_mse < min_mse:
                            min_mse = validation_mse
                            optimal_parameter[self.kernel_name]["sigma"] = sigma
                            optimal_parameter[self.kernel_name]["lambd"] = lambd

            # Apply logarithmic transformation to the validation MSE values
            log_validation_mses = np.log10(validation_mses)

            # Apply logarithmic transformation to the lambda values
            log_lambd_values = np.log10(lambd_values)

            # Apply logarithmic transformation to the sigma values
            log_sigma_mses = np.log10(sigma_values)

            # Create a 3D scatter plot of the log-transformed validation MSE values
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(log_sigma_mses, log_lambd_values, log_validation_mses, c=log_validation_mses, cmap='viridis')

            ax.set_xlabel('Log(Sigma)')
            ax.set_ylabel('Log(Lambda)')
            ax.set_zlabel('Log(Validation MSE)')
            ax.set_title('3D Scatter Plot of Log-Transformed Validation MSE')

            # Add a color bar for reference
            fig.colorbar(scatter)

            plt.savefig(f'{self.kernel_name}_{self.standardized}_log_validation_mse_3d_scatter_plot.png')
            plt.close()

        if self.kernel_name == "Linear":
            lambd_space = [10 ** i for i in range(-9, 3)]
            min_mse = float('inf')

            lambd_values = []
            validation_mses = []

            with open(f'{self.kernel_name}_girdsearch_results.txt', 'w') as f:
                for lambd in lambd_space:
                    parameters['lambd'] = lambd
                    train_mse, _ = self.train(train_data, train_label, **parameters)
                    validation_mse, _ = self.test(train_data, validation_data, validation_label, **parameters)

                    lambd_values.append(lambd)
                    validation_mses.append(validation_mse)

                    f.write(f"Parameters: lambd={lambd}\n")
                    f.write(f"Train MSE: {train_mse}\n")
                    f.write(f"Validation MSE: {validation_mse}\n\n")
                    f.write("---------------------------------------\n")

                    if validation_mse < min_mse:
                        min_mse = validation_mse
                        optimal_parameter[self.kernel_name]["lambd"] = lambd

            plt.style.use('seaborn-darkgrid')

            # Create the plot
            plt.figure(figsize=(8, 6))
            plt.plot(np.log10(lambd_values), np.log10(validation_mses), marker='o', linestyle='-', color='b',
                     markersize=8, linewidth=2)

            plt.grid(True, which="both", ls="--", linewidth=0.5)
            plt.title('log(Validation MSE) vs log(Lambda)', fontsize=14)
            plt.xlabel('log(Lambda)', fontsize=12)
            plt.ylabel('log(Validation MSE)', fontsize=12)

            plt.savefig('validation_mse_vs_lambda_loglog.png', dpi=300, bbox_inches='tight')

        if self.kernel_name == "Polynomial":
            lambd_space = [10 ** i for i in range(-4, 3)]
            degree_space = [3]
            c_space = [0.1, 1, 10, 100]
            min_mse = float('inf')

            c_values = []
            lambd_values = []
            validation_mses = []

            with open(f'{self.kernel_name}_girdsearch_results.txt', 'w') as f:
                for degree in degree_space:
                    for c in c_space:
                        for lambd in lambd_space:

                            parameters['degree'] = degree
                            parameters['c'] = c
                            parameters['lambd'] = lambd

                            train_mse, _ = self.train(train_data, train_label, **parameters)
                            validation_mse, _ = self.test(train_data, validation_data, validation_label, **parameters)

                            c_values.append(c)
                            lambd_values.append(lambd)
                            validation_mses.append(validation_mse)

                            f.write(f"Parameters: degree={degree}, c={c}, lambd={lambd}\n")
                            f.write(f"Train MSE: {train_mse}\n")
                            f.write(f"Validation MSE: {validation_mse}\n\n")
                            f.write("---------------------------------------\n")

                            if validation_mse < min_mse:
                                min_mse = validation_mse
                                optimal_parameter[self.kernel_name]["lambd"] = lambd
                                optimal_parameter[self.kernel_name]["degree"] = degree
                                optimal_parameter[self.kernel_name]["c"] = c

            log_validation_mses = np.log10(validation_mses)

            log_lambd_values = np.log10(lambd_values)

            log_c_values = np.log10(c_values)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the scatter points with the log-transformed MSE values
            scatter = ax.scatter(log_c_values, log_lambd_values, log_validation_mses,
                                c=log_validation_mses, cmap='viridis')

            ax.set_xlabel('Log(c)')
            ax.set_ylabel('Log(Lambda)')
            ax.set_zlabel('Log(Validation MSE)')
            ax.set_title('3D Scatter Plot of Log-Transformed Validation MSE')

            # Add a color bar for reference
            fig.colorbar(scatter)

            plt.savefig(
                f'{self.kernel_name}_{self.standardized}_log_validation_mse_3d_scatter_plot.png')
            plt.close()

        return optimal_parameter
