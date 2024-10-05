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
    def get_kernel_parallel(self, name, local_data, **parameters):
        def compute_local_kernel(X_local, X):
            return self.get_kernel(name, X_local, X, **parameters)

        local_kernel = [0] * self.size
    
        for round in range(self.size):
            send_data = local_data.copy()
            recv_data = np.empty_like(send_data)
            self.comm.Sendrecv(send_data, dest=(self.rank + round) % self.size, recvbuf=recv_data, source=(self.rank - round) % self.size)
            
            kernel_block = compute_local_kernel(local_data, recv_data)
            print(f'Round {round} on Rank {self.rank}')
            local_kernel[(self.rank - round) % self.size] = kernel_block
        
        print(f'Wait for Kernel Computation to complete on Rank {self.rank}')
        self.comm.Barrier()

        K_local = np.hstack(local_kernel)
        return K_local
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
        self.standardized = args.standard
        if args.do_parallel:
            self.comm = comm
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
    def conjugate_gradient(self, A, y, threshold=5.):
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
    def conjugate_gradient_parallel(self, A_local, y_local, tol=1e-2, max_iter=1000):

        self.tol = tol
        self.max_iter = max_iter
        self.N = A_local.shape[1]

        alpha = np.zeros(self.N)
        r_local = y_local - A_local.dot(alpha)
        p_local = r_local.copy()
        local_rs_old = np.dot(r_local, r_local)
        rs_old = self.comm.allreduce(local_rs_old, op=MPI.SUM)
        error_list = [rs_old]

        for i in range(self.max_iter):
            
            print(f"Rank {self.rank} step {i+1} : {rs_old}")
            p_global = np.zeros(self.N)
            self.comm.Allgather(p_local, p_global)

            local_Ap = np.dot(A_local, p_global)
            local_pAp = np.dot(p_local, local_Ap)
            global_pAp = self.comm.allreduce(local_pAp, op=MPI.SUM)
            alpha_step = rs_old / global_pAp
            alpha += alpha_step * p_global

            r_local -= alpha_step * local_Ap
            local_rs_new = np.dot(r_local, r_local)
            rs_new = self.comm.allreduce(local_rs_new, op=MPI.SUM)

            if self.rank == 0:
                error_list.append(np.sqrt(rs_new))

            if np.sqrt(rs_new) < self.tol:
                if self.rank == 0:
                    print(f"Converged in {i + 1} iterations.")
                break

            p_local = r_local + (rs_new / rs_old) * p_local
            rs_old = rs_new

            self.comm.Barrier()
        alpha = np.array_split(alpha, self.size, axis=0)[self.rank]
        if self.rank == 0:
            return alpha, error_list
        else:
            return alpha, None
    def train(self, train_data, train_label, **parameters):
        self.lambd = parameters.pop("lambd")
        if self.standardized == True:
            self.train_label_mean = train_label.mean(axis=0)
            self.train_label_std = np.std(train_label, axis=0)
            train_label = (train_label - self.train_label_mean) / self.train_label_std
        if not self.do_parallel:
            n_samples = train_data.shape[0]
            K = self.kernel.get_kernel(self.kernel_name, train_data, train_data, **parameters)
            K += self.lambd * np.eye(n_samples)
            alpha, error_list = self.conjugate_gradient(K, train_label)
            self.alpha = alpha
            self.save_alpha()
            return error_list
        else:
            print(f"rank{self.rank} calculating kernel now")
            K = self.kernel.get_kernel_parallel(self.kernel_name, train_data, **parameters)
            n_samples = K.shape[1]
            K += self.lambd * np.array_split(np.eye(n_samples), self.size, axis=0)[self.rank]

            alpha, error_list = self.conjugate_gradient_parallel(K, train_label)
            self.alpha = alpha
            self.save_alpha()
            if self.rank==0:
                return error_list
            else:
                return None
    def save_alpha(self):
        if self.do_parallel:
            np.save(f'alpha_{self.rank}.npy', self.alpha)
        else:
            np.save('alpha.npy', self.alpha)
    
    def load_alpha(self):
        if self.do_parallel:
            self.alpha = np.load(f'alpha_{self.rank}.npy')
        else:
            self.alpha = np.load(f'alpha.npy')

    def predict(self, train_data, pred_data, **parameters):
        K = self.kernel.get_kernel(self.kernel_name, pred_data, train_data, **parameters)
        if not self.do_parallel:
            predicted_label = np.dot(K, self.alpha)
        else:
            predicted_label = np.dot(K, self.alpha)
            predicted_label = self.comm.reduce(predicted_label, op=MPI.SUM, root=0)
        return predicted_label
    def test(self, train_data, test_data, test_label,**parameters):
        if not self.do_parallel: 
            predicted_label = self.predict(train_data, test_data, **parameters)
            if self.standardized == True:
                predicted_label = predicted_label * self.train_label_std + self.train_label_mean
            print(predicted_label)
            print(test_label)
            # print(predicted_label - test_label)
            test_mse = np.mean((predicted_label - test_label) ** 2)
            return test_mse, predicted_label
        else:
            predicted_label = self.predict(train_data, test_data, **parameters)
            if self.rank == 0:
                if self.standardized == True:
                    predicted_label = predicted_label * self.train_label_std + self.train_label_mean
                print(predicted_label)
                print(test_label)
                test_mse = np.mean((predicted_label - test_label) ** 2)
                return test_mse, predicted_label
            else:
                return None, None
    
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
                        error_list = self.train(train_data, train_label, **parameters)
                        validation_mse, _ = self.test(train_data, validation_data, validation_label, **parameters)

                        # Store values for plotting
                        sigma_values.append(sigma)
                        lambd_values.append(lambd)
                        validation_mses.append(validation_mse)

                        f.write(f"Parameters: sigma={sigma}, lambd={lambd}\n")
                        f.write(f"Validation MSE: {validation_mse}\n\n")
                        f.write("---------------------------------------\n")

                        if validation_mse < min_mse:
                            min_mse = validation_mse
                            optimal_parameter[self.kernel_name]["sigma"] = sigma
                            optimal_parameter[self.kernel_name]["lambd"] = lambd

            log_validation_mses = np.log10(validation_mses)
            log_lambd_values = np.log10(lambd_values)
            log_sigma_mses = np.log10(sigma_values)

            
            fig = plt.figure(figsize=(10, 6)) # Create a 3D scatter plot of the log-transformed validation MSE values
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(log_sigma_mses, log_lambd_values, log_validation_mses, c=log_validation_mses, cmap='viridis')

            ax.set_xlabel('Log(Sigma)')
            ax.set_ylabel('Log(Lambda)')
            ax.set_zlabel('Log(Validation MSE)')
            ax.set_title('3D Scatter Plot of Log-Transformed Validation MSE')

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
                    error_list = self.train(train_data, train_label, **parameters)
                    validation_mse, _ = self.test(train_data, validation_data, validation_label, **parameters)

                    lambd_values.append(lambd)
                    validation_mses.append(validation_mse)

                    f.write(f"Parameters: lambd={lambd}\n")
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
            lambd_space = [10 ** i for i in range(0, 2)]
            degree_space = [3]
            c_space = [1, 10]
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

                            error_list = self.train(train_data, train_label, **parameters)
                            validation_mse, _ = self.test(train_data, validation_data, validation_label, **parameters)

                            c_values.append(c)
                            lambd_values.append(lambd)
                            validation_mses.append(validation_mse)

                            f.write(f"Parameters: degree={degree}, c={c}, lambd={lambd}\n")
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

            scatter = ax.scatter(log_c_values, log_lambd_values, log_validation_mses,
                                c=log_validation_mses, cmap='viridis')

            ax.set_xlabel('Log(c)')
            ax.set_ylabel('Log(Lambda)')
            ax.set_zlabel('Log(Validation MSE)')
            ax.set_title('3D Scatter Plot of Log-Transformed Validation MSE')

            fig.colorbar(scatter)

            plt.savefig(
                f'{self.kernel_name}_{self.standardized}_log_validation_mse_3d_scatter_plot.png')
            plt.close()

        return optimal_parameter
