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
        if name == "Linear":
            kernel = self.get_linear_kernel(X1,X2, **parameters)
        if name == "Polynomial":
            kernel = self.get_poly_kernel(X1, X2, **parameters)
        return kernel
    def get_kernel_parallel(self, name, local_data, **parameters):
        def compute_local_kernel(X_local, X):
            return self.get_kernel(name, X_local, X, **parameters)

        # def gather_kernel_matrix(K_local, rank):
        #     recvbuf = self.comm.gather(K_local, root=0)
        #     if rank == 0:
        #         recvbuf = np.vstack([np.hstack(local_data) for local_data in recvbuf])
        #         print('Dim of K full: ', recvbuf.shape[0], '*', recvbuf.shape[1])
        #         print(recvbuf)
        #         return recvbuf
        #     else:
        #         return None

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
    def conjugate_gradient_parallel(self, A_local, y_local, tol=10, max_iter=1000):

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
            # Save the array as a .npy file
            np.save(f'alpha_{self.rank}.npy', self.alpha)
        else:
            np.save('alpha.npy', self.alpha)
    
    def load_alpha(self):
        # Load the array from the .npy file
        if self.do_parallel:
            # Save the array as a .npy file
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
        
    def test(self, train_data, test_data, test_label, **parameters):
        if not self.do_parallel:
            predicted_label = self.predict(train_data, test_data, **parameters)
            print(predicted_label)
            print(test_label)
            test_mse = np.mean((predicted_label - test_label) ** 2)
            return test_mse, predicted_label
        else:
            predicted_label = self.predict(train_data, test_data, **parameters)
            if self.rank == 0:
                print(predicted_label)
                print(test_label)
                test_mse = np.mean((predicted_label - test_label) ** 2)
                return test_mse, predicted_label
            else:
                return None, None
    
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