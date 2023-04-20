import numpy as np
import copy
import warnings
from cvxopt import solvers, matrix 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern 


class KernelMeanMatching():
    def __init__(self,kernel,source_data,target_data,target_response):
        """
        Constructor for KernelMeanMatching class.
        
        Args:
            kernel (str): Name of the kernel function to use.
            source_data (numpy array): n x d numpy array containing the source data features.
            target_data (numpy array): m x d numpy array containing the target data features.
            target_response (numpy array): m x 1(t) numpy array containing the target response values.
        """
        self.kernel = kernel
        # n data points
        self.source_data = source_data
        # m data points
        self.target_data = target_data
        self.target_response = target_response
        warnings.filterwarnings('ignore')

    def call_kernel(self,para = None):
        """
        Helper function to call the kernel function specified by the user.
        
        Args:
            para (numpy array): Parameters to use when instantiating the kernel function. Defaults to None.
        
        Returns:
            A kernel function object based on the user-specified kernel function and parameters.
        """
        if para is None :
            if self.kernel == 'RBF':
                kernel = RBF(length_scale_bounds=(0.01,100))  
            elif self.kernel == 'DotProduct':
                kernel = DotProduct(sigma_0_bounds=(0.01,100))
            elif self.kernel == 'WhiteKernel':
                kernel = WhiteKernel(noise_level_bounds=(0.01,100))
            elif self.kernel == 'Matern':
                kernel = Matern(length_scale_bounds=(0.01,100))
            else:
                print("Unknown kernel !")
                print('Only the following kernel functions are legal')
                print('RBF | DotProduct | WhiteKernel | Matern')
        else :
            if self.kernel == 'RBF':
                kernel = RBF(para,"fixed")
            elif self.kernel == 'DotProduct':
                kernel = DotProduct(para,"fixed")
            elif self.kernel == 'WhiteKernel':
                kernel = WhiteKernel(para,"fixed")
            elif self.kernel == 'Matern':
                kernel = Matern(para,"fixed")
            else:
                print("Unknown kernel !")
                print('Only the following kernel functions are legal')
                print('RBF | DotProduct | WhiteKernel | Matern')
        return kernel
    
    def Instantiate(self, ):
        """
        Helper function to instantiate the Gaussian Process Regressor and set the kernel parameters.
        
        Returns:
            An instantiated kernel function object.
        """
        Xtrain = np.array(self.target_data)
        Ytrain = np.array(self.target_response)
        ker = self.call_kernel()
        GPr = GPR(kernel=ker).fit(Xtrain,Ytrain)
        print('Trained Kernel Function :', GPr.kernel_)
        para = np.exp(GPr.kernel_.theta)

        Inst_kernel = self.call_kernel(para)
        return Inst_kernel


    def cal_beta(self, B=1,tao = None):
        """
        Helper function to calculate the beta values that balance the source and target data distributions.
        
        Args:
            B (float): Bound on the beta values. Defaults to 1.
        
        Returns:
            A numpy array containing the calculated beta values.
        """
        
        n = len(self.source_data)
        m = len(self.target_data)
        if tao is None:
            tao = B/np.sqrt(n)/2
        elif type(tao) == float:
            tao = tao
        else:
            print('The input parameter tao must be a float, like 0.1')
        
        """
        1-tao <= sum(beta_i) <= 1+tao

        E(beta) = 1 , E is expectation 
        """
        kernel = self.Instantiate()
        # n x n matrix
        K_SS = kernel(self.source_data)
        # m x n matrix
        K_ST = kernel(self.source_data,self.target_data)


        """
        standard quadratic programming problem 

        minimize    (1/2) * X.T * P * X + q.T * X
        subject to  G * X <= h
                    A * X = b

        M_P: a square matrix of size (n, n) representing the quadratic coefficients of the objective function. The matrix must be symmetric and positive semidefinite, meaning that x.T M_P x >= 0 for any vector x.

        V_q: a column vector of size (n, 1) representing the linear coefficients of the objective function.

        M_G: a matrix of size (m, n) representing the linear constraints of the problem. Each row of M_G represents a constraint, and each column represents a variable.

        h: a column vector of size (m, 1) representing the right-hand side of the linear constraints. Each element of h corresponds to a row of M_G.

        M_A: an optional matrix of size (p, n) representing additional linear equality constraints of the problem. Each row of M_A represents a constraint, and each column represents a variable.

        b: an optional column vector of size (p, 1) representing the right-hand side of the additional linear equality constraints. Each element of b corresponds to a row of M_A.
        """

        # In KMM 
        M_P =  K_SS
        # matrix multiplication
        V_q = - n/m * np.sum(K_ST,axis = 1)
        M_G1 = -np.eye(n)
        M_G2 = np.eye(n)
        M_A1 = np.ones((1,n))/n
        M_A2 = -np.ones((1,n))/n
        V_h1 = np.zeros((n,1))
        V_h2 = B * np.ones((n,1))

        # convert to matrix
        M_P = matrix(M_P)
        V_q = matrix(V_q)
        M_G1_U = copy.deepcopy(matrix(M_G1))
        M_G2_U = copy.deepcopy(matrix(M_G2))
        M_A1 = matrix(M_A1)
        M_A2 = matrix(M_A2)
        V_h1_U = copy.deepcopy(matrix(V_h1))
        V_h2_U = copy.deepcopy(matrix(V_h2))
        M_G = matrix(np.r_[M_G1_U, M_G2_U,M_A1,M_A2])
        h1 = matrix(1+tao)
        h2 = matrix(tao-1)
        h = matrix(np.r_[V_h1_U, V_h2_U,h1,h2])

        

        solvers.options['show_progress'] = True
        sol = solvers.qp(M_P,V_q,M_G,h)
        beta = sol['x']
        return np.array(beta)
