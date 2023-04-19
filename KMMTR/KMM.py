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
                kernel = RBF()  
            elif self.kernel == 'DotProduct':
                kernel = DotProduct()
            elif self.kernel == 'WhiteKernel':
                kernel = WhiteKernel()
            elif self.kernel == 'Matern':
                kernel = Matern()
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
        para = np.exp(GPr.kernel_.theta)

        Inst_kernel = self.call_kernel(para)
        return Inst_kernel


    def cal_beta(self, B=1):
        """
        Helper function to calculate the beta values that balance the source and target data distributions.
        
        Args:
            B (float): Bound on the beta values. Defaults to 1.
        
        Returns:
            A numpy array containing the calculated beta values.
        """
        n = len(self.source_data)
        m = len(self.target_data)
        kernel = self.Instantiate()
        # n x n matrix
        K_SS = kernel(self.source_data)
        # m x n matrix
        K_TS = kernel(self.target_data,self.source_data)

        """
        standard quadratic programming problem 

        min 1/2 * X.T * P * X + q.T * X
            s.t.,
            GX <= h
            AX = b
        """

        # In KMM 
        M_P = 2/n**2 * K_SS
        V_I = np.ones((1,m))
        # matrix multiplication
        V_q = - 2/m/n * np.dot(V_I,K_TS).T
        M_G1 = -np.eye(n)
        M_G2 = np.eye(n)
        M_A = np.ones((1,n))
        V_h1 = np.zeros((n,1))
        V_h2 = B * np.ones((n,1))

        # convert to matrix
        M_P = matrix(M_P)
        V_q = matrix(V_q)
        M_G1_U = copy.deepcopy(matrix(M_G1))
        M_G2_U = copy.deepcopy(matrix(M_G2))
        M_A = matrix(M_A)
        V_h1_U = copy.deepcopy(matrix(V_h1))
        V_h2_U = copy.deepcopy(matrix(V_h2))
        M_G = matrix([M_G1_U, M_G2_U])
        h = matrix([V_h1_U, V_h2_U])

        b = matrix(float(n))


        solvers.options['show_progress'] = False
        sol = solvers.qp(M_P,V_q,M_G,h,M_A,b)
        beta = sol['x']
        return np.array(beta)
