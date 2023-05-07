import numpy as np
import warnings
from .KMM import KernelMeanMatching


class KMMTransferReg():
    """
    KMMTransferReg:
    KMMTransferReg is a transfer learning regression model based on Kernel Mean Matching (KMM) algorithm. It maps
    the source and target datasets to a common feature space, where the marginal distributions of the source and target
    data are as close as possible, while maintaining the conditional distribution of the source data. The mapped data
    is then used to train a regression model that can make predictions on the target dataset.

    Author:
    ----------
    Bin Cao, ZheJiang LAB, Hangzhou, CHINA.

    Parameters
    ----------
    Regressor : str or object, default='RF'
        A regression model used to fit the mapped data. If a string is passed, it should be one of {'RF', 'LR'}
        representing RandomForestRegressor and LinearRegression, respectively. Otherwise, an object with fit and
        predict methods that implements a regression algorithm can be passed.
    UpBound : float, default=1
        The upper bound of the beta for the target dataset during kernel mean matching.
    kernel : str, default='RBF'
        Kernel function used for kernel mean matching. It should be one of {'RBF', 'DotProduct', 'WhiteKernel', 'Matern'}
        representing the Radial Basis Function kernel, Dot Product kernel, White noise kernel and Matern kernel,
        respectively.
    Targets : int, default=1
        The number of target variables to be predicted.

    Methods
    -------
    fit(source_dataset,target_dataset,test_data)
        Fit the transfer learning regression model to the source dataset and target dataset, and predict the target variable
        on the test dataset.
    """
    def __init__(self, Regressor='RF',UpBound=1,kernel = 'RBF', Targets = 1):
        """
        Initialize KMMTransferReg model.

        Parameters
        ----------
        Regressor : str or object, default='RF'
            A regression model used to fit the mapped data. If a string is passed, it should be one of {'RF', 'LR'}
            representing RandomForestRegressor and LinearRegression, respectively. Otherwise, an object with fit and
            predict methods that implements a regression algorithm can be passed.
        UpBound : float, default=1
            The upper bound for beta coefficients. 
        kernel : str, default='RBF'
            The kernel to use for KMM. Can be 'RBF', 'DotProduct', 'WhiteKernel', 'Matern'.
        Targets : int, default=1
            The number of target variables in the dataset.
        """
        # A sklearn regression model 
        if type(Regressor) == str:
            self.Regressor = GenerateReg(Regressor)
        else:
            self.Regressor = Regressor
        self.UpBound = UpBound
        self.kernel = kernel
        self.Targets = Targets
        warnings.filterwarnings('ignore')


    def fit(self,source_dataset,target_dataset,test_data,tao=None):
        """
        Fit the transfer model on source and target datasets and return the predictions
        on the test data along with beta coefficients.

        Parameters
        ----------
        source_dataset : array-like of shape (n_samples, n_features)
            The source dataset, where n_samples is the number of samples and n_features
            is the number of features including the target variable(s).
        target_dataset : array-like of shape (n_samples, n_features)
            The target dataset, where n_samples is the number of samples and n_features
            is the number of features including the target variable(s).
        test_data : array-like of shape (n_samples, n_features - Targets)
            The test dataset, where n_samples is the number of samples and n_features
            is the number of features excluding the target variable(s).
        tao : float, default=None
            used in KMM : 1-tao <= sum(beta_i) <= 1+tao , E(beta) = 1 , E is expectation 
            if tao == None, tao =  B/np.sqrt(n), n is the number of source domain data.
        Returns
        -------
        predictions : array-like of shape (n_samples,)
            The predicted values for the test data.
        beta : array-like of shape (n_samples,)
            The beta coefficients derived by the KMM weighting scheme.
        """
        source_data = np.array(source_dataset)[:, :-self.Targets]
        source_response = np.array(source_dataset)[:, -self.Targets:]
        target_data = np.array(target_dataset)[:, :-self.Targets]
        target_response = np.array(target_dataset)[:, -self.Targets:]

        KMM = KernelMeanMatching(self.kernel,source_data,target_data,target_response)
        beta = KMM.cal_beta(B = self.UpBound,tao=tao)

        X = np.concatenate((source_data, target_data), axis=0)
        Y = np.concatenate((source_response, target_response), axis=0)

        # test the Regressor
        attribute_list = ['fit', 'predict',]
        check_attributes(self.Regressor, attribute_list)

        check_weight(self.Regressor.fit)

        data_weight = np.concatenate(
            (beta,np.ones(
                (len(target_data),1)
                        )))


        return self.Regressor.fit(X,Y, sample_weight = data_weight.flatten()).predict(test_data), beta.flatten()


def check_attributes(estimator, attribute_list):
    for attribute in attribute_list:
        if hasattr(estimator, attribute) :
            print(f"The estimator has a {attribute} attribute.")
            pass
        else:
            print(f"The estimator does not have a {attribute} attribute.")
            print('Please provide another Regressor')
            raise ValueError("Error of Regressor")


def check_weight(estimator,):
        if 'sample_weight' in estimator.__code__.co_varnames:
            print("The estimator.fit has 'sample_weight' attribute.")
            pass
        else:
            print("The estimator.fit does not have 'sample_weight' attribute.")
            print('Please provide another Regressor')
            raise ValueError("Error of Regressor")

def GenerateReg(Regressor_name):
    if Regressor_name == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        mdoel = RandomForestRegressor()
    elif Regressor_name == 'LR':
        from sklearn import linear_model
        mdoel = linear_model.LinearRegression()
    else:
        print('Sorry, Bin did not define this function for you, please pass it in yourself')
    return mdoel

