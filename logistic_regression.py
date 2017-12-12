import numpy as np
import matplotlib.pyplot as plt

from utils import holdout
from utils import sigmoid
from utils import confusion_mat

class LogisticReg(object):
    """docstring for LogisticReg"""
    def __init__(self, X,y):
        super(LogisticReg, self).__init__()
        self.X = X
        self.y = y 
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.X_ext = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
        
    def loglik(self, beta):
        """Summary
        
        Args:
            beta (TYPE): (D+1)-dim parameter (including bias) parametrising conditional probilities in logistic regression.
        
        Returns:
            TYPE: scalar (total loglikehood of data | not per data point) 

        Flag:  To account for asymptotic behaviour of log(sigma(beta*X_n)) as   beta*X_n -> -inf.
        """
        vect = np.zeros(self.y.shape)
        prod = np.dot(self.X_ext, beta.T)
        sigmas = sigmoid(prod)
        if np.min(prod) < -1e4:
            for n in range(self.N):
                if prod[n] <  -1e4:
                    vect[n,:] = (2*self.y[n,:] - 1)*prod[n]
                else:
                    temp = (self.y*np.log(sigmas) 
                    + (1.0 - self.y)*np.log(1.0 - sigmas))
                    vect[n,:] = temp[n,:]
                    # import pdb; pdb.set_trace()
        else:
            vect = (self.y*np.log(sigmas) 
                    + (1.0 - self.y)*np.log(1.0 - sigmas))
        return sum(vect)


    def gradient(self, beta):
        """Summary
        
        Args:
            beta (TYPE): (D+1)-dim parameter (including bias) parametrising conditional probilities in logistic regression.
        
        Returns:
            TYPE: (D+1)-dim gradient of total loglikelihood with respect to beta
        """
        sigmas = sigmoid(np.dot(self.X_ext, beta.T))
        vect = (self.y - sigmas)*self.X_ext
        return sum(vect)
    
    def gradient_descent(self,
                         lr=0.01,
                         tol=0.01):
        """Summary
            Trains model finding optimal beta to min. neg. loglikehood
        Args:
            lr (float, optional): learning rate
            tol (float, optional): rolerance
        """
        self.beta = np.random.normal(0,1,(1, self.D + 1)).reshape(1,-1)
        lik_list, i = [], 0
        while np.linalg.norm(self.gradient(self.beta), ord=1) > tol:
            lik_list.append(self.loglik(self.beta))
            self.beta += lr * self.gradient(self.beta)
            if i > 500: 
                print "Convergence not reached after 500 iterations"
                break
            i += 1
        plt.plot(lik_list, label=(lr))
#         plt.ylim(-600,-350)
        plt.ylabel("loglikelihood")
        plt.xlabel("iterations")
        plt.legend()
        
    def predictions(self, beta, tau=0.5): 
        """Summary
        
        Args:
            beta (TYPE): (D+1)-dim parameter (including bias) parametrising conditional probilities in logistic regression.
            tau (float, optional): threshold for predictions
        
        Returns:
            TYPE: 1-D array with predicted labels assigning 1 if conditional prob of 1 (given by logistic regression) > tau
        """
        sigmas = sigmoid(np.dot(self.X_ext, beta.T))
        self.y_pred = np.array(sigmas > tau * np.ones((self.N,1)), dtype=int)
        return self.y_pred
    
    def RBF_features(self, X_train, ls=0.1):
        """Summary
            Radial basis function feature extensions 
        Args:
            X_train (TYPE): Training set acting as centres of RBFs
            ls (float, optional): lengthscale of 
        
        Returns:
            TYPE: (N vectors of dimension N_train)
        """
        N_train = X_train.shape[0]
        X_rbf = np.zeros((self.N, N_train))
        for m in range(N_train):
            sq_diff = np.power(self.X - X_train[m,:].reshape(1,-1), 2)
            X_rbf[:,m] = np.exp(- np.sum(sq_diff,axis=1)/(2.0 * ls**2))
        return X_rbf 
            
    def grad_descent_MAP(self,
                        lr=0.01,
                        tol=0.01):      
        """Summary
        Performs gradient descent to find min. of the negative MAP (maximum a posteriori) objective for standard gaussian priors on beta: -loglik + beta^T*beta/2.
        Args:
            lr (float, optional): learning rate
            tol (float, optional): tolerance
        """
        self.beta = np.random.normal(0,1,(1, self.D + 1)).reshape(1,-1)
        lik_list, i = [], 0
        while np.linalg.norm(self.gradient(self.beta), ord=1) > tol:
            lik_list.append(self.loglik(self.beta))
            self.beta += lr * (self.gradient(self.beta) - self.beta)
            if i > 500: 
                print "Convergence not reached after 500 iterations"
                break
            i += 1
        plt.plot(lik_list, label=(lr))
#         plt.ylim(-600,-350)
        plt.ylabel("loglikelihood")
        plt.xlabel("iterations")
        plt.legend()
