import numpy as np
import matplotlib.pyplot as plt

def holdout(X, y , trainRatio): 
    """Summary
    
    Args:
        X (TYPE): D-dimensional dataset
        y (TYPE): labels
        trainRatio (TYPE): Percentage of points of dataset in training set.
    
    Returns:
        TYPE: Description
    """
    N = X.shape[0]
    N_train = int(np.floor(N*trainRatio))
    ind_train = list(np.random.choice(np.arange(N), N_train, replace=False))
    ind_test = list(set(range(N)) - set(ind_train))
    X_train, X_test = X[ind_train, :], X[ind_test, :]
    y_train, y_test = y[ind_train], y[ind_test]
    return X_train, X_test, y_train, y_test 

def sigmoid(x): 
    """Summary
    
    Args:
        x (TYPE): scalar
    
    Returns:
        TYPE: 1-dim sigmoif function
    """
    return 1.0/(1.0 + np.exp(-x))

def confusion_mat(y, y_pred): 
    """Summary
    
    Args:
        y (TYPE): 1-d array of labels
        y_pred (TYPE): 1-d array of predicted labels
    
    Returns:
        TYPE: Computes confusion matrix of a model
    """
    true_pos = sum((y_pred == 1) * (y == 1))
    true_neg = sum((y_pred == 0) * (y == 0))
    false_neg = sum((y_pred == 0) * (y == 1))
    false_pos = sum((y_pred == 1) * (y == 0))
    n_0 = float(sum(np.array(y == 0, dtype = int)))
    n_1 = float(sum(np.array(y == 1, dtype = int)))
    return np.array( [[true_neg/n_0 , false_pos/n_0] ,
                   [false_neg/n_1 , true_pos/n_1]], dtype=float)

# def roc_curve(X, y, beta):
#     x_axis, y_axis = [], []
#     for tau in np.linspace(0,1,100):
#         test = LogisticReg(X,y)
#         y_pred = test.predictions(beta, tau=tau)
#         conf = confusion_mat(y, y_pred)
#         x_axis.append(conf[0,1])
#         y_axis.append(conf[1,1])
#     plt.plot(x_axis, y_axis, label='MLE of $beta')
#     plt.plot(x_axis, x_axis, label='Chance')
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     plt.ylabel("True positives")
#     plt.xlabel("False Positives")
#     plt.legend()
#     area = np.trapz(-np.array(y_axis).T, np.array(x_axis).T)
#     return area