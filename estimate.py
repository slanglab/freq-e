"""
freq-e : Class frequency estimation 

Also known as 
- quantification (data mining)
- prevalance estimation (statistics, epidemiology)
- class prior estimation (machine learning) 

This package is built upon sklearn and numpy.   

Authors: 
Katie Keith (kkeith@cs.umass.edu)
Brendan O'Connor (brenocon@cs.umass.edu)
"""
from __future__ import division, print_function 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
from scipy.stats import beta
import ecdf 

"""
TODO: 
- what are the alternatives to the grid search? 
- double check MLL curve is working and matches the math!

Clean up ecdf and make sure you understand it all!  
"""
trange = np.arange(0.001, 1.0, 0.001) 

def get_conf_interval(log_post_probs, conf_level): 
    """
    Gets CI interval for the given confidence level  
    """
    value_to_unorm_logprob = {t: ll for t, ll in zip(trange, log_post_probs)}
    catdist = ecdf.dist_from_logprobs(value_to_unorm_logprob)
    conf_interval = catdist.hdi(conf_level)
    return conf_interval 

def generative_get_map_est(log_post_probs):
    '''
    Returns the map estimate of the posterior for generative models 

    In the case of ties for the max posterior we take the middle index
    '''
    assert len(log_post_probs) == len(trange)

    #check multi-modal posterior!  
    mx = np.max(log_post_probs)
    if len(log_post_probs[log_post_probs == mx]) >= 2:
        where = (np.where(log_post_probs == mx))[0]
        warnings.warn("You have a multimodal posterior distribution where "+str(where)+' Resolving tie by picking the middle value.')
        middle_index = int(np.floor(len(where)/2))
        map_est = trange[where[middle_index]]

    else: 
        map_est = trange[np.argmax(log_post_probs)]
    
    return map_est


def get_beta_prior():
    '''
     (2) Beta(1+eps, 1+eps)
        we need to add this prior to the mll probs in the case where 
        we have a flat mll curve('i.e. high regularization')
    '''
    #prior (2)
    eps = 0.0001
    a, b = 1.0+eps, 1.0+eps
    log_prior = beta.logpdf(trange, a, b) 
    assert len(log_prior) == len(trange)
    return log_prior 

def mll_curve_stable(logodds, pi):
    """
    Parameters:
        logodds: nparray size (ndocs, )
            log(p(y=1 | w) / p(y=0 | w))
            = wx + b
            (which comes out of sklearn logistic regression decision_function)

        pi: prevelance of positive class
            estimated from the training data 
            
    Returns:
        mll_curve: numpy array, size (len(trange),) 

    """
    def mll_curve_onedoc_onetheta_stable(logodds_onedoc, log_pi_ratio, theta):
        term = 1.0 - theta + np.exp(np.log(theta) + logodds_onedoc + log_pi_ratio)
        return np.log(term)

    def mll_curve_onedoc_stable(logodds_onedoc, log_pi_ratio):
        curve = np.array([mll_curve_onedoc_onetheta_stable(logodds_onedoc, log_pi_ratio, theta) for theta in trange])
        return curve 

    curve = np.zeros(len(trange))
    log_pi_ratio = np.log((1.0 - pi)/ pi)
    for logodds_onedoc in logodds:
        curve += mll_curve_onedoc_stable(logodds_onedoc, log_pi_ratio)
    return curve

class FreqEstimate(): 
    """
    Description 

    Parameters
    ----------
    disc_classifier : str, default: Logistic Regression 
        Specify the sklearn class discriminitive classifier 
        of your choice. This will be used within the 
        discriminitive-generative framework 

    conf_level : float, default=0.95
        Confidence level for the confidence intervals that 
        will be constructed 
    
    Attributes
    ----------



    """

    def __init__(self, conf_level=0.95): 
        self.conf_level = conf_level
        self.train_mean_acc = None 

    def fit(self, X, y):
        """
        Fits logistic regression estimator via 
        - grid search over L1 penalty 
        - finding the best model by minimizing log loss on 10-fold cross-validation 
        """
        assert type(X) == type(y) == np.ndarray
        print('TRAINING DISCRIMINATIVE MODEL')
        parameters = {'C': [1.0/2.0**reg for reg in np.arange(-12, 12)]}
        lr = LogisticRegression(penalty='l1')
        grid_search = GridSearchCV(lr, parameters, cv=10, refit=True, 
                                   scoring=make_scorer(log_loss, greater_is_better=True))
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print(best_model)
        train_mean_acc = best_model.score(X, y)
        print('Training mean accuracy=', train_mean_acc)
        self.train_mean_acc = train_mean_acc 
        return best_model 

    def predict_freq(self, trained_model, X_train, y_train, X_test):
        """
        "LR-Implicit" or "Implict likelihood generative reinterpretation" method 
        from Keith and O'Connor 2018


        Point estimate and confidence intervals  
        """ 
        train_prior = np.mean(y_train)
        log_odds = trained_model.decision_function(X_test)
        log_post_probs = mll_curve_stable(log_odds, train_prior)
        log_prior = get_beta_prior()
        log_post_probs = np.add(log_post_probs, log_prior)
        map_est = generative_get_map_est(log_post_probs)
        conf_interval = get_conf_interval(log_post_probs, self.conf_level)
        return {'point': map_est, 'conf_interval': conf_interval}

