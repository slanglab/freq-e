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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
from freq_e import ecdf

DEFAULT_THETA_GRID_EPSILON = .001
DEFAULT_THETA_GRID = np.arange(0, 1 + DEFAULT_THETA_GRID_EPSILON, DEFAULT_THETA_GRID_EPSILON)

def is_scalar(x):
    """Type checks for typical numbers. Include np.float since, e.g.
    np.log(5) returns np.float64, not a standard float."""
    return isinstance(x, (int,float, np.float))

def mll_curve_simple(pred_logodds, label_prior, theta_grid=DEFAULT_THETA_GRID):
    """
    pred_logodds: vector length N (num docs), logodds of each's positive class
    label_prior: scalar, a probability
    return vector of l(theta) for each theta in the grid with values
    l(theta) =
    =     sum_d log[theta p(d|yd=1)         +  [1-theta] p(d|yd=0)]
    = C + sum_d log[theta p(yd=1|d)/p(y=1)  +  [1-theta] p(yd=0|d)/p(y=0)]
    This will return using a 'C' that, of course, does not necessarily
    correspond to the true MLL. Being off by a constant does not matter for
    finding the mode, nor for HDI inference since we'll have to normalize
    anyway.
    """

    # typecheck and sanity silliness
    if not isinstance(pred_logodds, np.ndarray): # allows for passing in list
        pred_logodds = np.array(pred_logodds)
    Ndoc = len(pred_logodds)
    assert pred_logodds.shape == (Ndoc,)
    assert is_scalar(label_prior)
    assert 0 <= label_prior <= 1
    label_prior = float(label_prior)

    ret = np.zeros(len(theta_grid))
    pos_odds = np.exp(pred_logodds)
    pos_prob = pos_odds / (1+pos_odds)
    neg_prob = 1-pos_prob

    adj_pos_prob = pos_prob / label_prior
    adj_neg_prob = neg_prob / (1-label_prior)

    for grid_index in range(len(theta_grid)):
        theta = theta_grid[grid_index]
        for d in range(Ndoc):
            log_mar_prob = np.log(theta * adj_pos_prob[d] + (1-theta) * adj_neg_prob[d])
            ret[grid_index] += log_mar_prob
    return ret

def get_conf_interval(log_post_probs, conf_level): 
    """
    Gets CI interval for the given confidence level  
    """
    value_to_unorm_logprob = {t: ll for t, ll in zip(DEFAULT_THETA_GRID, log_post_probs)}
    catdist = ecdf.dist_from_logprobs(value_to_unorm_logprob)
    conf_interval = catdist.hdi(conf_level)
    return conf_interval 

def generative_get_map_est(log_post_probs):
    '''
    Returns the map estimate of the posterior for generative models 

    In the case of ties for the max posterior we take the middle index
    '''
    assert len(log_post_probs) == len(DEFAULT_THETA_GRID)

    #check multi-modal posterior and break ties with the middle index  
    mx = np.max(log_post_probs)
    if len(log_post_probs[log_post_probs == mx]) >= 2:
        where = (np.where(log_post_probs == mx))[0]
        warnings.warn("You have a multimodal posterior distribution where "+str(where)+' Resolving tie by picking the middle value.')
        middle_index = int(np.floor(len(where)/2))
        map_est = DEFAULT_THETA_GRID[where[middle_index]]
    else: 
        map_est = DEFAULT_THETA_GRID[np.argmax(log_post_probs)]
    
    return map_est

def calc_log_odds(pos_probs): 
    """
    pos_probs : probabilites of the positive class
    """
    return np.log(pos_probs/(1.0 - pos_probs))

class FreqEstimate(): 

    def __init__(self): 
        self.trained_model = None 
        self.train_prior = None  

    def train_cross_val(self, X, y):
        """
        Trains logistic regression estimator via 
        - grid search over L1 penalty 
        - finding the best model by minimizing log loss on 10-fold cross-validation 
        """
        assert X.shape[0] == y.shape[0]
        #check to make sure y_train is binary (only 0's and 1's)
        assert np.array_equal(y, y.astype(bool)) == True 
        assert type(X) == type(y) == np.ndarray

        print('TRAINING LOGISTIC REGRESSION MODEL')
        parameters = {'C': [1.0/2.0**reg for reg in np.arange(-12, 12)]}
        lr = LogisticRegression(penalty='l1', solver='liblinear')
        grid_search = GridSearchCV(lr, parameters, cv=10, refit=True, 
                                   scoring=make_scorer(log_loss, greater_is_better=False))
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print(best_model)
        train_mean_acc = best_model.score(X, y)
        print('Training mean accuracy=', train_mean_acc)
        self.trained_model = best_model
        self.train_prior = np.mean(y)

    def infer_freq_obj(self, X_test, conf_level=0.95):
        """
        "LR-Implicit" or "Implict likelihood generative reinterpretation" method 
        from Keith and O'Connor 2018

        Point estimate and confidence intervals

        Parameters
        ----------
        X_test : numpy.ndarray 
            Numpy array of the test X matrix 

        conf_level : float, default: 0.95
            The confidence level for the inferred confidence intervals 
            Must be between 0.0 and 1.0  
        """
        assert type(X_test) == np.ndarray
        assert conf_level > 0.0 and conf_level < 1.0 

        if self.trained_model == None or self.train_prior == None:
            raise Exception('must call .fit() function first ')
        elif self.trained_model != None:  
            log_odds = self.trained_model.decision_function(X_test)

        log_post_probs = mll_curve_simple(log_odds, self.train_prior)
        map_est = generative_get_map_est(log_post_probs)
        conf_interval = get_conf_interval(log_post_probs, conf_level)
        return {'point': map_est, 'conf_interval_'+str(int(conf_level*100))+'%': conf_interval}

def infer_freq(X_test, label_prior, conf_level=0.95, trained_model=None, test_pred_probs=None):
    """
    "LR-Implicit" or "Implict likelihood generative reinterpretation" method 
    from Keith and O'Connor 2018

    Point estimate and confidence intervals

    Parameters
    ----------
    label_prior : float
        Positive class label prior 
        Most often, this is estimated empirically via the training set
        (i.e. label_prior = np.mean(y_train))

    X_test : numpy.ndarray 
        Numpy array of the test X matrix 

    conf_level : float, default: 0.95
        The confidence level for the inferred confidence intervals 
        Must be between 0.0 and 1.0  

    trained_model : skelarn.linear_model class, default : None 
        As an alternative to test_pred_probs, a user may pass in a trained_model
        which will directly find the predicted probabilities on the test instances

    test_pred_probs : numpy.ndarray
        predicted probability (of the positive class) on the 
        test set  
    """
    if trained_model == None and type(test_pred_probs) == type(None): 
        raise Exception('Must specify EITHER a trained model (sklearn.linear_model class) OR test_pred_probs (predicted probabilities on the test instances)') 

    assert type(X_test) == np.ndarray
    assert conf_level > 0.0 and conf_level <1.0 

    if trained_model == None:
        if type(test_pred_probs) != np.ndarray: raise Exception('test_pred_probs must be a numpy array')
        assert test_pred_probs.shape[0] == X_test.shape[0] 
        log_odds = calc_log_odds(test_pred_probs)
    elif trained_model != None:  
        try: 
            log_odds = trained_model.decision_function(X_test)
        except: 
            print('trained_model must be a sklearn trained classifier that has a .decision_function()')

    log_post_probs = mll_curve_simple(log_odds, label_prior)
    map_est = generative_get_map_est(log_post_probs)
    conf_interval = get_conf_interval(log_post_probs, conf_level)
    return {'point': map_est, 'conf_interval_'+str(int(conf_level*100))+'%': conf_interval}

