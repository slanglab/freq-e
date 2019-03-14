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
    """
    Type checks for typical scalar numbers. Include np.float since, e.g.
    np.log(5) returns np.float64, not a standard float.
    Apparently bool is included here.
    """
    return isinstance(x, (int, float, np.number))

def mll_curve(pred_logodds, label_prior, theta_grid=DEFAULT_THETA_GRID):
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
    assert 0 < label_prior < 1
    Ndoc = len(pred_logodds)
    label_prior = float(label_prior)
    pos_odds = np.exp(pred_logodds)
    pos_prob = pos_odds / (1+pos_odds)
    neg_prob = 1-pos_prob
    adj_pos_prob = pos_prob / label_prior
    adj_neg_prob = neg_prob / (1-label_prior)
    # doc vectorization is faster if more docs than grid points.
    return fill_mll_curve_vectorize_docs(Ndoc, theta_grid, adj_pos_prob, adj_neg_prob)

def fill_mll_curve_slow(Ndoc, theta_grid, adj_pos_prob, adj_neg_prob):
    # assumes no 0 or 1 probs. Clip first.
    ret = np.zeros(len(theta_grid))
    for grid_index in range(len(theta_grid)):
        theta = theta_grid[grid_index]
        for d in range(Ndoc):
            log_mar_prob = np.log(theta * adj_pos_prob[d] + (1-theta) * adj_neg_prob[d])
            ret[grid_index] += log_mar_prob
    return ret

def fill_mll_curve_vectorize_grid(Ndoc, theta_grid, adj_pos_prob, adj_neg_prob):
    # assumes no 0 or 1 probs. Clip first.
    ret = np.zeros(len(theta_grid))
    theta = theta_grid
    for d in range(Ndoc):
        grid_log_mar_probs = np.log(theta * adj_pos_prob[d] + (1-theta) * adj_neg_prob[d])
        ret += grid_log_mar_probs
    return ret

def fill_mll_curve_vectorize_docs(Ndoc, theta_grid, adj_pos_prob, adj_neg_prob):
    # assumes no 0 or 1 probs. Clip first.
    ret = np.zeros(len(theta_grid))
    for ti in range(len(theta_grid)):
        theta = theta_grid[ti]
        doc_log_mar_probs = np.log(theta * adj_pos_prob + (1-theta) * adj_neg_prob)
        ret[ti] += np.sum(doc_log_mar_probs)
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
        warnings.warn("You have a multimodal posterior distribution, with mode indexes "+str(where)+'. Resolving tie by picking the middle value.')
        middle_index = int(np.floor(len(where)/2))
        map_est = DEFAULT_THETA_GRID[where[middle_index]]
    else: 
        map_est = DEFAULT_THETA_GRID[np.argmax(log_post_probs)]
    
    return map_est

def calc_log_odds(pos_probs): 
    """
    pos_probs : probabilites of the positive class
    """
    # Deal with prob=1.0 or prob=0.0.
    p = np.float64(pos_probs)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p/(1.0 - p))

class FreqEstimator(): 

    def __init__(self): 
        self.trained_model = None 
        self.train_prior = None  

    def train_cross_val(self, X: np.ndarray, y: np.ndarray, verbose=True):
        """
        Trains a logistic regression model on supplied training data, and
        stores it.  After this method is called, the FreqEstimator will be
        ready to infer prevalence in new test sets.
        
        This method uses the following approach for training:
        - grid search over L1 penalty 
        - finds best model by minimizing log loss on 10-fold cross-validation 

        Parameters
        ----------
        X : shape (num examples) by (num features)
            Training data feature matrix, a 2-D array.
        y : shape (num examples)
            Training data's labels, a 1-D array.
        """
        assert X.shape[0] == y.shape[0]
        #check to make sure y_train is binary (only 0's and 1's)
        assert np.array_equal(y, y.astype(bool)) == True 
        assert type(X) == type(y) == np.ndarray

        if verbose: print('TRAINING LOGISTIC REGRESSION MODEL')
        parameters = {'C': [1.0/2.0**reg for reg in np.arange(-12, 12)]}
        lr = LogisticRegression(penalty='l1', solver='liblinear')
        grid_search = GridSearchCV(lr, parameters, cv=10, refit=True, 
                                   scoring=make_scorer(log_loss, greater_is_better=False))
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        if verbose: print('Best model:', best_model)
        train_mean_acc = best_model.score(X, y)
        if verbose: print('Training mean accuracy=', train_mean_acc)

        self.set_trained_model(best_model, np.mean(y))

    def set_trained_model(self, trained_model, label_prior: float):
        """
        Instead of having this object train its own model, you can train your
        own and set it with this method.  
        
        After this is called, the FreqEstimator object will be ready to infer
        prevalence in new test sets.

        Parameters
        ----------
        trained_model : skelarn.linear_model class
            A trained model, which FreqEstimator will use to predict
            probabilities on the test instances.  (Technically, only
            .decision_function() is used, and interpreted as log odds.)

        label_prior : float
            The training-time prior distribution of the positive class.
            This should be set to the empirical mean from the training data
            (that is, the proportion of training data with the positive class).
        """
        if not hasattr(trained_model, 'decision_function'):
            raise Exception('trained_model must be a sklearn trained classifier that has a .decision_function()')
        
        self.trained_model = trained_model
        self.train_prior = label_prior

    def infer_freq(self, X_test: np.ndarray, conf_level=0.95):
        """
        Using the trained model, infer class prevalence for a new set of
        examples, with the "LR-Implicit" or "Implicit likelihood generative
        reinterpretation" method from Keith and O'Connor 2018.

        Return a point estimate and confidence interval for prevalence of the
        positive class.

        Parameters
        ----------
        X_test : numpy.ndarray 
            Numpy array of the test X matrix 

        conf_level : float, default: 0.95
            The confidence level for the inferred confidence interval.
            Must be between 0.0 and 1.0.
        """
        assert type(X_test) == np.ndarray
        assert 0.0 < conf_level < 1.0
        if self.trained_model is None or self.train_prior is None:
            raise Exception('Must first call .train_cross_val() or .set_trained_model() first.')
        log_odds = self.trained_model.decision_function(X_test)
        return _infer_freq_from_pred_logodds(log_odds, self.train_prior, conf_level)

def infer_freq_from_predictions(test_pred_probs: np.ndarray, label_prior: float, conf_level=0.95):
    """
    Infer class prevalence of a test set, from a supervised model's individual
    predictions, with the "LR-Implicit" or "Implicit likelihood generative
    reinterpretation" method from Keith and O'Connor 2018.

    Return a point estimate and confidence interval for prevalence of the
    positive class.

    Parameters
    ----------
    test_pred_probs : numpy.ndarray
        1-D array of predicted probabilities of the positive class, for each example.

    label_prior : float
        The training-time prior distribution of the positive class.
        This should be set to the empirical mean from the training data
        (that is, the proportion of training data with the positive class).

    conf_level : float, default: 0.95
        The confidence level for the inferred confidence interval.
        Must be between 0.0 and 1.0.
    """
    if type(test_pred_probs) != np.ndarray: raise Exception('test_pred_probs must be a numpy array')
    assert 0.0 < conf_level < 1.0
    log_odds = calc_log_odds(test_pred_probs)
    return _infer_freq_from_pred_logodds(log_odds, label_prior, conf_level)

def _infer_freq_from_pred_logodds(log_odds, label_prior, conf_level):
    """
    Infer class prevalence based on predicted log-odds for each example.
    """
    log_post_probs = mll_curve(log_odds, label_prior)
    map_est = generative_get_map_est(log_post_probs)
    conf_interval = get_conf_interval(log_post_probs, conf_level)
    return {'point': map_est, 'conf_interval': conf_interval, 'conf_level': conf_level}
