"""
To run all unit tests (test_*):
    py.test go_test.py

To run and show all output:
    py.test -vs go_test.py

The -k flag is also useful to run only some tests
"""

import sys, os, pytest
import numpy as np
# import pdb  

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '../py'))
from freq_e.ecdf import CategDist
from freq_e import estimate

def test_scalars():
    s = estimate.is_scalar
    assert s(5)
    assert s(1)
    assert s(-1)
    assert not s(None)
    assert not s('a')
    assert s(np.log(5))
    assert s(np.float32(5))
    assert s(np.float64(5))
    assert s(np.int32(5))
    assert s(np.int64(5))
    assert s(np.arange(5)[2])
    assert s(np.arange(5.0)[2])
    assert s(np.nan)
    assert s(np.inf)
    assert not s(np.arange(1))

def test_hdi():
    d = CategDist({0:1,1:2,2:7})
    x,y = d.hdi(0.0001)
    assert x==y                   # any single-point interval attains this coverage request
    assert d._hdi_sort(0.0001) == (2,2)    # prefers the highest-prob point

    assert d.hdi(0.7) == (2,2)
    assert d._hdi_sort(0.9 - 1e-5) == (1,2)  # subtract epsilon to deal with floating point issue

    # these were designed for hdi_sort.
    assert d._hdi_sort(0.1) == (2,2)    # set {2}, though {1} would satisfy also
    assert d._hdi_sort(0.15) == (2,2)   # {1} or {2} would satisfy
    assert d._hdi_sort(0.2) == (2,2)    # {1} or {2} would satisfy
    # assert d._hdi_sort(0.9) == (1,2)    # fails only due to floating point issues

    # below is behavior under hdi_quadratic, which more conservatively returns
    # smaller-coverage intervals to match smaller requests.
    assert d._hdi_quadratic(0.1) == (0,0)    # set {0}
    assert d._hdi_quadratic(0.15) == (1,1)   # Better to use set {1} instead of {0,1}
    assert d._hdi_quadratic(0.2) == (1,1)
    assert d._hdi_quadratic(0.9) == (1,2)

def test_hdi_unif():
    ## Ties for the 0.5 interval
    d = CategDist({ 1:1, 2:1, 3:1, 4:1 })
    # print d.cdf(4)-d.cdf(2),d.cdf(3)-d.cdf(1)
    assert 0.5==d.cdf(4)-d.cdf(2)==d.cdf(3)-d.cdf(1)

    # break tie on centralness within the CLOSED interval
    # this is done explicitly in hdi_sort
    # and implicitly in hdi_sort. need test to confirm it works
    assert d.hdi(0.5) == (2,3)

    # trickier: among 40% intervals, both {2,3} and {3,4} have equal close-to-centerness.
    # we tiebreak to leftmost
    d = CategDist({ 1:1, 2:1, 3:1, 4:1, 5:1 })
    assert d.hdi(0.4)==(2,3)
    assert d.hdi(0.4) in { (2,3), (3,4) }

def interval_contains(A,B):
    """does A contain B?"""
    a_lo,a_hi = A
    b_lo,b_hi = B
    return a_lo <= b_lo <= b_hi <= a_hi

def test_interval_contains():
    assert interval_contains([1,3], [1,2])
    assert interval_contains([1,3], [1,3])
    assert not interval_contains([1,3], [1,5])
    assert not interval_contains([1,3], [2,5])
    assert not interval_contains([1,3], [8,9])
    assert not interval_contains([1,3], [-11,-10])
    assert not interval_contains([1,3], [-11,2])
    assert not interval_contains([1,3], [-11,1])

def test_hdi_coverage_simple():
    def monotonic_interval_test(d):
        intervals = [d.hdi(c) for c in np.arange(.1, 1.1, .1)]
        for i in range(len(intervals)-1):
            for j in range(i, len(intervals)-1):
                int1,int2 = intervals[i], intervals[j]
                assert interval_contains(int2, int1)
    d = CategDist({ 1:10, 2:20, 3:30, 4:10 })
    monotonic_interval_test(d)
    for i in range(100):
        n = 100
        counts = np.random.random_integers(0,1000, size=n)
        d = CategDist( {k:v for k,v in zip(range(n), counts) })
        monotonic_interval_test(d)
    
def test_functions():
    # Note: these were developed after observing output.  So a chance something
    # may have been specified incorrectly.
    # These are intended to be run with "py.test -v ecdf.py" from the "pytest" package.
    d = CategDist({0:1, 1:1, 2:8})
    assert d.size==3
    assert d.pmf(0) == 0.1
    assert d.pmf(0.0) == 0.1  # interesting question when does float/integer stuff work here?
    assert d.pmf(2) == 0.8
    with pytest.raises(Exception): d.pmf(3, mode='strict')
    with pytest.raises(Exception): d.pmf("asdf", mode='strict')
    with pytest.raises(Exception): d.pmf(0.3134234324, mode='strict')
    assert d.cdf(2)==1
    assert d.cdf(1.99) == 0.2
    assert d.cdf(42) == 1
    assert d.cdf(-1) == 0
    assert d.cdf(0) == 0.1
    assert d.cdf(1) == 0.2

    assert d.icdf(0, mode='toobig') == 0
    with pytest.raises(Exception): d.icdf(0, mode='toosmall')
    assert d.icdf(1, mode='toobig') == 2
    assert d.icdf(1, mode='toosmall') == 2
    assert d.icdf(0.7, mode='toobig') == 2
    assert d.icdf(0.7, mode='toosmall') == 1
    assert d.icdf(0.2, mode='toobig') == 1
    assert d.icdf(0.2, mode='toosmall') == 1



def test_mll(): 
    pos_odds = np.array([0.2, 0.7])
    pred_logodds = np.log(pos_odds/(1.0 - pos_odds))
    label_prior = 0.7 
    theta_grid_test = np.array([0.3, 0.6])

    #worked out this example by hand
    ans = np.array([0.669050, 0.213574])
    output = estimate.mll_curve_simple(pred_logodds, label_prior, theta_grid=theta_grid_test)
    assert np.allclose(ans, output)

def test_infer_freq():
    freq_e = estimate.FreqEstimate()
    #test to make sure you can only put in one or the other but not both 
    with pytest.raises(Exception): freq_e.infer_freq(np.zeros(3), np.zeros((3, 3)), trained_model=None, test_pred_probs=None)

    #test importing a list to pred_probs
    with pytest.raises(Exception): freq_e.infer_freq(np.zeros(3), np.zeros((3, 3)), trained_model=None, test_pred_probs=[1, 2, 3])


