from __future__ import division
import sys
import numpy as np
import scipy.misc
from collections import Counter

class CategDist:
    """
    Represents an explicit categorical distribution.
    This includes CDF, Inverse CDF, and Highest Density Interval
    implementations that assume the the domain is ordered (for CDF and ICDF)
    and is reasonable to do numerical subtraction on (for HDI).
    This is intended to support empirical CDF-type work with samples or
    probability evaluations on a grid.

    Todo: 
    1. investigate scipy.stats.rv_histogram(), which looks like the same
    thing, or very similar.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html#scipy.stats.rv_histogram
    2. in general see if much of scipy.stats' distributions API can be supported;
    for example they do mean(), var(), etc.
    """
    def __init__(self, value_to_unnorm_prob={}, normalizer=None):
        """
        Create parallel domain and probs lists.
        unnorm probs could just be counts.
        Would have to change the implementation within here if we need to pass
        in logprobs instead.
        """
        # note: technically, maybe 'domain' should really be called 'support'?
        # we don't distinguish between these things.  
        self.domain = sorted(value_to_unnorm_prob.keys())
        self.size = len(self.domain)
        if normalizer is None:
            normalizer = sum(value_to_unnorm_prob.values())
        self.probs = [value_to_unnorm_prob[x]/normalizer for x in self.domain]
        self.value_to_prob = dict(zip(self.domain, self.probs))

        #get the mean of the distribution 
        self.get_mean()
        self.get_posterior_variance()

        # cached cumulative sums to be faster
        # self.cached_cdf = np.cumsum(probs)  ## note that CDF=0 is NOT included in this list.
        # assert np.abs(self.cached_cdf[-1] - 1.0) < 1e-10, "normalization bug?"
        # self.cached_ccdf = 1.0 - self.cached_cdf

    def get_pmf_cdf(self):
        self.cdf_values = {}
        self.pmf_values = {}
        for x in self.domain:
            self.pmf_values[x] = self.pmf(x)
            self.cdf_values[x] = self.cdf(x)

    def get_mean(self):
        #do I need to check to make sure the probabilities are actually normalized first??
        self.post_mean = np.sum(np.array([val*prob for val, prob in self.value_to_prob.iteritems()]))

    def get_posterior_variance(self):
        #get the variance of the distribution 
        self.post_var = np.sum(np.array([(val**2)*prob for val, prob in self.value_to_prob.iteritems()])) - self.post_mean**2

    def view(self):
        print "Domain size %s" % self.size
        print "x, P(X=x), P(X<=x)"
        print "=================="
        for x in self.domain:
            # print x, self.pmf(x), self.cdf(x)
            print "%6g %6g %6g" % (x, self.pmf(x), self.cdf(x))

    def pmf(self, x, mode='strict'):
        """The prob of value x.
        Internally, we don't really distinguish the domain versus the support.
        If you know your distribution actually has a domain larger than the
        support, use mode=lax and this function returns 0 if it doesn't know
        about the queried value.
        But if what we internally call the domain really IS the mathematical
        domain here, then use mode=strict.
        """
        if mode=='lax':
            return self.value_to_prob.get(x, 0.0)
        elif mode=='strict':
            assert x in self.domain, "Can only evaluate PMF for the domain in strict mode."
            return self.value_to_prob[x]
        else:
            assert False, "Bad mode %s" % repr(mode)

    def cdf_slow(self, x):
        """
        P(X <= x)  (not <) 
        Can ask about ANY x, including things not in the support/domain.
        """
        # todo maybe: faster cdf with cached values. but check against this
        prob = 0
        for i in xrange(self.size):
            xprime = self.domain[i]
            if xprime > x: break   ## since self.domain is sorted
            prob += self.probs[i]
        return prob

    def cdf(self,x): return self.cdf_slow(x)

    def icdf(self, p, mode='toobig'):
        """ 
        Inverse CDF a.k.a. "quantile" of "ppf"
        If there exists exactly an x in the domain where P(X<=x)=p, then return that.
        
        mode determines what to do when there is no such x (a gap situation):
         - "toobig":  choose the smallest x'>x where P(X<=x')>p
         - "toosmall": choose the largest x'<x where P(X<=x')<p.
           Note this isn't defined if p is smaller than min_x P(X<=x).
        This is similar to numpy.percentile() or R quantile(), except those
        work on individual samples as input and who knows how they handle the
        gap situation.
        Also similar to R qnorm(), qbeta(), and other q* functions for
        continuous distributions, or scipy.stats.*.ppf(), or Julia's
        Distributions package.
        """
        assert 0 <= p <= 1, "Desired CDF prob must be between 0 and 1 inclusive"

        if mode=='toobig':
            for i in xrange(self.size):
                x = self.domain[i]
                # if cdf=p, we're done.
                # if cdf>p, that means we just went past a gap, and we're now
                # returning the smallest too-big value.
                if self.cdf(x) >= p:
                    return x
            else:
                # This should never happen since p is no more than 1.0, and
                # always the last item in the domain has CDF=1.0.
                assert False, "Didn't find value with prob at least %s" % p
        elif mode=='toosmall':
            # go in descending order
            # list(xrange(5,-1,-1)) => [5, 4, 3, 2, 1, 0]
            for i in xrange(self.size-1, -1, -1):
                x = self.domain[i]
                if self.cdf(x) <= p:
                    return x
            else:
                # The problem now is, the user requested a CDF less then the
                # CDF of the smallest value in the support (and we don't have a
                # separate notion of the domain).
                # Not sure what the right behavior is.  The 'toosmall'
                # criterion doesn't make sense here.
                assert False, "Didn't find value with prob <= %s" % p
        else:
            assert False, "bad mode given"

    def hdi(self, alpha):
        """
        Return a pair of values (x1,x2) that is the highest-density *interval*,
        where it represents a CLOSED interval.  Specifically, it has:
         A) x1<=x2,
         B) Fulfills coverage: P(x1 <= X <= x2) >= alpha, and
         C) (x2-x1) is smallest among all (x1,x2) pairs with alpha coverage.

        This assumes the domain has subtraction semantics (condition C).  If
        the domain is strings, for example, this function won't work.
        This is the highest density *interval*, not *region*; the latter is
        often seen in the Bayesian stats literature, defined as the
        smallest-volume subset of the domain with prob mass at least alpha.  If
        the PMF is multimodal, the HDregion may be different than the
        HDinterval; it could consist of multiple intervals.  If the PMF is
        unimodal, I think the HDregion is always a single interval.

        Also note the stats literature conventionally uses (1-alpha) to refer
        to coverage.  Maybe we should use a different greek letter here.
        """
        # return self._hdi_quadratic(alpha)
        return self._hdi_sort(alpha)

    def _hdi_sort(self, alpha):
        # sort domain in descending PMF order
        # tiebreak by closeness to the median, in order to hopefully encourage symmetric intervals in severe circumstances
        # and then prefer leftmost, just to have an arbitrary but stable tiebreak

        # Potential issue: say the maximum point has way more
        # prob mass than requested.  then this method wil just return the
        # maximum point, giving a higher-prob-mass interval than requested,
        # even if there is somewhere else an interval whose prob mass is closer
        # to (and still higher than) to the request.

        value_indexes = range(self.size)
        median = np.median(self.domain)  # it does correctly average in case of a tie
        value_indexes.sort(key=lambda i: (-self.probs[i], abs(self.domain[i]-median), self.domain[i]))
        current_coverage = 0.0
        selected_inds = []
        for i in value_indexes:
            selected_inds.append(i)
            current_coverage += self.probs[i]
            # print "---- %.17g %.17g %s" % (current_coverage, alpha, selected_inds)
            if current_coverage >= alpha: break
        # indexset could be returned as a more principled highest-density set.
        # in fact right now it might incorporate multiple modes
        # but we're using interval semantics, so convert.
        # if we're multimodal, the interval will include a bunch of stuff between them too.
        return (self.domain[min(selected_inds)], self.domain[max(selected_inds)])

    def _hdi_quadratic(self,alpha):
        # Try all starting left-points, and for each, extend to the right until coverage is achieved.
        # Then choose the shortest interval among all left-points.
        # there should be a faster way to do this...

        # cache the CDF
        cdf = np.cumsum(self.probs)
        # print "CDF",cdf,list(cdf)
        # assemble candidates for each left-side starting point
        candidates = {}
        for i in xrange(self.size):
            # This woulda been easier just adding the PMFs iteratively.  But
            # since we're subtracting CDF values, and the domain is discrete,
            # for the the left side we actually want P(X<x1) not
            # P(X<=x1).Consider x1=x2=min(domain); then
            # P(x1<=X<=x2)=P(x1)=P(x2)=CDF(x2)-0.
            left_cdf = 0 if i==0 else cdf[i-1]
            for j in xrange(i, self.size):
                integral = cdf[j]-left_cdf
                if integral < alpha:
                    continue
                x1,x2 = self.domain[i], self.domain[j]
                center_q = left_cdf + integral/2.0
                candidates[i,j] = (x2-x1, integral, center_q)
                break  # move to next left-side starting point

        # Sort by interval size and obscure tie-breaking criteria
        keys = list(candidates)
        if not keys:
            assert False, "Couldn't find an interval of length alpha %s" % alpha

        keys.sort(key=lambda (i,j): (
            candidates[i,j][0], # interval size. if ties, then look at:
            # coverage no bigger than necessary - closest to alpha
            candidates[i,j][1],
            # interval closer to center; thus prefer symmetric interval if
            # everything is tied
            abs(0.5 - candidates[i,j][2]),
            # leftmost, only because we need to break the tie somehow
            i,
            ))
        ret_i,ret_j = keys[0]

        # print "Returning interval", [self.domain[ret_i],self.domain[ret_j]], "with interval size, coverage, center q:", candidates[ret_i,ret_j]
        # for i,j in keys[:10]:
        #     print "interval",[self.domain[i],self.domain[j]], " CDF values", (0 if i==0 else cdf[i-1], cdf[j]), "interval size, coverage (integral), center q:",candidates[i,j]

        return self.domain[ret_i], self.domain[ret_j]

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

    # example of failure due to numerical instability, since cdf[j]-cdf[i] is
    # not always same as probs[i] (since floating point arithmetic disobeys
    # associativity!)
    # interval [1, 2]  CDF values (0, 0.4) interval size, coverage, center q: (1, 0.4, 0.1)
    # interval [2, 3]  CDF values (0.2, 0.6000000000000001) interval size, coverage, center q: (1, 0.4000000000000001, 0.30000000000000004)
    # interval [3, 4]  CDF values (0.4, 0.8) interval size, coverage, center q: (1, 0.4, 0.5)
    # and note
    # >>> np.cumsum(d.probs)
    # array([0.2, 0.4, 0.6, 0.8, 1. ])
    # >>> x=np.cumsum(d.probs)
    # >>> list(x)
    # [0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
    # >>> x[2]-x[1]
    # 0.20000000000000007
    # >>> x[3]-x[2]
    # 0.19999999999999996

    
    

def dist_from_logprobs(value_to_unnorm_logprob):
    """An attempt to reuse the current __init__ code above, which works in prob
    space, by converting up from logprob input.  Not sure yet this is the right
    thing to do -- maybe it would be better to use log-domain arithmetic to do
    all the code in log-domain.
    
    input to this function is a dictionary.
    """
    unnorm_logprobs = np.array(value_to_unnorm_logprob.values())
    logZ = scipy.misc.logsumexp(unnorm_logprobs)
    value_to_norm_logprobs = {k: ulp-logZ for k,ulp in value_to_unnorm_logprob.items()}
    # Danger step!  could get unjustified 0s.
    value_to_norm_probs = {k: np.exp(lp) for k,lp in value_to_norm_logprobs.items()}
    # If logsumexp is implemented well, the prob sum should now be 1.0.
    # (I actually don't know if floating-point semantics allow it to be the
    # case that that can be guaranteed.)
    # We could pass in normalizer=1.0 to the constructor to save a bit of time.
    # But just in case we'll let it recompute it.
    return CategDist(value_to_norm_probs)

def dist_from_samples(xs):
    """This gives an ECDF: input xs is just a list of numbers or whatever"""
    counts = Counter(xs)
    return CategDist(counts)

def test_functions():
    import pytest
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


"""
BASICS
>>> import ecdf; reload(ecdf)
>>> d=ecdf.CategDist({0:1,1:1,2:8})
>>> d.view()
Domain size 3
x, P(X=x), P(X<=x)
==================
0 0.1 0.1
1 0.1 0.2
2 0.8 1.0
>>> d.pmf(0)
0.1
>>> d.pmf(1)
0.1
>>> d.pmf(2)
0.8
>>> d.pmf(3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "ecdf.py", line 27, in pmf
    assert x in self.domain, "Can only evaluate PMF for the domain."
AssertionError: Can only evaluate PMF for the domain.
>>> d.pmf(-1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "ecdf.py", line 27, in pmf
    assert x in self.domain, "Can only evaluate PMF for the domain."
AssertionError: Can only evaluate PMF for the domain.
>>> d.cdf(0)
0.1
>>> d.cdf(-1)
0
>>> d.cdf(1)
0.2
>>> d.cdf(2)
1.0
>>> d.cdf(10)
1.0
>>> d.cdf(100)
1.0
... Checking the inverse CDF.  Trickier.
>>> d.view()
Domain size 3
x, P(X=x), P(X<=x)
==================
0 0.1 0.1
1 0.1 0.2
2 0.8 1.0
>>> d.icdf(1)
2
>>> d.icdf(0)
0
>>> d.icdf(0,mode='toosmall')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "ecdf.py", line 93, in icdf
    assert False, "Didn't find value with prob <= %s" % p
AssertionError: Didn't find value with prob <= 0
>>> d.icdf(.3)
2
>>> d.icdf(.3, mode='toosmall')
1
Constructing from samples
>>> reload(ecdf);d=ecdf.ecdf_from_samples([random.randrange(10) for i in xrange(10000)])
>>> d.cdf(0)
0.1053
>>> d.cdf(8)
0.8975
>>> d.cdf(11)
1.0
>>> d.icdf(.25), d.icdf(.75)
(2, 7)
Using log-space input.  Oh fun, see the ties in the CDF.
Hopefully the use of logsumexp guarantees the last value has prob 1.0.
>>> ecdf.dist_from_logprobs({0: -3.0, 1:-500, 2:-6.0}).view()
Domain size 3
x, P(X=x), P(X<=x)
==================
0 0.952574126822 0.952574126822
1 1.3631425533e-216 0.952574126822
2 0.0474258731776 1.0
To doublecheck that logsumexp normalization is working.  Yeah, its supposed to
handle a big constant shift to all of the logprobs (which a naive exp() would
not).
>>> ecdf.dist_from_logprobs({0: -1003.0, 1:-1500, 2:-1006.0}).view()
Domain size 3
x, P(X=x), P(X<=x)
==================
0 0.952574126822 0.952574126822
1 1.3631425533e-216 0.952574126822
2 0.0474258731776 1.0
HDI.  See test_hdi()
Funny things can happen with a small lumpy discrete distribution.
As you request a higher alpha, its not guaranteed that the returned interval is
always strictly a superset of previous intervals.
Thats OK!  I think.
>>> d=ecdf.ecdf_from_samples([random.randrange(10) for i in xrange(10)])
>>> d.hdi(.3)
(8, 8)
>>> d.hdi(.5)
(8, 9)
>>> d.hdi(.6)
(7, 9)
>>> d.hdi(.7)
(5, 9)
>>> d.hdi(.8)
(1, 8)
>>> d.hdi(.9)
(1, 9)
>>> 
"""
