import numpy as np
import scipy.misc
from collections import Counter

class CategDist:
    """
    Represents a discrete distribution over an explicitly enumerated numeric
    domain, supporting various queries.
    (The CDF and Inverse CDF implementations only assume the domain is ordered,
    but everything is designed/tested for integer and floating-point domains.)

    This is intended to support empirical CDF, ICDF, and HDI functionality
    from either
    (1) 1-D numerical samples (all with equal weights) -- e.g. dist_from_samples()
    (2) Probabilities for points on a 1-D grid -- e.g. dist_from_logprobs()
    """
    def __init__(self, value_to_unnorm_prob={}, normalizer=None):
        """
        `value_to_unnorm_prob` maps every value in the domain to its unnormalized probability.
        Note the unnormalized probs could just be counts, or even a Counter object.
        """
        # self.domain, self.probs are *parallel* lists.
        self.domain = sorted(value_to_unnorm_prob.keys())
        self.size = len(self.domain)
        if normalizer is None:
            normalizer = sum(value_to_unnorm_prob.values())
        self.probs = [value_to_unnorm_prob[x]/normalizer for x in self.domain]
        self.value_to_prob = dict(list(zip(self.domain, self.probs)))

        self.store_mean()
        self.store_variance()

    def store_pmf_cdf(self):
        self.cdf_values = {}
        self.pmf_values = {}
        for x in self.domain:
            self.pmf_values[x] = self.pmf(x)
            self.cdf_values[x] = self.cdf(x)

    def store_mean(self):
        self.mean = np.sum(np.array([val*prob for val, prob in self.value_to_prob.items()]))

    def store_variance(self): 
        self.var = np.sum(np.array([(val**2)*prob for val, prob in self.value_to_prob.items()])) - self.mean**2

    def view(self):
        print("Domain size %s" % self.size)
        print("x, P(X=x), P(X<=x)")
        print("==================")
        for x in self.domain:
            print("%6g %6g %6g" % (x, self.pmf(x), self.cdf(x)))

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
        prob = 0
        for i in range(self.size):
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
            for i in range(self.size):
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
            # list(range(5,-1,-1)) => [5, 4, 3, 2, 1, 0]
            for i in range(self.size-1, -1, -1):
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

    def hdi(self, conf_level):
        """
        Return a pair of values (x1,x2) that is the highest-density *interval*,
        where it represents a CLOSED interval.  Specifically, it has:
         A) x1<=x2,
         B) Fulfills coverage: P(x1 <= X <= x2) >= conf_level, and
         C) (x2-x1) is smallest among all (x1,x2) pairs with conf_level coverage.

        This assumes the domain has subtraction semantics (condition C).  If
        the domain is strings, for example, this function won't work.
        This is the highest density *interval*, not *region*; the latter is
        often seen in the Bayesian stats literature, defined as the
        smallest-volume subset of the domain with prob mass at least conf_level.  If
        the PMF is multimodal, the HDregion may be different than the
        HDinterval; it could consist of multiple intervals.  If the PMF is
        unimodal, I think the HDregion is always a single interval.

        Conventionally, conf_level = 1-alpha, where 'alpha' is the miscoverage
        rate (or false positive rate for null hypothesis testing).
        """
        return self._hdi_sort(conf_level)

    def _hdi_sort(self, conf_level):
        # sort domain in descending PMF order
        # tiebreak by closeness to the median, in order to hopefully encourage symmetric intervals in severe circumstances
        # and then prefer leftmost, just to have an arbitrary but stable tiebreak

        # Potential issue: say the maximum point has way more
        # prob mass than requested.  then this method wil just return the
        # maximum point, giving a higher-prob-mass interval than requested,
        # even if there is somewhere else an interval whose prob mass is closer
        # to (and still higher than) to the request.

        value_indexes = list(range(self.size))
        median = np.median(self.domain)  # it does correctly average in case of a tie
        value_indexes.sort(key=lambda i: (-self.probs[i], abs(self.domain[i]-median), self.domain[i]))
        current_coverage = 0.0
        selected_inds = []
        for i in value_indexes:
            selected_inds.append(i)
            current_coverage += self.probs[i]
            # print "---- %.17g %.17g %s" % (current_coverage, conf_level, selected_inds)
            if current_coverage >= conf_level: break
        # indexset could be returned as a more principled highest-density set.
        # in fact right now it might incorporate multiple modes
        # but we're using interval semantics, so convert.
        # if we're multimodal, the interval will include a bunch of stuff between them too.
        return (self.domain[min(selected_inds)], self.domain[max(selected_inds)])

    def _hdi_quadratic(self,conf_level):
        # Try all starting left-points, and for each, extend to the right until coverage is achieved.
        # Then choose the shortest interval among all left-points.
        # there should be a faster way to do this...

        # cache the CDF
        cdf = np.cumsum(self.probs)
        # print "CDF",cdf,list(cdf)
        # assemble candidates for each left-side starting point
        candidates = {}
        for i in range(self.size):
            # This woulda been easier just adding the PMFs iteratively.  But
            # since we're subtracting CDF values, and the domain is discrete,
            # for the the left side we actually want P(X<x1) not
            # P(X<=x1).Consider x1=x2=min(domain); then
            # P(x1<=X<=x2)=P(x1)=P(x2)=CDF(x2)-0.
            left_cdf = 0 if i==0 else cdf[i-1]
            for j in range(i, self.size):
                integral = cdf[j]-left_cdf
                if integral < conf_level:
                    continue
                x1,x2 = self.domain[i], self.domain[j]
                center_q = left_cdf + integral/2.0
                candidates[i,j] = (x2-x1, integral, center_q)
                break  # move to next left-side starting point

        # Sort by interval size and obscure tie-breaking criteria
        keys = list(candidates)
        if not keys:
            assert False, "Couldn't find an interval of length conf_level %s" % conf_level

        keys.sort(key=lambda i_j: (
            candidates[i_j[0],i_j[1]][0], # interval size. if ties, then look at:
            # coverage no bigger than necessary - closest to conf_level
            candidates[i_j[0],i_j[1]][1],
            # interval closer to center; thus prefer symmetric interval if
            # everything is tied
            abs(0.5 - candidates[i_j[0],i_j[1]][2]),
            # leftmost, only because we need to break the tie somehow
            i_j[0],
            ))
        ret_i,ret_j = keys[0]
        return self.domain[ret_i], self.domain[ret_j]
    

def dist_from_logprobs(value_to_unnorm_logprob):
    """An attempt to reuse the current __init__ code above, which works in prob
    space, by converting up from logprob input.  Not sure yet this is the right
    thing to do -- maybe it would be better to use log-domain arithmetic to do
    all the code in log-domain.
    
    input to this function is a dictionary.
    """
    unnorm_logprobs = np.array(list(value_to_unnorm_logprob.values()))
    #logZ = scipy.misc.logsumexp(unnorm_logprobs) #deprecated, use new logsumexp
    logZ = scipy.special.logsumexp(unnorm_logprobs)
    value_to_norm_logprobs = {k: ulp-logZ for k,ulp in list(value_to_unnorm_logprob.items())}
    # Danger step!  could get unjustified 0s.
    value_to_norm_probs = {k: np.exp(lp) for k,lp in list(value_to_norm_logprobs.items())}
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

