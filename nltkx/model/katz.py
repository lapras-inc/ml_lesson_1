# Natural Language Toolkit: Corrected Katz backoff
# Based on Jurafsky & Martin 2nd edition, section 4.7
# Includess a number of 'fixes' to cover the cases where
#  the cleanliness assumptions of the published code are
#  not satisfied -- see comments below
#
# Author: Henry S. Thompson

import math
from itertools import chain

from nltk.model.ngram import NgramModel
from nltk.util import ingrams

from nltk.probability import ConditionalProbDist, ConditionalFreqDist, MLEProbDist, GoodTuringProbDist, FreqDist

LowHacked = 'lowHacked'
P_1Guess = 0.03                         # Our finger-in-the air guess for
#  probability of unseen items
#  if we have no good evidence for
#  a principled estimate


class KGoodTuringProbDist(GoodTuringProbDist):
    """
    Modify basic GoodTuring to include a cut-off parameter, use
    un-adjusted counts above that, and properly discount all
    probabilities

    Based on Jurafsky & Martin 2nd edition, p. 137

    Includes fixes for cases where the input data isn't clean enough
    for the original code to work
    """

    def __init__(self, freqdist, k=5, v=1,
                 liveDangerously=False, ctxt='unknown'):
        """
        Creates a Good-Turing probability distribution estimate with a
        Katz threshhold and fixups for less-than-perfect data.  This
        method calculates the probability mass to assign to events
        with zero or low counts based on the number of events with
        higher counts.

        @param freqdist:    The frequency counts upon which to base the
                            estimation.
        @type  freqdist:    C{FreqDist}
        @param k:           The threshhold above which counts are assumed
                            to be reliable.  Defaults to 5.
        @type  k:           C{Int}
        @param v:           The number of unseens.  Defaults to 1.
        @type  v:           C{Int}v
        @param liveDangerously: If False, check that the total probability
                            mass after all adjustments is close to 1.
                            Defaults to False.
        @type  liveDangerously: C{Boolean}
        @param ctxt:        If this dist is part of a conditional probability
                            distribution, the context.  Defaults to 'unknown'.
        @type  ctxt:        C{String}
        """
# The basic problem with the published approach in practice is that it
# assumes that there will be samples at all frequencies <= k+1, and
# that may be false
# In practice, there seem to be two ways in which
# this happens:
#  1) The population is well-covered, even over-covered,
#     by the sample, so all items are well-represented, i.e. there are
#     only high counts
#  2) The population is sparse, so there are gaps, at
#     worst not even any hapaxes, in the region below k+1
#
# There's a third way to lose: The Katz threshhold story assumes that
# Nr(i) > Nr(i+1) for all i<=k -- if it isn't, the discounting of values
# below k is seriously wacky. . .
#
# This implementation, rather than do smoothing, which seems
# unjustified in any of the problem cases identified above, takes some
# _ad hoc_ corrective measures, described by cases below
        self._freqdist = freqdist
        self._v = float(v)
        self._ctxt = ctxt
        assert k > 0, 'k parameter must be a positive integer'
        self._k = k
        assert freqdist.N() > 0, 'freqdist must have content'
        self._N = float(freqdist.N())
        Nns = sum(freqdist.Nr(i) > 0 for i in range(1, k + 1))
        N1 = freqdist.Nr(1)
        kcn = freqdist.Nr(k + 1)
        if Nns is 0:
            # If there's lots of mass high up, we're probably OK,
            #  because the chances of something unseen are small
            # We hack out a tiny bit just to avoid ever failing
            if self._N > 100 or freqdist.B() > 20:  # completely arbitrary
                self.status = 'bigSkewed'
            else:
                self.status = 'weak'
            self._N1 = P_1Guess * self._N
            m = freqdist.max()
            # discount this guy to make up for the hack
            #  on the grounds the he can afford it
            mn = freqdist[m]
            freqdist[m] = float(mn) - self._N1
            # print ('hack_1',mn,freqdist[m],mn/self._N,freqdist[m]/self._N,self._N1,self._N,P_1Guess)
        elif ((Nns < k) or kcn == 0 or
              sum(freqdist.Nr(i) > freqdist.Nr(i + 1) for i in range(1, k + 1)) != k):
            # Some gaps, or upsidedown, or not well-behaved <=k+1, hack it
            # Leave ourselves a note to discount any count < k
            #  equally -- not principled, but at least it will work
            self.status = LowHacked
            Nsmall = sum(freqdist.Nr(i) for i in range(2, k + 2))
            # figure out what we have
            #  to work with
            if N1 > 0:
                # We have some hapaxes, just go with that
                self._N1 = float(N1)
                # Discount evenly, but tweak if _only_ hapaxes,
                #  so they don't end up discounted out of existence
                if Nns == 1:
                    self._N1 = self._N1 / 2
                Nsmall += N1
                assert self._N1 > 0, "y %s %s" % (N1, Nsmall)
            else:
                # Otherwise use P_1Guess, but never give it more than half
                self._N1 = min(P_1Guess * self._N, float(Nsmall) / 2.0)
                assert self._N1 > 0, "x %s %s" % (
                    P_1Guess * self._N, float(Nsmall) / 2.0)
            self._subkDiscount = self._N1 / (Nsmall - freqdist.Nr(k + 1))
            assert self._subkDiscount > 0, "skd=0: %s %s %s" % (
                self._N1, Nsmall, self.status)
        else:
            # we're good
            self.status = 'normal'
            self._N1 = float(N1)
            self._kp = float(k + 1) * float(kcn) / self._N1
            self._kpi = 1.0 - self._kp
            assert abs(self._kp * self._kpi) > 0, "uh-oh: %s %s %s %s %s" % (k,
                                                                             kcn, N1, self._kp, self._kpi)
            while self._someNeg(freqdist):
                # cheat even harder!
                freqdist._Nr_cache[1] += 1
                self._N1 = freqdist.Nr(1)
                self._kp = float(k + 1) * float(kcn) / self._N1
                self._kpi = 1.0 - self._kp
                assert abs(
                    self._kp * self._kpi) > 0, "uh-oh: %s %s %s %s %s" % (k, kcn, N1, self._kp, self._kpi)
        if liveDangerously:
            return
        pp = self.prob('UNK') * self._v
        for s in freqdist.samples():
            p = self.prob(s)
            if p <= 0:
                print(('oops1', ctxt, s, p, self._kp, self._kpi, self._N,
                       self.status, self._v))
                print([freqdist.Nr(i) for i in range(1, 10)])
            pp += p
        if abs(pp - 1.0) > 0.001:
            print(('oops', pp, ctxt, self.status, self._N, self._N1,
                   k, self._v, len(freqdist)))

    def _someNeg(self, freqdist):
        for s in freqdist.samples():
            if self.prob(s) <= 0:
                return True
        return False

    def prob(self, sample):
        c = self._freqdist[sample]
        if c == 0:
            # unseen - use the fallback
            c = self._N1 / self._v
        elif c > self._k:
            # use the raw data
            c = float(c)
        elif self.status == LowHacked:
            # Apply bogus discount, see above
            c = float(c) - self._subkDiscount
        else:
            # katz discount
            nc = self._freqdist.Nr(c)
            ncn = self._freqdist.Nr(c + 1)
            assert nc * ncn > 0, \
                "can't estimate because of missing data 2 %s,%s,%s,%s -- reduce k or get more data!" % (
                    c, nc, ncn, sample)
            c = ((float(c + 1) * (float(ncn) / float(nc))) - (float(c) * self._kp)) / \
                self._kpi

        return c / self._N

    def discount(self):
        return self._N1 / self._N

    def __contains__(self, item):
        return item in self._freqdist

    def __getitem__(self, item):
        return self._freqdist[item]

    def __repr__(self):
        """
        @rtype: C{string}
        @return: A string representation of this C{ProbDist}.
        """
        return '<KGoodTuringProbDist based on %d samples>' % self._freqdist.N()


class KBNgramModel(NgramModel):
    """
    Katz-threshholded backoff on top of NLTK NgramModel
    """

    def __init__(self, n, train, k=5, v=None,
                 liveDangerously=False, quiet=False):
        """
        Creates an Katz-threshholded Ngram language model to capture
        patterns in n consecutive words of training text.
        Uses the KGoodTuringProbDist to estimate the conditional and unigram probabilities,
        to provide coverage of Ngrams not seen during training.

        @param n: the order of the language model (ngram size)
        @type n: C{int}
        @param train: the training text
        @type train: C{list} of C{string}
        @param k: The threshhold above which counts are assumed
                  to be reliable.  Defaults to 5.
        @type  k: C{Int}
        @param v: The number of unseens of degree 1.  Defaults to the
                  number of types in the training set
        @type  v: C{Int}
        @param liveDangerously: If False, for each model check that
                                the total probability mass after all
                                adjustments is close to 1.  Defaults
                                to False.
        @type  liveDangerously: C{Boolean}
        @param quiet: Various information will be printed during model
                       construction unless this is True.  Defaults to False.
        @type  quiet: C{Boolean}
        """
        self._n = n
        self._N = 1 + len(train) - n
        fd = FreqDist(train)
        if v is None:
            v = fd.B()
        print(('v', v))
        if n == 1:
            # Treat this case specially
            self._model = KGoodTuringProbDist(fd, k, v, liveDangerously, ())
            if not quiet:
                print("%s entries for %s tokens at degree 1, %s" % (len(fd),
                                                                    fd.N(),
                                                                    self._model.status))
        else:
            def estimator(fdist, ctxt): return KGoodTuringProbDist(fdist, k, v,
                                                                   liveDangerously,
                                                                   ctxt)

            cfd = ConditionalFreqDist()

            for ngram in ingrams(train, n):
                # self._ngrams.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                cfd[context].inc(token)

            self._model = ConditionalProbDist(cfd, estimator, True)
            if not quiet:
                statuses = {'normal': 0, 'bigSkewed': 0,
                            'weak': 0, LowHacked: 0}
                for ctx in cfd.conditions():
                    statuses[self[ctx].status] += 1
                print("%s conditions at degree %s" %
                      (len(cfd.conditions()), n))
                for s in list(statuses.keys()):
                    print(" %s %6d" % (s, statuses[s]))

            # recursively construct the lower-order models
            self._backoff = KBNgramModel(n - 1, train, k, v, liveDangerously)

    # Katz Backoff probability
    def prob(self, word, context, verbose=False):
        '''Evaluate the probability of this word in this context.
        @param word: The item to estimate for.
        @type word: C{String}
        @param context: The left context (or () if none)
        @type context: C{Tuple}
        @param verbose: Print a record of backoffs if True.  Defaults to False.
        @type verbose: C{Boolean}
        '''

        context = tuple(context)
        if self._n == 1:
            # KGoodTuring will do the right thing for unigrams
            return self._model.prob(word)
        elif (context in self._model and
              word in self[context].freqdist()):
            return self[context].prob(word)  # already discounted
        else:
            if verbose:
                print("backing off from %s to %s for %s" %
                      (self._n, self._n - 1, context + (word,)))
            return self._alpha(context) * self._backoff.prob(word, context[1:])

    def logprob(self, word, context, verbose=False):
        '''Log (base 2) of the probability of this word in this context
        @param word: The item to estimate for.
        @type word: C{String}
        @param context: The left context (or () if none)
        @type context: C{Tuple}
        @param verbose: Print a record of backoffs if True.  Defaults to False.
        @type verbose: C{Boolean}
        '''
        return math.log(self.prob(word, context, verbose), 2)

    def entropy(self, text, verbose=False, perItem=False):
        '''Evaluate the total entropy of a text with respect to this model.
        This is the sum of the log probability of each word in the message.
        @param text: The items to estimate for.
        @type word: C{List} of C{String}
        @param verbose: Print a record of backoffs if True.  Defaults to False.
        @type verbose: C{Boolean}
        @param perItem: Normalise for length of text if True.
                        Defaults to False.
        @type perItem: C{Boolean}
        '''

        e = 0.0
        m = len(text)
        cl = self._n - 1
        for i in range(cl, m):
            context = tuple(text[i - cl: i])
            token = text[i]
            e -= self.logprob(token, context, verbose)
        if perItem:
            return e / (m - cl)
        else:
            return e

    def perplexity(self, text, perItem=False):
        '''Perplexity of the whole text -- 2 to the power of the entropy'''
        return math.pow(2.0, self.entropy(text, perItem))

    def __contains__(self, item):
        return item in self._model

    def __getitem__(self, item):
        return self._model[item]

    def __repr__(self):
        return '<KBNgramModel with %d %d-grams>' % (self._N, self._n)


def demo():
    from nltk.corpus import brown
    lm = KBNgramModel(3, brown.words(categories='news'))
    print(lm)
    print(lm.entropy(['the', 'city', 'should', 'provide', 'a', 'CD', 'headquarters', 'so', 'that', 'pertinent',
                      'information', 'about', 'the', 'local', 'organization', 'would', 'be', 'centralized'],
                     True, True))


if __name__ == '__main__':
    demo()
