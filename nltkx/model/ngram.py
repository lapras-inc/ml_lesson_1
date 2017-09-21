# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2009 NLTK Project
# Author: Steven Bird <sb@csse.unimelb.edu.au>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

import random
from itertools import chain
from math import log

from nltk.probability import (ConditionalProbDist, ConditionalFreqDist,
                              MLEProbDist, FreqDist)
try:
    from nltk.util import ingrams
except Exception:
    from nltkx.util import ingrams

from .api import *


class NgramModel(ModelI):
    """
    A processing interface for assigning a probability to the next word.
    """

    # add cutoff
    def __init__(self, n, train, estimator=None):
        """
        Creates an ngram language model to capture patterns in n consecutive
        words of training text.  An estimator smooths the probabilities derived
        from the text and may allow generation of ngrams not seen during
        training.

        @param n: the order of the language model (ngram size)
        @type n: C{int}
        @param train: the training text
        @type train: C{list} of C{string}
        @param estimator: a function for generating a probability distribution
        @type estimator: a function that takes a C{ConditionalFreqDist} and
              returns a C{ConditionalProbDist}
        """
        self._n = n
        self._N = 1 + len(train) - n

        if estimator is None:
            def estimator(fdist, bins): return MLEProbDist(fdist)

        if n == 1:
            fd = FreqDist(train)
            self._model = estimator(fd, fd.B())
        else:
            cfd = ConditionalFreqDist()
            self._ngrams = set()
            self._prefix = ('',) * (n - 1)

            for ngram in ingrams(chain(self._prefix, train), n):
                self._ngrams.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                cfd[context][token] += 1

            self._model = ConditionalProbDist(cfd, estimator, len(cfd))
        # recursively construct the lower-order models
        if n > 1:
            self._backoff = NgramModel(n - 1, train, estimator)

    # Katz Backoff probability
    def prob(self, word, context, verbose=False):
        """
        Evaluate the probability of this word in this context.
        """
        context = tuple(context)
        if self._n == 1:
            if not(self._model.SUM_TO_ONE):
                # Smoothing models should do the right thing for unigrams
                #  even if they're 'absent'
                return self._model.prob(word)
            else:
                try:
                    return self._model.prob(word)
                except Exception:
                    raise RuntimeError("No probability mass assigned"
                                       "to unigram %s" % (word))
        if context + (word,) in self._ngrams:
            return self[context].prob(word)
        else:
            alpha = self._alpha(context)
            if alpha > 0:
                if verbose:
                    print("backing off for %s" % (context + (word,),))
                return alpha * self._backoff.prob(word, context[1:], verbose)
            else:
                if verbose:
                    print("no backoff for %s as model doesn't do any smoothing" % word)
                return alpha

    def _alpha(self, tokens):
        return self._beta(tokens) / self._backoff._beta(tokens[1:])

    def _beta(self, tokens):
        # print self,self._n,self._model,tokens
        if tokens in self:
            return self[tokens].discount()
        else:
            return 1

    def logprob(self, word, context, verbose=False):
        """
        Evaluate the (negative) log probability of this word in this context.
        """

        return -log(self.prob(word, context, verbose), 2)

    # NB, this will always start with same word since model
    # is trained on a single text
    def generate(self, num_words, context=()):
        '''Generate random text based on the language model.'''
        text = list(context)
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text

    def _generate_one(self, context):
        context = (self._prefix + tuple(context))[-self._n + 1:]
        # print "Context (%d): <%s>" % (self._n, ','.join(context))
        if context in self:
            return self[context].generate()
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return '.'

    def entropy(self, text, verbose=False, perItem=False):
        """
        Evaluate the total entropy of a text with respect to the model.
        This is the sum of the log probability of each word in the message.
        """

        e = 0.0
        m = len(text)
        cl = self._n - 1
        for i in range(cl, m):
            context = tuple(text[i - cl: i])
            token = text[i]
            e += self.logprob(token, context, verbose)
        if perItem:
            return e / (m - cl)
        else:
            return e

    def __contains__(self, item):
        try:
            return item in self._model
        except Exception:
            try:
                # hack if model is an MLEProbDist, more efficient
                return item in self._model._freqdist
            except Exception:
                return item in self._model.samples()

    def __getitem__(self, item):
        return self._model[item]

    def __repr__(self):
        return '<NgramModel with %d %d-grams>' % (self._N, self._n)


def demo():
    from nltk.corpus import brown
    from nltk.probability import LidstoneProbDist, WittenBellProbDist

    def estimator(fdist, bins): return LidstoneProbDist(fdist, 0.2)

#    estimator = lambda fdist, bins: WittenBellProbDist(fdist, 0.2)
    lm = NgramModel(3, brown.words(categories='news'), estimator)
    print(lm)
#    print lm.entropy(sent)
    text = lm.generate(100)
    import textwrap
    print('\n'.join(textwrap.wrap(' '.join(text))))


if __name__ == '__main__':
    demo()
