from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


"""
@Project       : ML4Science Project
@File          : cider.py
@Author        : Yiyang Feng
@Date          : 2022/12/20 18:33
@Version       : 1.0
"""

"""
define a class for computing cider score.
"""




from cider.cider_scorer import CiderScorer


class Cider(object):
    """
    Main Class to compute the CIDEr metric
    """

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute(self, res, gts):
        """
        Main function to compute CIDEr score
        :param  res: list with candidate sentence
        :param  gts: list with reference sentences
        :return: cider (float): computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for idx in range(len(gts)):
            hypo = res[idx]
            ref = gts[idx]

            # Sanity check.
            assert(isinstance(hypo, str))
            assert(isinstance(ref, list))
            assert(len(ref) > 0)

            cider_scorer += (hypo, ref)

        return cider_scorer.compute_score()

    @staticmethod
    def method():
        return "CIDEr"