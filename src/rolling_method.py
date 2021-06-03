# -*- coding: utf-8 -*-
"""
A collection of rolling methods for stat.
"""

from __future__ import division
from __future__ import print_function

import abc
import six

import pandas as pd
import math

#@six.add_metaclass(abc.ABCMeta)
class BaseRollingMetric(object):
    """
    Abstract class for all Rolling Metric.

    Parameters
    ----------
    cycle : int in (1, +inf), optional (default=500)
            This Method is for cycle data or cycle-like data, need not equal to real cycle, nearby is OK
    """

    #@abc.abstractmethod
    def __init__(self, cycle=500):
        if not (0 < cycle):
            raise ValueError("contamination must be in (0, +inf], "
                             "got: %f" % cycle)
        
        self.cycle = cycle

    @abc.abstractmethod
    def score(self, X):
        """
        Process rolling windows computing method. By now it just support n_features=1

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features=1)
            The input samples.

        Returns
        -------
        score : dataframe of shape (n_samples, n_features=1)
            rolling windows score.
        """ 
        pass   

    #@abc.abstractmethod
    def fit(self, X):
        """
        fit function to match sklearn function style

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features=1)
            The input samples.

        Returns
        -------
        fit : Null
        """
        df_X = pd.DataFrame(X)
        self._score = self.score(df_X)

    #@abc.abstractmethod
    def decision_function(self, X):
        """
        fit function to match sklearn function style

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features=1)
            The input samples.

        Returns
        -------
        fit : Null
        """ 
        return self._score

class RollingCount(BaseRollingMetric):
    """
    Process rolling windows Count method. count Number is not-Nan

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """

    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).count()

class RollingSum(BaseRollingMetric):
    """
    Process rolling windows Sum method. compute Sum is non-Nan

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """

    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).sum()
        #return X.rolling_sum(cycle,min_periods=1,center=True)

class RollingMean(BaseRollingMetric):
    """
    Process rolling windows Mean method. compute Mean

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).mean()
        #return X.rolling_mean(cycle,min_periods=1,center=True)

class RollingMedian(BaseRollingMetric):
    """
    Process rolling windows Median method. Get the Median value

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).median()
        #return X.rolling_median(cycle,min_periods=1,center=True)

class RollingVar(BaseRollingMetric):
    """
    Process rolling windows Moving variance. Get the Moving variance.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).var()
        #return X.rolling_var(cycle,min_periods=1,center=True)

class RollingStd(BaseRollingMetric):
    """
    Process rolling windows Moving standard deviation. Get Moving standard deviation.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).std()
        #return X.rolling_std(cycle,min_periods=1,center=True)

class RollingCorr(BaseRollingMetric):
    """
    Process rolling windows Moving standard deviation. Get Moving standard deviation.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return X.rolling(self.cycle,min_periods=1,center=True).median()
        #return X.rolling_corr(cycle,min_periods=1,center=True)

class RollingCov(BaseRollingMetric):
    """
    Process rolling windows Unbiased moving covariance. Get Unbiased moving covariance.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return X.rolling(self.cycle, min_periods=1, center=True).cov()
        #return X.rolling_cov(cycle,min_periods=1,center=True)

class RollingMAA(BaseRollingMetric):
    """
    Process rolling windows Absolute Average Number. Get Absolute Average Number,
    it is very useful for change.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        #return X.applymap(math.log10).abs().rolling_mean(cycle,min_periods=1,center=True)
        return X.abs().rolling(self.cycle, min_periods=1, center=True).mean()
        #X.abs().rolling_mean(cycle,min_periods=1,center=True)

class RollingLogMAA(BaseRollingMetric):
    """
    Process rolling windows Absolute Average Number to log value. Get Absolute Average Number to log value
    it is very useful for change.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features=1)
        The input samples.

    Returns
    -------
    score : dataframe of shape (n_samples, n_features=1)
        rolling windows score.
    """
    def score(self, X):
        return (X.abs()+0.0001).applymap(math.log).rolling(self.cycle,min_periods=1,center=True).mean()
        #return X.abs().applymap(math.log10).rolling_mean(cycle,min_periods=1,center=True)
        #return X.abs().log().rolling_mean(cycle,min_periods=1,center=True)