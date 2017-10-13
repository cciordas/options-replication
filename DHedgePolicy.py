import abc
import numpy as np
from PricerBS import PricerBlackScholes

class DHedgePolicy(object):
    """
    Implements a delta hedging algorithm.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass
        
    @abc.abstractmethod
    def calculate_hedges(self, spots, t):
        """
        PURPOSE: Calculates the delta hedge according to the current hedging policy
                 for a given set of spot prices, and a given time to expiration
        PARAMS:  spots - spot prices for which to calculate the delta hedge [numpy arrray of floats]
                 t     - time to expiration                                 [datetime.datetime]
        RETURNS: an array, having the same shape as the 'spots' argument, storing the hedges 
        """
        pass

        
class DHedgePolicyBS(DHedgePolicy):
    """
    Calculates deltas using the Black-Scholes formula.
    """
    
    def __init__(self, payoff, vol, r):
        """
        PURPOSE: Initialize the hedger.
        PARAMS:  payoff - the payoff at expiration          [PayoffConstExpiration]
                 vol    - annualized constant volatility    [float]
                 r      - annualized constant interest rate [float]
        """
        DHedgePolicy.__init__(self)

        self.vol      = vol
        self.r        = r
        self.pricerBS = PricerBlackScholes(payoff)
        
    def calculate_hedges(self, spots, t):
        """
        PURPOSE: Calculates the BS deltas, for a given set of spot prices, and a given time to expiration
        PARAMS:  spots - spot prices for which to calculate the delta hedge [numpy arrray of floats]
                 t     - time to expiration                                 [datetime.datetime]
        RETURNS: an array, having the same shape as the 'spots' argument, storing the hedges 
        """
        return self.pricerBS.price(t, spots, self.vol, self.r)[1]        

    
class DHedgePolicyRandom(DHedgePolicy):
    """
    Calculates deltas by sampling from a uniform distribution.
    """
    
    def __init__(self, mindelta, maxdelta):
        """
        PURPOSE: Initialize the hedger.
        PARAMS:  range within which the delta takes values
        """
        DHedgePolicy.__init__(self)
        self._m = mindelta
        self._M = maxdelta
        
    def calculate_hedges(self, spots, t):
        return np.random.uniform(self._m, self._M, spots.shape)        


class DHedgePolicyConstant(DHedgePolicy):
    """
    Returns a constant value.
    """
    
    def __init__(self, val):
        """
        PURPOSE: Initialize the hedger.
        PARAMS:  the constant value returned for the hedge
        """
        DHedgePolicy.__init__(self)
        self._val = val
        
    def calculate_hedges(self, spots, t):
        return self._val
    

class DHedgePolicyLogistic(DHedgePolicy):
    """
    """

    def __init__(self, K, vol, expiry, Lambda):
        DHedgePolicy.__init__(self)
        self.K      = K
        self.vol    = vol
        self.expiry = expiry
        self.Lambda = Lambda

    def calculate_hedges(self, spots, t):
        tdelta   = self.expiry - t
        tdelta_y = tdelta.days / 365.0 + tdelta.seconds / 86400.0 / 365.0 
        if tdelta_y == 0:
            tdelta_y = 0.000001

        x = (spots - self.K)/float(self.K)/self.vol
        return 1.0 / (1 + np.exp(-self.Lambda*x/tdelta_y))        
