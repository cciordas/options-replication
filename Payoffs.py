
import numpy  as np
from datetime import date

class PayoffSimple:
    """
    The payoff of a Put or Call option.
    """

    def __init__(self, strike, expiry, type, size):
        """
        PURPOSE: Initialize class.
        PARAMS:  option strike         [float]
                 option expiration     [date]
                 whether a Put or Call ['P'/'C']
                 payoff size           [int]
        """        
        # error check
        if not isinstance(expiry, date):   raise TypeError ("bad option expiration: must be represented as a 'datetime.date'")
        if type.upper() not in ["C", "P"]: raise ValueError("bad option type: must be 'C' / 'P'")            

        # save payoff description
        self.strike = strike
        self.expiry = expiry
        self.type   = type.upper()
        self.size   = size


    def intrinsic_value(self, spot):
        """
        PURPOSE: Calculates the intrinsic value of the payoff, for a given spot.
        PARAMS:  spot price      [float]
        RETURNS: intrinsic value [float]
        """        
        if self.type == 'C': return self.size * (spot - self.strike) if spot > self.strike else 0.0
        if self.type == 'P': return self.size * (self.strike - spot) if spot < self.strike else 0.0


    def intrinsic_values(self, spots):
        """
        PURPOSE: Calculates the intrinsic value of the payoff, for a given array of spots.
        PARAMS:  spot prices      [numpy array of float]
        RETURNS: intrinsic values [numpy array of float]
        """        
        strikes = self.strike * np.ones(spots.shape)
        zeroes  = np.zeros(spots.shape)
        
        if self.type == 'C': return self.size * np.maximum(spots - strikes, zeroes)
        if self.type == 'P': return self.size * np.maximum(strikes - spots, zeroes)


    def expiration_delta(self, spot):
        """
        PURPOSE: Calculates the delta at expiration, for a given spot.
        PARAMS:  spot price          [float]
        RETURNS: delta at expiration [float]
        """        
        if self.type == 'C': return +1 * self.size if spot > self.strike else 0.0
        if self.type == 'P': return -1 * self.size if spot < self.strike else 0.0


    def expiration_deltas(self, spots):
        """
        PURPOSE: Calculates the delta at expiration, for a given array of spots.
        PARAMS:  spot prices         [1-D numpy array of float]
        RETURNS: delta at expiration [1-D numpy array of float]
        """        
        K      = self.strike
        S      = spots.flatten()
        deltas = np.ones_like(S)

        for i in range(len(S)):
            if self.type == 'C': deltas[i] = +1 * self.size if S[i] > K else 0.0
            if self.type == 'P': deltas[i] = -1 * self.size if S[i] < K else 0.0

        return deltas.reshape(spots.shape)


    def __neg__(self):
        """
        PURPOSE: Creates the payoff opposite to the current one.
        RETURNS: A new PayoffSimple instance.
        """
        return PayoffSimple(self.strike, self.expiry, self.type, -self.size)

# ---------- #

class PayoffConstExpiration:
    """
    A linear combination of Put and Call payoffs, for the same underlying and having the same expiration.
    """

    def __init__(self, simple_payoffs):
        """
        PURPOSE: Initialize class.
        PARAMS:  list of Put/Call payoffs having the same expiry [list of PayoffSimple]
        """
        # error checks
        if len(simple_payoffs) == 0:
            raise ValueError("must contain at least one simple payoff")
        for pof in simple_payoffs:
            if not isinstance(pof, PayoffSimple):
                raise TypeError("payoffs must be instances of PayoffSimple")
        for pof in simple_payoffs:
            if pof.expiry != simple_payoffs[0].expiry:
                raise ValueError("payoffs must have same expiration")

        # save inputs
        self.simple_payoffs = simple_payoffs
        self.expiry         = simple_payoffs[0].expiry 


    def intrinsic_value(self, spot):
        """
        PURPOSE: Calculates the intrinsic value of the payoff, for a given spot.
        PARAMS:  spot price      [float]
        RETURNS: intrinsic value [float]
        """        
        intrval = 0
        for pof in self.simple_payoffs:
            intrval += pof.intrinsic_value(spot)
        return intrval


    def intrinsic_values(self, spots):
        """
        PURPOSE: Calculates the intrinsic value of the payoff, for a given array of spots.
        PARAMS:  spot prices      [numpy array of float]
        RETURNS: intrinsic values [numpy array of float]
        """        
        intrvals = np.zeros(spots.shape)
        for pof in self.simple_payoffs:
            intrvals += pof.intrinsic_values(spots)
        return intrvals


    def expiration_delta(self, spot):
        """
        PURPOSE: Calculates the delta at expiration, for a given spot.
        PARAMS:  spot price          [float]
        RETURNS: delta at expiration [float]
        """        
        delta = 0
        for pof in self.simple_payoffs:
            delta += pof.expiration_delta(spot)
        return delta


    def expiration_deltas(self, spots):
        """
        PURPOSE: Calculates the delta at expiration, for a given array of spots.
        PARAMS:  spot prices         [1-D numpy array of float]
        RETURNS: delta at expiration [1-D numpy array of float]
        """        
        deltas = 0
        for pof in self.simple_payoffs:
            deltas += pof.expiration_deltas(spots)
        return deltas


    def __add__(self, rhs):
        """
        PURPOSE: Adds two payoffs.
        RETURNS: A new PayoffConstExpiration instance.
        """
        return PayoffConstExpiration(self.simple_payoffs + rhs.simple_payoffs)


    def __sub__(self, rhs):
        """
        PURPOSE: Substracts two payoffs.
        RETURNS: A new PayoffConstExpiration instance.
        """
        return PayoffConstExpiration(self.simple_payoffs + (-rhs).simple_payoffs)
        

    def __neg__(self):
        """
        PURPOSE: Creates the payoff opposite to the current one.
        RETURNS: A new PayoffConstExpiration instance.
        """
        return PayoffConstExpiration([-pof for pof in self.simple_payoffs])

# ========== #
