
import numpy     as np
from scipy.stats import norm
from Payoffs     import *

class PricerBlackScholes:
    """
    A simple Black-Scholes options pricer.
    Can price a linear combination of Put/Call payoffs having the same expiration (and underlying).
    Assumes constant interest rates and no dividends.
    Uses calendar days (1 year = 365 days), no holidays and 24 hours trading days.
    """

    def __init__(self, payoff):
        """
        PURPOSE: Initialize the pricer.
        PARAMS:  the payoff at expiration [PayoffConstExpiration]
        """
        if not isinstance(payoff, PayoffConstExpiration):
            raise TypeError("payoff argument must be instance of PayoffConstExpiration")
        self._payoff = payoff


    def price(self, asof, spots, vol, r):
        """
        PURPOSE: Calculates the premium and the greeks.
        PARAMS:  asof  - date as of which the payoff is being priced      [datetime.datetime]
                 spots - spot prices for which the payoff is being priced [numpy arrray of floats]
                 vol   - annualized constant volatility                   [float]
                 r     - annualized constant interest rate                [float]
        RETURNS: a tuple of numpy arrays, each array having the same shape as the 'spots' argument
                 the arrays in the tuple are: premiums, deltas, gammas
        """
        if asof > self._payoff.expiry:
            raise ValueError("options may be priced only as of prior to expiry")

        # price at expiration
        if asof == self._payoff.expiry:
            premiums = self._payoff.intrinsic_values (spots).reshape(spots.shape)
            deltas   = self._payoff.expiration_deltas(spots).reshape(spots.shape)
            gammas   = np.zeros(spots.shape)
            return (premiums, deltas, gammas)

        premiums = np.zeros(spots.shape)
        deltas   = np.zeros(spots.shape)
        gammas   = np.zeros(spots.shape)

        tdelta = self._payoff.expiry - asof                                # datetime.timedelta
        t      = tdelta.days / 365.0 + tdelta.seconds / 86400.0 / 365.0    # time to expiration in years
        df     = np.exp(-r*t)                                              # discount factor

        # add the contribution of each simple (P/C) payoff
        for pof in self._payoff.simple_payoffs:
            strike    = pof.strike
            sz        = pof.size
            vol_total = vol * np.sqrt(t)

            d1 = (np.log(spots/strike) + (r + 0.5 * vol * vol) * t) / vol_total
            d2 = d1 - vol_total
            
            if pof.type == "C":
                premiums += sz * (spots * norm.cdf(d1) - df * strike * norm.cdf(d2))
                deltas   += sz * (norm.cdf(d1))
                gammas   += sz * (norm.pdf(d1) * spots * np.sqrt(t))

            if pof.type == "P":
                premiums += sz * (df * strike * norm.cdf(-d2) - spots * norm.cdf(-d1))
                deltas   += sz * (norm.cdf(d1) - 1)
                gammas   += sz * (norm.pdf(d1) * spots * np.sqrt(t))

        return (premiums, deltas, gammas)

# ========== #
