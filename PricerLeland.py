
import numpy     as np
from scipy.stats import norm
from Payoffs     import *

class PricerLeland:
    """
    Implements the Leland correction to the Black-Scholes model.
    Assumes constant interest rates and no dividends.
    Uses calendar days (1 year = 365 days), no holidays and 24 hours trading days.
    WARNING: the Leland formula applies only to portfolio of options with a constant gamma sign.
             To keep things simple, this pricer can only process simple payoffs (one Put or one Call).
    """

    def __init__(self, payoff, dt, trans_cost, delta_cost):
        """
        PURPOSE: Initialize the pricer.
        PARAMS:  payoff     - the payoff at expiration (either a Put or Call)                  [PayoffSimple]
                 dt         - rehedge interval in years (1 year = 365 days)                    [float]
                 trans_cost - hedging transaction costs as % of spot price                     [float]
                 delta_cost - whether to factor in the cost of setting the initial delta hedge [bool]
        """
        # error check
        if not isinstance(payoff, PayoffSimple):
            raise TypeError("payoff argument must be instance of PayoffSimple")

        # save configuration values
        self._payoff     = payoff
        self._dt         = dt
        self._trans_cost = trans_cost
        self._delta_cost = delta_cost


    def price(self, asof, spots, vol, r):
        """
        PURPOSE: Calculates the option bid and ask premium.
        PARAMS:  asof  - date as of which the option is being priced      [datetime.datetime]
                 spots - spot prices for which the option is being priced [numpy arrray of floats]
                 vol   - annualized constant volatility                   [float]
                 r     - annualized constant interest rate                [float]
        RETURNS: two numpy arrays having the same shape as the 'spots' argument, the first stores the bids, the seconds the asks
        """        
        if asof >= self._payoff.expiry:
            raise ValueError("options may be priced only as of prior to expiry")

        bids = np.zeros(spots.shape)
        asks = np.zeros(spots.shape)        

        tdelta     = self._payoff.expiry - asof                               # datetime.timedelta
        t          = tdelta.days / 365.0 + tdelta.seconds / 86400.0 / 365.0   # time to expiration in years
        df         = np.exp(-r*t)                                             # discount factor
        strike     = self._payoff.strike
        sz         = self._payoff.size
        opt_type   = self._payoff.type
        trans_cost = self._trans_cost
        dt         = self._dt

        # Leland correction to volatility
        vol_bid = vol * np.sqrt(1 - trans_cost / vol * np.sqrt(8 / (np.pi * dt)))
        vol_ask = vol * np.sqrt(1 + trans_cost / vol * np.sqrt(8 / (np.pi * dt)))            

        vol_total_bid = vol_bid * np.sqrt(t)
        vol_total_ask = vol_ask * np.sqrt(t)
        
        d1_bid = (np.log(spots/strike) + (r + 0.5 * vol_bid * vol_bid) * t) / vol_total_bid
        d1_ask = (np.log(spots/strike) + (r + 0.5 * vol_ask * vol_ask) * t) / vol_total_ask

        d2_bid = d1_bid - vol_total_bid
        d2_ask = d1_ask - vol_total_ask            

        if opt_type == "C":
            bids = sz * (spots * norm.cdf(d1_bid) - df * strike * norm.cdf(d2_bid))
            asks = sz * (spots * norm.cdf(d1_ask) - df * strike * norm.cdf(d2_ask))            
            
        if opt_type == "P":
            bids = sz * (df * strike * norm.cdf(-d2_bid) - spots * norm.cdf(-d1_bid))
            asks = sz * (df * strike * norm.cdf(-d2_ask) - spots * norm.cdf(-d1_ask))

        # factor in the cost of the initial (static) delta hedge
        if self._delta_cost:

            d1 = (np.log(spots/strike) + (r + 0.5 * vol * vol) * t) / (vol * np.sqrt(t))

            if opt_type == "C": deltas = sz * (norm.cdf(d1))
            if opt_type == "P": deltas = sz * (norm.cdf(d1) - 1)

            bids = bids - trans_cost * deltas * spots
            asks = asks + trans_cost * deltas * spots            
        
        return (bids, asks)

# ========== #
