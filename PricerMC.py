import abc
import numpy      as np
from datetime     import datetime,timedelta
from DHedgePolicy import *
from Payoffs      import *
from PricerBS     import PricerBlackScholes
from scipy.stats  import norm, skew, kurtosis

class PricerMC(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, spots, vol, r, npaths, npoints_day, asof, antithetic, ctrlvariate):
        """
        PARAMS:
        spots       - spot prices for which the payoff is being priced             [1-D numpy arrray of floats]
        vol         - annualized constant volatility                               [float]
        r           - annualized constant interest rate                            [float]
        npaths      - the number of paths to generate for each starting spot price [int]
        npoints_day - nr of points generated for each calendar day                 [int]
        asof        - date as of which the payoff is being priced                  [datetime.datetime]
        antithetic  - generate twice the number of paths - include 'mirror' paths  [bool]
        ctrlvariate - use a control variate                                        [bool]
        """
        

        self.spots       = spots
        self.vol         = vol
        self.r           = r
        self.npaths      = npaths
        self.npoints_day = npoints_day
        self.asof        = asof
        self.antithetic  = antithetic
        self.ctrlvariate = ctrlvariate
        self.paths       = []
        

    def generate_paths(self, expiry):
        """
        PURPOSE: 
        Generate the paths used by the MC pricer.
        Saves a 3-D numpy array of shape ((2 if antithetic else 1) * npaths, npoints, spots), 
        storing the path prices in 'self.paths'.
        PARAMS:
        how far out to generate the paths ('self.npoints_day' for each day between 'self.asof' and 'expiry')
        """

        dt  = 1.0 / (365.0 * self.npoints_day)   # <-- the time span between each two points along a path (in years)
        vol = self.vol
        r   = self.r        
        c0  = r - 0.5 * vol * vol
        c1  = vol * np.sqrt(dt)

        # initialize the 3-D array that stores all the paths
        npoints = int((expiry - self.asof).days * self.npoints_day + 1)   # <-- the total number of points along each path
        nspots  = len(self.spots)
        npaths  = self.npaths
        Npaths  = (2 if self.antithetic else 1) * self.npaths
        paths   = np.zeros((Npaths, npoints, nspots), dtype=float)             

        # initialize the start of each path
        paths[:, 0, :] = np.tile(self.spots, Npaths).reshape(Npaths, nspots)

        # step through time
        for counter in range(npoints-1):

            # generate random returns for all paths, for the current time step
            noises = np.random.normal(0.0, 1.0, npaths * nspots).reshape(npaths, nspots)      

            # convert returns into prices (notice the use of antithetic variables)
            if self.antithetic:
                paths[:npaths, counter + 1, :] = paths[:npaths, counter, :] * np.exp(c0 * dt + c1 * noises)
                paths[npaths:, counter + 1, :] = paths[npaths:, counter, :] * np.exp(c0 * dt - c1 * noises)
            else:
                paths[:, counter + 1, :] = paths[:, counter, :] * np.exp(c0 * dt + c1 * noises)
                
        self.paths = paths
        
        
    @abc.abstractmethod
    def price(self, payoff):
        """
        PURPOSE: Calculates the premium of the options payoff.
        PARAMS:  payoff - the payoff at expiration [PayoffConstExpiration]
        RETURNS: Derived class implementations must return the results of the simulation.
        """        
        if self.asof >= payoff.expiry:                    raise ValueError("options may be priced only as of prior to expiry")
        if not isinstance(payoff, PayoffConstExpiration): raise TypeError ("payoff argument must be instance of PayoffConstExpiration")
        
# ---------- #
    
class PricerMC_NoHedging(PricerMC):
    """
    Calculates the option price by simulating lognormal paths and by averaging the final payoffs.
    Can price a linear combination of Put/Call payoffs having the same expiration (and underlying).
    Assumes constant interest rates and no dividends.
    Uses calendar days (1 year = 365 days), no holidays and 24 hours trading days.

    How to use:
    ----------

    EXP    = datetime.today() + timedelta(days=30)
    K      = 100
    SZ     = 1
    VOL    = 0.2
    R      = 0.0
    SPOTS  = np.arange(90, 110, 1, dtype = float)
    PAYOFF = PayoffSimple(K, EXP, "C", 1)
    NPATHS = 10

    pricer = PricerMC_NoHedging(SPOTS, VOL, R, NPATHS, 1)
    pricer.generate_paths(EXP)
    pxs = pricer.price(payoff)
    """
    
    def __init__(self, spots, vol, r, npaths, npoints_day, asof=datetime.today(), antithetic=True):    
        """
        See 'PricerMC.__init__(...)' documentation.
        """        
        PricerMC.__init__(self, spots, vol, r, npaths, npoints_day, asof, antithetic, True)

        
    def price(self, payoff):
        """
        See 'PricerMC.price(...)' documentation.
        RETURNS: a 1-D numpy array having the same number of entries as 'spots'
                 storing the premium for the associated spot price.
        """        
        PricerMC.price(self, payoff)
               
        # for each path, replace final underlying price by option portfolio payoff
        self.paths[:, -1, :] = payoff.intrinsic_values(self.paths[:, -1, :])
        
        # calculate path averages (one average for each starting spot price)
        t = (payoff.expiry - self.asof).days / 365.0    # <-- time to expiry (in years)
        return np.exp(-self.r*t) * np.average(self.paths[:,-1,:], axis = 0), self.paths[:,-1,:]
    
# ------------------------------ #

class PricerMC_DynamicHedging(PricerMC):
    """
    Calculates the option price by simulating lognormal paths and by averaging the P&L 
    of dynamic hedging along each path. Transaction consts can be included as a constant
    percentage of the notional traded.    
    Can price a linear combination of Put/Call payoffs having the same expiration (and underlying).

    For simple payoffs the output coincides with the Leland formula. 
    Leland's formula applies only if sqrt(8/(pi*dt)) * (k/vol) < 1  => k < k_critical with
    k_critical = 0.63 * vol * sqrt(dt), where k represents transaction costs. In practice we notice
    that the agreement between MC simulations and the Leland formula for the bid-ask spread breaks
    down for k > k_critical/2. This might be due to second order terms, which are not ignored by
    Leland, and which become relevant for higher transaction costs.

    Assumes constant interest rates and no dividends.
    Uses calendar days (1 year = 365 days), no holidays and 24 hours trading days.

    How to use:
    ----------

    EXP    = datetime.today() + timedelta(days=30)
    K      = 100
    SZ     = 1
    VOL    = 0.2
    R      = 0.0
    SPOTS  = np.arange(90, 110, 1, dtype = float)
    PAYOFF = PayoffSimple(K, EXP, "C", 1)
    NPATHS = 10

    pricer = PricerMC_DynamicHedging(SPOTS, VOL, R, NPATHS, 1, 0.04, 0.04, DHedgerBS(PAYOFF, VOL, R))
    pricer.generate_paths(EXP)
    bids, asks = pricer.price(PAYOFF)
    """

    def __init__(self, spots, vol, r, npaths, npoints_day, trans_cost, delta_cost, hedge_policy, asof=datetime.today(),
                 antithetic=True, ctrlvariate=True):
        """
        See 'PricerMC.__init__(...)' documentation.
        PARAMS:  
        trans_cost   - hedging transaction costs as % of notional traded                [float]
        delta_cost   - whether to factor in the cost of setting the initial delta hedge [bool]                 
        hedge_policy - used to calculate the delta hedge                                [instance of class derived from DHedgePolicy]
        """        
        if not isinstance(asof, datetime):
            raise TypeError("payoff argument must be instance of datetime.datetime")
        PricerMC.__init__(self, spots, vol, r, npaths, npoints_day, asof, antithetic, ctrlvariate)
        self.trans_cost = trans_cost
        self.delta_cost = delta_cost
        self.hedger     = hedge_policy
        

    def price(self, payoff):
        """
        See 'PricerMC.price(...)' documentation.
        RETURNS: two 1-D numpy array having the same number of entries as the 'spots' argument
                 the first array stores the bid for the option payoff at different spots
                 the second array stores the ask.
        """        
        PricerMC.price(self, payoff)
        Npaths = (2 if self.antithetic else 1) * self.npaths
        nspots = len(self.spots)

        # create arrays to store the dynamic hedging PL along each path        
        PLpaths_nocost = np.zeros((Npaths, nspots), dtype=float)   # PL if long  the option payoff and     zero transaction costs
        PLpaths_long   = np.zeros((Npaths, nspots), dtype=float)   # PL if long  the option payoff and non-zero transaction costs
        PLpaths_shrt   = np.zeros((Npaths, nspots), dtype=float)   # PL if short the option payoff and non-zero transaction costs
        PLstep_expiry  = np.zeros((Npaths, nspots), dtype=float)

        # calculate the time stamps at which we re-hedge 
        tstamps = []                                                            # <-- 'datetime.timedelta' values
        npoints = int((payoff.expiry - self.asof).days * self.npoints_day + 1)  # <--  the total number of points along each path
        dt_secs = 24 * 3600 / self.npoints_day                                  # <--  the time span between each two points (in secs)
        for n in range(npoints):            
            tstamps.append(self.asof + timedelta(seconds = n * dt_secs))

        vol     = self.vol
        r       = self.r
        dt      = 1.0 / (365.0 * self.npoints_day)   # <-- the time span between each two points (in years)
        gf_step = np.exp(+r*dt)
        df_step = np.exp(-r*dt)        
        
        # calculate the initial delta hedge
        deltas = -self.hedger.calculate_hedges(self.paths[:,0,:], self.asof)
        
        # if requested to, calculate the costs of setting up the inital delta hedge
        sprd = self.trans_cost
        if self.delta_cost:
            initial_delta_cost = sprd * self.paths[:,0,:] * np.abs(deltas)
            PLpaths_long += -initial_delta_cost
            PLpaths_shrt += -initial_delta_cost
            
        # step through time and dynamically hedge
        for n in range(1, npoints):

            # recalculate the delta
            deltas_new = -self.hedger.calculate_hedges(self.paths[:,n,:], tstamps[n])
            
            # Calculate the PL on the current delta hedge, generated during the time step just completed.
            # Assume that:
            #   - at t=0 we borrow the amount needed to buy the delta hedge and purchase stock at S(t=0);
            #     if selling stock, we loan out the sale proceeds. The amount of stock transacted is delta(t=0)
            #   - at t=1 we unwind the *entire* delta position @ S(t=1) and settle the loan balance
            #   - at t=1 we borrow the amount needed to buy delta(t=1) shares @ S(t=1) (we loan the
            #     proceeds if we sell stock instead)
            #   - .... etc
            # Therefore, at each time step we completely unwind the previous delta hedge and set up a new one.
            # The net cash flow from setting up the new delta hedge is zero, therefore the entire delta PL for
            # the period comes from the net cash flows generated when unwinding the hedge.
            # While a trader will not completely unwind the current delta hedge just to set up a new one right
            # away (but instead trade the incremental delta), this assumption simplifies the calculation of PL.
            # When including the effect of transaction costs however, we will only consider the incremental delta.
            #
            # Note the we approximated the market value of the delta hedge by using the mid market spot (instead
            # of using the ask if buying stock, the bid if selling). This causes a small error when calculating
            # the cost of financing the trade, which we chose to ignore.
            #
            # The formula below represents the PL from hedging *a long* position in the option payoff.

            Df = df_step ** n
            PLstep_delta = Df * deltas * (self.paths[:,n,:] - self.paths[:,n-1,:] * gf_step)
            
            # Add the impact of transaction costs.
            # Now is important to recognize that only the incremental delta is actually traded.

            PLstep_trcosts = sprd * self.paths[:,n,:] * np.abs(deltas_new - deltas)
            
            # At expiration, get/pay payoff intrinsic value
            if n == npoints-1:
                PLstep_expiry = np.exp(-r*n*dt) * payoff.intrinsic_values(self.paths[:, -1, :])
                
            # add up all PL contributions generated during the previous step
            PLpaths_nocost += ( PLstep_delta + PLstep_expiry )
            PLpaths_long   += ( PLstep_delta + PLstep_expiry - PLstep_trcosts )
            PLpaths_shrt   += (-PLstep_delta - PLstep_expiry - PLstep_trcosts )

            deltas = deltas_new

        # Add a control variate.
        # The path PL w/o transaction costs is highly correlated with the path PL with transactions costs,
        # and we also know its average: the Black-Scholes option premium.
        if self.ctrlvariate:
            pricerBS   = PricerBlackScholes(payoff)
            BSpremiums = pricerBS.price(self.asof, self.spots, self.vol, self.r)[0]
            for i in range(len(self.spots)):
                cov_long = np.cov(PLpaths_long[:,i],  PLpaths_nocost[:,i])
                cov_shrt = np.cov(PLpaths_shrt[:,i], -PLpaths_nocost[:,i])            

                PLpaths_long[:,i] -= cov_long[0][1] / cov_long[1][1] * ( PLpaths_nocost[:,i] - BSpremiums[i])
                PLpaths_shrt[:,i] -= cov_shrt[0][1] / cov_shrt[1][1] * (-PLpaths_nocost[:,i] + BSpremiums[i])            
        
        return np.average(PLpaths_long, axis = 0), -np.average(PLpaths_shrt, axis = 0), PLpaths_nocost

# ------------------------------ #

class PricerMC_DynamicHedging_Portfolio(PricerMC):
    """
    Calculates the option price by simulating lognormal paths and by averaging the P&L 
    of dynamic hedging along each path. Transaction consts can be included as a constant
    percentage of the notional traded. 

    Can price a linear combination of Put/Call payoffs having the same expiration (and underlying).
    The payoff is priced both in isolation and relative to an existing payoff (of same expiry).
    The relative pricing is implemented as follows:
      (1) price a long position in the existing payoff
      (2) price a long position in the existing payoff + the new payoff
      (3) price a long position in the existing payoff - the new payoff
    The bid for the new payoff (considering the existing payoff) is (2) - (1)
    The ask for the new payoff (considering the existing payoff) is (1) - (3)

    Also see comments for 'PricerMC_DynamicHedging' regarding matching the Leland formula for simple payoffs.

    Assumes constant interest rates and no dividends.
    Uses calendar days (1 year = 365 days), no holidays and 24 hours trading days.
    """

    def __init__(self, spots, vol, r, npaths, npoints_day, trans_cost, delta_cost, portfolio, asof=datetime.today(),
                 antithetic=True, ctrlvariate=True):
        """
        See 'PricerMC.__init__(...)' documentation.
        PARAMS:  
        trans_cost  - hedging transaction costs as % of notional traded                         [float]
        delta_cost  - whether to factor in the cost of setting the initial delta hedge          [bool]                 
        portfolio   - existing option portfolio, relative to which we are pricing new payoffs   [PayoffConstExpiration]
        """        
        if not isinstance(portfolio, PayoffConstExpiration):
            raise TypeError("'portfolio' argument must be instance of PayoffConstExpiration")

        PricerMC.__init__(self,  spots, vol, r, npaths, npoints_day, asof, antithetic, ctrlvariate)
        self._pricer_aux = PricerMC_DynamicHedging(spots, vol, r, npaths, npoints_day,
                                                   trans_cost, delta_cost, asof, antithetic, ctrlvariate)

        self._payoff_existing = portfolio   # existing payoff
        self._payoff_pricing  = None        # payoff we want to price
        self._payoff_plus     = None        # existing payoff + payoff we want to price
        self._payoff_minus    = None        # existing payoff - payoff we want to price


    def price(self, payoff):
        """
        See 'PricerMC.price(...)' documentation.
        RETURNS: four 1-D numpy array having the same number of entries as the 'spots' argument
                 the first 2 arrays store the bid & ask for the new payoff in isolation at different spots
                 the last  2 arrays store the bid & ask for the new payoff relative to the existing payoff at different spots
        """        

        if payoff.expiry != self._payoff_existing.expiry:
            raise ValueError("the existing and the new payoffs must have the same expiry")

        # we're going to use the auxiliary pricer to price various payoffs
        self._pricer_aux.paths = self.paths

        # payoff being priced
        self._payoff_pricing = payoff        
        self._payoff_plus    = (self._payoff_existing + payoff) 
        self._payoff_minus   = (self._payoff_existing - payoff) 
        
        # price various payoffs
        bids_isolated, asks_isolated = self._pricer_aux.price(self._payoff_pricing )   # new payoff in isolation
        bids_existing, asks_existing = self._pricer_aux.price(self._payoff_existing)   # existing payoff
        bids_plus    , asks_plus     = self._pricer_aux.price(self._payoff_plus    )   # existing + new payoffs
        bids_minus   , asks_minus    = self._pricer_aux.price(self._payoff_minus   )   # existing - new payoffs  

        # calculate the bid & ask of the new paylod relative to the existing one
        bids_inportfolio  = bids_plus     - bids_existing
        asks_inportfolio  = bids_existing - bids_minus

        return bids_isolated, asks_isolated, bids_inportfolio, asks_inportfolio
            
# ========== #
