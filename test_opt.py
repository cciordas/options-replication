#! /usr/bin/env python

from Payoffs      import *
from PricerBS     import PricerBlackScholes
from PricerLeland import PricerLeland
from PricerMC     import PricerMC_Simple, PricerMC_PathTC, PricerMC_PathTC_Portfolio
from datetime     import datetime, timedelta
import pylab as pl
import numpy as np

# ----- GENERAL ----- #

TDAY  = datetime(2016, 9, 17, 0, 0, 0)
EXP   = TDAY + timedelta(days=30, seconds=0)
K     = 100
SZ    = 1
VOL   = 0.2
R     = 0.0
SPOTS = np.arange(80, 120, 1, dtype = float)

# ----- Leland ----- #

NUM_REHEDGE_DAILY = 1
TRANS_COSTS = 0.0004 / np.sqrt(NUM_REHEDGE_DAILY)
DELTA_COSTS = False

# ----- MonteCarlo ----- #

NPATHS = 1000

payoff_zero  = PayoffSimple(K  , EXP, "C",   0)
payoff_call1 = PayoffSimple(K  , EXP, "C",  SZ)
payoff_call2 = PayoffSimple(K-5, EXP, "C",  SZ)
payoff_call3 = PayoffSimple(K+5, EXP, "C", -SZ)

pricer_mcp = PricerMC_PathTC_Portfolio(NPATHS, NUM_REHEDGE_DAILY, TRANS_COSTS, DELTA_COSTS)
pricer_mcp.load_existing_payoff(PayoffConstExpiration([ payoff_call2,  payoff_call3]))
pricer_mcp.load_pricing_payoff (PayoffConstExpiration([ payoff_call1]))
bids_isolated, asks_isolated, bids_inportfolio, asks_inportfolio = pricer_mcp.price(TDAY, SPOTS, VOL, R)

sprds_isolated    = asks_isolated    - bids_isolated
sprds_inportfolio = asks_inportfolio - bids_inportfolio
    
pl.plot(SPOTS, sprds_isolated   , "ro")
pl.plot(SPOTS, sprds_inportfolio, "bo")
pl.show()
