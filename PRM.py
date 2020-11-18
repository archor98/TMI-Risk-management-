# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.cla import CLA
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import norm
#from pypfopt.objective_functions import negative_cvar


# Define Argumentfunction so that we can add arguments
def get_args():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--target',type=float,default=0.02,
                        help="Desired Target Return of the Portfolio\n"
                        "Default to 0.02 Risk-Free Return",required=False)

    parser.add_argument('--invest',type=float,default=0.0,
                        help="Dollar Amount of Investment\n"
                        "Default to 0.0",required=False)

    parser.add_argument("--OptMet",type=str,
                        help="Optimization Methods:\n"
                              "MPT: Optimize by Morden Portfolio Theory; \n"
                              "CVaR: Minimizing the 5 percent Worst Case Loss;\n")

    parser.add_argument("--ForMet",type=str,
                        help="Forecasting Methods:\n"
                              "Hist: Forecasting by Historical Data; \n"
                              "MontCar: Forecasting by Monte Carlo Simulation;\n")

    parser.add_argument("--short",action="store_true",default=False,required=False,
                        help="Allow Short Position in the Portfolio\n")

    return parser.parse_args()
#------------------------------------End-----------------------------------

# Define Private MPT_User Function
def MPT_sharpe(ef,invest):
    wSharpe=ef.max_sharpe()
    print('Weight of Maximized Sharpe Ratio:')
    clean_weights=ef.clean_weights()
    print(clean_weights)
    print('Expected Performance')
    P=ef.portfolio_performance(verbose=True)
    if invest !=0:
        gain = P[0]*invest
        print('Expected Gain:',gain)
        if gain>0:
            z=-P[0]/P[1]
            prob=norm.cdf(z)
            print('Probability to lose Money:',prob)
    plt.scatter(P[1],P[0],c='r',marker='^',label='Maximum Sharpe Ratio')

def MPT_return(ef,target,invest):
    wReturn=ef.efficient_return(target_return=target)
    print('Weight of Maximized Efficient Return:')
    clean_weights=ef.clean_weights()
    print(clean_weights)
    print('Expected Performance')
    P=ef.portfolio_performance(verbose=True)
    if invest !=0:
        gain = P[0]*invest
        print('Expected Gain:',gain)
        if gain>0:
            z=-P[0]/P[1]
            prob=norm.cdf(z)
            print('Probability to lose Money:',prob)
    plt.scatter(P[1],P[0],c='m',marker='^',label='Desired Return')

def MPT_vol(ef,invest):
    wVol=ef.min_volatility()
    print('Weight of Minimizing Volatility:')
    clean_weights=ef.clean_weights()
    print(clean_weights)
    print('Expected Performance')
    P=ef.portfolio_performance(verbose=True)
    if invest !=0:
        gain = P[0]*invest
        print('Expected Gain:',gain)
        if gain>0:
            z=-P[0]/P[1]
            prob=norm.cdf(z)
            print('Probability to lose Money:',prob)
    plt.scatter(P[1],P[0],c='c',marker='^',label='Minimum Volatility')

#------------------------------------End-----------------------------------

# Define Monte Carlo Simulation
def MontC(mu,e_cov,prices):               #mu and sigma are arrays!
    total_steps = 365                # One Year
    N           = 10000
    ndata       = np.shape(prices)[1]
    yearly_return  = np.zeros((ndata,N))
    mPad          = np.zeros((ndata,total_steps))

    #Build Up mPad Array
    for i in range (total_steps):
        mPad[:,i]     = mu[:]


# Create the Monte Carlo simulated runs
    for n in range(N):
        # Compute simulated path of length total_steps for correlated returns
        # Gaussian is assumed
        correlated_randomness = e_cov @ norm.rvs(size = (ndata,total_steps))
        # Adjust simulated path by number of total_steps and mean of returns

        daily_return = mPad * (1/total_steps) + correlated_randomness * np.sqrt(1/total_steps)

        yearly_return[:, n] =  daily_return.sum(axis=1)

    ''' Code for DeBug    Ignore it for normal run!!
    print(np.shape(yearly_return))
    fig = plt.figure(figsize=(10,10),facecolor='white')
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    fig.tight_layout()
    ax0.hist(yearly_return[0],bins=100)
    ax1.hist(yearly_return[1],bins=100)
    ax2.hist(yearly_return[2],bins=100)
    ax3.hist(yearly_return[3],bins=100)
    plt.show()
    quit()
    '''

    Mu  = np.average(yearly_return,axis=1)
    Cov = np.cov(yearly_return)
    return Mu,Cov


#------------------------------------End-----------------------------------


def main(args):
    target = args.target
    OptMet = args.OptMet
    ForMet = args.ForMet
    short  = args.short
    invest = args.invest

    if OptMet   == 'MPT':
        print('[Configure]:MPT is chosen for Optimization!\n')
    elif OptMet == 'CVaR':
        print('[Configure]:CVaR is chosen for Optimization\n')
    else:
        print('OptMet can only be MPT or CVaR\n')
        quit()

    if ForMet   == 'Hist':
        print('[Configure]:Historical Data is used for Prediction!\n')
    elif ForMet == 'MontCar':
        print('[Configure]:Monte Carlo Simulation is used for Prediction\n')
    else:
        print('ForMet can only be Hist or MontCar\n')
        quit()

    if short:
        print('[Configure]:Short Position is Enabled!\n')
    else:
        print('[Configure]:Short Position is Disabled\n')
# Read The Data
    prices = pd.read_csv("portfolio.csv",index_col=0)
    prices.index=pd.to_datetime(prices.index,format='%Y/%m/%d')

    N = np.shape(prices)[0]   #number of trading days

# Compute the annualized average (mean) historical return and the Covariance
    if ForMet=='Hist':   #Computation with Historical Data
        mean_returns = mean_historical_return(prices, frequency = N)
        efficient_cov = CovarianceShrinkage(prices).ledoit_wolf()
    elif ForMet=='MontCar':
        mu = mean_historical_return(prices, frequency = N)
        e_cov = CovarianceShrinkage(prices).ledoit_wolf()
        keys  = mu.keys()
        columns=e_cov.keys()
        mean_returns,efficient_cov = MontC(mu,e_cov,prices)
        # A Trick to put mean_returns array in dataframe quickly
        mu[:]=mean_returns[:]
        mean_returns=mu



# Calculate the weights
    if OptMet=='MPT':

        if short:
            ef1=EfficientFrontier(mean_returns,efficient_cov,weight_bounds=(-1,1))
            ef2=EfficientFrontier(mean_returns,efficient_cov,weight_bounds=(-1,1))
            ef3=EfficientFrontier(mean_returns,efficient_cov,weight_bounds=(-1,1))
        else:
            ef1=EfficientFrontier(mean_returns,efficient_cov)
            ef2=EfficientFrontier(mean_returns,efficient_cov)
            ef3=EfficientFrontier(mean_returns,efficient_cov)


        print('==============MPT Results==============\n')
    # Maximizing the Sharpe's Ratio

        MPT_sharpe(ef1,invest)

        print('===================================\n')
    # Efficient Return Method
        MPT_return(ef2,target,invest)

        print('===================================\n')
    # Minimizing Volatility

        MPT_vol(ef3,invest)
        print('===================================\n')

    # Construct the Efficient Frontier Curve
        cla = CLA(mean_returns, efficient_cov)
        (ret, vol, weights) = cla.efficient_frontier()
    # Plot the Efficient EfficientFrontier Curve
        plt.scatter(vol, ret, s = 4, c = 'g', marker = '.', label = 'Efficient Frontier')
        plt.title('EfficientFrontier')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.legend()
        plt.show()


    elif OptMet=='CVaR':
        print('CVaR Method is not Available yet! Please wait for the updates')
        quit()
        '''
        ef = pypfopt.efficient_frontier.EfficientFrontier(None, efficient_cov)
        min_cvar_weights = ef.custom_objective(negative_cvar, mean_returns)
        print(min_cvar_weights)
        '''

if __name__ == '__main__':
   # If this file is called from cmd line
   args = get_args()
   main(args)
