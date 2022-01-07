# 3220 Final Script
# Jack Young, jry28
# Advay Koranne, ak845
# Blaze Ezrakowski, be96

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple
from IPython.display import display
import yfinance as yf
import scipy.optimize as optimize
from sympy import *


# Generates a list of random weights in range [0, 1] that sum to 1
# Length of returned array is based on the number of stocks in the portfolio
# i.e if you have 5 stocks it will generate 5 random numbers that sum to 1.
# ptfl : portfolio of assets
def random_weights(ptfl):
    rand_weights = np.random.rand(len(ptfl))  # W random vector in R^N
    rand_weights = rand_weights / rand_weights.sum()  # W.normalized()
    return rand_weights


# Returns the expected portfolio return based on the weights and the mean daily % returns of the stocks.
# exp_returns : expected returns of assets vector
# w : asset weights
def portfolio_expected_return(exp_returns, w):
    return np.dot(exp_returns, w)


# returns the portfolio standard deviation based on the weights and the co-variance of the stocks.
# cov : asset covariance matrix
# w : asset weights
def portfolio_return_variance(cov, w):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))

# returns correlation coefficients of assets at index1, 2, in portfolio
# cov : asset covariance matrix
def asset_correlation_coefficients(cov):
    l = len(cov)
    coeffs = np.zeros((len(cov), len(cov)))
    for i in range(l):
        for j in range(l):
            coeffs[i, j] = cov[i, j] / (sqrt(cov[i, i]) * sqrt(cov[j, j]))
    return coeffs

# Expression at which minima define efficient frontier
# w : weights vector
# cov : asset covariance matrix
# q : risk tolerance scalar [0, inf)
# R : expected returns vector
def expression_to_minimize(w, cov, q, R):


    return np.dot(w.T, np.dot(cov, w)) - q * np.dot(R.T, w)


# Gradient of minimization expression at weights
# Forward finite difference approximator
# w : weights vector
# cov : asset covariance matrix
# q : risk tolerance scalar [0, inf)
# R : expected returns vector
def gradient_of_expression_to_minimize(w, cov, q, R):
    Fw = expression_to_minimize(w, cov, q, R)
    num_assets = len(w)
    divs = np.zeros(num_assets)
    hVal = 0.000001
    eps = np.zeros(num_assets)
    eps[0] = hVal
    for i in range(num_assets):
        if i >= 1:
            eps[i] = hVal
            eps[i-1] = 0
        divs[i] = (Fw - expression_to_minimize(w + eps, cov, q, R)) / hVal
    return divs

# Returns step size for gradient descent
# w : weights vector
# Pk : step direction
# c : gradient descent sufficiency coefficient
# cov : asset covariance matrix
# q : risk tolerance scalar [0.01, 0.025...)
# R : expected returns vector
def find_step_size(weights, Pk, c, cov, q, R):
    a = 1
    fW = -Pk
    for i in range(100):
        new_weights = weights + a * Pk
        for i in range(len(weights)):
            if new_weights[i] < 0:
                new_weights[i] = 0
            if new_weights[i] == 0 and Pk[i] < 0:
                Pk[i] = 0
        new_weights = weights + a * Pk
        # Somehow hold constraint that weights are non-negative
        new_weights = new_weights / new_weights.sum()
        if expression_to_minimize(new_weights, cov, q, R) < expression_to_minimize(weights, cov, q, R):# + c*a*fW*Pk:
            return a
        a = a / 2
    return a

# Returns negative gradient of expression to minimize
# w : weights vector
# cov : asset covariance matrix
# q : risk tolerance scalar [0, inf)
# R : expected returns vector
def find_step_direction(w, cov, q, R):
    fW = gradient_of_expression_to_minimize(w, cov, q, R)
    return -1 * fW


def backtest_model(portfolio, weights, start_date, end_date):
  modl_df = yf.download(portfolio, start=start_date, end= end_date)
  modl_df = np.log(1+ modl_df['Adj Close'].pct_change().dropna())
  test_df = yf.download(portfolio, start=end_date)
  test_df = np.log(1+ test_df['Adj Close'].pct_change().dropna())

  return np.dot(test_df.mean(), weights)*100*52


if __name__ == '__main__':
    p1 = false
    # true for big portfolio
    if p1:
        portfolio = ['BTC-USD', 'ETH-USD', 'XLM-USD', 'ADA-USD', 'DOGE-USD',
                     'SPY', 'GLD', 'GSG', 'NLY', 'MSFT', 'TSLA', 'AMZN',
                     'BABA', 'BA', 'AMD', 'V', 'F', 'JPM', 'VZ', 'SCHH',
                     'NFLX', 'WM', 'BYND', 'SLV', 'AAPL', 'BLU', 'PFE']

    else:  portfolio = ['SPY', 'AMD', 'V', 'F', 'JPM']

    num_assets = len(portfolio)

    data = yf.Ticker(portfolio[0]).history(period='2y', interval='1wk')
    num_weeks = len(data) + 5

    # Isolate all weekly returns of past 2y
    portfolio_returns = np.zeros((num_assets, num_weeks))
    for i in range(num_assets):
        data = yf.Ticker(portfolio[i]).history( interval='1wk', start='2020-01-01' )
        data = np.log(1 + data['Close'].pct_change()).dropna()  # Normalized, interval % return over ~2 years
        portfolio_returns[i, 0:len(data)] = data

    # Avg weekly returns over past 2y
    avg_portfolio_returns = np.zeros(num_assets)
    for i in range(num_assets):
        sum = 0
        num = 0
        for j in range(num_weeks):
            wk_ret = portfolio_returns[i, j]
            if wk_ret != 0:
                sum += wk_ret
                num += 1
        avg_portfolio_returns[i] = sum / num

    # Sample weekly covariances
    asset_covariances = np.zeros((num_assets, num_assets))

    for i in range(num_assets):
        for j in range(num_assets):
            sum = 0
            num = 0
            for k in range(num_weeks):
                xi = portfolio_returns[i, k]
                yi = portfolio_returns[j, k]
                # Only calculate covariance at this week if we have data for both asset i and j
                if (xi != 0) and (yi != 0):
                    sum += ( avg_portfolio_returns[i] - xi ) * ( avg_portfolio_returns[j] - yi )
                    num += 1
            asset_covariances[i, j] = sum / (num-1) # n-1 for sample covariance

    asset_correlations = asset_correlation_coefficients(asset_covariances)

    # Graph some randomly weighted portfolios
    if p1:
        trials = 100000
    else:
        trials = 10000
    returns, variances, w = [], [], []
    for i in range(trials):
        weights = random_weights(portfolio)
        returns.append(portfolio_expected_return(avg_portfolio_returns, weights))
        variances.append(portfolio_return_variance(asset_covariances, weights))
        w.append(weights)

    print(asset_correlations)

    plt.figure(figsize=(10, 7))
    plt.scatter(variances, returns, c=returns, cmap='RdBu', marker='o', s=10, edgecolors="black", linewidth=0.3)
    plt.colorbar()
    plt.title('A Set of Randomly Weighted Diverse Portfolios')
    plt.xlabel('Portfolio Variance (risk)')
    plt.ylabel('Portfolio Expected Weekly Returns')

    num_steps = 400
    qstep = 1 / num_steps
    Qrange = [0.0,  num_steps*qstep]


    BFGD = true
    # true = brute force, false = gradient descent

    returns, variances, ws = [], [], []
    while Qrange[0] < Qrange[1]:
        Qrange[0] += qstep
        if BFGD == false:
            # Gradient descent
            if Qrange[0] - qstep == 0:
                print('Doing Gradient Descent:\n')
            c = 0.001
            weights = random_weights(portfolio)
            # Components of Pk must sum to 0 to maintain the constraint that the weights sum to 1
            for j in range(num_steps): # Hard limit on num steps, gradient never 0 at efficient frontier
                Pk = find_step_direction(weights, asset_covariances, Qrange[0], avg_portfolio_returns)
                ak = find_step_size(weights, Pk, c, asset_covariances, Qrange[0], avg_portfolio_returns)
                weights = weights + ak * Pk

            returns.append(portfolio_expected_return(avg_portfolio_returns, weights))
            variances.append(portfolio_return_variance(asset_covariances, weights))
            ws.append(weights)
        else:
            # Brute force minimization
            if Qrange[0] - qstep == 0:
                print('Doing Brute Force Minimization:\n')
            current_best_weights = []
            max_iter = 10000 # Keep at order e4 for reasonable runtime
            min_val = 10000
            for i in range(1, max_iter):
                w = random_weights(portfolio)
                min = expression_to_minimize(w, asset_covariances, Qrange[0], avg_portfolio_returns)
                if min < min_val:
                    current_best_weights = w
                    min_val = min
            returns.append(portfolio_expected_return(avg_portfolio_returns, current_best_weights))
            variances.append(portfolio_return_variance(asset_covariances, current_best_weights))
            ws.append(current_best_weights)
    
    plt.scatter(variances, returns, c=returns, marker='o', s=50, edgecolors="black", linewidth=0.3)

    plt.show()
    print(portfolio)

    q_to_test = 0.8
    index = (int)(q_to_test / num_steps / qstep)

    print(backtest_model(portfolio, ws[index], "2020-01-01", "2020-02-01"))
    print(backtest_model(['SPY'], [1], "2020-01-01", "2020-02-01"))