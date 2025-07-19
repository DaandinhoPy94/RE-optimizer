
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go


class VastgoedPortfolioOptimizer:
    """
    Optimaliseert vastgoedportfolio's op basis van Modern Portfolio Theory
    """

    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risicovrije rente

    def calculate_returns(self, price_history):
        """
        Bereken historische rendementen
        """
        returns = price_history.pct_change().dropna()
        return returns

    def calculate_portfolio_metrics(self, weights, returns):
        """
        Bereken portfolio rendement en risico
        """
        # Check of returns al jaarlijks zijn (CBS data) of dagelijks
        # Als datum frequency jaarlijks is, geen extra annualisatie
        annualization_factor = 1  # Voor jaarlijkse data

        portfolio_return = np.sum(
            returns.mean() * weights) * annualization_factor
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(
            returns.cov() * annualization_factor, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / \
            portfolio_volatility

        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def optimize_portfolio(self, returns, target_return=None):
        """
        Vind de optimale portfolio weights
        """
        n_assets = returns.shape[1]

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

    if target_return:
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: np.sum(returns.mean() * x) - target_return  # Verwijder * 252
        })

     # Bounds (0-100% per asset)
     bounds = tuple((0, 1) for _ in range(n_assets))

      # Initial guess (equal weights)
      initial_weights = np.array([1/n_assets] * n_assets)

       # Optimize for maximum Sharpe ratio
       def negative_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights, returns)
            return -metrics['sharpe_ratio']

        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        optimal_metrics = self.calculate_portfolio_metrics(
            optimal_weights, returns)

        return optimal_weights, optimal_metrics

    def generate_efficient_frontier(self, returns, n_portfolios=50):
        """
        Genereer de efficient frontier
        """
        # Bereken min en max mogelijk rendement
        min_return = returns.mean().min()
        max_return = returns.mean().max()

        target_returns = np.linspace(min_return, max_return, n_portfolios)

        frontier_volatilities = []
        frontier_returns = []

        for target_return in target_returns:
            try:
                weights, metrics = self.optimize_portfolio(
                    returns, target_return)
                frontier_volatilities.append(metrics['volatility'])
                frontier_returns.append(metrics['return'])
            except:
                continue

        return frontier_returns, frontier_volatilities
