import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    """
    Evaluates the performance of the trading strategy.
    """
    def __init__(self, data, periods_per_year=252):
        self.data = data.copy()
        self.periods_per_year = periods_per_year

    def compute_sharpe_ratio(self):
        """
        Computes the Sharpe Ratio of the strategy.
        """
        strategy_returns = self.data['Strategy_Return'].dropna()
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()

        if std_return == 0:
            print("Standard deviation of strategy returns is zero. Sharpe Ratio is undefined.")
            return np.nan

        annualized_return = mean_return * self.periods_per_year
        annualized_std = std_return * np.sqrt(self.periods_per_year)
        sharpe_ratio = annualized_return / annualized_std
        return sharpe_ratio

    def compute_max_drawdown(self):
        """
        Computes the Maximum Drawdown of the strategy.
        """
        cumulative = self.data['Cumulative_Return'].dropna()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def plot_cumulative_returns(self, title='Cumulative Returns'):
        """
        Plots the cumulative returns over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Cumulative_Return'], label='Cumulative Return')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.show()
