import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluates the performance of the trading strategy.
    """
    def __init__(self, data):
        self.data = data.copy()

    def compute_sharpe_ratio(self):
        """
        Computes the Sharpe Ratio of the strategy.
        """
        daily_return = self.data['Strategy_Return'].mean() * (390)
        daily_std = self.data['Strategy_Return'].std() * np.sqrt(390)
        sharpe_ratio = daily_return / daily_std
        return sharpe_ratio

    def compute_max_drawdown(self):
        """
        Computes the Maximum Drawdown of the strategy.
        """
        cumulative = self.data['Cumulative_Return']
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
