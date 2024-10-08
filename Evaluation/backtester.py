import matplotlib.pyplot as plt

class Backtester:
    """
    Simulates trading to evaluate strategy performance.
    """

    def __init__(self, data, dependent_var, independent_var, hedge_ratio, transaction_cost=0.002):
        self.data = data.copy()
        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.hedge_ratio = hedge_ratio
        self.transaction_cost = transaction_cost

    def backtest(self):
        """
        Performs backtesting of the strategy.
        """
        # calculate returns
        self.data['Return_Dependent'] = self.data[self.dependent_var].pct_change()
        self.data['Return_Independent'] = self.data[self.independent_var].pct_change()

        # make sure no nan value
        self.data.dropna(subset=['Return_Dependent', 'Return_Independent'], inplace=True)

        # calculate strategy returns
        self.data['Strategy_Return'] = self.data['Position'].shift(1) * (
            self.data['Return_Dependent'] - self.hedge_ratio * self.data['Return_Independent']
        )

        # add cost
        self.data['Trade'] = self.data['Position'].diff().abs()
        self.data['Strategy_Return'] -= self.data['Trade'] * self.transaction_cost

        # calculate cumulative returns
        self.data['Cumulative_Return'] = (1 + self.data['Strategy_Return']).cumprod()


        return self.data

    def plot_performance(self, title='Strategy Performance'):
        """
        Plots the cumulative returns of the strategy.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Cumulative_Return'], label='Strategy Return')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.show()
