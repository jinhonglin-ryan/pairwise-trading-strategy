class SpreadCalculator:
    """
    Calculates the spread and z-score based on the hedge ratio.
    """
    def __init__(self, data, hedge_ratio, dependent_var, independent_var, window=300):
        self.data = data.copy()
        self.hedge_ratio = hedge_ratio
        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.window = window

    def compute_spread(self):
        """
        Computes the spread using the hedge ratio.
        """
        self.data['Spread'] = self.data[self.dependent_var] - self.hedge_ratio * self.data[self.independent_var]

    def compute_zscore(self):
        """
        Computes the z-score of the spread.
        """
        self.data['Mean'] = self.data['Spread'].rolling(window=self.window).mean()
        self.data['Std'] = self.data['Spread'].rolling(window=self.window).std()
        self.data.dropna(inplace=True)
        self.data['ZScore'] = (self.data['Spread'] - self.data['Mean']) / self.data['Std']
        return self.data
