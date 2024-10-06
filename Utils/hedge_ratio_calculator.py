from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class HedgeRatioCalculator:
    """
    Calculates the hedge ratio using ordinary least squares regression.
    """
    def __init__(self, training_data, dependent_var, independent_var, downsample_interval=10):
        self.training_data = training_data
        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.downsample_interval = downsample_interval
        self.hedge_ratio = None

    def calculate_hedge_ratio(self):
        """
        Performs regression to calculate the hedge ratio.
        """
        # Downsample data to reduce computation
        if self.downsample_interval > 1:
            data_ds = self.training_data.iloc[::self.downsample_interval]
        else:
            data_ds = self.training_data

        Y = data_ds[self.dependent_var]
        X = data_ds[[self.independent_var]]
        X = add_constant(X)
        model = OLS(Y, X).fit()
        self.hedge_ratio = model.params[self.independent_var]
        return self.hedge_ratio
