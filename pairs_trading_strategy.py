from Data.data_fetcher import DataFetcher
from Data.data_preprocessor import DataPreprocessor
from Utils.signal_generator import SignalGenerator
from Utils.spread_calculator import SpreadCalculator
from Utils.hedge_ratio_calculator import HedgeRatioCalculator
from Evaluation.evaluator import Evaluator
from Evaluation.backtester import Backtester


class PairsTradingStrategy:
    """
    Orchestrates the pairs trading strategy workflow.
    """

    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_fetchers = {}
        self.data = {}
        self.training_data = None
        self.test_data = None
        self.hedge_ratio = None
        self.training_results = None
        self.test_results = None

    def run(self):
        """
        Executes the strategy workflow.
        """
        self.fetch_data()
        self.preprocess_data()
        self.calculate_hedge_ratio()
        self.calculate_spread_and_zscore()
        self.generate_signals()
        self.backtest_strategy()
        self.evaluate_strategy()

    def fetch_data(self):
        """
        Fetches data for all symbols.
        """
        for symbol in self.symbols:
            fetcher = DataFetcher(symbol, self.start_date, self.end_date)
            self.data_fetchers[symbol] = fetcher
            self.data[symbol] = fetcher.fetch_data()

    def preprocess_data(self):
        """
        Merges and splits the data.
        """
        preprocessor = DataPreprocessor(self.data)
        merged_data = preprocessor.merge_data()
        self.training_data, self.test_data = preprocessor.split_data()

    def calculate_hedge_ratio(self):
        """
        Calculates the hedge ratio using training data.
        """
        calculator = HedgeRatioCalculator(
            training_data=self.training_data,
            dependent_var=f'Price_{self.symbols[0]}',
            independent_var=f'Price_{self.symbols[1]}',
            downsample_interval=10
        )
        self.hedge_ratio = calculator.calculate_hedge_ratio()
        print(f"Hedge Ratio: {self.hedge_ratio}")

    def calculate_spread_and_zscore(self):
        """
        Calculates spread and z-score for both training and test data.
        """
        # Training data
        spread_calculator_train = SpreadCalculator(
            data=self.training_data,
            hedge_ratio=self.hedge_ratio,
            dependent_var=f'Price_{self.symbols[0]}',
            independent_var=f'Price_{self.symbols[1]}',
            window=300
        )
        spread_calculator_train.compute_spread()
        self.training_data = spread_calculator_train.compute_zscore()

        # Test data
        spread_calculator_test = SpreadCalculator(
            data=self.test_data,
            hedge_ratio=self.hedge_ratio,
            dependent_var=f'Price_{self.symbols[0]}',
            independent_var=f'Price_{self.symbols[1]}',
            window=300
        )
        spread_calculator_test.compute_spread()
        self.test_data = spread_calculator_test.compute_zscore()

    def generate_signals(self):
        """
        Generates trading signals for both training and test data.
        """
        # Training data
        signal_generator_train = SignalGenerator(self.training_data)
        self.training_data = signal_generator_train.generate_signals()

        # Test data
        signal_generator_test = SignalGenerator(self.test_data)
        self.test_data = signal_generator_test.generate_signals()

    def backtest_strategy(self):
        """
        Backtests the strategy on both training and test data.
        """
        # Training data
        backtester_train = Backtester(
            data=self.training_data,
            dependent_var=f'Price_{self.symbols[0]}',
            independent_var=f'Price_{self.symbols[1]}'
        )
        self.training_results = backtester_train.backtest()
        backtester_train.plot_performance(title='Training Data Performance')

        # Test data
        backtester_test = Backtester(
            data=self.test_data,
            dependent_var=f'Price_{self.symbols[0]}',
            independent_var=f'Price_{self.symbols[1]}'
        )
        self.test_results = backtester_test.backtest()
        backtester_test.plot_performance(title='Test Data Performance')

    def evaluate_strategy(self):
        """
        Evaluates the strategy performance on test data.
        """
        evaluator = Evaluator(self.test_results)
        sharpe_ratio = evaluator.compute_sharpe_ratio()
        max_drawdown = evaluator.compute_max_drawdown()
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        evaluator.plot_cumulative_returns(title='Cumulative Returns on Test Data')
