from datetime import datetime, timedelta
from pairs_trading_strategy import PairsTradingStrategy

# Main Execution Workflow
if __name__ == "__main__":
    # Define parameters
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    symbols = ['GLD', 'GDX']

    # Instantiate and run the strategy
    strategy = PairsTradingStrategy(symbols, start_date, end_date)
    strategy.run()