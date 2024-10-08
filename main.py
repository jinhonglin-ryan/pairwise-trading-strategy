from datetime import datetime, timedelta
from pairs_trading_strategy import PairsTradingStrategy
import pytz

# Main Execution Workflow
if __name__ == "__main__":
    # Define parameters
    # start_date = datetime.now() - timedelta(days=90)
    # end_date = datetime.now()
    symbols = ['GLD', 'GDX']

    timezone = pytz.timezone('America/New_York')
    end_date = timezone.localize(datetime.now())  # End date as the current date and time

    # Calculate the start date by subtracting 5 years from the end date
    start_date = end_date - timedelta(days=365 * 5)

    # Instantiate and run the strategy
    strategy = PairsTradingStrategy(symbols, start_date, end_date)
    strategy.run()
