from ib_insync import *
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


class DataFetcher:
    """
    Fetches historical minute-level price data for a given symbol between start_date and end_date using IBroker API.
    """

    def __init__(self, symbol, start_date, end_date, ib_port=7497, client_id=1):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.ib_port = ib_port
        self.client_id = client_id
        self.data = None
        self.ib = None

    def connect(self):
        """
        Establishes connection to the IBroker API.
        """
        self.ib = IB()
        self.ib.connect('127.0.0.1', self.ib_port, clientId=self.client_id)

    def disconnect(self):
        """
        Disconnects from the IBroker API.
        """
        if self.ib:
            self.ib.disconnect()

    def fetch_data(self):
        """
        Fetches the previous 90 days of minute-level historical close price data using the IBroker API.
        The data is fetched in 30-day chunks per iteration to comply with API limitations.

        Returns:
            pd.DataFrame or None: A DataFrame containing the close prices indexed by date,
                                   or None if an error occurs.
        """
        try:
            self.connect()

            # Define the contract
            contract = Stock(self.symbol, 'SMART', 'USD')

            data_frames = []
            current_end_date = self.end_date

            while current_end_date > self.start_date:
                # Calculate the duration to fetch (30 days or remaining days)
                remaining_days = (current_end_date - self.start_date).days
                fetch_days = 30 if remaining_days >= 30 else remaining_days
                duration_str = f"{fetch_days} D"  # e.g., '30 D'

                # Format the endDateTime as required by IBroker API (YYYYMMDD HH:MM:SS)
                end_datetime_str = current_end_date.strftime("%Y%m%d %H:%M:%S")

                print(f"Fetching data for {self.symbol} from {end_datetime_str} back {duration_str}.")

                # Request historical data
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_datetime_str,
                    durationStr=duration_str,
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False
                )

                if not bars:
                    print(f"No bars returned for {self.symbol} ending at {current_end_date}.")
                    break

                # Convert bars to DataFrame
                df = util.df(bars)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df[['close']]
                df.rename(columns={'close': 'Price'}, inplace=True)

                # Append to list
                data_frames.append(df)

                # Update current_end_date for next iteration
                earliest_date = df.index.min()
                current_end_date = earliest_date - timedelta(minutes=1)

                print(f"Fetched {len(df)} records. Next end_date: {current_end_date}")

                # Sleep to comply with rate limits
                time.sleep(2)

            # Concatenate all DataFrames
            if data_frames:
                self.data = pd.concat(data_frames)
                self.data.sort_index(inplace=True)
                # Filter data within the start_date and end_date
                self.data = self.data[(self.data.index >= self.start_date) & (self.data.index <= self.end_date)]
                print(f"Total records fetched: {len(self.data)}")
            else:
                print(f"No data fetched for symbol {self.symbol}")

            return self.data

        except Exception as e:
            print(f"Error fetching data for symbol {self.symbol}: {e}")
            return None
        finally:
            self.disconnect()
