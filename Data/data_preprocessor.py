import pandas as pd


class DataPreprocessor:
    """
    Merges and preprocesses the data from multiple assets.
    """

    def __init__(self, data_dict):
        """
        data_dict: Dictionary with symbol as key and DataFrame as value.
        """
        self.data_dict = data_dict
        self.merged_data = None

    def merge_data(self):
        """
        Merges the data on the DateTime index.
        """
        data_frames = [df.rename(columns={'Price': f'Price_{symbol}'}) for symbol, df in self.data_dict.items()]
        self.merged_data = pd.concat(data_frames, axis=1).dropna().sort_index()
        return self.merged_data

    def split_data(self, ratio=0.66):
        """
        Splits the data into training and test sets based on the specified ratio.
        """
        split_point = int(len(self.merged_data) * ratio)
        training_data = self.merged_data.iloc[:split_point]
        test_data = self.merged_data.iloc[split_point:]
        return training_data, test_data
