import pandas as pd
import numpy as np


class SignalGenerator:
    """
    Generates trading signals based on z-score thresholds with partial positions.
    """

    def __init__(self, data, entry_threshold=2.5, exit_threshold=0.5, max_position=1.0):
        """
        Initializes the SignalGenerator.

        :param data: DataFrame containing the 'ZScore' column.
        :param entry_threshold: Z-score threshold to enter a position.
        :param exit_threshold: Z-score threshold to exit a position.
        :param max_position: Maximum position size (e.g., 1.0 for full position).
        """
        self.data = data.copy()
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position = max_position

    def calculate_position_size(self, zscore):
        """
        Calculates the position size based on the z-score.

        :param zscore: The z-score value.
        :return: Position size (positive for long, negative for short, zero for no position).
        """
        if zscore > self.entry_threshold:
            # Short spread: position size increases with z-score
            return -min((zscore - self.entry_threshold) / self.entry_threshold, self.max_position)
        elif zscore < -self.entry_threshold:
            # Long spread: position size increases with the absolute z-score
            return min((-zscore - self.entry_threshold) / self.entry_threshold, self.max_position)
        elif abs(zscore) < self.exit_threshold:
            # Exit positions
            return 0
        else:
            # Maintain existing position (no change)
            return np.nan  # Will be forward-filled later

    # def calculate_position_size(self, zscore):
    #     """
    #     Simplified position sizing: Full positions only.
    #     """
    #     if zscore > self.entry_threshold:
    #         return -self.max_position  # Short spread
    #     elif zscore < -self.entry_threshold:
    #         return self.max_position  # Long spread
    #     elif abs(zscore) < self.exit_threshold:
    #         return 0  # Exit positions
    #     else:
    #         return np.nan  # Maintain existing position

    def generate_signals(self):
        """
        Generates signals and positions with partial positions.

        :return: DataFrame with 'Signal' and 'Position' columns.
        """
        # Apply the position size calculation to each z-score
        self.data['Signal'] = self.data['ZScore'].apply(self.calculate_position_size)

        # Forward-fill the positions where 'Signal' is NaN to maintain existing positions
        self.data['Position'] = self.data['Signal'].fillna(method='ffill').fillna(0)

        # Ensure that positions do not exceed the maximum allowed
        self.data['Position'] = self.data['Position'].clip(-self.max_position, self.max_position)

        return self.data
