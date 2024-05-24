import pandas as pd

__all__ = ['DataBatcher']

class DataBatcher:
    """
    A class to manage batching operations for datasets with various sorting strategies.

    Attributes:
        data (pd.DataFrame): The dataframe to be processed.
        predict_col (str): Column name for prediction values.
        target_long_col (str): Column name for long target values.
        target_short_col (str): Column name for short target values.
        date_col (str, optional): Column name for date values.
    """

    def __init__(self, data, predict_col, target_long_col, target_short_col, date_col=None):
        """
        Initializes the DataBatcher with a dataframe and column names.

        Parameters:
            data (pd.DataFrame): DataFrame containing the data.
            predict_col (str): Name of the column containing prediction values.
            target_long_col (str): Name of the column containing long target values.
            target_short_col (str): Name of the column containing short target values.
            date_col (str, optional): Name of the column containing date values, if any.
        """
        self.data = data
        self.predict_col = predict_col
        self.target_long_col = target_long_col
        self.target_short_col = target_short_col
        self.date_col = date_col

    def shuffle(self, random_state=0):
        """Randomly shuffles the DataFrame."""
        self.data = self.data.sample(frac=1, random_state=random_state)

    def sort_by_date(self):
        """Sorts the DataFrame by date, if date column is not entirely NaN."""
        if self.date_col and not self.data[self.date_col].isna().all():
            self.data = self.data.sort_values(self.date_col)
        else:
            print("All values in column 'date' are None! Sorting by 'date' was not performed.")

    def sort_by_square_error(self):
        """Sorts the DataFrame by the square error calculated from predictions and target_long."""
        square_error = (self.data[self.predict_col] - self.data[self.target_long_col]) ** 2
        self.data = self.data.iloc[square_error.argsort()]

    def get_batches(self, size_batch=100, method='random', random_state=0):
        """
        Divides the DataFrame into batches according to the specified method.

        Parameters:
            size_batch (int): The number of samples per batch.
            method (str): The method used to prepare data for batching ('random', 'sorted_date', or 'sorted_square_error').
            random_state (int): Seed for the random number generator (if method is 'random').

        Returns:
            pd.DataFrame: The modified dataframe with an additional 'batch' column indicating batch number.
        """
        if method == 'random':
            self.shuffle(random_state)
        elif method == 'sorted_date':
            self.sort_by_date()
        elif method == 'sorted_square_error':
            self.sort_by_square_error()
        else:
            print('No valid sorting method provided! Defaulting to random.')
            self.shuffle(random_state)

        n = self.data.shape[0]
        n_batches = max(1, int(n / size_batch))  # Ensure at least one batch
        self.data['batch'] = pd.qcut(range(n), n_batches, labels=False)

        return self.data