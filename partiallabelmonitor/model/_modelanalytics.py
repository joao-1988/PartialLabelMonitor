import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt

class RatioModel():

  def __init__(self):
    self.coef_ = None

  def predict(self,x):
    return x.iloc[:,0]*self.coef_

  def fit(self,x,y):
    import numpy as np
    self.coef_ = np.mean(y/x.iloc[:,0])
    return self

class ModelAnalytics:
    def __init__(self, data, predict_col, target_long_col, target_short_col, metric, by='batch'):
        """
        Initialize the ModelAnalytics.

        Parameters:
            data (pd.DataFrame): DataFrame containing the data.
            predict_col (str): Name of the column containing prediction values.
            target_long_col (str): Column name for long-term target values.
            target_short_col (str): Column name for short-term target values.
            metric (callable): A callable that calculates a metric with format parameters y and pred.
            by (str): Column name to group by.
        """
        #self.data = data
        self.predict_col = predict_col
        self.target_long_col = target_long_col
        self.target_short_col = target_short_col
        self.data_metric = self.__metric_estimation(data, metric, by)


    def __metric_estimation(self, data, metric, by):
        """
        Calculate specified metrics for each target grouped by a column.

        Parameters:
            metric (callable): A callable that calculates a metric with format parameters y and pred.
            by (str): Column name to group by.

        Returns:
            pd.DataFrame: DataFrame containing calculated metrics for each group.
        """
        if by:
            grouped = data.groupby(by)
        else:
            grouped = data

        results = []
        for target in [self.target_long_col, self.target_short_col]:
            metric_result = grouped.apply(lambda df: metric(df[target], df[self.predict_col]))
            results.append(metric_result)

        df_metric = pd.concat(results, axis=1)
        df_metric.columns = ['metric_long', 'metric_short']
        return df_metric
   
    def split(self, test_size=0.2, random_state=1):
        """
        Splits the data into training and validation sets without stratification.

        Parameters:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed used by the random number generator for reproducibility.

        Returns:
            df_train (DataFrame): Training subset of the original data.
            df_val (DataFrame): Validation subset of the original data.
        """
        np.random.seed(random_state)
        # Shuffle the data
        indices = self.data_metric.index
        shuffled_indices = np.random.permutation(indices)
        test_set_size = int(len(indices) * test_size)
        self.train_indices = shuffled_indices[test_set_size:]
        self.test_indices = shuffled_indices[:test_set_size]

        return

    def train(self, model):
        """
        Train the model using the provided data.

        Parameters:
            model (model): Machine learning model to be trained.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.

        Returns:
            model: Trained model.
            DataFrame: Training subset of the original data.
            DataFrame: Validation subset of the original data.
        """
        df_train = self.data_metric.iloc[self.train_indices]
        model.fit(df_train[['metric_short']], df_train['metric_long'])
        return model

    def __mae(self, y, pred):
        """Calculate Mean Absolute Error between observations and predictions."""
        return np.mean(np.abs(y - pred))

    def __ratio_mae(self, y, pred, naive):
        """Calculate the ratio of MAE to a naive prediction's MAE."""
        return 1 - (self.__mae(y, pred) / self.__mae(y, naive))

    def __uncertainty(self, y, pred, conf=0.9):
        """Calculate the conformal prediction interval based on confidence level."""
        abs_err = np.abs(y - pred)
        return np.quantile(abs_err, conf)

    def evaluate(self, model, conf=0.9):
        """
        Evaluate the model using the validation data.

        Parameters:
            model (model): Trained machine learning model.
            df_val (DataFrame): Validation data used for evaluation.
            conf (float): Confidence level for conformal prediction.

        Returns:
            float: Conformal prediction value.
            float: Pearson correlation coefficient.
            float: Mean Absolute Error.
            float: Ratio MAE.
        """
        df_train = self.data_metric.iloc[self.train_indices]
        df_test = self.data_metric.iloc[self.test_indices]

        pred = model.predict(df_test[['metric_short']])
        y = df_test['metric_long']
        naive = df_train['metric_long'].mean()

        uncertainty = self.__uncertainty(y, pred, conf)
        pearson_corr = stats.pearsonr(y, pred)[0]
        mae = self.__mae(y, pred)
        ratio_mae = self.__ratio_mae(y, pred, naive)

        dict = {'Uncertainty': uncertainty
                ,'Pearson Correlation': pearson_corr
                ,'MAE': mae
                ,'Ratio MAE': ratio_mae}

        return dict
    
    def apply_model(self, model, uncertainty):
        """
        Applies the provided model to the data and calculates Conformal Predictions.

        Parameters:
            model (model): The model to be applied.
            uncertainty (float): The uncertainty factor used to calculate the Conformal Predictions.

        Adds the following columns to the DataFrame:
        - 'metric_predict': The predicted values from the model.
        - 'cp_inf': Lower bound of the Conformal Predictions.
        - 'cp_sup': Upper bound of the Conformal Predictions.
        """
        self.data_metric['metric_predict'] = model.predict(self.data_metric[['metric_short']])
        self.data_metric['cp_inf'] = self.data_metric['metric_predict'] - uncertainty
        self.data_metric['cp_sup'] = self.data_metric['metric_predict'] + uncertainty

    def plot_estimation(self, figsize=(10, 4), ylim=[0.7,1]):
        """
        Generates a plot for metrics with Conformal Predictions.

        Parameters:
            df (pd.DataFrame): DataFrame containing the metrics and Conformal Predictions data.

        The DataFrame must contain the following columns:
        - 'metric_long'
        - 'metric_predict'
        - 'cp_inf'
        - 'cp_sup'
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.data_metric.index, self.data_metric['metric_long'], label='Metric Long')
        ax.plot(self.data_metric.index, self.data_metric['metric_predict'], label='Metric Predict')
        ax.fill_between(self.data_metric.index, self.data_metric['cp_inf']
                        , self.data_metric['cp_sup'], color='gray'
                        , alpha=0.3, label='Conformal Interval')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylim(ylim)
        ax.legend()

        plt.title("Metric Long vs. Metric Predict with Conformal Predictions")
        #plt.xlabel("Index")
        plt.ylabel("Metric Values")
        plt.grid(True, linestyle='--', alpha=0.5)  # Make grid less prominent
        plt.show()       
