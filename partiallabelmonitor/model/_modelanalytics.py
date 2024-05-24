import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

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
    def __init__(self, data, predict_col, target_long_col, target_short_col):
        """
        Initialize the ModelAnalytics with a DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing the data.
            predict_col (str): Name of the column containing prediction values.
            target_long_col (str): Column name for long-term target values.
            target_short_col (str): Column name for short-term target values.
        """
        self.data = data
        self.predict_col = predict_col
        self.target_long_col = target_long_col
        self.target_short_col = target_short_col

    def mae(self, y, pred):
        """Calculate Mean Absolute Error between observations and predictions."""
        return np.mean(np.abs(y - pred))

    def ratio_mae(self, y, pred, naive):
        """Calculate the ratio of MAE to a naive prediction's MAE."""
        return 1 - (self.mae(y, pred) / self.mae(y, naive))

    def metric_estimation(self, metric, by='batch'):
        """
        Calculate specified metrics for each target grouped by a column.

        Parameters:
            metric (callable): A callable that calculates a metric with format parameters y and pred.
            by (str): Column name to group by.

        Returns:
            pd.DataFrame: DataFrame containing calculated metrics for each group.
        """
        if by:
            grouped = self.data.groupby(by)
        else:
            grouped = self.data

        results = []
        for target in [self.target_long_col, self.target_short_col]:
            metric_result = grouped.apply(lambda df: metric(df[target], df[self.predict_col]))
            results.append(metric_result)

        df_metric = pd.concat(results, axis=1)
        df_metric.columns = ['metric_long', 'metric_short']
        self.data_metric = df_metric
        return

    def get_conformal_prediction(self, y, pred, conf=0.9):
        """Calculate the conformal prediction interval based on confidence level."""
        abs_err = np.abs(y - pred)
        return np.quantile(abs_err, conf)

    def train_and_evaluate(self, model=RatioModel(), test_size=0.2, random_state=1, conf=0.9):
        """
        Train a model and evaluate its performance.

        Parameters:
            model (model): Machine learning model to be trained.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
            conf (float): Confidence level for conformal prediction.

        Returns:
            tuple: Trained model, conformal prediction value, validation DataFrame
        """
        df_train, df_val = train_test_split(self.data_metric, test_size=test_size, random_state=random_state)
        model.fit(df_train[['metric_short']], df_train['metric_long'])

        pred_val = model.predict(df_val[['metric_short']])
        df_val['metric_predict'] = pred_val

        conf_pred = self.get_conformal_prediction(df_val['metric_long'], pred_val, conf)

        print('Conformal Prediction:', conf_pred)
        print('Pearson Correlation:', stats.pearsonr(pred_val, df_val['metric_long'])[0])
        print('MAE:', self.mae(pred_val, df_val['metric_long']))
        print('Ratio MAE:', self.ratio_mae(df_val['metric_long'], pred_val, naive=df_train['metric_long'].mean()))

        self.model_metric = model

        return model, conf_pred, df_val

