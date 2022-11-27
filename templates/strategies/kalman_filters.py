# Before the beginning, you have to install library 'pykalman'
from pykalman import *
import pandas as pd
import numpy as np

class SimpleKalmanFilter(BaseStrategy):
    def train_model(self, train_data):
        self.lookback_period = 250

        self.lookback_freq = 100

        self.lookback_data = train_data.iloc[-self.lookback_period:]

        self.trade_cnt = 0
        
    def trade(self, daily_data : dict) -> dict:

        # Place a recent date into self.lookback_data
        self.lookback_data = \
                    self.lookback_data.append(daily_data, ignore_index=True)

        # Fill infinite values with nan values to interpolate
        self.lookback_data = self.lookback_data.replace([np.inf, -np.inf], np.nan)

        # Check for any nan values resulted from missing data and interpolate them
        if self.lookback_data.isnull().values.any():
            self.lookback_data = self.lookback_data \
                                        .interpolate(method='polynomial', order=1)

        # Delete the latest value from lookback data
        self.lookback_data = self.lookback_data.iloc[1:]

        if self.trade_cnt % self.lookback_freq == 0:
            # Get copy of data from lookback for easier work
            data = self.lookback_data[['er_cdx_ig_long', 'spx']].copy()
            # Get a ratio of two assets for their portfolio weights
            ratio = data['spx']/data['er_cdx_ig_long']

            # We will use UnscentedKalmanFilter() class from pykalman library
            kalman = UnscentedKalmanFilter()
            # Get means and covariances of assets
            means, covariances = kalman.filter(ratio.values)
            means, covariances = kalman.smooth(ratio.values)
            means, covariances = means.squeeze(), covariances.squeeze()
            # Get Pearson's correlation coeficient between two assets
            data['ratio'] = data['spx']/data['er_cdx_ig_long']
            data['mean'] = means
            data['deviations_spx'] = (data['spx'] - data['mean'])**2
            data['deviations_er_cdx_ig_long'] = (data['spx'] - data['mean'])**2
            data['variances_spx'] = sum(data['deviations_spx'])/len(data['deviations_spx'])
            data['variances_er_cdx_ig_long'] = sum(data['deviations_er_cdx_ig_long'])/len(data['deviations_er_cdx_ig_long'])
            data['covariance'] = covariances
            data['corr'] = data['covariance']/np.sqrt(data['variances_spx']*data['variances_er_cdx_ig_long'])

            """ If coeficient of correlation is in [-0.5; 0.5], we give the assets the 'minimal' weights.
            If it's not, we give assets weights, which depend on the ratio of assets"""
            for i in range (data.shape[0]):
                if data['corr'].iloc[i] < 0.5 and data['corr'].iloc[i] > -0.5:
                    self.portfolio = {'spx': .1, 'er_cdx_ig_long': -.1}
                elif data['corr'].iloc[i] >= 0.5:
                    self.portfolio = {'spx': .3, 'er_cdx_ig_long': -10/(round(data['ratio'].iloc[i],1))}
                elif data['corr'].iloc[i] <= -0.5:
                    self.portfolio = {'spx': -.3, 'er_cdx_ig_long': 10/(round(data['ratio'].iloc[i],1))}
                else:
                    self.portfolio = {'er_cdx_ig_long' : -.1, 'spx' : .1}

        self.trade_cnt += 1
        return self.portfolio
