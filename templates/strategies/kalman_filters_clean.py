# before the beginning, you have to install library 'pykalman'
from pykalman import *
import pandas as pd
import numpy as np

class SimpleKalmanFilter(BaseStrategy):
    def train_model(self, train_data):
        self.lookback_period = 252

        self.lookback_freq = 100

        self.lookback_data = train_data.iloc[-self.lookback_period:]

        self.trade_cnt = 0
    def trade(self, daily_data : dict) -> dict:
        self.lookback_data = self.lookback_data.append(daily_data, ignore_index=True)
        self.lookback_data = self.lookback_data.iloc[1:]

        self.portfolio = {}

        if self.trade_cnt % self.lookback_freq == 0:          
          self.lookback_data = self.lookback_data.fillna(0)
          self.lookback_data = self.lookback_data[['er_cdx_ig_long', 'spx']]
          data = self.lookback_data
          ratio = data['spx']/data['er_cdx_ig_long']

          kalman = KalmanFilter()
          means, covariances = kalman.filter(ratio.values)
          means, covariances = means.squeeze(), covariances.squeeze()
          data['ratio'] = data['spx']/data['er_cdx_ig_long']
          data['mean'] = means
          data['deviations_spx'] = (data['spx'] - data['mean'])**2
          data['deviations_er_cdx_ig_long'] = (data['spx'] - data['mean'])**2
          data['variances_spx'] = sum(data['deviations_spx'])/len(data['deviations_spx'])
          data['variances_er_cdx_ig_long'] = sum(data['deviations_er_cdx_ig_long'])/len(data['deviations_er_cdx_ig_long'])
          data['covariance'] = covariances
          data = data.fillna(0)
          data['corr'] = data['covariance']/np.sqrt(data['variances_spx']*data['variances_er_cdx_ig_long'])


          for i in range (data.shape[0]):
            if data['corr'].iloc[i] < 0.5 and data['corr'].iloc[i] > -0.5:
              portfolio = {'spx': -.1, 'er_cdx_ig_long': .1}
            elif data['corr'].iloc[i] >= 0.5:
              portfolio = {'spx': .3, 'er_cdx_ig_long': -10/(round(data['ratio'].iloc[i],1))}
            elif data['corr'].iloc[i] <= -0.5:
              portfolio = {'spx': -.3, 'er_cdx_ig_long': 10/(round(data['ratio'].iloc[i],1))}
            else:
              portfolio = {'er_cdx_ig_long' : .1, 'spx' : -.1}
          else:
            portfolio = {}
        self.trade_cnt += 1
        return self.portfolio