"""
Example of cointegration strategy
implemented by Arthur Arifullin
GitHub: ArturArifullin
"""

import scipy.stats as sps
from statsmodels.tsa.stattools import adfuller
import numpy as np
import math

class CointegrationStrategy(BaseStrategy):
    """
    Cointegration strategy with rebalancing 
    each 100 days based on latest 252 trading days.
    The decision rule is formed according to
    the degree of correlation between assets
    """
    def train_model(self, train_data):
        """
        Set lookback lenght and frequency
        """

        # Define lookback period in days
        self.lookback_period = 252

        # Set lookback frequency
        self.lookback_freq = 100

        # Get lookback data
        self.lookback_data = train_data.iloc[-self.lookback_period:]

        # Set counter for tracking trades
        self.trade_cnt = 0

        # Set correlation
        self.corr = 0.8

  
    def trade(self, daily_data : dict) -> dict:
        """
        Create cointegrating portfolio based
        """
        # Place a recent date into self.lookback_data
        self.lookback_data = self.lookback_data.append(daily_data, 
                                                       ignore_index=True)
        self.lookback_data = self.lookback_data.iloc[1:].dropna()
        # Now you have new lookback data including the latest date

        x = self.lookback_data['spx'].to_numpy().reshape((1, -1))[0]
        y = self.lookback_data['er_cdx_ig_long'].to_numpy().reshape((1, -1))[0]
      
        if ( x.size > 0 and y.size > 0 ):
          if self.trade_cnt % self.lookback_freq == 0:
            #print(x, y)
            #print(np.corrcoef(x, y))
            if ( np.corrcoef(x, y)[0, 1] >= self.corr ):
              #print(np.corrcoef(x, y)[0, 1])
              slope, intercept, rvalue, pvalue, stderr = sps.linregress(np.log(y)
              , np.log(x))
              eps_spread = np.log(y) - slope * np.log(x)
              df_test = adfuller( eps_spread )
              if ( df_test[0] > df_test[4]['5%'] ): #df test is done 
                #print('yes')
                self.mean = eps_spread.mean()
                self.std  = eps_spread.std()
                self.slope = slope 
                self.to_trade = True
              else:
                #print('no')
                self.to_trade = False
            else:
              self.to_trade = False
        else:
          self.to_trade = False

        self.current_portfolio = {}
        if self.to_trade == True:
          self.z_t = (np.log(daily_data['er_cdx_ig_long']) - 
                      self.slope*np.log(daily_data['spx']) - self.mean)/self.std
          #print('x', self.z_t)
          #print('a', self.opened_position)
          if ( math.isnan(self.z_t) and self.opened_position != None ):
            self.current_portfolio = self.opened_position 
          elif ( self.z_t < -2 ):
            self.current_portfolio = {'spx' : -.1, 'er_cdx_ig_long' : .1} 
            self.opened_position = self.current_portfolio
           
          elif ( self.z_t > 2  ):
            self.current_portfolio = {'spx' : .1, 'er_cdx_ig_long' : -.1} 
            self.opened_position = self.current_portfolio
          elif ((self.z_t <= -0.75) and self.opened_position != None ):
            self.current_portfolio = self.opened_position 
          elif ( (self.z_t >= 0.5) and self.opened_position != None ):
            self.current_portfolio = self.opened_position
          elif ( self.opened_position != None ):
            self.current_portfolio = {}
            self.opened_position = None
          else:
            self.current_portfolio = {}
        #CHECK SIDE
        # Update self.trade_cnt
        self.trade_cnt += 1
        #print(self.current_portfolio)
        return self.current_portfolio
