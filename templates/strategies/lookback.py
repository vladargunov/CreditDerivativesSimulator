"""
Template for lookback type strategies
"""

class CustomLookBackStrategy(BaseStrategy):

    """
    This strategy template implements a simple
    lookback strategy, which allows to process
    specific number of past dates at each call
    of trade. Additionally, the data processing
    of lookback strategy can be executed with
    specified frequency.
    """
    def train_model(self, train_data):
        """
        Set lookback lenght and frequency
        """
        # Define lookback period in days
        self.lookback_period = 10

        # Set lookback frequency
        self.lookback_freq = 252

        # Get lookback data
        self.lookback_data = train_data.iloc[-self.lookback_period:]

        # Set counter for tracking trades
        self.trade_cnt = 0
    def trade(self, daily_data : dict) -> dict:
        """
        Trade based on lookback data
        """
        # Place a recent date into self.lookback_data
        self.lookback_data.append(daily_data, ignore_index=True)
        self.lookback_data = self.lookback_data.iloc[1:]
        # Now you have new lookback data including the latest date

        if self.trade_cnt % self.lookback_freq == 0:
          # Do some code with lookback data here
          pass

        # Update self.trade_cnt
        self.trade_cnt += 1
        return {}
