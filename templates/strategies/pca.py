"""
PCA strategy developed by
Ivan Alpatov
Github: ialpatov
"""

class PCA_strategy(BaseStrategy):

    """
    PCA Strategy
    """
    def __init__(self, regression_period = 60, res_estimation_period = 60, 
                 use_pca = True, n_components=1, stock_decomposed = True, currency = 'usd',
                 scale_factor=10):
        """ 
        PC - principal components
        Attributes:
        regression_period - window length for calculating PC of data series
        res_estimation_period - window length for residual calculation
        use_pca - if we use PCA method or not
        n_components - number of components for PCA method
        stock_decomposed - True if we decompose stock index by credit indices, False otherwise
        currency - determines group of stock and credit derivatives indices by their currency
        """ 
        self.res_estimation_period = res_estimation_period
        self.regression_period = regression_period
        self.n_components = n_components
        self.use_pca = use_pca
        self.stock_decomposed = stock_decomposed
        self.scale_factor = scale_factor

        self.all_traded_tickers = ['er_cdx_ig_long', 'hyg', 'lqd', 'vix', \
                                   'er_itraxx_main_long', 'er_itraxx_xover_long', 'ihyg', 'ieac',
                                   'spx', 'sx5e']

        # thresholds for s-score
        self.thresholds = [-3.55, -1.4, 1.15, 3.44]
        
        stock_ticker_usd = 'spx'
        stock_ticker_eur = 'sx5e'
        credit_tickers_usd = ['er_cdx_ig_long', 'hyg', 'lqd', 'vix']
        credit_tickers_eur = ['er_itraxx_main_long', 'er_itraxx_xover_long', 'ihyg', 'ieac']
        
        pairs = {'usd': {'stock': stock_ticker_usd, 'credit': credit_tickers_usd},
                      'eur': {'stock': stock_ticker_eur, 'credit': credit_tickers_eur}}
        self.current_pair = pairs[currency]

        self.arr = [self.current_pair['stock'], *self.current_pair['credit']]
        self.portfolio = dict(zip(self.arr, [0 for x in self.arr]))
        self.etf_dec_portfolio = [dict(zip(self.arr, [0 for x in self.arr])) for i in self.arr[1:]]
        self.position_opened = None if self.stock_decomposed == True else [None for x in self.arr[1:]]
        

        self.lookback_freq = 1
        self.trade_cnt = 0

    def train_model(self, train_data):
        """
        Set train data
        """
        # Get lookback data
        self.lookback_data = train_data.iloc[-self.regression_period:]
        
       
    def trade(self, daily_data : dict) -> dict:
        """
        Trade based on lookback data and using PCA
        """
        # Place a recent date into self.lookback_data
        self.lookback_data = \
                    self.lookback_data.append(daily_data, ignore_index=True)

        # Fill infinite values with nan values to interpolate
        self.lookback_data = self.lookback_data.replace([np.inf, -np.inf], np.nan)

        # Check for any nan values resulted from missing data and interpolate them
        if self.lookback_data.isnull().values.any():
            for asset in self.lookback_data.columns:
              try:
                self.lookback_data[asset] = self.lookback_data[asset] \
                                        .interpolate(method='polynomial', order=1)
              except ValueError:
                continue
        self.lookback_data = self.lookback_data.ffill()
        # Get returns for current lookback_data
        current_returns = self.lookback_data.pct_change()

        # Drop first row with nan values which resulted from pct_change()
        # from current data
        self.lookback_data = self.lookback_data.iloc[1:]
        # update daily_data so that there would be no nans there
        daily_data = self.lookback_data.iloc[-1].to_dict()

        # Drop first row with nan values from current_lookback_data
        current_returns = current_returns.iloc[1:]

        # Get current etfs and stocks
        etfs = current_returns[self.current_pair['credit']]
        stock = current_returns[self.current_pair['stock']]

        factors, components, sigmas = self.get_factors(etfs, stock)
        
        # here we could use Ridge but it can perform badly
        if self.stock_decomposed == True:
          X, beta_0, betas = self.regression(factors, stock)
        else:
          X, beta_0, betas = self.regression(factors, etfs)

        if self.stock_decomposed == True:
          s_score = self.calc_s_score(list(X), beta_0)
        else:
          s_score = []
          for index, etf_name in enumerate(X):
            s_score.append(self.calc_s_score(list(X[etf_name]), beta_0[index]))


        # ЗДЕСЬ ПРОИСХОДИТ РАБОТА С ПОРТФОЛИО
        # self.etf_dec_portfolio нужен если stock_decomposed == False 
        # в этом случае self.portfolio представляет собой массив словарей ! 
        # (просто открываем позиции отдельно для каждого credit индекса)
        self.portfolio = self.portfolio if self.stock_decomposed == True else self.etf_dec_portfolio
        self.position_opened, new_portfolio = self.update_portfolio(s_score, components, sigmas, betas, daily_data)
        self.portfolio = new_portfolio if self.stock_decomposed == True else reduce(self.sum_dicts, new_portfolio)
        self.etf_dec_portfolio = self.etf_dec_portfolio if self.stock_decomposed == True else new_portfolio
        # здесь возвращается уже портфолио в формате словаря

        # ADDITIONS OF VLADARGUNOV
        # Check portfolio for existence of nan values and replace them to zeros
        self.portfolio = self.replace_nans(self.portfolio)

        # Scale portfolio by multiplying all values by a constant
        return_portfolio = self.scale_portfolio(portfolio=self.portfolio.copy(),
                                              scale_factor=self.scale_factor)

        return return_portfolio

    
    def get_factors(self, etfs, stock):
      """
      Calculates factors for regression and performs PCA if necessary
      """

      factors = pd.DataFrame()
      # scaled data for PCA
      if self.use_pca == True:
        scaler = StandardScaler()
        Y = scaler.fit_transform(etfs)
        sigmas = scaler.scale_
        # Compute PCA
        pca = PCA(n_components=self.n_components)
        pca.fit(Y)
        
        if self.stock_decomposed == False:
          factors['stock'] = stock
        for index, eigenvector in enumerate(pca.components_):
          factors[f'{index}'] = (np.dot(etfs, eigenvector / sigmas))
        return factors, pca.components_, sigmas
      elif self.stock_decomposed == True: 
        factors = etfs
      else:
        factors['stock'] = stock
      return factors, None, None
        

    def regression(self, factors, y):
      """
      Performs regression of index by factors

      Attributes:
      factors - regressors
      y - vector to regress

      Returns residual ndarray and regression coefficients (betas)
      """

      clf = LinearRegression()
      clf.fit(factors, y)
      beta_0 = clf.intercept_
      betas = clf.coef_
      eps = y - clf.predict(factors)

      X = eps[-self.res_estimation_period:].cumsum(axis=0)
      return X, beta_0, betas

    def calc_s_score(self, X, beta_0):
      """
      Performs autoregression of residuals and calculates s-score

      Attributes:
      X - residual vector
      beta_0 - regression coef
      """

      adfuller_test = adfuller(X)
      if adfuller_test[0] < adfuller_test[4]['5%']:
        mod = ARIMA(X, order=(1, 0, 0)).fit()
        a, b, sigma2 = mod.params

        k = -np.log(b)*252
        if k < 252*2/self.res_estimation_period - 0.1:
          return np.nan
        s_score = (X[-1] - a/(1-b))*np.sqrt(1 - b**2)/(np.sqrt(sigma2))
        s_modified = s_score - beta_0*np.sqrt(1 - b**2)/(-np.log(b)*np.sqrt(sigma2)) 
        return s_score
      else:
        return np.nan
    
    def update_portfolio(self, s_score, components, sigmas, betas, daily_data):
      """
      Updates portfolio based on the s-score for all indices

      Attributes:
      s_score - s-score, float if stock_decomposed == True or List[float] otherwise
      components - eigenvectors of PCA
      sigmas - standard deviations of etfs
      betas - regression coefs
      daily_data - current prices in dict format

      This function provides new values of position_opened and portfolio in 
      unified format applicable for all cases
      """

      if self.stock_decomposed == True:
        signal = self.trading_signal(s_score, self.position_opened)
        new_position_opened, new_portfolio = self.interpret_signal(signal, self.portfolio, 
                                                              components, sigmas, betas, 
                                                              self.position_opened, 'stock', daily_data)
      elif self.stock_decomposed == False:
        new_position_opened = []
        new_portfolio = []
        for i, etf_name in enumerate(self.arr[1:]):
          signal = self.trading_signal(s_score[i], self.position_opened[i])
          pos_i, port_i = self.interpret_signal(signal, self.portfolio[i], components, sigmas, 
                                          betas[i], self.position_opened[i],
                                          etf_name, daily_data)
          new_position_opened.append(pos_i)
          new_portfolio.append(port_i)

      return new_position_opened, new_portfolio

    
    def trading_signal(self, s_score, position_opened):
      """
      Calculates trading signal based on s-score

      Attributes:
      s_score - s-score value for index
      position_opened - previous value of position_opened for this index
      """

      if np.isnan(s_score):
        return None
      s_open_long, s_close_short, s_close_long, s_open_short = self.thresholds
      if position_opened is None:
        if s_open_long < s_score < s_open_short:
          return None
        elif s_score <= s_open_long:
          return 'open_long'
        elif s_score >= s_open_short:
          return 'open_short'
      elif position_opened == 'short':
        if s_score > s_close_short:
          return None
        elif  s_open_long < s_score <= s_close_short:
          return 'close_short'
        elif s_score <= s_open_long:
          return 'close_short_open_long'
      elif position_opened == 'long':
        if s_score < s_close_long:
          return None
        elif  s_close_long <= s_score < s_open_short:
          return 'close_long'
        elif s_score >= s_open_short:
          return 'close_long_open_short'


    def interpret_signal(self, signal, portfolio, components, sigmas, betas, position_opened, index_name, daily_data):
      """
      Interprets signal obtained from trading_signal func

      Attributes:
      signal - trading signal (output of trading_signal func)
      portfolio - previous portfolio
      components - eigenvectors of PCA
      sigmas - standard deviations of etfs
      betas - regression coefs
      position_opened - previous value of position_opened for this index
      index_name - name of the index to open position in (not necessary if it is stock index)
      daily_data - current prices in dict format

      This function forms new values for position_opened and portfolio for
      only 1 (!!!) index
      """
      zero_portfolio = dict(zip(self.arr, [0 for x in self.arr]))
      if signal is None:
        return position_opened, portfolio
      elif signal == 'open_long' or signal == 'close_short_open_long':
        new_portfolio = self.open_position(components, sigmas, betas, index_name, 'long')

        # ЗДЕСЬ ПРОИСХОДИТ СКЕЙЛИНГ ЗНАЧЕНИЙ ПОРТФОЛИО ИЗ ДОЛЛАРОВ В ШТУКИ ИНДЕКСОВ
        # (ПРОСТО ПРОИСХОДИТ ДЕЛЕНИЕ ВЛОЖЕННЫХ ДОЛЛАРОВ НА ЦЕНЫ)
        new_portfolio = self.divide_dicts(new_portfolio, daily_data)
        return 'long', new_portfolio
      elif signal == 'open_short' or signal == 'close_long_open_short':
        new_portfolio = self.open_position(components, sigmas, betas, index_name, 'short')

        # ЗДЕСЬ ПРОИСХОДИТ СКЕЙЛИНГ ЗНАЧЕНИЙ ПОРТФОЛИО ИЗ ДОЛЛАРОВ В ШТУКИ ИНДЕКСОВ
        # (ПРОСТО ПРОИСХОДИТ ДЕЛЕНИЕ ВЛОЖЕННЫХ ДОЛЛАРОВ НА ЦЕНЫ)
        new_portfolio = self.divide_dicts(new_portfolio, daily_data)
        return 'short', new_portfolio
      elif signal == 'close_long' or signal == 'close_short':
        return None, zero_portfolio


    def open_position(self, components, sigmas, betas, index_name, pos):
      """
      Opens position in 1 index in absolute values (in units of currency)

      Attributes:
      components - eigenvectors of PCA
      sigmas - standard deviations of etfs
      betas - regression coefs
      index_name - name of the index to open position in (not necessary if it is stock index)
      pos - position type, either 'long' or 'short'
      """

      portfolio = dict(zip(self.arr, [0 for x in self.arr]))
      exponent = 0 if pos == 'long' else 1

      if self.stock_decomposed == True:
        if self.use_pca == False:
            for index, name in enumerate(self.arr[1:]):
              portfolio[name] = ((-1)**(1 - exponent))*betas[index]
        else:
          for index, component in enumerate(components):
            portfolio = self.sum_dicts(portfolio, 
                                    dict(zip(self.arr, [0, *((-1)**(1 - exponent))*betas[index]*(component/sigmas)]))
                                    )
        portfolio[self.arr[0]] = (-1)**exponent
      elif self.stock_decomposed == False:
        portfolio[index_name] = (-1)**exponent
        portfolio[self.arr[0]] = (-1)**(1 - exponent)*betas[0]
        if self.use_pca == True:
          for index, component in enumerate(components):
            portfolio = self.sum_dicts(portfolio, dict(zip(self.arr, [0, *((-1)**(1 - exponent))*betas[1:][index]*(component/sigmas)])))

      # ЗДЕСЬ ВОЗВРАЩАЕТСЯ ПОРТФОЛИО СО ЗНАЧЕНИЯМИ В ДОЛЛАРАХ, ВЛОЖЕННЫХ
      # В СООТВЕТСТВУЮШИЙ АКТИВ
      return portfolio


    def scale_portfolio(self, portfolio, scale_factor=100):
        """
        Scale the portfolio by scale_factor is it is possible
        and to sum up to 1 otherwise
        """
        total_weights = sum([abs(value) for value in portfolio.values()])

        if total_weights * scale_factor >= 1:
          scale_factor = 1 / (total_weights + 0.001)
        for key, value in portfolio.items():
          portfolio[key] = scale_factor * value

        return portfolio

    def sum_dicts(self, dict1, dict2):
      return {x: dict1[x] + dict2[x] for x in dict1}

    def divide_dicts(self, dict1, dict2):
      return {x: dict1[x] / dict2[x] for x in dict1}

    def replace_nans(self, portfolio):
      for key, value in portfolio.items():
        if np.isnan(value):
          print('Nan value detected')
          portfolio[key] = 0
      return portfolio
