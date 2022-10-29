"""
Simulator deals with executing the agents
strategy and logging values
"""

from typing import Optional

import numpy as np

from tqdm import tqdm
import wandb

from src.datamodule import DataModule

class Simulator():
    """
    Simulator handles all the execution of any strategy
    inherited from base_strategy
    """
    def __init__(self, train_test_split_time : str='2019-01-02',
                 use_wandb : bool=True, debug_mode : bool=False,
                 run_name : Optional[str]='SampleStrategy1',
                 project_name : str='Test'):

        assert project_name in ['Test', 'Final', 'Development'], \
        'Specify correct project name! Available options are "Test" and "Final"'
        self.project_name = project_name

        self.datamodule = DataModule()
        self.datamodule.setup()

        # if self.debug_mode = True, then do not log any values
        self.use_wandb = use_wandb and not debug_mode
        self.debug_mode = debug_mode

        self.train_test_split_time = train_test_split_time

        if project_name == 'Final':
            self.train_test_split_time = '2019-01-02'
            print('For project "Final" the train_test_split_time is ' + \
                  'set at 2019-01-02. If you wish to set another date, ' + \
                  'use "Test" project')
        self.wandb_run = None
        self.run_name = run_name

        self.train_data = None
        self.test_data = None

        self.test_data_returns = None

        self.current_return_cache = []
        self.value_portfolio_cache = []

        self._configure_simulator()
        self._configure_wandb()


    def _configure_simulator(self):
        """
        Configure simulator by splitting the
        data according to train_test_split_time
        """
        self.train_data = self.datamodule.get_data(start_date=-1,
                                           end_date=self.train_test_split_time)

        self.test_data = self.datamodule.get_data(start_date=self.train_test_split_time,
                                          end_date=-1)
        self.test_data_returns = self.test_data.pct_change()

    def get_availiable_assets(self):
        """
        Return available assets
        """
        return self.datamodule.get_asset_names()


    def _configure_wandb(self):
        """
        Configure wandb logging
        """
        if self.use_wandb:
            self.wandb_run = wandb.init(reinit=True, name=self.run_name,
                                        entity="cmf-credit-derivatives",
                                        project=self.project_name)

    def simulate(self, strategy, verbose: bool=True):
        """
        Simulates a strategy according the predefined
        configuration
        """

        value_portfolio = 1
        # Firstly train the strategy
        strategy.train_model(train_data=self.train_data)

        # Then test it
        for idx in tqdm(range(self.test_data.shape[0])):
            current_prices = self.test_data.iloc[idx]
            current_date = self.test_data.index[idx]

            if idx != 0:
                current_return = self._update_value(portfolio, idx)
                value_portfolio *= (1 + current_return)
                # Append values to cache
                self.current_return_cache.append(current_return)
                self.value_portfolio_cache.append(value_portfolio)

                # Compute drawdown over the last 252 days (year) and overall
                total_drawdown = self.get_max_drawdown()
                annual_drawdown = self.get_max_drawdown(trailing_days=252)

                # Compute annualised returns over the last 252 days and overall
                total_ann_ret = self.get_annualised_return()
                annual_ann_ret = self.get_annualised_return(trailing_days=252)

                # Compute sharpe over the last 252 days and overall
                total_sharpe = self.get_sharpe(current_date=current_date)
                # Trailing date is needed for sharpe calculation
                trailing_date = self.test_data.index[max(0, idx - 252)]
                annual_sharpe = self.get_sharpe(current_date=current_date,
                                                trailing_days=252,
                                                trailing_date=trailing_date)

                if self.use_wandb:
                    wandb.log({'daily return' : current_return,
                               'portfolio value' : value_portfolio,
                               'drawdown' : total_drawdown,
                               '1Y drawdown' : annual_drawdown,
                               'annualised_return' : total_ann_ret,
                               '1Y return' : annual_ann_ret,
                               'sharpe' : total_sharpe,
                               '1Y sharpe' : annual_sharpe,
                               'date' : current_date
                               },
                               step=idx)
                    self._log_portfolio(portfolio, step=idx)
                if self.debug_mode:
                    break

            # Create a portfolio for next step
            portfolio = strategy.trade(daily_data=current_prices.to_dict())

        if verbose:
            print(f'\nFinal value of portfolio {value_portfolio}')

        if self.use_wandb:
            print('Logging completed!')
            wandb.finish()

    def _update_value(self, current_portfolio : dict, idx : int) -> float:
        """
        Update the value of the portfolio according to the
        weights of the portfolio
        """
        total_portfolio = sum([abs(weight) for weight in current_portfolio.values()])
        assert total_portfolio <= 1, 'You have exceeded overall allocations! \
                                      All weights must sum to 1, including the negative allocations!'


        current_returns = self.test_data_returns.iloc[idx].to_dict()
        return sum([current_portfolio[asset] * current_returns[asset] \
                                for asset in current_portfolio.keys()])

    def _log_portfolio(self, portfolio : dict, step : int):
        """
        Log values of the given portfolio
        """
        assets = self.get_availiable_assets()
        logging_values = portfolio.copy()
        # Add zero values
        for asset in assets:
            if asset not in portfolio.keys():
                logging_values[asset] = 0.0
        # Log values into wandb
        wandb.log(logging_values, step=step)


    def get_sharpe(self, current_date=None, trailing_days=None,
                   trailing_date=None) -> float:
        """
        Sharpe - calculated as mean of daily returns minus the
        yield treasury curve rate 1month averaged and converted
        to daily return
        Args:
           current_date <- represents the current date needed for risk-free
           interest rate retrieval
           trailing_days <- number of trailing_days to compute the metric over
           trailing_date <- the date corresponding to the start_date if
           trailing_days is specified
        Note that is trailing_days is None or trailing_date is None, the sharpe
        ratio is calculated assuming all test_history, so for trailing metric
        one need to specify both trailing_days and trailing_date
        """
        # Get risk free rate
        if trailing_days is None or trailing_date is None:
            risk_free_rate = self.datamodule.get_risk_free_rate(
                                            start_date=self.train_test_split_time,
                                            end_date=str(current_date))

            return_cache = self.current_return_cache
        else:
            risk_free_rate = self.datamodule.get_risk_free_rate(
                                            start_date=str(trailing_date),
                                            end_date=str(current_date))

            return_cache = self.current_return_cache[-trailing_days:]

        sharpe_daily = (np.array(return_cache).mean() - risk_free_rate) \
                          / np.array(return_cache).std()
        sharpe_yearly = sharpe_daily * (252 ** .5)
        return sharpe_yearly

    def get_max_drawdown(self, trailing_days : Optional[int]=None) -> float:
        """
        Calculates the maximum drawdown over the specified
        trailing days. Trailing days can be None, then the maximum drawdown
        is computed over all history
        Formula for it is
        MDD = (min_value - max_value) / max_value
        Source:
        https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp

        Args:
        trailing_days - number of days to compute mdd over
        """
        if trailing_days is None:
            current_value_portfolio = self.value_portfolio_cache.copy()
        else:
            current_value_portfolio = \
                            self.value_portfolio_cache[-trailing_days:]

        min_value = min(current_value_portfolio)
        max_value = max(current_value_portfolio)

        mdd = (min_value - max_value) / max_value

        return mdd

    def get_annualised_return(self, trailing_days : Optional[int]=None) -> float:
        """
        Calculates the annualised return over the specified
        trailing days. Trailing days can be None, then the annualised return
        is computed over all history
        Formula for annualised return:
        annual_return = (1+r_1) * (1+r_2) * ... (1+r_n) ** (252 / n) - 1
        Source:
        https://www.investopedia.com/terms/a/annualized-total-return.asp
        """
        if trailing_days is None:
            current_value_portfolio = self.current_return_cache.copy()
        else:
            current_value_portfolio = \
                           self.current_return_cache[-trailing_days:]

        num_days = len(current_value_portfolio)

        # Convert to numpy array and add 1
        current_value_portfolio = np.array(current_value_portfolio) + 1

        annualised_ret = np.prod(current_value_portfolio) ** (252 / num_days) - 1

        return annualised_ret

    def get_training_data(self):
        """
        Get training data as used in train_model()
        function
        """
        return self.train_data.copy()


    def get_test_data(self):
        """
        Iterator of test data as
        given to the simulator,
        used for testing and debugging
        """
        for idx in range(self.test_data.shape[0]):
            yield self.test_data.iloc[idx].to_dict()
