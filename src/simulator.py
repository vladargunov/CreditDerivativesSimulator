"""
Simulator deals with executing the agents
strategy and logging values
"""

import os
import subprocess

from typing import Optional

import numpy as np
import pandas as pd

from tqdm import tqdm
import wandb

from src.datamodule import DataModule

class Simulator():
    """
    Simulator handles all the execution of any strategy
    inherited from base_strategy
    """
    def __init__(self, train_test_split_time : str='2019-01-02',
                 transaction_costs : float=.0005,
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
        self.transaction_costs = transaction_costs

        if project_name == 'Final':
            self.train_test_split_time = '2019-01-02'
            self.transaction_costs = 0.0005
            print('For project "Final" the train_test_split_time is ' + \
                  'set at 2019-01-02 and transaction costs are set at 0.5%. If you wish to set another date or costs, ' + \
                  'use "Test" project')
            
        self.wandb_run = None
        self.run_name = run_name

        self.train_data = None
        self.test_data = None

        self.test_data_returns = None

        self.current_return_cache = []
        self.value_portfolio_cache = []

        self.value_portfolio_cache_with_costs = []
        self.current_return_cache_with_costs = []
        self.transaction_costs_cache = []

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

        # Create csv file with dates as indices for test data
        if 'supplementary_data' not in os.listdir():
            subprocess.run('mkdir supplementary_data'.split(), check=True)

        pd.DataFrame(self.test_data.index).reset_index(level=0) \
            .rename(columns={'index' : 'Step'}) \
            .to_csv('supplementary_data/test_data_steps_dates.csv', index=False)

    def get_available_assets(self):
        """
        Return available assets
        """
        return self.datamodule.get_asset_names()


    def _configure_wandb(self):
        """
        Configure wandb logging
        """
        if self.use_wandb:
            project_name = self.project_name + '_v1-2'
            self.wandb_run = wandb.init(reinit=True, name=self.run_name,
                                        entity="cmf-credit-derivatives",
                                        project=project_name)

    def simulate(self, strategy, verbose: bool=True):
        """
        Simulates a strategy according the predefined
        configuration
        """

        # Firstly train the strategy
        strategy.train_model(train_data=self.train_data)

        # Then simulate on test dataset
        previous_portfolio = {}
        portfolio = None
        current_transaction_costs = None
        value_portfolio = 1
        value_portfolio_with_costs = 1

        for idx in tqdm(range(self.test_data.shape[0])):
            current_prices = self.test_data.iloc[idx]
            current_date = self.test_data.index[idx]

            if idx != 0:
                # Compute metrics without transaction costs
                value_portfolio, current_metrics = self._compute_metrics(
                               portfolio=portfolio, current_date=current_date,
                               value_portfolio=value_portfolio,
                               idx=idx, transaction_costs=None)

                # Compute metrics with transaction costs
                value_portfolio_with_costs, current_metrics_with_costs = \
                           self._compute_metrics(portfolio=portfolio,
                           current_date=current_date, idx=idx,
                           value_portfolio=value_portfolio_with_costs,
                           transaction_costs=current_transaction_costs)

                if self.use_wandb:
                    # Log current metrics without transaction costs

                    wandb.log(current_metrics, step=idx)

                    # Log current metrics with transaction costs
                    current_metrics_with_costs = {key + ' (clean)' : item  \
                                for key, item in current_metrics_with_costs.items()}
                    wandb.log(current_metrics_with_costs, step=idx)

                    # Log current portfolio
                    self._log_portfolio(portfolio, step=idx)
                if self.debug_mode:
                    break

            # Create a portfolio for next step
            if idx != 0:
                previous_portfolio = portfolio

            portfolio = strategy.trade(daily_data=current_prices.to_dict())

            # Compute transaction costs
            current_transaction_costs = self._compute_transaction_costs(
                                        previous_portfolio=previous_portfolio,
                                        current_portfolio=portfolio,
                                        value_portfolio=value_portfolio_with_costs)

        if verbose:
            print(f'\nFinal value of portfolio {value_portfolio}')

        if self.use_wandb:
            print('Logging completed!')
            wandb.finish()

    def _compute_metrics(self, portfolio : dict, current_date : str, idx : int,
                 value_portfolio : float, transaction_costs : Optional[float]):
        """
        Computes the required metrics to be
        logged in wandb including transaction costs
        """
        metrics = {}
        # Subtract transaction_costs from portfolio
        if transaction_costs is not None:
            value_portfolio_no_costs = value_portfolio

            value_portfolio -= transaction_costs
            metrics['transaction costs'] = transaction_costs

            transaction_costs_flag = True
            self.transaction_costs_cache.append(transaction_costs)
            metrics['accumulated transaction costs'] = \
                                            sum(self.transaction_costs_cache)

            return_no_costs = self._update_value(portfolio, idx)

            value_portfolio *= (1 + return_no_costs)
            metrics['portfolio value'] = value_portfolio

            metrics['daily return'] = value_portfolio / value_portfolio_no_costs - 1

        else:
            transaction_costs_flag = False

            # Compute daily return
            metrics['daily return'] = self._update_value(portfolio, idx)
            # Compute portfolio valiue
            value_portfolio *= (1 + metrics['daily return'])
            metrics['portfolio value'] = value_portfolio

        # Append values to cache depending whether it is clean or dirty PnL
        if transaction_costs is None:
            self.value_portfolio_cache.append(metrics['portfolio value'])
            self.current_return_cache.append(metrics['daily return'])
        else:
            self.value_portfolio_cache_with_costs.append(metrics['portfolio value'])
            self.current_return_cache_with_costs.append(metrics['daily return'])

        # Compute drawdown over the last 252 days (year) and overall
        metrics['drawdown'] = self.get_max_drawdown(
                                    transaction_costs_flag=transaction_costs_flag)
        metrics['1Y drawdown'] = self.get_max_drawdown(trailing_days=252,
                                    transaction_costs_flag=transaction_costs_flag)

        # Compute annualised returns over the last 252 days and overall
        metrics['annualised return'] = self.get_annualised_return(
                                    transaction_costs_flag=transaction_costs_flag)
        metrics['1Y return'] = self.get_annualised_return(trailing_days=252,
                                    transaction_costs_flag=transaction_costs_flag)

        # Compute sharpe over the last 252 days and overall
        metrics['sharpe'] = self.get_sharpe(current_date=current_date, idx=idx,
                                transaction_costs_flag=transaction_costs_flag)
        metrics['1Y sharpe'] = self.get_sharpe(current_date=current_date, idx=idx,
                    trailing_days=252, transaction_costs_flag=transaction_costs_flag)

        # Log current date
        metrics['date'] = current_date

        return value_portfolio, metrics


    def _compute_transaction_costs(self, previous_portfolio : dict,
                            current_portfolio : dict, value_portfolio : float):
        """
        Computes the current transaction costs based on changes from
        previous_portfolio to current_portfolio
        """
        transaction_costs = 0
        for asset in self.get_available_assets():
            transaction_costs += abs(current_portfolio.get(asset,0) \
                                 - previous_portfolio.get(asset,0)) \
                                 * self.transaction_costs * value_portfolio
        return transaction_costs


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
        assets = self.get_available_assets()
        logging_values = portfolio.copy()
        # Add zero values
        for asset in assets:
            if asset not in portfolio.keys():
                logging_values[asset] = 0.0
        # Log values into wandb
        wandb.log(logging_values, step=step)


    def get_sharpe(self, current_date=None, trailing_days=None, idx : int=1,
                   transaction_costs_flag : bool=False) -> float:
        """
        Sharpe - calculated as mean of daily returns minus the
        yield treasury curve rate 1month averaged and converted
        to daily return
        Args:
           current_date <- represents the current date needed for risk-free
           interest rate retrieval
           trailing_days <- number of trailing_days to compute the metric over
        Note that if trailing_days is None, the sharpe ratio is calculated
        assuming all test_history, so for trailing metric one need to
        specify both trailing_days and trailing_date
        """
        # Get risk free rate
        if trailing_days is None:
            risk_free_rate = self.datamodule.get_risk_free_rate(
                                            start_date=self.train_test_split_time,
                                            end_date=str(current_date))
            if transaction_costs_flag:
                return_cache = self.current_return_cache_with_costs
            else:
                return_cache = self.current_return_cache
        else:
            # Compute the trailing_date
            trailing_date = self.test_data.index[max(0, idx - trailing_days)]

            risk_free_rate = self.datamodule.get_risk_free_rate(
                                            start_date=str(trailing_date),
                                            end_date=str(current_date))

            if transaction_costs_flag:
                return_cache = self.current_return_cache_with_costs[-trailing_days:]
            else:
                return_cache = self.current_return_cache[-trailing_days:]

        sharpe_daily = (np.array(return_cache).mean() - risk_free_rate) \
                          / np.array(return_cache).std()
        sharpe_yearly = sharpe_daily * (252 ** .5)
        return sharpe_yearly

    def get_max_drawdown(self, trailing_days : Optional[int]=None,
                         transaction_costs_flag : bool=False) -> float:
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
            if transaction_costs_flag:
                current_value_portfolio = self.value_portfolio_cache_with_costs.copy()
            else:
                current_value_portfolio = self.value_portfolio_cache.copy()
        else:
            if transaction_costs_flag:
                current_value_portfolio = \
                            self.value_portfolio_cache_with_costs[-trailing_days:]
            else:
                current_value_portfolio = \
                            self.value_portfolio_cache[-trailing_days:]

        min_value = min(current_value_portfolio)
        max_value = max(current_value_portfolio)

        mdd = (min_value - max_value) / max_value

        return mdd

    def get_annualised_return(self, trailing_days : Optional[int]=None,
                              transaction_costs_flag : bool=False) -> float:
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
            if transaction_costs_flag:
                current_value_portfolio = self.current_return_cache_with_costs.copy()
            else:
                current_value_portfolio = self.current_return_cache.copy()
        else:
            if transaction_costs_flag:
                current_value_portfolio = \
                           self.current_return_cache_with_costs[-trailing_days:]
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
