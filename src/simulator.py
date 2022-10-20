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
                 run_name : Optional[str]='SampleStrategy1'):
        self.datamodule = DataModule()
        self.datamodule.setup()

        # if self.debug_mode = True, then do not log any values
        self.use_wandb = use_wandb and not debug_mode
        self.debug_mode = debug_mode

        self.train_test_split_time = train_test_split_time
        self.wandb_run = None
        self.run_name = run_name

        self.train_data = None
        self.test_data = None

        self.test_data_returns = None

        self.current_return_cache = []

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
                                        project='CMF-Credit-Derivatives')

    def simulate(self, strategy, verbose: bool=True):
        """
        Simulates the strategy according the predefined
        configuration
        """

        value_portfolio = 1
        # Firstly train the strategy
        strategy.train_model(train_data=self.train_data)

        # Then test it
        for idx in tqdm(range(self.test_data.shape[0])):
            current_prices = self.test_data.iloc[idx]

            if idx != 0:
                current_return = self._update_value(portfolio, idx)
                value_portfolio *= (1 + current_return)
                self.current_return_cache.append(current_return)

                if self.use_wandb:
                    wandb.log({'daily_return' : current_return})
                    wandb.log({'portfolio_value' : value_portfolio})
                if self.debug_mode:
                    break

            # Create a portfolio for next step
            portfolio = strategy.trade(daily_data=current_prices.to_dict())

        final_metrics = self._calculate_final_metrics()

        if verbose:
            print(f'\nFinal value of portfolio {value_portfolio}')
            print(f"Sharpe of the porfolio {final_metrics['sharpe']}")

        if self.use_wandb:
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

    def _calculate_final_metrics(self) -> dict:
        """
        Calculate the final metrics for the strategy
        reporting
        Metrics calculated:
        Sharpe - calculated as mean of daily returns minus the return of
        a portfolio consisting of only spx long stock, divided by the
        standard deviation of daily returns
        """
        # Calculate a portfolio of investing only in spx
        mean_spx = self.test_data_returns['spx'].mean()

        sharpe = (np.array(self.current_return_cache).mean() - mean_spx) \
                          / np.array(self.current_return_cache).std()

        final_metrics = {'sharpe' : sharpe}
        if self.use_wandb:
            wandb.log(final_metrics)

        return final_metrics
