import os

import numpy as np
import pandas as pd

from typing import Optional, Union

from src.datamodule import DataModule
from tqdm import tqdm

import wandb


class Simulator():
    def __init__(self, train_test_split_time : str='2013-01-02',
                 use_wandb : bool=True, debug_mode : bool=False,
                 run_name : Optional[str]='SampleStrategy1'):
        self.dm = DataModule()
        self.dm.setup()

        self.use_wandb = use_wandb
        self.debug_mode = debug_mode

        self.train_test_split_time = train_test_split_time
        self.wandb_run = None
        self.run_name = run_name

        self.train_data = None
        self.test_data = None

        self.test_data_returns = None

        self._configure_simulator()
        self._configure_wandb()


    def _configure_simulator(self):
        self.train_data = self.dm.get_data(start_date=-1,
                                           end_date=self.train_test_split_time)

        self.test_data = self.dm.get_data(start_date=self.train_test_split_time,
                                          end_date=-1)
        self.test_data_returns = self.test_data.pct_change()


    def _configure_wandb(self):
        if (not self.use_wandb) or (self.debug_mode):
            os.environ['WANDB_MODE'] = 'disabled'

        self.wandb_run = wandb.init(reinit=True, name=self.run_name,
                                    project='CMF-Credit-Derivatives')

    def simulate(self, strategy, verbose=True):

        value_portfolio = 1
        # Firstly train the strategy
        strategy.train_model(train_data=self.train_data)

        # Then test it
        previous_prices = 1
        for idx in tqdm(range(self.test_data.shape[0])):
            current_prices = self.test_data.iloc[idx]

            if idx != 0:
                current_return = 1 + self._update_value(portfolio, idx)
                wandb.log({'return_portfolio' : current_return})
                self.value_portfolio *= (1 + current_return)
                if self.debug_mode:
                    break

            # Create a portfolio for next step
            portfolio = strategy.trade(daily_data=prices.to_dict())

        # After loop completion, report the results
        wandb.log({'value_portfolio' : value_portfolio})

        if verbose:
            print(f'Final value of portfolio {value_portfolio}')

    def _update_value(self, current_portfolio : dict, idx : int):
        total_portfolio = sum([abs(weight) for weight in current_portfolio.value()])
        assert total_portfolio <= 1, 'You have exceeded overall allocations! \
                                      All weights must sum to 1, including the negative allocations!'


        current_returns = self.test_data_returns.iloc[idx]
        return sum([current_portfolio[asset] * current_returns[asset] \
                                for asset in current_portfolio.keys()])
