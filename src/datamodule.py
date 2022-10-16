"""
Datamodule class is responsible for extraction and storage of data
"""

import os
import subprocess

import numpy as np
import pandas as pd

from typing import Optional, Union
from datetime import date, timedelta

class DataModule():
    """
    Extracts data from dropbox and creates a pandas DataFrame with
    all assets as columns and Date as index
    """
    def __init__(self):
        self.path_data = 'https://www.dropbox.com/s/' + \
                         'djnwwsdqdk5osaj/cmf_credit_data.zip?dl=0'
        self.asset_names = None
        self.data = None

    def setup(self):
        """
        Downloads and prepares data
        """
        # Download and prepare dataset
        if 'data' not in os.listdir():
            print('...Start Data Download...')
            download_zip = f'wget {self.path_data}' + \
                            ' -O data.zip -q'
            subprocess.run(download_zip.split(), check=True)

            unzip_data = 'unzip -q data.zip'
            subprocess.run(unzip_data.split(), check=True)

            rename_folder = 'mv cmf_credit_data data'
            subprocess.run(rename_folder.split(), check=True)

            remove_zip_file = 'rm -r data.zip'
            subprocess.run(remove_zip_file.split(), check=True)

            remove_mac_os_files = 'rm -r data/.DS_Store __MACOSX'
            subprocess.run(remove_mac_os_files.split(), check=True)
            print('...Data Download Completed...')

        # Create available asset names
        self.asset_names = [asset[:-4] for asset in os.listdir('data') \
                            if not (asset.startswith('.') or asset.startswith('_'))]
        # Create dataframe with assets as columns
        self.data = pd.DataFrame()
        for asset in self.asset_names:
            path_asset = os.path.join('data', asset + '.csv')
            asset_series = pd.read_csv(path_asset, sep=';', skiprows=1)
            asset_series['Date'] = asset_series['Date'].apply(lambda x: date(int(x[-4:]), int(x[3:5]), int(x[:2])))
            asset_series = asset_series.set_index('Date')
            self.data[asset] = asset_series['Last Price'] \
                               .apply(lambda x: float(str(x).replace(',','.')) \
                               if isinstance(x, (float, str)) else np.nan)

        # Reverse data <- latest date if first row,
        # closest date <- last row
        self.data = self.data[::-1]

    def get_asset_names(self):
        """
        Get list of asset names
        """
        return self.asset_names

    def get_data(self, start_date : Union[int, str],
                 end_date : Union[int, str]):
        """
        Get data from start_date to end_date specified as strings
        in the form yyyy-mm-dd, first date is inclusive and
        last date is excusive

        if start_date equals to -1, then the first available date is
        taken
        if end_date equals to -1, then the last available date is taken
        """
        if isinstance(start_date, str):
            start_date = date(int(start_date[:4]), \
                              int(start_date[5:7]), int(start_date[8:]))
        if isinstance(end_date, str):
            end_date = date(int(end_date[:4]), \
                            int(end_date[5:7]), int(end_date[8:]))

        if start_date == -1:
            start_date = self.data.index[0]
        if end_date == -1:
            end_date = self.data.index[-1] + timedelta(days=1)
        return self.data[(self.data.index >= start_date) & \
                         (self.data.index < end_date)]
