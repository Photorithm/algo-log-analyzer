
import numpy as np
import os
from account_scraper import AccountScraper
import pandas as pd



root = os.getcwd()
# step out cd to 'log_analyzer' if needed
if os.path.basename(root) != 'log_analyzer':
    os.chdir('..')
    root = os.getcwd()
mac_path = r'/Users/matanb/Downloads/All Version Users-data-2024-04-21 07_16_03.csv'
macs = pd.read_csv(mac_path)

monitoring_users = pd.read_csv(r'/Users/matanb/Downloads/New Users-data-2024-05-02 10_02_08.csv')
accs = monitoring_users['AccountId'].unique()
date_range = pd.date_range(start='2024-05-25', end='2024-05-27')

# filter accs
accs = accs[accs < 1000]

large_df = pd.DataFrame()
for _date in date_range:
    for acc in accs:
        try:
            mac_id = macs[macs['AccountId'] == acc]['BaseMAC'].values[0]
            AS = AccountScraper(acc_id=acc, mac_id=mac_id, dates=[_date], mac_path=mac_path)
            df = AS.get_fields()
            df['acc_id'] = acc
            large_df = pd.concat([large_df, df])
            large_df.to_csv('large_df.csv', index=False)
        except Exception as e:
            print(e)
            continue

large_df = pd.read_csv('large_df.csv')