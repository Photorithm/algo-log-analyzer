import numpy as np
import os
from account_scraper import AccountScraper
import pandas as pd
import datetime


root = os.getcwd()
# step out cd to 'log_analyzer' if needed
if os.path.basename(root) != 'log_analyzer':
    os.chdir('..')
    root = os.getcwd()
mac_path = r'/Users/matanb/Downloads/All Version Users-data-2024-04-21 07_16_03.csv'
macs = pd.read_csv(mac_path)


def get_last_calibrating(df):
    # get a list of all rows where the state is calibrating decision and the next state is not calibrating decision
    temp_df = df.copy()
    stop = temp_df[temp_df['notification'].apply(lambda x: 'Stopped' in x or 'NotFound' in x if x is not None else False)]
    # drop rows where state is None
    temp_df = temp_df[temp_df['state'].notnull()]


    temp_df['next_state'] = temp_df['state'].shift(-1)


    # add median of last 6 seconds of bc1
    temp_df['bc_median'] = temp_df['bc1'].rolling(window=6).median()
    calibrating = temp_df[temp_df['state'] == 'calibrating decision']
    calibrating = calibrating[(calibrating['next_state'] == 'breathing micro') | (calibrating['next_state'] == 'no breathing')]
    # get the last one beofre a stop
    last_last_calibrating = []
    for idx, row in stop.iterrows():
        last_calibrating = calibrating[calibrating['time'] < row['time']].tail(1)
        last_last_calibrating.append(last_calibrating)
    last_last_calibrating = pd.concat(last_last_calibrating)


    last_last_calibrating[last_last_calibrating['next_state'] == 'no breathing']['pow']
    last_last_calibrating[last_last_calibrating['next_state'] == 'no breathing']['bc_median']

    # output df
    output_df = pd.DataFrame(columns = ['stop', 'time', 'last_pow', 'last_bc_median'])
    output_df['stop'] = stop['notification']
    output_df['time'] = last_last_calibrating['time']
    output_df['last_pow'] = last_last_calibrating['pow'].values
    output_df['last_bc_median'] = last_last_calibrating['bc_median'].values

    return output_df




accounts = pd.read_csv(r'/Users/matanb/Downloads/Unjustified Stops-data-2024-04-21 07_12_37.csv')
fresh_start = False
if fresh_start:
    last_calibrating = pd.DataFrame(columns = ['acc_id', 'date', 'stop', 'time', 'last_pow', 'last_bc_median'])
else:
    last_calibrating = pd.read_csv('last_calibrating.csv')
for acc_idx, acc_row in accounts.iterrows():
    _date = pd.to_datetime(acc_row['alert_date'])
    _date = _date + datetime.timedelta(days=1)
    mac_id = acc_row['MAC']
    acc_id = macs[macs['BaseMAC'] == mac_id]['AccountId'].values[0]
    if last_calibrating[(last_calibrating['acc_id'] == acc_id) & (last_calibrating['date'] == _date)].shape[0] > 0:
        continue
    AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date], mac_path = mac_path)
    if not hasattr(AS, 'lines'):
        continue
    df = AS.get_fields()
    try:
        last_calibrating = get_last_calibrating(df)
    except:
        continue
    last_calibrating['acc_id'] = acc_id
    last_calibrating['date'] = _date
    last_calibrating.to_csv('last_calibrating.csv', index=False, mode='a')
    print('Done:', acc_id, _date)



