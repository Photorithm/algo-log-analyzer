from account_scraper import AccountScraper
import pandas as pd
import numpy as np
import datetime
import os

macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2023-11-19 14_31_57.csv')

users = [299,208,182,812,179,629,383]
_date = pd.to_datetime('2024-01-01')
for acc_id in users:
    mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]


    AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date])
    df = AS.get_fields()
    notifs = df[df['notification'].apply(lambda x: (str(x).lower() != 'nan') & (str(x).lower() != 'none'))]
    notifs = notifs[notifs['notification'].apply(lambda x: 'nomeasure' in str(x).lower())]
    for idx, row in notifs.iterrows():
        _time = row['time']
        _time = pd.to_datetime(_time)

        latest_state = _df[_df['time'] < _time]['state'].tolist()
        # take last non none vlue
        latest_state = latest_state[latest_state != 'None']
        print(latest_state)

        # see id it was critical or not
        _df = df[(pd.to_datetime(df['time']) > _time - datetime.timedelta(minutes=1)) & (pd.to_datetime(df['time']) < _time + datetime.timedelta(minutes=1))]
        print(len(_df[_df['is_critical'] ==True]))
        print(np.max(_df['bc_buffer']))


users = [299,208,182,812,179,629,383]
users = [789,416]
_date = pd.to_datetime('2024-01-01')
for acc_id in users:
    mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]


    AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date])

    try:
        df = AS.get_fields()

        notifs = df[df['notification'].apply(lambda x: (str(x).lower() != 'nan') & (str(x).lower() != 'none'))]
        notifs = notifs[notifs['notification'].apply(lambda x: 'nomeasure' in str(x).lower())]
        for idx, row in notifs.iterrows():
            _time = row['time']
            _time = pd.to_datetime(_time)

            latest_state = df[(df['time'] < _time) & (df['time'] > _time - datetime.timedelta(minutes=3))]
            # drop nones
            latest_state = latest_state[latest_state['state'].apply(lambda x: str(x).lower() != 'none')]
            last_calibrating_decision = latest_state[latest_state['state'].apply(lambda x: 'calibrating' in str(x).lower())]
            # get last calibrating decision
            latest_state = latest_state['state'].tolist()[-1]
            if latest_state == 'no breathing':
                bc_last = last_calibrating_decision['bc1'].tolist()[-1]
                print(bc_last)
    except:
        pass

