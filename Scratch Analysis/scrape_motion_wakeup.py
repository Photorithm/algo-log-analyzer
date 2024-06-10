import pandas as pd
from account_scraper import AccountScraper
import datetime
import os
from google.cloud import bigquery
import pickle


notFound = pd.read_csv(r'/Users/matanb/Downloads/MonitoringStopped - No Measure-data-2023-12-24 18_02_08.csv')
JustifiedNoMeasure = pd.read_csv(r'/Users/matanb/Downloads/Unjustified Stops-data-2023-12-21 07_33_08.csv')
motions = pd.read_csv(r'/Users/matanb/Downloads/Motion [m]-data-2023-12-24 17_45_35.csv')


# drop zero rows from notFound
notFound = notFound[notFound['AccountID'] != 0]

wakeups = pd.DataFrame(columns=['acc_id', 'time'])
for idx, acc_id in enumerate(JustifiedNoMeasure['acc_id'].unique()):
    wakeups = pd.concat([wakeups, pd.DataFrame({'acc_id': [acc_id], 'time': [pd.to_datetime(JustifiedNoMeasure.iloc[idx]['alert_date'])]})])
for idx, _time in enumerate(notFound.iterrows()):
    wakeups = pd.concat([wakeups, pd.DataFrame({'acc_id': [_time[1]['AccountID']], 'time': [pd.to_datetime(_time[1]['Time'])]})])

macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2023-11-19 14_31_57.csv')


data = []
output_df = pd.DataFrame(columns = ['acc_id', 'time', 'reason','conf'])
# start from the last date
for wakeup_row in enumerate(wakeups.iterrows()):
    # look for motions in the 5 minutes before and 2 minutes after the wakeup
    acc_id = wakeup_row[1][1]['acc_id']
    time = wakeup_row[1][1]['time']
    time = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S')
    try:
        # get time coll and col = 'h ' + str(acc_id)
        user_motions = motions[['Time', 'h ' + str(acc_id)]]
    except:
        continue

    # get all rows that are close to this time
    _df = user_motions[(pd.to_datetime(user_motions['Time']) > time - datetime.timedelta(minutes=25)) & (pd.to_datetime(user_motions['Time']) < time + datetime.timedelta(minutes=2))]
    _df = _df[_df['h ' + str(acc_id)] != '.']
    print('user ', acc_id, ' has ', len(_df), ' motions','date: ', time)

    # count motion before wakepu
    before_wakeup = _df[pd.to_datetime(_df['Time']) < time]
    print('before wakeup: ', len(before_wakeup))
    data.append(before_wakeup)

# plot histogram of motion before wakeup
import matplotlib.pyplot as plt
plt.hist([len(x) for x in data], bins=50)
plt.xlabel('Number of motions before wakeup')
plt.ylabel('Count')
plt.title('Histogram of number of motions before wakeup')
plt.show()





# show distribution of reasons, grouped by whetehr conf was > 0.1 or not
small = []
large = []
micro = []
for element in data:
    element = element[element['state'].notnull()]
    _df = element
    micro_motion = _df[_df['state'].str.contains('micro motion')]
    micro_motion_duration = micro_motion['time'].count() / 60

    small_motion = _df[_df['state'].str.contains('small motion')]
    small_motion_duration = small_motion['time'].count() / 60

    large_motion = _df[_df['state'].str.contains('large motion')]
    large_motion_duration = large_motion['time'].count() / 60

    print('small micro diff: ',small_motion_duration - micro_motion_duration)


    small.append(small_motion_duration)
    large.append(large_motion_duration)
    micro.append(micro_motion_duration)

import matplotlib.pyplot as plt
plt.hist(small, bins=10)
plt.hist(large, bins=10)
plt.hist(micro, bins=10)
# add legend
plt.legend(['small motion', 'large motion', 'micro motion'])
plt.show()


