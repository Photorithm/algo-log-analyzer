import datetime
import numpy as np
import pandas as pd
from account_scraper import AccountScraper

macs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2023-11-01 09_00_38.csv')
unjustified_stops = pd.read_csv(r'/Users/matanb/Downloads/Notification Analysis_2023-11-13_unjustified_nomeasure.csv')

monitoring_users = [1055,1028,940,923,950,1045,1011,912,880,978,927,1032]
monitoring_users = [880]
dates = pd.date_range(start='2023-11-12', end='2023-11-14')
total_reasons = pd.DataFrame(columns=['acc_id', 'time', 'reason'])
for acc_id in monitoring_users:
    for date in dates:
        try:
            AS = AccountScraper(acc_id=acc_id,  dates=[date])
            df = AS.get_fields()
            reason = AS.resolve_reason()
            # add acc_id and time
            reason['acc_id'] = acc_id
            total_reasons = pd.concat([total_reasons, reason])
            total_reasons.to_csv(r'/Users/matanb/Downloads/total_reasons.csv', index=False)


        except:
            continue

# for each of the rows, see if there is a row in unjustified stops that is close to it, and same acc_id
unjustified_times = unjustified_stops['date'].tolist()
unjustified_times = [pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S') for x in unjustified_times]
for re in total_reasons.iterrows():
    # get the time
    time = re[1]['time']
    acc_id = re[1]['acc_id']
    _unjustified_stops = unjustified_stops[unjustified_stops['acc_id'] == acc_id]
    unjustified_times = unjustified_stops['date'].tolist()
    unjustified_times = [pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S') for x in unjustified_times]
    if len(_unjustified_stops) == 0:
        total_reasons['reason'][re[0]] = 'other'
        print('no unjustified stops')
        continue
    closest_time = min(unjustified_times, key=lambda x: abs(x - time))
    if abs(closest_time - time) < datetime.timedelta(minutes=5):
        total_reasons['reason'][re[0]] = 'unjustified_stop'
    else:
        total_reasons['reason'][re[0]] = 'other'



# ======================================================================================================================
# Analyze the unjsutiifed measure csv
# ======================================================================================================================


df = pd.read_csv(r'/Users/matanb/Downloads/Notification Analysis_2023-11-14_unjustified_nomeasure.csv')
# data addin
missing_daat = {'E06290517873':  977, 'E062906E7072': 956,
'E0629072574B':1045, 'E06290756A76':1044, 'E06290595130':998,
'E062902B7074':895,
'E06290333454':962,
'E06290363067':1028,
'E0629039316D':1032,
'E062904C4559':1015,
'E06290506F53':978}


# GEt only account =0  and >850
df = df[(df['acc_id'] == 0) | (df['acc_id'] > 850)]
df = df[df.Trigger == 'monitoringStoppedNoMeasure']
df = df[df['min_conf']>0.5]

# add missing data
for mac in missing_daat.keys():
    df['acc_id'][df['MAC'] == mac] = missing_daat[mac]


# for each mac, if there are two rows closer in time thean 2 minutes, take the first
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
df = df.sort_values(by='date')
df = df.reset_index(drop=True)
new_df = pd.DataFrame(columns=df.columns)
for row in df.iterrows():
    if len(new_df) == 0:
        new_df = pd.concat([new_df, row[1].to_frame().transpose()])
        continue
    # closest in df
    same_mac = df[df['MAC'] == row[1]['MAC']]
    # a new df without the current row
    if len(same_mac) == 1:
        new_df = pd.concat([new_df, row[1].to_frame().transpose()])
        continue
    same_mac = same_mac[same_mac['date'] != row[1]['date']]
    closest = min(same_mac['date'], key=lambda x: abs(x - row[1]['date']))
    if abs(closest - row[1]['date']) > datetime.timedelta(minutes=2):
        new_df = pd.concat([new_df, row[1].to_frame().transpose()])
        continue


print('Users affected:'  + str(len(df['MAC'].unique().tolist())))
users_with_more_than_1 = df.groupby('MAC').count()
users_with_more_than_1 = users_with_more_than_1[users_with_more_than_1['acc_id'] > 1]
# get macs and account ids
macs = users_with_more_than_1.index.tolist()
acc_ids = []
for mac in macs:
    acc_ids.append(df[df['MAC'] == mac]['acc_id'].tolist()[0])
# print acc_id and number of times
for acc_id in acc_ids:
    print(acc_id, len(df[df['acc_id'] == acc_id]))
