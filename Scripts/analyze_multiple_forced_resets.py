import pandas as pd
from account_scraper import AccountScraper
import datetime
import os
from google.cloud import bigquery
import pickle

monitoring_users = pd.read_csv('/Users/matanb/Downloads/Sleep Cycle Per User-data-2023-12-24 09_39_26.csv')
macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2023-11-19 14_31_57.csv')

acc_ids = [int(x[2:]) for x in monitoring_users.columns.tolist()[1:]]
dates = pd.date_range(start='2023-12-01', end='2023-12-23')

data = []
output_df = pd.DataFrame(columns = ['acc_id', 'time', 'reason','conf'])
# start from the last date
for _acc in acc_ids:
    for _date in dates:

        cdate = pd.to_datetime(_date)
        cdate = cdate + datetime.timedelta(days=1)
        try:
            AS = AccountScraper(acc_id=_acc, mac_id = macs[macs['AccountId'] == _acc]['BaseMAC'].tolist()[0], dates=[cdate])
            df = AS.get_algo_events('forced')

            consecutive = 0
            #  look for event rows which have the same second or second+1
            consec_blocks = {'time':[], 'duration':[]}
            for idx,row in enumerate(df.iterrows()):
                if row[1]['time'].second == df.iloc[idx-1]['time'].second or row[1]['time'].second == df.iloc[idx-1]['time'].second+1:
                    consecutive += 1
                elif consecutive > 0:
                    consec_blocks['time'].append(df.iloc[idx-1]['time'])
                    consec_blocks['duration'].append(consecutive)
                    consecutive = 0

            # ============================================================


            # ====== Add your results to the total_df ======
            temp_df = pd.DataFrame(consec_blocks)
            temp_df['acc_id'] = _acc


            output_df = pd.concat([output_df, temp_df])
            output_df.to_csv('consecutive_forced.csv', index=False)

        except Exception as e:
            print(e)
            continue

problems = output_df[output_df['duration'] > 1]
problems = problems[problems['acc_id'] != 622]

#histogram of duration. high resolution near 0-10

# example of duration > 35 and <55
sample = problems[(problems['duration'] < 500) & (problems['duration']>50) ].sample(1)

ts = sample['time'].tolist()[0]
ts = ts - datetime.timedelta(days=1)
ts_end = ts + datetime.timedelta(days=2)
# to timestamp in milliseconds
ts = int(ts.timestamp())*1000
ts_end = int(ts_end.timestamp())*1000
acc_id = sample['acc_id'].tolist()[0]
url =  f'https://grafana.nanobebe.io/d/UzDNjfSIz/connectivity-single-user?orgId=1&var-AccountId={acc_id}&var-Datasource=Timescale-Prod&var-intervalMinutes=60&var-mysqlDatasource=nanobebe-production&var-query0=&from={ts}&to={ts_end}'
print(url)
print(f'Account ID: {acc_id} \nStart time: {sample["time"].tolist()[0]} \nDuration: {sample["duration"].tolist()[0]}')


# download the log manually
# get the log from the log bucket
from google.cloud import storage
storage_client = storage.Client()
mac = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
list_files = storage_client.list_blobs('production-nanobebe-iot-logs', prefix=f'analysis/{mac}')
blobs = []
for idx, file in enumerate(list_files):
    if pd.to_datetime(file.time_created).tz_localize(None).day!=sample["time"].tolist()[0].tz_localize(None).day + 1 or pd.to_datetime(file.time_created).tz_localize(None).month!=sample["time"].tolist()[0].tz_localize(None).month:
        continue
    blobs.append(file)
    pass
blob = blobs[-1]
blob.download_to_filename(f'./{sample["acc_id"].tolist()[0]}.tgz')
print('saved to ./'+str(sample["acc_id"].tolist()[0])+'.tgz')


# titile
import matplotlib.pyplot as plt
hist = plt.hist(problems['duration'], bins=100)
plt.title('Histogram of consecutive forced events')
# x label
plt.xlabel('Consecutive Resets (frames)')
# y label
plt.ylabel('Count')
