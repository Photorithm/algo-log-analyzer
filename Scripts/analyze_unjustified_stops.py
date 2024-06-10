# disocver how many _no breathing was there before the breathing alerts
import pandas as pd
import datetime
import os
from account_scraper import AccountScraper


alerts = pd.read_csv('/Users/matanb/Downloads/Unjustified Stops-data-2024-03-11 06_34_44.csv')

alerts['alert_date'] = pd.to_datetime(alerts['alert_date'])
# get only alerts btween yesterday and 2 days ago
alerts = alerts[alerts['alert_date'] > datetime.datetime.now() - datetime.timedelta(days=3)]
alerts = alerts[alerts['alert_date'] < datetime.datetime.now() - datetime.timedelta(days=1)]
# keep only once every mac
alerts = alerts.drop_duplicates(subset=['acc_id', 'MAC'])
macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2024-02-13 14_45_25.csv')
macs = macs.rename(columns={'AccountId': 'accountid'})


# collect all logs to a df
output_df = pd.DataFrame()
for idx, row in alerts.iterrows():
    cdate = pd.to_datetime(row['alert_date'])
    cdate = cdate + datetime.timedelta(days=1)
    _acc = int(row['acc_id'])
    MAC = row['MAC']
    AS = AccountScraper(acc_id=_acc, mac_id=MAC, dates=[cdate])
    if not hasattr(AS, 'lines'):
        continue
    df = AS.get_fields()
    temp_df = df.copy()
    temp_df['acc_id'] = _acc
    temp_df['MAC'] = MAC
    output_df = pd.concat([output_df, temp_df])



# for each acc_id, check the power dsitribution before each breathingStopped
no_measure = output_df[output_df['notification'].apply(lambda x: 'NoMeasure' in x if x is not None else False)]
analysis_df = pd.DataFrame()
i=0
for idx, row in no_measure.iterrows():
    i += 1
    # if i==2:
    #     break
    temp_df = output_df[(output_df['MAC'] == row['MAC'])]
    post_df = temp_df[temp_df['time'] > row['time']]
    temp_df = temp_df[temp_df['time'] > row['time']- datetime.timedelta(minutes=60)]
    temp_df = temp_df[temp_df['time'] < row['time'] + datetime.timedelta(minutes=1)]
    temp_df['confidence_head'] = temp_df['confidence_head'].ffill()
    # when was the lastest monitoringn notificaiton
    ons = temp_df[temp_df['notification'].apply(lambda x: 'monitoringOn' in x if x is not None else False)]
    if ons.shape[0] == 0:
        time_since_on = 10000
    else:
        ons = ons['time'].max()
        time_since_on = row['time'] - ons
        time_since_on = time_since_on.total_seconds() / 60

    head_conf = temp_df['confidence_head'].values[-1]

    # when was the latest breathin micro state?
    breathing_micro = temp_df[temp_df['state'] == 'breathing micro']
    if breathing_micro.shape[0] == 0:
        time_since_breathing = 0
    else:
        breathing_micro = breathing_micro['time'].max()
        time_since_breathing = row['time'] - breathing_micro
        time_since_breathing = time_since_breathing.total_seconds() / 60

    next_breathing = post_df[post_df['state'] == 'breathing micro']
    if next_breathing.shape[0] == 0:
        next_breathing = 1000
    else:
        next_breathing = next_breathing['time'].min()
        next_breathing = next_breathing - row['time']
        next_breathing = next_breathing.total_seconds() / 60

    # latest_classifier_counter
    latest_classifier = temp_df['bc_buffer'].values[-1]

    analysis_df = pd.concat([analysis_df, pd.DataFrame({'MAC': [row['MAC']],
                'date': [row['time']],
                'time_since_on': [time_since_on],
                'time_since_breathing': [time_since_breathing],
                'head_conf': [head_conf],
                'next_breathing': [next_breathing],
                'latest_classifier': [latest_classifier]
                })])

temp_df = analysis_df.copy()
temp_df = temp_df[temp_df['head_conf'] > 0.7]

# plot scatter of  time_since_breathing vs lastest_classifier
import seaborn as sns
sns.scatterplot(x='time_since_breathing', y='latest_classifier', data=temp_df)
