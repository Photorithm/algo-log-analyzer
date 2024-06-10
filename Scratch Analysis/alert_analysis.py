# disocver how many _no breathing was there before the breathing alerts
import pandas as pd
import datetime
import os
from account_scraper import AccountScraper


alerts_df = pd.read_csv('/Users/matanb/Downloads/Latest Breathing Alerts-data-2024-03-26 14_09_18.csv')
macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2024-03-24 09_14_51.csv')
# join the macs to the alerts
#alerts_df = alerts_df.merge(macs, on='AccountId', how='left')


# collect all logs to a df
output_df = pd.DataFrame()
for idx, row in alerts_df.iterrows():
    cdate = pd.to_datetime(row['LastAlertDate'])
    cdate = cdate + datetime.timedelta(days=1)
    cdate = cdate.replace(hour=0, minute=0, second=0)
    _acc = int(row['AccountId'])
    MAC = macs[macs['AccountId'] == _acc]['BaseMAC'].tolist()[0]
    AS = AccountScraper(acc_id=_acc, mac_id=MAC, dates=[cdate])
    if not hasattr(AS, 'lines'):
        continue
    df = AS.get_fields()
    temp_df = df.copy()
    temp_df['acc_id'] = _acc
    output_df = pd.concat([output_df, temp_df])


# for each account get the latest breathing micro before breathingStopped
results = pd.DataFrame()
for idx, row in alerts_df.iterrows():
    temp_df = output_df[output_df['acc_id'] == row['AccountId']]
    temp_df = temp_df[temp_df['time'] < row['LastAlertDate']]

    temp_df = temp_df[temp_df['state'] == 'calibrating']
    if temp_df.shape[0] == 0:
        continue
    latest_breathing = temp_df['time'].max()
    # latest non breathing event

    alert_time = pd.to_datetime(row['LastAlertDate'])
    #alert time is a string. convert to datetime
    alert_time = pd.to_datetime(alert_time)
    time_since_breathing = alert_time - latest_breathing

    time_since_breathing = time_since_breathing.total_seconds() / 60
    results = pd.concat([results, pd.DataFrame({'acc_id': [row['AccountId']], 'alert_date': [row['LastAlertDate']], 'time_since_breathing': [time_since_breathing]})])



# for each acc_id, check the power dsitribution before each breathingStopped
# get all the breathing stopped events
# get the power before each event
breathing_stopped = output_df[output_df['notification'].apply(lambda x: 'breathingStopped' in x if x is not None else False)]
analysis_df = pd.DataFrame()
for idx, row in breathing_stopped.iterrows():
    temp_df = output_df[output_df['acc_id'] == row['acc_id']]
    temp_df = temp_df[temp_df['time'] < row['time']- datetime.timedelta(minutes=2)]
    temp_df = temp_df[temp_df['time'] > row['time'] - datetime.timedelta(minutes=15)]
    temp_df = temp_df[temp_df['monitoring'] > 0]
    temp_df = temp_df[temp_df['state'] == 'breathing micro']
    mean_power = temp_df['pow'].mean()
    per_25 = temp_df['pow'].quantile(0.25)
    per_75 = temp_df['pow'].quantile(0.75)
    dur = temp_df['time'].max() - temp_df['time'].min()

    analysis_df = pd.concat([analysis_df, pd.DataFrame({'acc_id': [row['acc_id']], 'mean_power': [mean_power], 'per_25': [per_25],
                                                        'per_75': [per_75], 'dur': [dur]})])

# plot bar chart with error bars
import plotly.express as px
fig = px.bar(analysis_df, x='acc_id', y='mean_power', error_y='per_75', error_y_minus='per_25')
# add text with the duration of the breathing stopped event
analysis_df['dur'] = analysis_df['dur'].dt.total_seconds() / 60 / 60
analysis_df['dur'] = analysis_df['dur'].round(1)
for idx, row in analysis_df.iterrows():
    fig.add_annotation(x=row['acc_id'], y=200, text=row['dur'], showarrow=False)
fig.show('browser')