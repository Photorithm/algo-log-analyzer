import pandas as pd
from account_scraper import AccountScraper
import datetime
import os
from google.cloud import bigquery
import pickle

monitoring_users = pd.read_csv(r'/Users/matanb/Downloads/Breathing Issue-data-2024-03-04 10_14_57.csv')
macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2024-02-13 14_45_25.csv')
monitoring_stops = pd.read_csv(r'/Users/matanb/Downloads/Unjustified Stops-data-2024-03-21 07_11_50.csv')


acc_ids = monitoring_stops['acc_id'].unique()
dates = pd.date_range(start='2023-03-1', end='2023-03-4')



data = []
output_df = pd.DataFrame(columns = ['acc_id', 'time', 'reason','conf'])
# start from the last date
for _acc in acc_ids:
    for _date in dates:

        cdate = pd.to_datetime(_date)
        cdate = cdate + datetime.timedelta(days=1)
        try:
            AS = AccountScraper(acc_id=_acc, mac_id = macs[macs['AccountId'] == _acc]['BaseMAC'].tolist()[0], dates=[cdate])
            df = AS.get_fields()
            df = df[df['monitoring'] > 0  ]
            df = df[df['state'] == 'breathing micro']

            breathing_stopped = df[
                df['notification'].apply(lambda x: 'breathingStopped' in x if x is not None else False)]
            for bs_id, bs_row in breathing_stopped.iterrows():
                temp_df = df[(df['time'] < bs_row['time']) & (df['time'] > bs_row['time'] - datetime.timedelta(minutes=3))]
                mean_power = temp_df['pow'].mean()
                per_25 = temp_df['pow'].quantile(0.25)
                per_75 = temp_df['pow'].quantile(0.75)
                dur = temp_df['time'].max() - temp_df['time'].min()

                data.append({'acc_id': _acc, 'mean_power': mean_power, 'per_25': per_25, 'per_75': per_75, 'dur': dur})

            power = df['pow']



            # ====== Add your results to the total_df ======
            temp_df = pd.DataFrame()
            temp_df['power'] = power
            temp_df['acc_id'] = _acc
            temp_df['time'] = cdate


            output_df = pd.concat([output_df, temp_df])
            output_df.to_csv('power.csv', index=False)

        except Exception as e:
            print(e)
            continue

       # plot histogram of aug and invalid. color is invalid


import plotly.express as px
import pandas as pd
df2 = pd.read_csv(r'/Users/matanb/PycharmProjects/log_analyzer/power.csv')
# histogram colored by acc id
fig = px.histogram(df2, x="power", color="acc_id", marginal="rug",
                   hover_data=df2.columns,histnorm='percent')
fig.show('browser')

# 25th percentile per acc
per = df2[['acc_id','power']].groupby('acc_id').quantile(0.25)
# clip the power to 0-100
per['power'] = per['power'].clip(0,10000)
# bar plot
fig = px.bar(per, x=per.index, y='power')
fig.show('browser')