import pandas as pd
from account_scraper import AccountScraper
import datetime
import os
from google.cloud import bigquery
import pickle

monitoring_users = pd.read_csv(r'/Users/matanb/Downloads/Unjustified stops - # users _ Monitoring Users-data-2024-02-05 06_47_10.csv')
macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2023-11-19 14_31_57.csv')
monitoring_stops = pd.read_csv(r'/Users/matanb/Downloads/Unjustified Stops-data-2024-02-12 10_18_04.csv')


acc_ids = monitoring_users['acc'].unique()
dates = pd.date_range(start='2023-01-26', end='2023-01-30')
rotations = [270, 90, -270, -90, 180, -1, 1, -180]
augmentations = [[270,0], [90,0], [270,1], [90,1]]


data = []
output_df = pd.DataFrame(columns = ['acc_id', 'time', 'reason','conf'])
# start from the last date
for _acc in acc_ids:
    for _date in dates:

        cdate = pd.to_datetime(_date)
        cdate = cdate + datetime.timedelta(days=1)
        try:
            AS = AccountScraper(acc_id=_acc, mac_id = macs[macs['AccountId'] == _acc]['BaseMAC'].tolist()[0], dates=[cdate])
            df = AS.get_algo_events('detector_inference')
            version = AS.get_version()

            # ============================================================
            for idx, row in df.iterrows():
                ctx = row['algo_event_ctx']
                #Out[27]: '"{"aug": 0, "conf": 0.103581, "rect": [323, 22, 641, 642], "invalid": 0}", "pid": 1739730'
                # extract aug and invalid
                # aug = int(ctx.split('aug": ')[1].split(',')[0])
                try:
                    invalid = int(ctx.split('invalid": ')[1].split('}')[0])
                except:
                    invalid = 1
                # data.append({'aug': aug, 'invalid': invalid})

                # old version
                flip = int(ctx.split('flp": ')[1].split(',')[0])
                rotation = int(ctx.split('rot": ')[1].split(',')[0])
                conf = float(ctx.split('conf": ')[1].split(',')[0])
                if conf < 0.4:
                    continue
                aug = augmentations.index([rotation, flip])
                data.append({'aug': aug, 'invalid': invalid})



            # ====== Add your results to the total_df ======
            temp_df = pd.DataFrame(data)
            temp_df['acc_id'] = _acc
            temp_df['time'] = cdate
            temp_df['version'] = version['version'].values[0]


            output_df = pd.concat([output_df, temp_df])
            output_df.to_csv('augmentatiuons.csv', index=False)

        except Exception as e:
            print(e)
            continue

       # plot histogram of aug and invalid. color is invalid


import plotly.express as px
import pandas as pd
df2 = pd.read_csv(r'/Users/matanb/PycharmProjects/log_analyzer/augmentatiuons.csv')
df2 = df2[df2['version'] != '0.13.44']
df2 = df2[df2['version'] != '0.13.45']

fig = px.histogram(df2, x="aug", color="invalid", histnorm='percent')
fig.show(renderer='browser')
