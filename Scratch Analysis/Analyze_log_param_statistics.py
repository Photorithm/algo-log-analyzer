import pandas as pd
import plotly.express as px
import numpy as np
import datetime



# read a concantnated log field ran by the accountscraoer
tot_df = pd.read_csv(r'/Users/matanb/Downloads/total_df.csv')

# filter by state
tot_df = tot_df[tot_df.state == 'breathing micro']

# Histogram
fig = px.histogram(tot_df, x='confidence', nbins=50, histnorm='probability density')
fig.update_layout(title_text='Detecting confidence in breathing micro')
fig.show('browser')

# CDF
conf_vals = tot_df['confidence'].tolist()
conf_vals = [x for x in conf_vals if str(x) != 'nan']
hist_vlas = np.histogram(conf_vals, bins=100)
cdf = np.cumsum(hist_vlas[0]) / np.sum(hist_vlas[0])
fig = px.line(x=hist_vlas[1][:-1], y=cdf)
fig.show('browser')


# get only rows fwithout nan notiofication - allows manual inspection of videos
non_nan = tot_df[tot_df['notification'].apply(lambda x:  str(x) != 'nan')]
new_df = pd.DataFrame()
for row in non_nan.iterrows():
    # get the time
    time = row[1]['time']
    time = pd.to_datetime(time)
    # get all rows that are close to this time
    _df = tot_df[(pd.to_datetime(tot_df['time']) > time - datetime.timedelta(minutes=2)) & (pd.to_datetime(tot_df['time']) < time + datetime.timedelta(minutes=2))]
    new_df = pd.concat([new_df, _df])


# group by account id and hour, keep first
new_df['time'] = pd.to_datetime(new_df['time'])
# now group by day, keep pnly the first one with conf < 0.6
new_df = new_df.groupby(['acc_id', pd.Grouper(key='time', freq='D')]).first().reset_index()


# ====== do this part over and over
# plot sample withrelevant info - acc id, date and time
samp = new_df[(new_df['confidence'] < 0.6) & (new_df['state'] == 'breathing micro')].sample(1)
acc_id = samp['acc_id'].tolist()[0]
date = samp['time'].tolist()[0].split(' ')[1]
time = samp['time'].tolist()[0]
conf = samp['confidence'].tolist()[0]
#closest notification
closest_notification = non_nan[non_nan['acc_id'] == acc_id]
closest_notification = closest_notification.iloc[[np.argmin(np.abs(pd.to_datetime(closest_notification['time']) - pd.to_datetime(time)))]]
print(f'acc_id: {acc_id}, date: {time},  conf: {conf} , closest notification: {closest_notification["notification"].tolist()[0]}')



# plot the realtion between bc1 and confidence, in breathing state
fig = px.scatter(tot_df[tot_df['state'] == 'breathing micro'], x='bc1', y='confidence')
fig.show('browser')


