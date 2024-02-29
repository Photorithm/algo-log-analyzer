import pandas as pd
import datetime
import os
from google.cloud import bigquery
# Imports the Google Cloud client library
import matplotlib.pyplot as plt
from google.cloud import storage
import os
import pandas as pd
import cv2
import numpy as np
import datetime
from google.cloud import bigquery
import os
import tqdm

client = bigquery.Client()
dataset_id = 'mlanalysis1'
table_id = 'unjustified_stop_1'

versions = pd.read_csv(r'/Users/matanb/Downloads/Base Versions-data-2024-02-19 06_52_11.csv')
macs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2024-02-19 06_53_37.csv')
# change AccountId to accountid in macs
macs = macs.rename(columns={'AccountId': 'accountid'})
# join based on accountid = AccountId
macs = macs.merge(versions, on='accountid', how='left')

# get all macs from the last 10 days
query = f"""
  SELECT
      *
  FROM
      `mlanalysis1.unjustified_stop_1`
  WHERE
      DATE(date) > DATE_SUB(CURRENT_DATE(), INTERVAL 10 DAY)
  """
query_job = client.query(query)
results = query_job.result()  # Waits for job to complete.
df = results.to_dataframe()
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# keep only combinations between mac and date that are in the df
for idx, row in macs.iterrows():
    mac_id = row['BaseMAC']
    date = row['date']
    date = pd.to_datetime(date)
    keep = (df['MAC'] == mac_id) & (df['day'] == date.day) & (df['month'] == date.month) & (df['year'] == date.year)
    # onlyl if version is null
    keep = keep & df['version'].isnull()
    macs.at[idx, 'keep'] = keep.any()

macs = macs[macs['keep']]
macs = macs.drop(columns=['keep'])
macs = macs.drop_duplicates()
# update the version in the table


progress = tqdm.tqdm(total=len(macs))
for idx, row in macs.iterrows():
    progress.update(1)
    acc_id = row['accountid']
    MAC = row['BaseMAC']
    version = row['basefwversion']
    date = row['date']
    date = pd.to_datetime(date)

    # create query by year month and day

    query = f"""
    UPDATE
        `mlanalysis1.unjustified_stop_1`
    SET
        version = "{version}"
    WHERE
        MAC = '{MAC}'
        AND EXTRACT(DAY FROM DATE(date)) = {date.day}
        AND EXTRACT(MONTH FROM DATE(date)) = {date.month}
        AND EXTRACT(YEAR FROM DATE(date)) = {date.year}
    """
    query_job = client.query(query)
    results = query_job.result()  # Waits for job to complete.
    affected_rows = results.num_dml_affected_rows
    if affected_rows > 0:
        print('Updated ', MAC, ' with version ', version, ' on date ', date)
    else:
        print('Failed to update ', MAC, ' with version ', version, ' on date ', date)

