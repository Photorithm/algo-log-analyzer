import pandas as pd
from account_scraper import AccountScraper
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


client = bigquery.Client()
dataset_id = 'mlanalysis1'
table_id = 'unjustified_stop_1'

# get a list of videos to update, where version=null
def get_list():
    query = """
    SELECT
      *
    FROM
        `mlanalysis1.unjustified_stop_1`
        WHERE
        version IS NULL
        """
    client = bigquery.Client()
    query_job = client.query(query)
    results = query_job.result()  # Waits for job to complete.
    df = results.to_dataframe()
    return df

df = get_list()
# filter out today
df = df[pd.to_datetime(df['date']) < pd.to_datetime(datetime.datetime.now().date())]

# group by mac and date
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date
df = df.groupby(['MAC', 'date']).first().reset_index()


# pass over the list and update the version
#use tqdm to show progress
from tqdm import tqdm

for idx, row in tqdm(df.iterrows()):
    # get the video
    # get the log
    AS = AccountScraper(acc_id=None, mac_id=row['MAC'], dates=[pd.to_datetime(row['date']) + datetime.timedelta(days=1)])
    version = AS.get_version()

    print(version)
    break
