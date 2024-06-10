import pandas as pd
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

macs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2024-03-24 09_14_51.csv')

query = """
    SELECT
      *
    FROM
        `mlanalysis1.unjustified_stop_1`
        WHERE
        acc_id = 0
        """
client = bigquery.Client()
query_job = client.query(query)
results = query_job.result()  # Waits for job to complete.
df = results.to_dataframe()
# keep only one row per mac
df = df.groupby('MAC').first().reset_index()


# order macs by accoutn id descending
macs = macs.sort_values(by='AccountId', ascending=False)
progress = tqdm.tqdm(total=len(macs))
for idx, row in df.iterrows():
    progress.update(1)
    acc = 0
    mac_id = row['MAC']
    acc = macs[macs['BaseMAC'] == mac_id]
    if len(acc) == 0:
        print(f'No account for {mac_id}')
        continue
    print(f'Found account for {mac_id}')
    acc = acc.iloc[0]['AccountId']


    query = f"""
    UPDATE `mlanalysis1.unjustified_stop_1`
    SET acc_id = {acc}
    WHERE MAC = '{mac_id}'
    """
    query_job = client.query(query)
    results = query_job.result()  # Waits for job to complete.
    print(f'Updated {mac_id} with {acc}', 'total ', results.total_rows, ' rows updated')


