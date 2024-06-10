import numpy as np
import os
from account_scraper import AccountScraper
import pandas as pd
from google.cloud import storage
import os
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/matanb/.config/gcloud/application_default_credentials.json'

# ======================================================================================================================
Trigger = 'subjectAsleep'
acc_id = 2094
_date = pd.to_datetime('2024-03-30 00:00:00')
# ======================================================================================================================

mac_path = r'/Users/matanb/Downloads/All Version Users-data-2024-03-17 08_00_50.csv'
macs = pd.read_csv(mac_path)

storage_client = storage.Client(project='nanobebe-production')

def get_video(acc_id, _date, Trigger):
    mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
    day_of_week = _date.dayofweek
    day_of_week = 6 if day_of_week == 0 else day_of_week
    prefix = f'{_date.year}/{_date.week}/{day_of_week}/{mac_id}/{Trigger}/data/'
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    found_blobs = []
    for b in blobs:
        if mac_id in b.name and Trigger in b.name:
            found_blobs.append(b)
            print(b.name)

    # get the latest video
    vid_dates = [pd.to_datetime(b.name.split('/')[-1].split('.')[0], unit='ms') for b in found_blobs]
    # get the closest timestamp
    closest_ts = np.argmin(np.abs([d.timestamp() for d in vid_dates]) - _date.timestamp())
    found_blobs = [found_blobs[closest_ts]]
    print('Closest video is at:', vid_dates[closest_ts])
    latest_blob = found_blobs[0]
    latest_blob_uri = f"gs://{latest_blob.bucket.name}/{latest_blob.name}"
    vid_file_name = latest_blob.name.replace('/','_')
    latest_blob.download_to_filename(f'./{vid_file_name}')
    print('saved to ./'+latest_blob.name)
    return latest_blob_uri

mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
# list bl
bucket_name = "production-nanobebe-video"
day_of_week = _date.dayofweek
day_of_week = 6 if day_of_week == 0 else day_of_week
prefix = f'{_date.year}/{_date.week}/{day_of_week}/{mac_id}/{Trigger}/data/'
blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
found_blobs = []
for b in blobs:
    if mac_id in b.name and Trigger in b.name:
        found_blobs.append(b)
        print(b.name)

# get the latest video
vid_dates = [pd.to_datetime(b.name.split('/')[-1].split('.')[0], unit='ms') for b in found_blobs]
# get the closest timestamp
closest_ts = np.argmin(np.abs([d.timestamp() for d in vid_dates]) - _date.timestamp())
found_blobs = [found_blobs[closest_ts]]
print('Closest video is at:', vid_dates[closest_ts])
latest_blob = found_blobs[0]
latest_blob_uri = f"gs://{latest_blob.bucket.name}/{latest_blob.name}"
vid_file_name = latest_blob.name.replace('/','_')
latest_blob.download_to_filename(f'./{vid_file_name}')
print('saved to ./'+latest_blob.name)
# ======================================================================================================================


