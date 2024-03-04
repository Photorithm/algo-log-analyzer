import numpy as np
import os
from account_scraper import AccountScraper
import pandas as pd



root = os.getcwd()
# step out cd to 'log_analyzer' if needed
if os.path.basename(root) != 'algo-log-analyzer':
    os.chdir('..')
    root = os.getcwd()
mac_path = os.path.join(root, 'All Version Users-data-2024-02-25 09_09_31.csv')
macs = pd.read_csv(mac_path)


# ======================================================================================================================
acc_id = 1976
_date = pd.to_datetime('2024-02-11')
# ======================================================================================================================


mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date], mac_path = mac_path)
# AS.get_logs_from_path(r'//Users/matanb/PycharmProjects/log_analyzer/E062904F6151_2024_1_6/varlog/syslog')
df = AS.get_fields()
# AS.get_videos()
# AS.annotate_video(vid_file_path=None,play=True)
AS.plot_fields(classifier_breathing_only=False, save=str(acc_id)+'_'+str(_date.date())+'.html')
events = AS.get_algo_events(event = 'prevented')

reason = AS.resolve_reason()
df = AS.get_algo_events('detector_inference')

# =========== downnload manually
from google.cloud import storage
storage_client = storage.Client()
list_files = storage_client.list_blobs('production-nanobebe-iot-logs', prefix=f'analysis/{mac_id}')
# download the file modified in 24-11

blobs = []
for idx, file in enumerate(list_files):
    if pd.to_datetime(file.time_created).tz_localize(None).day!=_date.tz_localize(None).day or pd.to_datetime(file.time_created).tz_localize(None).month!=_date.tz_localize(None).month:
        continue
    blobs.append(file)
    pass

blob = blobs[-1]
blob_uri = f"gs://{blob.bucket.name}/{blob.name}"
blob.download_to_filename(f'./{mac_id}.tgz')
print('saved to ./'+mac_id+'.tgz')


