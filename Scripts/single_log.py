import numpy as np

from account_scraper import AccountScraper
import pandas as pd
macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2024-02-19 06_53_37.csv')


# ======================================================================================================================
acc_id = 736
_date = pd.to_datetime('2024-02-28')
# ======================================================================================================================


mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date])
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




import re
conf_pattern = r'"confidence": (\d+\.\d+)'

# get cofidence distribution for 1429
dates = pd.date_range(start='2024-01-21', end='2024-01-31')
confidence = []
for _date in dates:
    AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date])
    df = AS.get_fields()
    # go to breathing micro and get the latest confidence which is above 0.4

    # get all the timestamps where state changed fro, soemthing to breathin gmicro
    transition_to_breathing = df[(df['state'] == 'breathing micro') & (df['state'].shift(1) != 'breathing micro')]


    # search in df for the latest >0.4 confidence
    _df = AS.get_algo_events('detector_inference')
    _df['conf'] = _df['algo_event_ctx'].apply(lambda x: re.findall(conf_pattern, x))
    _df['conf'] =_df['conf'].apply(lambda x: [float(x) for x in x])
    _df['conf'] =_df['conf'].apply(lambda x: x[0])
    _df = _df[_df['conf'] > 0.4]
    for idx, row in transition_to_breathing.iterrows():
        conf = _df[_df['time'] > row['time']]['conf']
        if len(conf) > 0:
            confidence.append(conf.values[0])
        else:
            confidence.append(np.nan)

# plot histogram
import matplotlib.pyplot as plt
plt.hist(confidence, bins=100)
# axis and title
plt.xlabel('confidence')
plt.ylabel('count')
plt.title('confidence distribution for 1429 (transition to brteathing micro)')

# print when confidence was below 0.5
df = AS.get_algo_events('detector_inference')

