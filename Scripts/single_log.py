import numpy as np
import os
from account_scraper import AccountScraper
import pandas as pd



root = os.getcwd()
# step out cd to 'log_analyzer' if needed
if os.path.basename(root) != 'log_analyzer':
    os.chdir('..')
    root = os.getcwd()
mac_path = r'All Version Users-data-2024-05-23 10_08_35.csv'
macs = pd.read_csv(mac_path)


# ======================================================================================================================
acc_id = 1257
_date = pd.to_datetime('2024-05-28')
# ======================================================================================================================


mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date], mac_path = mac_path)
# AS.get_logs_from_path(r'//Users/matanb/PycharmProjects/log_analyzer/E062904F6151_2024_1_6/varlog/syslog')
df = AS.get_fields()
# AS.get_videos()
#AS.annotate_video(vid_file_path=r'/Users/matanb/Downloads/2024_12_5_E06290795733_monitoringStoppedNoMeasure_data_1711001918159.mp4',play=True)
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



import pandas as pd
large_df = pd.read_csv('large_df.csv')
accounts = large_df['acc_id'].unique()
analysis_df = pd.DataFrame(columns = ['acc_id', 'breathing', 'motion', 'calibrating', 'calibrating decision', 'no_breathing', 'breathing_visibility', 'visible_from_breathing', 'head_conf', 'bc1'])
for acc in accounts:
    df = large_df[large_df['acc_id'] == acc]
    try:
        # from df, get all rows wwhere state is breathing micro and usr_rate is -1
        # calc overall freq visibility
        monitoring = df[df['monitoring'] == True]

        breathing = len(monitoring[monitoring['state'] == 'breathing micro']) / len(monitoring)
        motion = len(monitoring[(monitoring['state'].apply(lambda x: 'motion' in x if x is not None else False))]) / len(monitoring)
        calibrating_decision = len(monitoring[(monitoring['state'] == 'calibrating decision') & (monitoring['rate'] != -1)] )/ len(monitoring)
        calibrating = len(monitoring[(monitoring['state'] == 'calibrating')| ((monitoring['state'] == 'calibrating decision') & (monitoring['rate'] == -1) )]) / len(monitoring)
        no_breathing = len(monitoring[monitoring['state'] == 'no breathing']) / len(monitoring)

        # out of breathing micro, how many have usr_rate == -1
        breathing_visibility = len(monitoring[(monitoring['state'] == 'breathing micro') & (monitoring['usr_rate'] != -1)]) / len(monitoring)
        visible_from_breathing = len(monitoring[(monitoring['state'] == 'breathing micro') & (monitoring['rate'] != -1)]) / len(monitoring[monitoring['state'] == 'breathing micro'])


        # anakyze reasong for non visible
        monitoring['confidence_head'] = monitoring['confidence_head'].ffill()
        monitoring['confidence'] = monitoring['confidence'].ffill()

        # average head conf in breathing not visible
        breathing_not_visible = monitoring[(monitoring['state'] == 'breathing micro') & (monitoring['usr_rate'] == -1)]
        head_conf = breathing_not_visible['confidence_head'].mean()
        bc1 = breathing_not_visible['bc1'].mean()

        analysis_df = pd.concat([analysis_df, pd.DataFrame({'acc_id': acc, 'breathing': breathing, 'motion': motion,
                                                            'calibrating': calibrating, 'calibrating_decision': calibrating_decision,
                                                            'no_breathing': no_breathing, 'breathing_visibility': breathing_visibility,
                                                            'visible_from_breathing': visible_from_breathing, 'head_conf': head_conf, 'bc1': bc1}, index=[0])])

    except:
        continue




breathing = sum(analysis_df['breathing']) / len(analysis_df)
motion = sum(analysis_df['motion']) / len(analysis_df)
calibrating = sum(analysis_df['calibrating']) / len(analysis_df)
calibrating_decision = sum(analysis_df['calibrating_decision']) / len(analysis_df)
no_breathing = sum(analysis_df['no_breathing']) / len(analysis_df)
import matplotlib.pyplot as plt
labels = ['breathing', 'motion', 'calibrating', 'no_breathing', 'calibrating_decision']
sizes = [breathing, motion, calibrating, no_breathing, calibrating_decision]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.show()


# histogram of breathing visibility
import matplotlib.pyplot as plt
plt.hist(analysis_df['breathing'])
plt.show()


# histogram of motion
plt.hist(analysis_df['calibrating'])
plt.show()


# drawe pie chart

# do they sum up to 1?
sum(sizes)


calibrating_visibility = len(df[(df['state'] == 'calibrating decision') & (df['monitoring']) & (df['rate'] != -1)]) / len(df['monitoring'])


df['confidence_head'] = df['confidence_head'].ffill()
df['confidence'] = df['confidence'].ffill()
freq_not_shown = df[(df['state']  == 'breathing micro') & (df['monitoring']) & (df['usr_rate'] == -1)]
# keep confidence_head, bc1
freq_not_shown = freq_not_shown[['time', 'usr_rate', 'confidence_head', 'bc1']]



