import re
import pandas as pd
import numpy as np
import datetime
from google.cloud import storage
import tarfile
import glob
import os
import tqdm
# example url for log
# https://storage.cloud.google.com/nanobebe-iot-logs/analysis/E062904F6151/archive-230921-000002.tgz?authuser=1
#
macs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2023-09-27 10_02_05.csv')
macs = macs['BaseMAC'].tolist()

dates = pd.date_range(start='2023-09-12', end='2023-09-27')
df = pd.DataFrame(columns=['time', 'subj', 'status'])

storage_client = storage.Client()
bucket_name = "nanobebe-iot-logs"

for mac_id in macs:
    list_files = storage_client.list_blobs(bucket_name, prefix=f'analysis/{mac_id}')
    # get all blobs
    blobs = []
    for idx, file in enumerate(list_files):
        blobs.append(file)
        pass

    for blob in blobs:

        try:
            date = pd.to_datetime(blob.name.split('/')[2].split('-')[1], format='%y%m%d')
            if date not in dates:
                continue
            # download the file
            print(date)
            blob.download_to_filename(f'./{mac_id}.tgz')

            # open the file
            try:
                tar = tarfile.open(f'./{mac_id}.tgz')
                tar.extractall()
                tar.close()
            except:
                continue

            # get the syslog txt file
            syslog_path = glob.glob(r'prev/varlog/syslog')[0]
            # open the file
            with open(syslog_path) as f:
                lines = f.readlines()

            # create a dataframe
            # loop with jumps of 10 lines
            for line in lines[::5000]:
                # get the time
                # get monmitoring mode monitoring: true
                monitoring = re.findall(r'MONITORING', line)
                if len(monitoring) == 0:
                    continue

                time = re.findall(r'\d{2}:\d{2}:\d{2}', line)
                if len(time) == 0:
                    continue
                time = time[0]
                # the day month and year are in the file name
                time = pd.to_datetime(f'{date.year}-{date.month}-{date.day} {time}')

                # get the 'subj: 1 0.65' type
                subj = re.findall(r'subj: \d \d+\.\d\d', line)

                if len(subj) == 0:
                    continue
                if subj[0].split(' ')[1] == '0':
                    print('0')
                confidence = float(re.findall(r'\d+\.\d\d', subj[0])[0])
                # add to the dataframe
                df = pd.concat([df, pd.DataFrame({'time': [time], 'subj': [confidence], 'status': [monitoring[0]]})])

                # get breathing classifier, a part starting with bc and ending woth -
                bc = re.findall(r'bc.*-', line)
                # extract the two numbers
                if len(bc) == 0:
                    continue
                bc = bc[0]
                # look for the two decimal numbers
                bc = re.findall(r'\d+\.\d\d', bc)
                if len(bc) == 0:
                    # look for one decimal number
                    bc = re.findall(r'\d+\.\d', bc)
                    if len(bc) == 0:
                        continue
                # convert to float
                print(bc)

                bc = [float(x) for x in bc]
                # add to the dataframe
                df = pd.concat([df, pd.DataFrame({'time': [time], 'subj': [confidence], 'bc1': [bc[0]], 'bc2': [bc[1]]})])

                # delete the files
                try:
                    os.remove(f'./{mac_id}.tgz')
                except:
                    continue
            print(len(df))
            if len(df) > 1000:
                break
        except:
            continue


import seaborn as sns
import matplotlib.pyplot as plt
# divide to two box plots before and after 21.9.23
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
df['version'] = df['time'].apply(lambda x: 73 if x < datetime.datetime(2023, 9, 21) else 76)
df['subj'] = df['subj'].astype(float)
sns.histplot(data=df, x='subj', hue='version', bins=20)
plt.show()

# scatter(bc1 vs subj) and scatter(bc2 vs subj)
sns.scatterplot(data=df, x='bc2', y='subj', hue='version')
plt.show()
