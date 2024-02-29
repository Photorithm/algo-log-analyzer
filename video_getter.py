import re
import pandas as pd
import numpy as np
import datetime
from google.cloud import storage
import tarfile
import glob
import os
import tqdm
import datetime
from detectors import BabyDetector
import cv2

macs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2023-11-19 14_31_57.csv')
bd = BabyDetector(r'/Users/matanb/Downloads/yolov8x_beta_merged_320320_map0.972_recall0.92.onnx')


# ====== do this part over and over
# plot sample withrelevant info - acc id, date and time
samps = new_df[(new_df['confidence'] < 0.6) & (new_df['state'] == 'breathing micro')]

breathing_blanket_output = pd.DataFrame(columns=['acc_id', 'time', 'confidence', 'trigger', 'video_path', 'x_conf'])
for samp in samps.iterrows():
    samp = samp[1]
    acc_id = samp['acc_id']
    date = samp['time']
    time = samp['time']
    conf = samp['confidence']
    #closest notification
    closest_notification = non_nan[non_nan['acc_id'] == acc_id]
    closest_notification = closest_notification.iloc[[np.argmin(np.abs(pd.to_datetime(closest_notification['time']) - pd.to_datetime(time)))]]
    #  downalods videos from the production-nanobebe-video bucket

    # first try in script
    trigger = closest_notification['notification'].tolist()[0]
    trigger = trigger[2:-1]
    _date = pd.to_datetime(closest_notification['time'].tolist()[0])
    # get the number of week and dat
    week = _date.week
    day = _date.dayofweek
    year = _date.year

    # this is an example of a link to a video 2022/45/4/E06290545433/subjectawake/data/1667976989472.mp4
    # the date is 2022/45/4 and the iflenamee is a timestamp. get the closeset time stamp to the one given in the date
    try:
        mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
        vid_path = f'{year}/{week}/{day}/{mac_id}/{trigger}/data/'
    except:
        continue


    storage_client = storage.Client()
    # the workspace is nanobebe-production

    bucket_name = "production-nanobebe-video"
    blobs = storage_client.list_blobs(bucket_name, prefix=vid_path)
    vid_nominations = []
    ts = []
    for b in blobs:
        request_ts = _date.timestamp()
        blob_ts = int(b.name.split('/')[-1].split('.')[0])
        ts.append(blob_ts)
        vid_nominations.append(b)
    # get the index of the closest timestamp
    if len(ts) == 0:
        print('no videos found in ' + vid_path)
        continue
    idx = np.argmin(np.abs(np.array(ts) - request_ts))
    # get the video
    blob = vid_nominations[idx]
    blob.download_to_filename(f'./{blob.name.split("/")[-1]}')

    cap = cv2.VideoCapture(f'./{blob.name.split("/")[-1]}')
    ret, frame = cap.read()

    det = bd(frame)
    conf = det[1]
    # add to the dataframe
    breathing_blanket_output = pd.concat([breathing_blanket_output, pd.DataFrame([[acc_id, time, conf, trigger, f'./{blob.name.split("/")[-1]}',  conf]], columns=breathing_blanket_output.columns)])
    breathing_blanket_output.to_csv(r'breathing_blanket_output.csv')

    #delete the video
    os.remove(f'./{blob.name.split("/")[-1]}')

