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

class LogRetriever:

    def __init__(self, mac_path = r'/Users/matanb/Downloads/All Version Users-data-2023-11-01 09_00_38.csv'):
        self.storage_client = storage.Client()
        self.bucket_name = "nanobebe-iot-logs"
        self.macs = pd.read_csv(mac_path)


    def get_logs(self,  acc_id ,  dates, mac_id = None, output_path = None):
        # if acc_id is char turn it into int, if it is possible. there are someweird
        # drop weird strings from acc_id
        self.macs['AccountId'] = self.macs['AccountId'].apply(lambda x: re.sub(r'\D', '', str(x)))
        self.macs['AccountId'] = [int(x) for x in self.macs['AccountId'].tolist() ]

        if mac_id is None:
            mac_id = self.macs[self.macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
        elif acc_id is None or acc_id == 0:
            if len(self.macs[self.macs['BaseMAC'] == mac_id]['AccountId'].tolist()) == 0:
                acc_id = 0
            else:
                acc_id = self.macs[self.macs['BaseMAC'] == mac_id]['AccountId'].tolist()[0]
        else:
            if len(self.macs[self.macs['BaseMAC'] == mac_id]['AccountId'].tolist()) == 0:
                acc_id = 0
            else:
                acc_id = self.macs[self.macs['BaseMAC'] == mac_id]['AccountId'].tolist()[0]

        # dates is a pd_date_Range
        # if only one date is given, convert to list
        if type(dates) == pd.Timestamp:
            dates = [dates]
        if output_path is None:
            output_paths = []
            for date in dates:
                output_paths.append(f'./{mac_id}_{date.year}_{date.month}_{date.day}')
        list_files = self.storage_client.list_blobs(self.bucket_name, prefix=f'analysis/{mac_id}')
        # if no blobs at all, try the new bucket
        #if len(list(list_files)) == 0:
        # get all blobs
        blobs = []
        log_path_list = []
        for idx, file in enumerate(list_files):
            blobs.append(file)
            pass
        # if no blobs at all, try the new bucket
        list_files = self.storage_client.list_blobs('production-nanobebe-iot-logs', prefix=f'analysis/{mac_id}')
        for idx, file in enumerate(list_files):
            blobs.append(file)
            pass


        relevant_blobs = []
        output_paths = []
        for date in dates:
            for blob in blobs:
                if pd.to_datetime(blob.time_created).tz_localize(None).day!=date.tz_localize(None).day or pd.to_datetime(blob.time_created).tz_localize(None).month!=date.tz_localize(None).month:
                    continue
                relevant_blobs.append(blob)
                output_paths.append(f'./{mac_id}_{blob.time_created}')
        for i, blob in enumerate(tqdm.tqdm(relevant_blobs)):
            # download the file
            # check if the file exists
            filename = f'./temp/{mac_id}_{blob.time_created}.tgz'
            if ~os.path.exists(filename):
                blob.download_to_filename(filename)
            # open the file
            tar = tarfile.open(filename)
            tar.extractall()
            tar.close()
            # move the prev folder inside the output path
            # check if directory exists
            if not os.path.exists(output_paths[i]):
                os.rename('prev',  output_paths[i])
            log_path_list.append(glob.glob(output_paths[i] + '/varlog/syslog')[-1])

        return log_path_list

    def read_log(self, log_path):
        # open the log  file
        with open(log_path) as f:
            lines = f.readlines()
        for line in lines:
            # time and date. date is month MMM %d
            try:
                clock = re.findall(r'\d{2}:\d{2}:\d{2}', line)[0]
                time = pd.to_datetime(str(pd.to_datetime('today').year) + ' ' + line[:6] + ' ' + clock)
            except:
                continue

            line_splitted = re.split(r'\t+', line)
        return lines



