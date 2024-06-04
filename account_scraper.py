import pandas as pd
from log_retriever import LogRetriever
import re
import tqdm
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import subprocess
import datetime
import cv2
import json

# get video title
import os
from google.cloud import storage
from datetime import date as tdate


def draw_curved_rectangle(
    frame, x, y, w, h, corner_radius, thickness=2, color=(128, 128, 0)
):
    """
    Draws the outline of a curved rectangle on the given frame.

    Parameters:
    - frame: The image frame on which to draw.
    - x, y: The top-left corner coordinates of the rectangle.
    - w: The width of the rectangle.
    - h: The height of the rectangle.
    - corner_radius: The radius of the curved corners.
    - thickness: The thickness of the lines.
    - color: The color of the lines as a tuple (B, G, R).
    """
    # Ensure corner_radius is not greater than half the width or height
    corner_radius = min(corner_radius, w // 2, h // 2)

    # Calculate intermediate points
    p1 = (x + corner_radius, y)
    p2 = (x + w - corner_radius, y)
    p3 = (x + w, y + corner_radius)
    p4 = (x + w, y + h - corner_radius)
    p5 = (x + w - corner_radius, y + h)
    p6 = (x + corner_radius, y + h)
    p7 = (x, y + h - corner_radius)
    p8 = (x, y + corner_radius)

    # Draw straight lines
    cv2.line(frame, p1, p2, color, thickness)
    cv2.line(frame, p3, p4, color, thickness)
    cv2.line(frame, p5, p6, color, thickness)
    cv2.line(frame, p7, p8, color, thickness)

    # Draw arcs for the corners
    cv2.ellipse(
        frame,
        (x + corner_radius, y + corner_radius),
        (corner_radius, corner_radius),
        180,
        0,
        90,
        color,
        thickness,
    )
    cv2.ellipse(
        frame,
        (x + w - corner_radius, y + corner_radius),
        (corner_radius, corner_radius),
        270,
        0,
        90,
        color,
        thickness,
    )
    cv2.ellipse(
        frame,
        (x + w - corner_radius, y + h - corner_radius),
        (corner_radius, corner_radius),
        0,
        0,
        90,
        color,
        thickness,
    )
    cv2.ellipse(
        frame,
        (x + corner_radius, y + h - corner_radius),
        (corner_radius, corner_radius),
        90,
        0,
        90,
        color,
        thickness,
    )

    return frame


def get_week(date):
    return date.isocalendar()[1]


def weekDay(year, month, day):
    offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    week = [
        'Sunday',
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
    ]
    afterFeb = 1
    if month > 2:
        afterFeb = 0
    aux = year - 1700 - afterFeb
    # dayOfWeek for 1700/1/1 = 5, Friday
    dayOfWeek = 5
    # partial sum of days betweem current date and 1700/1/1
    dayOfWeek += (aux + afterFeb) * 365
    # leap year correction
    dayOfWeek += aux / 4 - aux / 100 + (aux + 100) / 400
    # sum monthly and day offsets
    dayOfWeek += offset[month - 1] + (day - 1)
    dayOfWeek %= 7
    return dayOfWeek, week[dayOfWeek]


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    '/Users/shirmilstein/.config/gcloud/application_default_credentials.json'
)
storage_client = storage.Client(project='nanobebe-production')


def figures_to_html(figs, filename="dashboard.html"):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")


def open_html(filename="dashboard.html"):
    webbrowser.open_new_tab(filename)


def get_packet_id(file_path):
    # Command to extract metadata using FFmpeg
    cmd = [
        'ffprobe',
        '-v',
        'quiet',
        '-print_format',
        'json',
        '-show_format',
        file_path,
    ]
    try:
        # Execute the command
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Parse the output to JSON
        metadata = json.loads(result.stdout)

        # Accessing the 'tags' field and then 'title' if available
        title = (
            metadata.get('format', {})
            .get('tags', {})
            .get('title', 'Title not found')
        )
        return title
    except Exception as e:
        print(f"Error getting video title: {e}")
        return None


class AccountScraper:

    def __init__(self, dates=None, mac_path=None, mac_id=None, acc_id=None):
        root = os.getcwd()
        mac_path = os.path.join(
            root, 'All Version Users-data-2024-02-25 09_09_31.csv'
        )
        macs = pd.read_csv(mac_path)
        if mac_id is not None and acc_id is not None:
            macs = pd.concat(
                [
                    macs,
                    pd.DataFrame(
                        [[acc_id, mac_id, mac_id]],
                        columns=['AccountId', 'CamMAC', 'BaseMAC'],
                    ),
                ]
            )
            macs.to_csv(mac_path, index=False)

        self.mac_id = mac_id
        self.acc_id = acc_id
        # self.acc_id = macs[macs['BaseMAC'] == mac_id]['AccountId'].tolist()[0]

        if dates is None:
            dates = pd.date_range(
                start=pd.to_datetime('today') - pd.Timedelta(days=2),
                end=pd.to_datetime('today'),
            )
        self.log_paths = self.download_logs(dates)
        self.date = dates[0].strftime('%Y-%m-%d')

    def download_logs(self, dates):

        if self.acc_id is None:
            self.acc_id = 0
        log_path = LogRetriever().get_logs(
            self.acc_id, dates, mac_id=self.mac_id
        )
        total_lines = []
        for log_path_ in log_path:
            with open(log_path_) as f:
                lines = f.readlines()
            # remove both extracted and tar files
            if 'log_analyzer' not in '/'.join(log_path_.split('/')[:-2]):
                subprocess.run(
                    ['rm', '-rf', '/'.join(log_path_.split('/')[:-2])]
                )
                tar_path = log_path_.split('.')[0] + '.tgz'
                subprocess.run(['rm', '-rf', tar_path])
            total_lines = total_lines + lines

        if len(total_lines) == 0:
            print('no logs')
            return None
        else:
            self.lines = total_lines

        return log_path

    def get_logs_from_path(self, log_path):
        with open(log_path) as f:
            lines = f.readlines()

        self.acc_id = 0  # fix weith macs pd
        self.acc_id = log_path.split('/')[-3].split('-')[0]
        self.date = ' '

        if len(lines) == 0:
            print('no logs')
            return None
        else:
            self.lines = lines
        return log_path

    def get_breathing_rate(self, monitoring=True):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Open the log file and read it line by line
        with open(self.log_paths[0]) as f:
            for line in tqdm.tqdm(f):
                if monitoring and 'MONITORING' not in line:
                    continue

                # Extract time and date
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    timestamp = timestamp_match.group()
                    date = line[:6]
                    time = f"{pd.to_datetime('today').year} {date} {timestamp}"

                if 'MONITORING' in line:
                    line_splitted = re.split(r'\t+', line)

                    # Find the index containing 'rate:'
                    rate_index = next(
                        (
                            i
                            for i, s in enumerate(line_splitted)
                            if 'rate:' in s
                        ),
                        None,
                    )

                    if rate_index is not None:
                        try:
                            rate = float(
                                line_splitted[rate_index].split(' ')[1]
                            )
                            data.append([time, rate])
                        except:
                            continue

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'breathing_rate'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_detector_confidence(self, monitoring=True):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Open the log file and read it line by line
        for line in self.lines:
            if monitoring and 'MONITORING' not in line:
                continue

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

            # Split the line by tabs
            line_splitted = re.split(r'\t+', line)

            # Find the index containing 'subj:'
            conf_index = next(
                (i for i, s in enumerate(line_splitted) if 'subj: ' in s), None
            )

            if conf_index is not None:
                try:
                    conf = float(line_splitted[conf_index].split(' ')[2])
                    data.append([time, conf])
                except:
                    continue

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'confidence'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_power(self, monitoring=True):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Open the log file and read it line by line
        for line in tqdm.tqdm(self.lines):
            # if monitoring and 'MONITORING' not in line:
            #     continue

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

            # Split the line by tabs
            line_splitted = re.split(r'\t+', line)

            # Find the index containing 'subj:'
            pow_index = next(
                (i for i, s in enumerate(line_splitted) if 'pwr' in s), None
            )

            if pow_index is not None:
                try:
                    pow_val = float(line_splitted[pow_index].split(' ')[1])
                    data.append([time, pow_val])
                except:
                    continue

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'power'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_notification(self):
        # this function returns a df with timestam as index as breathing rate as column
        # if monitoring is true, only return the monitoring hours

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Define a regular expression pattern for the date
        date_pattern = re.compile(r'[A-Z][a-z]{2} \d{1,2}')

        # Open the log file and read it line by line
        for line in self.lines:
            if '+++++ NOTIFICATION +++++' not in line:
                continue

            # Extract time and notification
            timestamp_match = timestamp_pattern.search(line)
            date_match = date_pattern.search(line)
            if timestamp_match and date_match:
                timestamp = timestamp_match.group()
                date = date_match.group()
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"
                notification = line.strip()  # Use line as notification
                notification = notification.split('+++++ NOTIFICATION +++++')[1]
                data.append([time, notification])

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'notification'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_breathing_classifier(self, breathing_only=False):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Define a regular expression pattern for 'bc: X.XX Y'
        bc_pattern = re.compile(r'bc: (\d+\.\d\d) \d+')

        # Open the log file and read it line by line
        for line in tqdm.tqdm(self.lines):
            line_splitted = re.split(' ', line)
            if '"breath_clsf":' not in line_splitted:
                continue
            # if breathing_only and 'breathing micro' not in line:
            #     continue

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

                # Extract 'breath_clsf' values
                # Extract the JSON part of the string
                if 'ALGO_SEC' in line:
                    try:
                        json_str = line.split('ALGO_SEC: ')[1].strip()
                        json_str = json_str.replace(
                            '"ctx": "{', '"ctx": {'
                        ).replace('}"', '}')
                        json_str = json_str.replace('[(', '[[').replace(
                            ')]', ']]'
                        )
                        # Parse the JSON string into a Python dictionary
                        json_data = json.loads(json_str)

                        # Extract the desired values
                        cnf_baby_dtct = json_data['baby_dtct']['cnf']
                        med_breath_clsf = json_data['breath_clsf']['med']
                        nbc_breath_clsf = json_data['breath_clsf']['nbc']
                        cnf_baby_dtct = med_breath_clsf
                        # data.append([time, cnf_baby_dtct,med_breath_clsf, nbc_breath_clsf ])
                        data.append([time, cnf_baby_dtct])
                    except json.JSONDecodeError as e:
                        print(f'Error parsing JSON: {e} in line: {line}')

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'breath_clsf'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_breathing_classifier_buffer(self):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Define a regular expression pattern for 'bc: X.XX Y'
        bc_pattern = re.compile(r'bc: (\d+\.\d\d) \d+')

        # Open the log file and read it line by line
        for line in tqdm.tqdm(self.lines):
            if 'breath_clsf' not in line:
                continue

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

                # Extract 'breath_clsf' values
                bc_match = bc_pattern.search(line)
                if bc_match:
                    bc_value = int(bc_match.group(0).split(' ')[-1])
                    data.append([time, bc_value])

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'bc_buffer'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_state(self):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Define a regular expression pattern for 'state: ..._'
        state_pattern = re.compile(r'state: (.*?)_')

        # Open the log file and read it line by line
        for line in tqdm.tqdm(self.lines):
            if 'state:' not in line:
                continue
            if 'JanusAdmin' in line or 'IoT' in line:
                continue

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

            # Extract 'state' value
            state_match = state_pattern.search(line)

            if state_match:
                state_value = state_match.group(1)
                # get the part until the first tab
                state_value = state_value.split('\t')[0]
                data.append([time, state_value])

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'state'])
        df['time'] = pd.to_datetime(df['time'])

        return df

    def get_rois(self, monitoring=True):

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')

        # Define a regular expression pattern for 'rect = [...]'
        roi_pattern = re.compile(r'rect = \[([0-9,]+)\]')

        # Open the log file and read it line by line
        for line in tqdm.tqdm(self.lines):
            if 'rect =' not in line:
                continue

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

            # Extract 'roi' values
            roi_match = roi_pattern.search(line)
            if roi_match:
                roi_value = list(map(int, roi_match.group(1).split(',')))
                data.append([time, roi_value])

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['time', 'roi'])

        return df

    def get_classifier_counter(self, df_bc):

        df_bc['breath_clsf'] = df_bc['breath_clsf'].rolling(10).quantile(0.5)
        df_state = self.get_state()
        df_counter = df_bc
        df_counter['no_breathing_counter'] = df_counter['breath_clsf'] > 0.5
        df_counter['no_breathing_counter'] = df_counter[
            'no_breathing_counter'
        ].astype(int)

        df_bc.at[0, 'no_breathing_counter'] = 0
        for i in range(1, len(df_counter)):
            if df_counter.at[i, 'no_breathing_counter'] == 1:
                df_counter.at[i, 'no_breathing_counter'] = (
                    df_counter.at[i - 1, 'no_breathing_counter'] + 20
                )
            else:
                df_counter.at[i, 'no_breathing_counter'] = (
                    df_counter.at[i - 1, 'no_breathing_counter'] - 20
                )

            if df_counter.at[i, 'no_breathing_counter'] < 0:
                df_counter.at[i, 'no_breathing_counter'] = 0
            curr_state = df_state[
                df_state['time'] < df_counter.at[i, 'time']
            ].iloc[-1]['state']
            if curr_state != 'breathing micro':
                df_counter.at[i, 'no_breathing_counter'] = 0
        return df_counter

    def annotate_video(
        self, vid_file_path=None, play=False, output_path=None, vid_idx=None
    ):

        if vid_idx is not None:
            vid_file_path = [self.vid_path_list[vid_idx]]

        for idx, _path in enumerate(self.vid_path_list):
            first_pid = int(get_packet_id(_path))
            cap = cv2.VideoCapture(_path)
            _trigger = self.vid_triggers[idx]

            # if there is no self.df, get it
            if not hasattr(self, 'df'):
                log_df = self.get_fields()
            else:
                log_df = self.df
            # fill  na with previous value of columns x,y,w,h
            log_df['x'] = log_df['x'].fillna(method='ffill')
            log_df['y'] = log_df['y'].fillna(method='ffill')
            log_df['w'] = log_df['w'].fillna(method='ffill')
            log_df['h'] = log_df['h'].fillna(method='ffill')
            log_df['x_head'] = log_df['x_head'].fillna(method='ffill')
            log_df['y_head'] = log_df['y_head'].fillna(method='ffill')
            log_df['w_head'] = log_df['w_head'].fillna(method='ffill')
            log_df['h_head'] = log_df['h_head'].fillna(method='ffill')

            log_df['state'] = log_df['state'].fillna(method='ffill')
            log_df['confidence'] = log_df['confidence'].fillna(method='ffill')
            log_df['confidence_head'] = log_df['confidence_head'].fillna(
                method='ffill'
            )

            # init params
            no_notif_counter = 0

            log_df['frame_idx'] = log_df['pid'] - first_pid

            vid_frame = 0
            prev_lines = pd.DataFrame()
            # set first fram
            while True:
                ret, frame = cap.read()
                vid_frame += 1
                if not ret:
                    break
                # get relevant lines
                lines = log_df[(log_df['frame_idx'] == vid_frame)]
                if len(lines) > 0:
                    prev_lines = lines
                    print('found lines', vid_frame)
                if len(prev_lines) > 0:
                    pid = prev_lines['pid']
                    power = prev_lines['pow']
                    x, y, w, h = (
                        prev_lines['x'].values[0],
                        prev_lines['y'].values[0],
                        prev_lines['w'].values[0],
                        prev_lines['h'].values[0],
                    )
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    conf = prev_lines['confidence'].values[0]
                    # draw box
                    if w > 0 and h > 0:
                        frame = draw_curved_rectangle(frame, x, y, w, h, 10)
                        cv2.putText(
                            frame,
                            str(conf),
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    try:
                        x_head, y_head, w_head, h_head = (
                            prev_lines['x_head'].values[0],
                            prev_lines['y_head'].values[0],
                            prev_lines['w_head'].values[0],
                            prev_lines['h_head'].values[0],
                        )
                        x_head, y_head, w_head, h_head = (
                            int(x_head),
                            int(y_head),
                            int(w_head),
                            int(h_head),
                        )
                        conf_head = prev_lines['confidence_head'].values[0]
                        # draw box
                        if w_head > 0 and h_head > 0:
                            frame = draw_curved_rectangle(
                                frame, x_head, y_head, w_head, h_head, 10
                            )
                            cv2.putText(
                                frame,
                                str(conf_head),
                                (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    except:
                        pass

                    # add state
                    cv2.putText(
                        frame,
                        prev_lines['state'].values[0],
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # add the pwr
                    cv2.putText(
                        frame,
                        str(power.values[0]),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # primary position
                    pp = prev_lines['primary_position'].values[0]
                    if pp is not None:
                        # add as a scatter
                        cv2.circle(
                            frame, (int(pp[0]), int(pp[1])), 5, (0, 0, 255), -1
                        )

                    # add notification
                    if len(prev_lines['notification']) > 0:
                        if prev_lines['notification'].values[0] is not None:
                            prev_notification = prev_lines[
                                'notification'
                            ].values[0]
                            # keep this prev notification for 2*20 frames
                    cv2.putText(
                        frame,
                        prev_lines['notification'].values[0],
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    no_notif_counter += 1
                    if no_notif_counter > 20 * 2:
                        prev_notification = None
                        no_notif_counter = 0

                    if play:
                        cv2.imshow(_trigger, frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        pass

    def get_videos(self):
        # this function returns a df with timestam as index as breathing rate as column
        # if monitoring is true, only return the monitoring hours

        # Initialize an empty list to collect data
        data = []

        # Define a regular expression pattern for the timestamp
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        # video pattern = Saved 1200 frames of video to /data/1707980402584.mp4
        vid_pattern = re.compile(r'Saved \d+ frames of video to (.+?\.mp4)')
        # Open the log file and read it line by line
        for idx, line in enumerate(tqdm.tqdm(self.lines)):
            if '.mp4' not in line:
                continue
            # Extract time and notification
            timestamp_match = timestamp_pattern.search(line)

            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"
                vid = vid_pattern.search(line)
                if vid:
                    data.append([time, '/' + vid.group(1)])

        vid_df = pd.DataFrame(data, columns=['time', 'video'])
        # drop duplicates
        vid_df = vid_df.drop_duplicates()
        # take only the ones stating with /data
        vid_df = vid_df[vid_df['video'].str.contains('/data')]
        # remove the /data
        vid_df['video'] = vid_df['video'].str.replace('/data/', '')

        year = pd.to_datetime(vid_df['time']).dt.year
        year = year[0]
        week = pd.to_datetime(vid_df['time']).dt.isocalendar().week
        week = week[0]
        month = pd.to_datetime(vid_df['time']).dt.month
        month = month[0]
        # add zero to month if needed
        if month < 10:
            month = '0' + str(month)
        day = pd.to_datetime(vid_df['time']).dt.day
        day = day[0]
        if day < 10:
            day = '0' + str(day)

        day_of_week = tdate.fromisoformat(
            str(year) + '-' + str(month) + '-' + str(day)
        ).weekday()
        day_of_week = day_of_week + 2
        if day_of_week == 8:
            day_of_week = 1

        mac_id = self.mac_id
        # list blobs with the prefix
        list_files = storage_client.list_blobs(
            'production-nanobebe-video',
            prefix=str(year)
            + '/'
            + str(week)
            + '/'
            + str(day_of_week)
            + '/'
            + mac_id,
        )
        list_vids = []
        triggers = []
        for idx, file in enumerate(list_files):
            list_vids.append(file.name)
            triggers.append(file.name.split('/')[4])

        # download the file
        root = '/Users/matanb/Downloads/'
        # if therer is no temo_vids folder, create it
        if not os.path.exists(root + 'temp_vids'):
            os.makedirs(root + 'temp_vids')
        paths = []
        for vid in tqdm.tqdm(list_vids[:3]):
            blob = storage.Blob(
                vid, storage_client.bucket('production-nanobebe-video')
            )
            filename = root + 'temp_vids/' + vid.split('/')[-1]
            blob.download_to_filename(filename)
            paths.append(filename)
            vid_df.loc[
                vid_df['video'] == '/' + vid.split('/')[-1], 'local_path'
            ] = filename

        self.vid_path_list = paths
        self.vid_triggers = triggers

        return vid_df

    def get_fields(self, fields=None, monitoring=True, new_fields=False):

        # Initialize an empty list to collect data
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        state_pattern = re.compile(r'state: (.*?)_')
        counter_pattern = re.compile(r'bc: (\d+\.\d\d) \d+')
        roi_pattern = re.compile(r'rect = \[([0-9,]+)\]')
        vid_pattern = re.compile(r'(\d{12})\.mp4')
        # focal is pattered like this: "- (0.36,0.12) -"
        focal_pattern = r"\((-?\d*\.\d+),(-?\d*\.\d+)\)"
        # init all dfs
        df = pd.DataFrame(
            columns=[
                'time',
                'confidence',
                'bc1',
                'bc2',
                'roi',
                'state',
                'pow',
                'notification',
                'rate',
                'bc_buffer',
                'counter',
            ]
        )
        data = {
            'time': [],
            'confidence': [],
            'bc1': [],
            'bc2': [],
            'roi': [],
            'state': [],
            'pow': [],
            'notification': [],
            'rate': [],
            'bc_buffer': [],
            'counter': [],
            'monitoring': [],
            'video': [],
            'focal': [],
            'pi': [],
        }

        for idx, line in enumerate(self.lines):

            if 'ALGO_SEC' in line:
                df = self.get_fields_new()
                self.df = df
                return df
            if (
                'MONITORING' not in line
                and '+++++ NOTIFICATION +++++' not in line
                and 'rect' not in line
                and 'state' not in line
                and '.mp4' not in line
            ):
                continue

            # video pattern 699662892411.mp4 saved 2:07 (Min:Sec)
            # Nov 11 00:34:33 Aura-v1 smartbeat-monitor[7522]: [Camera] [uploadVideo] Upload video [subjectLeft][f74ca447-7e2b-4fb0-a918-ab32cca1973c] started

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"

            # Split the line by tabs
            line_splitted = re.split(r'\t+', line)

            vid_match = vid_pattern.search(line)
            if vid_match:
                vid = vid_match.group(1)
                next_line_split = re.split(r'\t+', self.lines[idx + 1])
                if 'uploadVideo' in next_line_split[0]:
                    try:
                        trigger = (
                            next_line_split[0].split(' ')[-2].split('[')[1][:-1]
                        )
                        data['video'].append(trigger + '/' + vid)
                    except:
                        data['video'].append('video/' + vid)
                else:
                    data['video'].append('video/' + vid)
            else:
                data['video'].append(None)

            notif_index = next(
                (
                    i
                    for i, s in enumerate(line_splitted)
                    if '+++++ NOTIFICATION +++++' in s
                ),
                None,
            )
            if notif_index is not None:
                try:
                    notif = line_splitted[notif_index].split(
                        '+++++ NOTIFICATION +++++'
                    )[1]
                    data['notification'].append(notif)
                except:
                    data['notification'].append(None)
            else:
                data['notification'].append(None)

            if 'MONITORING' in line:
                data['monitoring'].append(True)
            else:
                data['monitoring'].append(False)
            # focal pattern is - than space than (x,y) than space than -
            focal_match = re.search(focal_pattern, line)

            if focal_match:
                numbers = focal_match.groups()
                focal = list(map(float, numbers))
                data['focal'].append(focal)
            else:
                data['focal'].append(None)

            # Find the index containing 'subj:'
            conf_index = next(
                (i for i, s in enumerate(line_splitted) if 'subj: ' in s), None
            )
            if conf_index is not None:
                try:
                    conf = float(line_splitted[conf_index].split(' ')[2])
                    data['confidence'].append(conf)
                except:
                    data['confidence'].append(None)
            else:
                data['confidence'].append(None)

            bc_index = next(
                (i for i, s in enumerate(line_splitted) if 'bc: ' in s), None
            )
            if bc_index is not None:
                try:
                    bc = float(line_splitted[bc_index].split(' ')[1])
                    data['bc1'].append(bc)
                except:
                    data['bc1'].append(None)
            else:
                data['bc1'].append(None)

            if bc_index is not None:
                try:
                    bc = float(line_splitted[bc_index].split(' ')[1])
                    data['bc2'].append(bc)
                except:
                    data['bc2'].append(None)
            else:
                data['bc2'].append(None)

            roi_index = next(
                (i for i, s in enumerate(line_splitted) if 'rect = ' in s), None
            )
            if roi_index is not None:
                try:
                    roi = list(
                        map(
                            int,
                            line_splitted[roi_index].split(' ')[2].split(','),
                        )
                    )
                    data['roi'].append(roi)
                except:
                    data['roi'].append(None)
            else:
                data['roi'].append(None)

            state_match = state_pattern.search(line)
            if state_match:
                state_value = state_match.group(1)
                # get the part until the first tab
                state_value = state_value.split('\t')[0]
                try:
                    state_value = state_value.split('(')[0]
                except:
                    pass

                data['state'].append(state_value)
            else:
                data['state'].append(None)

            pow_index = next(
                (i for i, s in enumerate(line_splitted) if 'pow: ' in s), None
            )
            if pow_index is not None:
                try:
                    pow = float(line_splitted[pow_index].split(' ')[1])
                    data['pow'].append(pow)
                except:
                    data['pow'].append(None)
            else:
                data['pow'].append(None)

            primary_index_index = next(
                (i for i, s in enumerate(line_splitted) if 'pi: ' in s), None
            )
            if primary_index_index is not None:
                try:
                    pi = float(line_splitted[primary_index_index].split(' ')[1])
                    data['pi'].append(pi)
                except:
                    data['pi'].append(None)
            else:
                data['pi'].append(None)

            rate_index = next(
                (i for i, s in enumerate(line_splitted) if 'rate: ' in s), None
            )
            if rate_index is not None:
                try:
                    rate = float(line_splitted[rate_index].split(' ')[1])
                    data['rate'].append(rate)
                except:
                    data['rate'].append(None)
            else:
                data['rate'].append(None)

            bc_buffer_index = counter_pattern.search(line)
            if bc_buffer_index:
                try:
                    bc_buffer = int(bc_buffer_index.group(0).split(' ')[-1])
                    data['bc_buffer'].append(bc_buffer)
                except:
                    data['bc_buffer'].append(None)
            else:
                data['bc_buffer'].append(None)

            counter_index = next(
                (i for i, s in enumerate(line_splitted) if 'bc: ' in s), None
            )
            if counter_index is not None:
                try:
                    counter = int(line_splitted[counter_index].split(' ')[-1])
                    data['counter'].append(counter)
                except:
                    data['counter'].append(None)
            else:
                data['counter'].append(None)

            data['time'].append(time)
        # print length of all fields

        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        self.df = df
        return df

    def get_algo_events(self, event='avoided alarm'):

        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        event_pattern = re.compile(r'ALGO_EVENT: (.*?)_')
        data = {'time': [], 'algo_event': [], 'algo_event_ctx': []}
        for idx, line in enumerate(self.lines):

            if event not in line:
                continue
            # append none to all fields
            data['time'].append(None)
            data['algo_event'].append(None)
            data['algo_event_ctx'].append(None)

            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"
                json_string = re.search(r'ALGO_EVENT: {"t":', line)
                if json_string:
                    json_string = line.split('ALGO_EVENT: {"t": ')[1]
                    json_string, context = (
                        json_string.split(', "ctx":')[0],
                        json_string.split(', "ctx":')[1],
                    )
                    json_string = json_string[1:-1]
                    context = context[1:-1]

                    data['algo_event'][-1] = json_string
                    data['algo_event_ctx'][-1] = context[:-1]
                    data['time'][-1] = time

        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        return df

    def plot_fields(
        self,
        fields=None,
        monitoring=True,
        classifier_breathing_only=True,
        save=None,
    ):
        df_conf = self.df.loc[self.df['confidence'].notnull()][
            ['time', 'confidence']
        ]
        df_bc = self.df.loc[self.df['bc1'].notnull()][['time', 'bc1']]
        df_bc_01 = df_bc.copy()
        # running quantile 10 points 0.7
        # df_bc['bc1'] = df_bc['bc1'].rolling(10).quantile(0.5)

        df_roi = self.df.loc[self.df['roi'].notnull()][['time', 'roi']]
        df_state = self.df.loc[self.df['state'].notnull()][['time', 'state']]
        df_pow = self.df.loc[self.df['pow'].notnull()][['time', 'pow']]
        df_notif = self.df.loc[self.df['notification'].notnull()][
            ['time', 'notification']
        ]
        df_rate = self.df.loc[self.df['rate'].notnull()][['time', 'rate']]
        df_bc_buffer = self.df.loc[self.df['bc_buffer'].notnull()][
            ['time', 'bc_buffer']
        ]
        df_counter = self.df.loc[self.df['counter'].notnull()][
            ['time', 'counter']
        ]
        df_mointoring = self.df.loc[self.df['monitoring'].notnull()][
            ['time', 'monitoring']
        ]
        df_video = self.df.loc[self.df['video'].notnull()][['time', 'video']]
        df_pi = self.df.loc[self.df['pi'].notnull()][['time', 'pi']]
        # get the date from one of the dfs
        try:
            self.date = df_conf['time'].iloc[0].strftime('%Y-%m-%d')
        except:
            self.date = ' '

        states = df_state
        notifs = df_notif

        states = states.set_index('time')
        states = states.sort_index()
        states = states[~states.index.duplicated(keep='first')]
        # make sure there are no overlapping states
        state_vals = states['state'].unique().tolist()
        map_states = {state_vals[i]: i for i in range(len(state_vals))}
        state_numeric = [map_states[x] for x in state_vals]
        state_vec_numeric = [map_states[x] for x in states['state'].tolist()]
        change_inds = np.where(np.diff(state_vec_numeric) != 0)[0]
        df = pd.DataFrame(columns=['state', 'start', 'end'])
        for i in range(len(change_inds)):
            if i == 0:
                start = states.index[0]
                end = states.index[change_inds[i]]
            else:
                start = states.index[change_inds[i - 1]]
                end = states.index[change_inds[i]]
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            'state': [states.iloc[change_inds[i]]['state']],
                            'start': [start],
                            'end': [end],
                        }
                    ),
                ]
            )
            if classifier_breathing_only:
                if states.iloc[change_inds[i]]['state'] != 'breathing micro':
                    df_bc.loc[
                        (df_bc['time'] > start) & (df_bc['time'] < end), 'bc1'
                    ] = None

        # create fig with seccondaary y axis
        tl_fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y=pd.Series(np.ones(len(df)) - 0.6),
            color="state",
            opacity=0.4,
        )
        tl_fig.update_yaxes(tickvals=[0, 0.8], ticktext=['0', '1'])
        tl_fig.update_layout(yaxis=dict(range=[0, 1]))
        df_conf.set_index('time', inplace=True)
        df_bc.set_index('time', inplace=True)
        df_pow.set_index('time', inplace=True)
        df_bc_buffer.set_index('time', inplace=True)
        df_mointoring.set_index('time', inplace=True)
        df_pi.set_index('time', inplace=True)
        df_rate.set_index('time', inplace=True)
        # df_rate.set_index('time', inplace=True)
        # df_counter.set_index('time', inplace=True)

        # add other signals and set them toggled off
        tl_fig.add_trace(
            go.Scatter(
                x=df_conf.index,
                y=df_conf['confidence'] * 0.8,
                name='conf',
                hovertext=df_conf['confidence'],
                mode='lines',
                hovertemplate='confidence: %{hovertext} \n time: %{x}',
            ),
            row=1,
            col=1,
        )
        tl_fig.add_trace(
            go.Scatter(
                x=df_bc.index,
                y=0.8 * df_bc["bc1"] / df_bc['bc1'].max(),
                name='breath_clsf',
                hovertext=df_bc['bc1'],
                mode='lines',
                hovertemplate='Breathing Classifier: %{hovertext} \n time: %{x}',
            ),
            row=1,
            col=1,
        )
        # add power
        tl_fig.add_trace(
            go.Scatter(
                x=df_pow.index,
                y=0.8 * df_pow["pow"] / df_pow['pow'].max(),
                name='power',
                hovertext=df_pow['pow'],
                mode='lines',
                hovertemplate='Power: %{hovertext} \n time: %{x}',
            ),
            row=1,
            col=1,
        )

        tl_fig.add_trace(
            go.Scatter(
                x=df_bc_buffer.index,
                y=0.8
                * df_bc_buffer["bc_buffer"]
                / df_bc_buffer['bc_buffer'].max(),
                name='bc_counter',
                hovertext=df_bc_buffer['bc_buffer'],
                mode='lines',
                hovertemplate='Breathing Classifier Buffer: %{hovertext} \n time: %{x}',
            ),
            row=1,
            col=1,
        )
        tl_fig.add_trace(
            go.Scatter(
                x=df_pi.index,
                y=0.8 * df_pi["pi"] / df_pi['pi'].max(),
                name='pi',
                hovertext=df_pi['pi'],
                mode='lines',
                hovertemplate='pi: %{hovertext} \n time: %{x}',
            ),
            row=1,
            col=1,
        )

        try:
            df_ap = self.df.loc[self.df['active_regions'].notnull()][
                ['time', 'active_regions']
            ]
            df_ap.set_index('time', inplace=True)
            tl_fig.add_trace(
                go.Scatter(
                    x=df_ap.index,
                    y=0.8
                    * df_ap["active_regions"]
                    / df_ap['active_regions'].max(),
                    name='active_regions',
                    hovertext=df_ap['active_regions'],
                    mode='lines',
                    hovertemplate='active_regions: %{hovertext} \n time: %{x}',
                ),
                row=1,
                col=1,
            )
            tl_fig.update_traces(
                visible='legendonly', selector=dict(name="active_regions")
            )
        except:
            pass

        # add rate
        tl_fig.add_trace(
            go.Scatter(
                x=df_rate.index,
                y=0.8 * df_rate["rate"] / df_rate['rate'].max(),
                name='breathing_rate',
                hovertext=df_rate['rate'],
                mode='lines',
                hovertemplate='Breathing Rate: %{hovertext} \n time: %{x}',
            ),
            row=1,
            col=1,
        )

        # plot black line everywhere monitoring = True
        df_mointoring['y'] = 0.8
        df_mointoring.loc[df_mointoring['monitoring'] == False, 'y'] = None
        df_mointoring.loc[df_mointoring['monitoring'] == True, 'y'] = 0.8
        # black color, thick line
        tl_fig.add_trace(
            go.Scatter(
                x=df_mointoring.index,
                y=df_mointoring['y'],
                name='monitoring',
                mode='lines',
                hovertemplate='time: %{x}',
                line=dict(color='black', width=4),
            ),
            row=1,
            col=1,
        )

        # toggle all signals except the first ogg
        tl_fig.update_traces(visible='legendonly', selector=dict(name="conf"))
        tl_fig.update_traces(visible='legendonly', selector=dict(name="bc"))
        tl_fig.update_traces(visible='legendonly', selector=dict(name="power"))
        tl_fig.update_traces(
            visible='legendonly', selector=dict(name="bc_counter")
        )
        tl_fig.update_traces(
            visible='legendonly', selector=dict(name="breathing_rate")
        )
        tl_fig.update_traces(visible='legendonly', selector=dict(name="pi"))
        tl_fig.update_traces(visible='legendonly', selector=dict(name="rate"))

        # plot power in a differnt subplot
        # add the notifications as vertical lines with text. group by notification and add each group as a different trace
        notif_vals = notifs['notification'].unique().tolist()
        for i in range(len(notif_vals)):
            # add oll the notifications of this type, as single trace with toggle on and text annotation
            notif = notif_vals[i]
            notif_df = notifs[notifs['notification'] == notif]
            notif_df['y'] = 0
            notif_df2 = notif_df.copy()
            notif_df3 = notif_df.copy()
            ## add 1 ms
            notif_df2['time'] = notif_df2['time'] + pd.Timedelta(milliseconds=1)
            notif_df3['time'] = notif_df3['time'] - pd.Timedelta(milliseconds=2)
            notif_df3['y'] = None
            notif_df2['y'] = 1
            notif_df_total = pd.concat([notif_df, notif_df2, notif_df3])
            notif_df_total = notif_df_total.sort_values(by=['time'])
            tl_fig.add_trace(
                go.Scatter(
                    x=notif_df_total['time'],
                    y=notif_df_total['y'],
                    name=notif,
                    mode='lines',
                    hovertemplate='time: %{x}',
                ),
                row=1,
                col=1,
            )
            tl_fig.update_traces(
                visible='legendonly', selector=dict(name=notif)
            )

        # do the same for videos
        plot_videos = False
        if plot_videos:
            video_vals = df_video['video'].unique().tolist()
            for i in range(len(video_vals)):
                # add oll the notifications of this type, as single trace with toggle on and text annotation
                video = video_vals[i]
                video_df = df_video[df_video['video'] == video]
                video_df['y'] = 0
                video_df2 = video_df.copy()
                video_df3 = video_df.copy()
                ## add 1 ms
                video_df2['time'] = video_df2['time'] + pd.Timedelta(
                    milliseconds=1
                )
                video_df3['time'] = video_df3['time'] - pd.Timedelta(
                    milliseconds=2
                )
                video_df3['y'] = None
                video_df2['y'] = 1
                video_df_total = pd.concat([video_df, video_df2, video_df3])
                video_df_total = video_df_total.sort_values(by=['time'])
                tl_fig.add_trace(
                    go.Scatter(
                        x=video_df_total['time'],
                        y=video_df_total['y'],
                        name=video,
                        mode='lines',
                        hovertemplate='time: %{x}',
                    ),
                    row=1,
                    col=1,
                )
                tl_fig.update_traces(
                    visible='legendonly', selector=dict(name=video)
                )

        # add title with date and acc id and mac id
        tl_fig.update_layout(
            title_text=f"Date: {self.date} Acc ID: {self.acc_id} MAC ID: {self.mac_id}"
        )
        tl_fig.show('browser')

        if save is not None:
            tl_fig.write_html(save)

    def get_version(self):
        # version template: version = "X-App-Version": "1.8.32-iot.0.13.28-2gb-staging", get the 0.13.28
        version_pattern = re.compile(r'X-App-Version": "(.*?)"')
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        version = None
        version_df = pd.DataFrame(columns=['time', 'version'])
        pattern = re.compile(r'version=\d+\.\d+\.\d+')
        for line in self.lines:
            if pattern.search(line):
                version_val = pattern.search(line).group().split('=')[1]
                print(version_val)
                break
        for line in self.lines:
            version_match = pattern.search(line)
            if version_match:
                # Extract time and date
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    timestamp = timestamp_match.group()
                    date = line[:6]
                    time = f"{pd.to_datetime('today').year} {date} {timestamp}"
                    version_val = pattern.search(line).group().split('=')[1]
                    version_df = pd.concat(
                        [
                            version_df,
                            pd.DataFrame(
                                {'time': [time], 'version': [version_val]}
                            ),
                        ]
                    )
        return version_df

    def get_all(
        self,
        vis=True,
        filter_notifs=['breathingStopped'],
        get=[
            'conf',
            'breath_clsf',
            'pow',
            'notif',
            'state',
            'roi',
            'bc_buffer',
            'rate',
        ],
    ):
        # get all the dataframes

        df_conf = self.get_detector_confidence()
        # df_rate = self.get_breathing_rate()
        df_notif = self.get_notification()
        # df_state = self.get_state()
        df_pow = self.get_power()
        # df_roi = self.get_rois()
        df_bc = self.get_breathing_classifier()
        df_bc_01 = df_bc.copy()
        # running quantile 10 points 0.7
        df_bc['breath_clsf'] = df_bc['breath_clsf'].rolling(10).quantile(0.5)

        df_counter = df_bc
        df_counter['no_breathing_counter'] = df_counter['breath_clsf'] > 0.5
        df_counter['no_breathing_counter'] = df_counter[
            'no_breathing_counter'
        ].astype(int)

        df_bc.at[0, 'no_breathing_counter'] = 0
        for i in range(1, len(df_counter)):
            if df_counter.at[i, 'no_breathing_counter'] == 1:
                df_counter.at[i, 'no_breathing_counter'] = (
                    df_counter.at[i - 1, 'no_breathing_counter'] + 20
                )
            else:
                # df_counter.at[i, 'no_breathing_counter'] = df_counter.at[i, 'no_breathing_counter'] * (0.99834 ** 60)
                df_counter.at[i, 'no_breathing_counter'] = (
                    df_counter.at[i - 1, 'no_breathing_counter'] - 20
                )

            if df_counter.at[i, 'no_breathing_counter'] < 0:
                df_counter.at[i, 'no_breathing_counter'] = 0
            # check what is the current state
            # curr_state = df_state[
            #     df_state['time'] < df_counter.at[i, 'time']
            # ].iloc[-1]['state']
            # if curr_state != 'breathing micro':
            #     df_counter.at[i, 'no_breathing_counter'] = 0

        df_bc_buffer = self.get_breathing_classifier_buffer()
        # df_all = pd.merge(df_conf, df_rate, on='time', how='outer')
        # merge them
        # states = df_state
        notifs = df_notif
        pow = df_pow
        if filter_notifs is not None:
            notifs = [
                notifs[notifs['notification'].str.contains(x)]
                for x in filter_notifs
            ]
            notifs = pd.concat(notifs)

        if vis:

            # states = states.set_index('time')
            # states = states.sort_index()
            # states = states[~states.index.duplicated(keep='first')]
            # # make sure there are no overlapping states
            # state_vals = states['state'].unique().tolist()
            # map_states = {state_vals[i]: i for i in range(len(state_vals))}
            # state_numeric = [map_states[x] for x in state_vals]
            # state_vec_numeric = [
            #     map_states[x] for x in states['state'].tolist()
            # ]
            # change_inds = np.where(np.diff(state_vec_numeric) != 0)[0]
            # df = pd.DataFrame(columns=['state', 'start', 'end'])
            # for i in range(len(change_inds)):
            #     if i == 0:
            #         start = states.index[0]
            #         end = states.index[change_inds[i]]
            #     else:
            #         start = states.index[change_inds[i - 1]]
            #         end = states.index[change_inds[i]]
            #     df = pd.concat(
            #         [
            #             df,
            #             pd.DataFrame(
            #                 {
            #                     'state': [states.iloc[change_inds[i]]['state']],
            #                     'start': [start],
            #                     'end': [end],
            #                 }
            #             ),
            #         ]
            #     )

            # create fig with seccondaary y axis
            # tl_fig = px.timeline(
            #     df,
            #     x_start="start",
            #     x_end="end",
            #     y=pd.Series(np.ones(len(df)) - 0.6),
            #     color="state",
            #     opacity=0.4,
            # )
            # tl_fig.update_yaxes(tickvals=[0, 0.8], ticktext=['0', '1'])
            # tl_fig.update_layout(yaxis=dict(range=[0, 1]))
            # df_conf.set_index('time', inplace=True)
            # df_bc.set_index('time', inplace=True)
            # # df_pow.set_index('time', inplace=True)
            # df_bc_buffer.set_index('time', inplace=True)
            # # df_rate.set_index('time', inplace=True)
            # # df_counter.set_index('time', inplace=True)

            # # add other signals and set them toggled off
            # tl_fig.add_trace(
            #     go.Scatter(
            #         x=df_conf.index,
            #         y=df_conf['confidence'] * 0.8,
            #         name='conf',
            #         hovertext=df_conf['confidence'],
            #         mode='lines',
            #         hovertemplate='confidence: %{hovertext} \n time: %{x}',
            #     ),
            #     row=1,
            #     col=1,
            # )
            # tl_fig.add_trace(
            #     go.Scatter(
            #         x=df_bc.index,
            #         y=0.8 * df_bc["breath_clsf"] / df_bc['breath_clsf'].max(),
            #         name='breath_clsf',
            #         hovertext=df_bc['breath_clsf'],
            #         mode='lines',
            #         hovertemplate='Breathing Classifier: %{hovertext} \n time: %{x}',
            #     ),
            #     row=1,
            #     col=1,
            # )
            # # tl_fig.add_trace(go.Scatter(x=df_pow.index, y=0.8 * df_pow["power"] / df_pow['power'].max(), name='power', hovertext=df_pow['power'], mode='lines', hovertemplate='Power: %{hovertext} \n time: %{x}'), row=1, col=1)
            # tl_fig.add_trace(
            #     go.Scatter(
            #         x=df_bc_buffer.index,
            #         y=0.8
            #         * df_bc_buffer["bc_buffer"]
            #         / df_bc_buffer['bc_buffer'].max(),
            #         name='bc_counter',
            #         hovertext=df_bc_buffer['bc_buffer'],
            #         mode='lines',
            #         hovertemplate='Breathing Classifier Counter: %{hovertext} \n time: %{x}',
            #     ),
            #     row=1,
            #     col=1,
            # )
            # # tl_fig.add_trace(go.Scatter(x=df_rate.index, y=0.8 * df_rate["breathing_rate"] / df_rate['breathing_rate'].max(), name='breathing_rate', hovertext=df_rate['breathing_rate'], mode='lines', hovertemplate='Breathing Rate: %{hovertext} \n time: %{x}'), row=1, col=1)
            # tl_fig.add_trace(
            #     go.Scatter(
            #         x=df_counter.index,
            #         y=0.8
            #         * df_counter["no_breathing_counter"]
            #         / df_counter['no_breathing_counter'].max(),
            #         name='no_breathing_counter',
            #         hovertext=df_counter['no_breathing_counter'],
            #         mode='lines',
            #         hovertemplate='No Breathing Counter: %{hovertext} \n time: %{x}',
            #     ),
            #     row=1,
            #     col=1,
            # )
            # # toggle all signals except the first ogg
            # tl_fig.update_traces(
            #     visible='legendonly', selector=dict(name="conf")
            # )
            # tl_fig.update_traces(visible='legendonly', selector=dict(name="bc"))
            # tl_fig.update_traces(
            #     visible='legendonly', selector=dict(name="power")
            # )
            # tl_fig.update_traces(
            #     visible='legendonly', selector=dict(name="bc_counter")
            # )
            # tl_fig.update_traces(
            #     visible='legendonly', selector=dict(name="breathing_rate")
            # )
            # tl_fig.update_traces(
            #     visible='legendonly', selector=dict(name="no_breathing_counter")
            # )

            # # plot power in a differnt subplot
            # # add the notifications as vertical lines with text. group by notification and add each group as a different trace
            # notif_vals = notifs['notification'].unique().tolist()
            # for i in range(len(notif_vals)):
            #     # add oll the notifications of this type, as single trace with toggle on and text annotation
            #     notif = notif_vals[i]
            #     notif_df = notifs[notifs['notification'] == notif]
            #     notif_df['y'] = 0
            #     notif_df2 = notif_df.copy()
            #     notif_df3 = notif_df.copy()
            #     ## add 1 ms
            #     notif_df2['time'] = notif_df2['time'] + pd.Timedelta(
            #         milliseconds=1
            #     )
            #     notif_df3['time'] = notif_df3['time'] - pd.Timedelta(
            #         milliseconds=2
            #     )
            #     notif_df3['y'] = None
            #     notif_df2['y'] = 0.8
            #     notif_df_total = pd.concat([notif_df, notif_df2, notif_df3])
            #     notif_df_total = notif_df_total.sort_values(by=['time'])
            #     tl_fig.add_trace(
            #         go.Scatter(
            #             x=notif_df_total['time'],
            #             y=notif_df_total['y'],
            #             name=notif,
            #             mode='lines',
            #             hovertemplate='time: %{x}',
            #         ),
            #         row=1,
            #         col=1,
            #     )
            #     tl_fig.update_traces(
            #         visible='legendonly', selector=dict(name=notif)
            #     )

            html = True
            if html:
                # crate the subplots. make the first row for onw graph, and the second row for the other two. use row span
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    specs=[
                        [
                            {"colspan": 1, "secondary_y": True},
                            {"secondary_y": True},
                        ],
                        [{"secondary_y": True}, {"secondary_y": True}],
                    ],
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    horizontal_spacing=0.02,
                )
                fig.add_trace(
                    px.line(pow, x="time", y="power").data[0], row=1, col=1
                )
                fig.add_trace(
                    px.line(df_bc, x="time", y="breath_clsf").data[0],
                    row=1,
                    col=2,
                )
                # title and grids
                fig.update_layout(title_text="Power")
                fig.update_yaxes(title_text="Power", row=1, col=1)
                fig.update_xaxes(title_text="Time", row=1, col=1)
                fig.update_xaxes(showgrid=True, row=1, col=1)
                fig.update_yaxes(showgrid=True, row=1, col=1)
                fig.update_yaxes(
                    title_text="Breathing Classifier", row=1, col=2
                )
                fig.update_xaxes(title_text="Time", row=1, col=2)
                fig.update_xaxes(showgrid=True, row=1, col=2)
                fig.update_yaxes(showgrid=True, row=1, col=2)

                # control tl_fig size
                # tl_fig.update_layout(height=500)
                # # link x axes in all figures
                # fig.update_layout(xaxis_range=tl_fig.layout.xaxis.range)
                figures_to_html([fig], filename="dashboard.html")
                subprocess.call(['open', 'dashboard.html'])
            # else:
            #     tl_fig.show('browser')

        return

    def get_video(self, trigger='NoMeasure'):

        # get the mac and date
        mac = self.mac_id
        date = self.date - pd.Timedelta(days=1)
        date = date.strftime('%Y-%m-%d')
        # get the video
        video = self.get_video_from_db(mac, date, trigger)

    def get_fields_new(
        self, fields=None, monitoring=True, classifier_breathing_only=True
    ):
        # Initialize an empty list to collect data

        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        state_pattern = re.compile(r'state: (.*?)_')
        counter_pattern = re.compile(r'bc: (\d+\.\d\d) \d+')
        roi_pattern = re.compile(r'rect = \[([0-9,]+)\]')
        vid_pattern = re.compile(r'(\d{12})\.mp4')
        # foca "emdp": {"e": 35, "px_cnt": 1009, "pos": [414.0, 406.0]
        focal_pattern = re.compile(r'pos": \[([0-9.,]+)\]')
        # priamry  "emdp": {"e": 35, "px_cnt": 1009, "pos": [414.0, 406.0]
        primary_pattern = re.compile(r'emdp')
        pid_pattern = re.compile(r'"pid": \d+')
        # init all dfs

        data = {
            'time': [],
            'confidence': [],
            'bc1': [],
            'bc2': [],
            'roi': [],
            'state': [],
            'pow': [],
            'notification': [],
            'rate': [],
            'bc_buffer': [],
            'counter': [],
            'monitoring': [],
            'video': [],
            'focal': [],
            'pi': [],
            'primary_position': [],
            'active_regions': [],
            'algo_event': [],
            'algo_event_ctx': [],
            'is_critical': [],
            'pid': [],
            'x': [],
            'y': [],
            'w': [],
            'h': [],
            'x_head': [],
            'y_head': [],
            'w_head': [],
            'h_head': [],
            'confidence_head': [],
        }

        for idx, line in enumerate(tqdm.tqdm(self.lines)):

            if (
                '+++++ NOTIFICATION +++++' not in line
                and '.mp4' not in line
                and 'ALGO_SEC' not in line
                and 'ALGO_EVENT' not in line
            ):
                continue

            # init all fields with None
            for field in data.keys():
                data[field].append(None)

            try:
                pid_line = float(line.split('}')[-2].split(' ')[-1])
                data['pid'][-1] = pid_line
            except:
                pass
            # Extract time and date
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                timestamp = timestamp_match.group()
                date = line[:6]
                time = f"{pd.to_datetime('today').year} {date} {timestamp}"
                data['time'][-1] = pd.to_datetime(time)

            if '+++++ NOTIFICATION +++++' in line:
                # get the notification
                notification = line.split('+++++ NOTIFICATION +++++')[1]
                data['notification'][-1] = notification
                continue

            if 'ALGO_EVENT' in line:
                # get the string after ALGO_EVENT: {"t":
                json_string = re.search(r'ALGO_EVENT: {"t":', line)
                if json_string:
                    json_string = line.split('ALGO_EVENT: {"t": ')[1]
                    if "baby_detector_inference" in json_string:
                        # Dec  2 00:10:11 Aura-v1 smartbeat-monitor[4590]: ALGO_EVENT: {"t": "baby_detector_inference", "ctx": {"flip": 0, "rotation": 270, "confidence": 0.728919, "rect": [146, 132, 760, 539]}}
                        # get the confidence
                        if 'confidence' in json_string:
                            confidence = re.search(
                                r'confidence": (\d+\.\d\d)', line
                            ).group(1)
                        elif 'conf' in json_string:
                            confidence = re.search(
                                r'conf": (\d+\.\d\d)', line
                            ).group(1)
                        data['confidence'][-1] = float(confidence)
                        rect = (
                            line.split('rect": [')[-1]
                            .split('}')[0]
                            .split('],')[0]
                        )
                        # join to a list of four numbers
                        try:
                            rect = list(map(int, rect.split(',')))
                            data['x'][-1] = rect[0]
                            data['y'][-1] = rect[1]
                            data['w'][-1] = rect[2]
                            data['h'][-1] = rect[3]
                        except:
                            pass
                    if 'head_detector_inference' in json_string:
                        if 'confidence' in json_string:
                            confidence = re.search(
                                r'confidence": (\d+\.\d\d)', line
                            ).group(1)
                        elif 'conf' in json_string:
                            confidence = re.search(
                                r'conf": (\d+\.\d\d)', line
                            ).group(1)
                        data['confidence_head'][-1] = float(confidence)
                        rect = (
                            line.split('rect": [')[-1]
                            .split('}')[0]
                            .split('],')[0]
                        )
                        # join to a list of four numbers
                        rect = list(map(int, rect.split(',')))
                        data['x_head'][-1] = rect[0]
                        data['y_head'][-1] = rect[1]
                        data['w_head'][-1] = rect[2]
                        data['h_head'][-1] = rect[3]

                    json_string, context = (
                        json_string.split(', "ctx":')[0],
                        json_string.split(', "ctx":')[1],
                    )
                    json_string = json_string[1:-1]
                    context = context[1:-1]

                    data['algo_event'][-1] = json_string
                    data['algo_event_ctx'][-1] = context

                    continue

            elif 'ALGO_SEC' in line:
                json_string = re.search(r'ALGO_SEC: (.+)', line).group(1)
                try:
                    line_data = json.loads(json_string)
                except:
                    continue

                if line_data['mntr_frms'] is not None:
                    data['monitoring'][-1] = True
                if line_data['crtc_frms'] is None:
                    data['is_critical'][-1] = False
                else:
                    data['is_critical'][-1] = True

                data['pi'][-1] = line_data['emdp']['e']
                data['primary_position'][-1] = line_data['emdp']['pos']
                data['active_regions'][-1] = float(
                    line_data['motion']['active_regs']
                )
                data['bc1'][-1] = line_data['breath_clsf']['cnf']
                data['state'][-1] = line_data['rut']['s']
                data['pow'][-1] = line_data['pwr']
                data['rate'][-1] = line_data['frq']
                data['bc_buffer'][-1] = line_data['breath_clsf']['nbc']

        df = pd.DataFrame(data)
        self.df = df
        return df

    def resolve_reason(self):

        df = self.df
        no_measure = df.loc[
            df['notification'] == '  monitoringStoppedNoMeasure\n'
        ]
        # for each notification, check bc in the past 5 minutes, check if there was motion in the past 3 minutes
        output = pd.DataFrame(columns=['time', 'reason'])
        for nm in no_measure.iterrows():
            # get the time
            time = nm[1]['time']
            # get the bc in the past 5 minutes
            bc = df[
                (df['time'] > time - datetime.timedelta(minutes=5))
                & (df['time'] < time)
            ]['bc1']
            bc_counter = df[
                (df['time'] > time - datetime.timedelta(minutes=5))
                & (df['time'] < time)
            ]['bc_buffer']
            # check if the state had 'motion' in it in the past 3 minutes
            # get the state 3 minutes before
            state = df[
                (df['time'] > time - datetime.timedelta(minutes=3))
                & (df['time'] < time)
            ]['state'].values
            # was there state with motion in the past 3 minutes
            motion = any(['motion' in s for s in state if s is not None])
            prev_state = df[
                (df['time'] > time - datetime.timedelta(minutes=3))
                & (df['time'] < time)
            ]['state'].values[-5]
            # see what is the power
            pow = df[
                (df['time'] > time - datetime.timedelta(minutes=5))
                & (df['time'] < time)
            ]['pow']

            # if prev stat is calibrating, and bc>0.5

            try:
                reason = []
                if 'calibratin' in prev_state and bc.values[-1] > 0.5:
                    reason.append('classifier prevented breathing micro')
                elif prev_state != 'breathing micro':
                    reason.append('prev state not breathing micro')
                if bc_counter.values[-1] > 779:
                    reason.append('thrown by classifier - 800')
                if motion:
                    reason.append('motion in the past 3 minutes')
                if bc_counter.values[-1] > 299:
                    reason.append('thrown by classifier - 300')
            except:
                reason = 'no reason found'
            output = pd.concat(
                [output, pd.DataFrame({'time': [time], 'reason': [reason]})]
            )

        return output


# object = AccountScraper(mac_id="E062906E3449", acc_id=2883)
# object.get_all()

object = AccountScraper(mac_id="E06290422B33", acc_id=3081)
object.get_all()

# print("hello")
