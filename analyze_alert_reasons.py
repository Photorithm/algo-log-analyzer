from account_scraper import AccountScraper
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import plotly.express as px

macs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2023-10-10 06_12_43.csv')
dates = pd.date_range(start='2023-10-13', end='2023-10-15')


states_data = []
states2_data = []
bc_buffer_data = []
motion_detected_data = []
acc = []
times = []
accs = macs['AccountId'].unique().tolist()
accs = [157,402,649,812,161,384,621]

for acc_id in tqdm.tqdm(accs):
    for date in dates:
        AS = AccountScraper(acc_id = acc_id, dates=[date])
        notifs = AS.get_notification()
        states = AS.get_state()
        notifs = notifs[notifs['notification'].str.contains('breathingStopped')]
        # for each notification, see if there was a motion stated in the 3 minutes before
        for idx, row in notifs.iterrows():
            try:
                # get the time of the notification
                time = row['time']
                # get the state of the 3 minutes before
                state = states[(states['time'] > time - pd.Timedelta(minutes=3)) & (states['time'] < time)]
                bc_buffer = AS.get_breathing_classifier_buffer()
                # if there was a motion, add the bc_buffer to the output
                if 'small motion' in state['state'].unique().tolist() or 'large motion' in state['state'].unique().tolist():
                    motion_detected = 1
                else:
                    motion_detected = 0
                # check the buffer close to the notification, the latest value before the notification
                bc_buffer = bc_buffer[bc_buffer['time'] < time]
                bc_buffer = bc_buffer.iloc[-1]['bc_buffer']

                #get the state before the notification
                state1 = state.iloc[-1]['state']
                state2 = state.iloc[-2]['state']

                states_data.append(state1)
                states2_data.append(state2)
                bc_buffer_data.append(bc_buffer)
                motion_detected_data.append(motion_detected)
                acc.append(acc_id)
                times.append(time)
            except:
                print('error')

        print(len(states_data))

output = pd.DataFrame({'state': states_data,'state2': states2_data, 'bc_buffer': bc_buffer_data, 'motion_detected': motion_detected_data, 'acc': acc, 'time': times})

