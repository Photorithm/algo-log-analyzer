import numpy as np
import os
import pandas as pd
import sys


# make sure everything is in the path
sys.path.append(os.getcwd())
os.chdir('..')
sys.path.append(os.getcwd())

root = os.getcwd()
# step out cd to 'log_analyzer' if needed
if os.path.basename(root) != 'log_analyzer':
    os.chdir('..')
    root = os.getcwd()
from account_scraper import AccountScraper
mac_path = r'All Version Users-data-2024-05-23 10_08_35.csv'
mac_path = os.path.join(root, mac_path)
macs = pd.read_csv(mac_path)


# ======================================================================================================================
acc_id = 1257
_date = pd.to_datetime('2024-05-28')
# ======================================================================================================================


mac_id = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()[0]
AS = AccountScraper(acc_id =acc_id, mac_id = mac_id, dates = [_date], mac_path = mac_path)
# AS.get_logs_from_path(r'//Users/matanb/PycharmProjects/log_analyzer/E062904F6151_2024_1_6/varlog/syslog')
df = AS.get_fields()
AS.plot_fields(classifier_breathing_only=False, save=str(acc_id)+'_'+str(_date.date())+'.html')




