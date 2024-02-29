from account_scraper import AccountScraper
import pandas as pd

# input, account_id, date and video filename
acc_id = 399
_date = pd.to_datetime('2023-11-13 17:35', format='%Y-%m-%d %H:%M')

# get the log
AS = AccountScraper(acc_id = 880,  dates  = [_date])
df = AS.get_fields()
AS.annotate_video(r'/Users/matanb/Downloads/1699621998646.mp4')
