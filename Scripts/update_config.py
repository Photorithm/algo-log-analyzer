
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import subprocess
import appscript
import time
# get the output text
import pyautogui






import time
import subprocess
import re
import pandas as pd


def copy_terminal_output():
    # Bring Terminal to the forefront (modify as needed for your terminal)
    # This command works for the default Terminal app on macOS
    # clear clipboard
    subprocess.run("pbcopy", text=True, input="")

    subprocess.call(["osascript", "-e", 'tell application "Terminal" to activate'])

    time.sleep(2)  # Wait for the application to be active

    # Simulate the keyboard shortcut for 'Select All' (Command + A)
    pyautogui.keyDown('command')
    pyautogui.press('a')  # Select all
    time.sleep(1)
    pyautogui.press('c')  # Copy
    pyautogui.keyUp('command')
    time.sleep(1)  # Wait for the clipboard to update

def extract_parameter_from_clipboard(parameter_name):
    # Get the current clipboard content
    clipboard_content = subprocess.run("pbpaste", capture_output=True, text=True).stdout
    # "parameter_name": "value"

    # now try without re. use classic string manipulation
    val = None
    clipboard_content = clipboard_content.split('\n')
    for line in clipboard_content:
        if parameter_name in line:
            val =  line.split(':')[1].split(',')[0]

    return val
def extract_url_from_clipboard():
    # Get the current clipboard content
    clipboard_content = subprocess.run("pbpaste", capture_output=True, text=True).stdout

    # Define a simple URL regex pattern (adjust this pattern as necessary)
    url_pattern = r"https?://\S+"
    match = re.search(url_pattern, clipboard_content)

    if match:
        return match.group(0)
    else:
        return None


macs = pd.read_csv('/Users/matanb/Downloads/All Version Users-data-2024-03-17 08_00_50.csv')
monitoring_users = pd.read_csv(r'/Users/matanb/Downloads/Bad Actors-data-2024-04-10 16_21_00.csv')
#monitoring_users = monitoring_users[monitoring_users['avg_hours_per_day'] > 5]

# bad_actors = pd.read_csv(r'/Users/matanb/Downloads/Bad Actors-data-2024-04-02 05_20_36.csv')
# bad_actors = bad_actors[bad_actors['stops_per_day'] > 0.2]

webdriver_path = '/opt/homebrew/bin/chromedriver'
user_data_dir = '/Users/matanb/Library/Application Support/Google/Chrome'

# Set up Chrome options
options = Options()
#options.add_argument(f"user-data-dir={user_data_dir}")
#options.add_argument("--profile-directory=Profile 2")
#options.add_argument("--no-sandbox")  # Disable sandbox for Chrome
#options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
# options.add_argument('--headless')
options.add_argument('--disable-infobars')
service = Service(webdriver_path)


accs = [151,157,161,1636,171,1719,1783,182,187,205, 208,2176,226,229, 242, 243, 276,299,316,323,334,344,379,380,383,384,393,397,407,416]
accs = pd.read_csv(r'/Users/matanb/Downloads/All Version Users-data-2024-04-29 18_31_59.csv')
accs = accs['AccountId'].tolist()
accs = [292,
303,653,772,348,379,380,384,418,423,616,630,632,646,648,652,679,688,776,778,794,149,157,161,175,182,192,195,205,208,211,226,229,235,242,276,292,299,303,316,334,348,378,379,380,383,384,393,397,399,402,407,416,423,607,616,
620,621,623,630,632,640,646,648,653,656,657,675,676,682,688,693,694,698,702,714,736,739,772,773,775,776,778,780,786,789,791,794,809,812,829,1074,1111,1636,1719,2176]
accs = [175,179,192,275,292,297,306,313,348,378,419,595,621,622,623,624,631,632,634,640,646,647,648,653,656,675,676,686,690,
        693,694,695,702,710,714,739,772,773,775,776,778,780,785,789,791,809,812,829]


mon_users = pd.read_csv(r'/Users/matanb/Downloads/Bad Actors-data-2024-05-16 05_57_05.csv')
mon_users = mon_users[mon_users['avg_hours_per_day'] > 5]
accs = mon_users['acc'].tolist()
#process_df = pd.DataFrame()
process_df = pd.read_csv('process_df.csv')
n = 0
for idx in range(len(accs)):
    #acc_id = row['accountid']
    acc_id = accs[idx]
    # skip if the account is not in the macs df
    try:
        if acc_id in process_df['acc_id'].tolist():
            continue
    except:
        pass

    # Define the SSH comma
    MAC = macs[macs['AccountId'] == acc_id]['BaseMAC'].tolist()
    if len(MAC) == 0:
        continue
    MAC = MAC[0]
    ssh_string = 'ssh {}@prod-ssh-2.nanobebe.io -p 22222'.format(MAC)
    if n == 0:
        terminal = appscript.app('Terminal')
    terminal.do_script(ssh_string)
    time.sleep(5)

    copy_terminal_output()
    # Extract the URL from the clipboard content
    url = extract_url_from_clipboard()

    if n ==0:
        driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    driver.implicitly_wait(5)  # Waits up to 10 seconds before throwing a TimeoutException

    if n == 0:
        curr_time = time.time()
        while time.time() - curr_time < 7:
            try:
                email = driver.find_element(By.XPATH, '//input[@type="email"]')
                email.send_keys('matanb@nanobebe.com')
            except:
                pass
            try:
                next_button = driver.find_element(By.XPATH, '//input[@type="submit"]')
                next_button.click()
            except:
                pass
            try:
                password = driver.find_element(By.XPATH, '//input[@name="passwd"]')
                password.send_keys('Dav81971')
            except:
                pass
            try:
                sign_in_button = driver.find_element(By.XPATH, '//input[@type="submit"]')
                sign_in_button.click()
            except:
                pass
            try:
                yes_button = driver.find_element(By.XPATH, '//input[@value="Yes"]')
                yes_button.click()
            except:
                pass
    else:
        curr_time = time.time()
        err = 0
        while time.time() - curr_time < 8 and err == 0:
            try:
                yes_button = driver.find_element(By.XPATH, '//input[@value="Yes"]')
                yes_button.click()
                err = 1
            except Exception as e:
                err = 0
                pass


    # now on the terminal, cloik enter
    subprocess.call(["osascript", "-e", 'tell application "Terminal" to activate'])
    pyautogui.click()
    pyautogui.press('enter')
    time.sleep(5)
    # wait for the session to start

    # check parameter
    check = False
    if check:
        subprocess.call(["osascript", "-e", 'tell application "Terminal" to activate'])
        get_json = 'cat /opt/smartbeat/config/algo.json'
        pyautogui.write(get_json)
        pyautogui.press('enter')
        copy_terminal_output()
        val = extract_parameter_from_clipboard('"weak_breathing_threshold"')

    # change
    subprocess.call(["osascript", "-e", 'tell application "Terminal" to activate'])
    pyautogui.press('enter')
    time.sleep(2)



    update_threshold = 'sed -i \'s/\\("weak_breathing_threshold": \\)[0-9]*\\(\\.[0-9]*\\)\\?/\\10/\' /opt/smartbeat/config/algo.json'
    update_retry = 'sed -i \'s/\\("retry_with_head_long": \\)[0-9]*\\(\\.[0-9]*\\)\\?/\\110/\' /opt/smartbeat/config/algo.json'
    update_head = 'sed -i \'s/\\("minimum_head_for_breathing": \\)[0-9]*\\(\\.[0-9]*\\)\\?/\\10.35/\' /opt/smartbeat/config/algo.json'
    update_aut = """sed -i 's/\("routing_enable_shadow_mode": \).*$/\\1true,/' /opt/smartbeat/config/algo.json"""
    enable_at = """sed -i 's/\("routing_enable_auto_monitoring": \).*$/\\1true,/' /opt/smartbeat/config/algo.json"""
    enable_at = 'sed -i \'s/\("routing_enable_auto_monitoring": \).*$/\\1true,/\'' ' /opt/smartbeat/SmartBeatDSP/config/algo.json'
    reduce_nv = 'sed -i \'s/\("subject_detector_nightvision_factor": \).*$/\\10.01,/\'' ' /opt/smartbeat/SmartBeatDSP/config/algo.json'

    restart = 'systemctl restart smartbeat-monitor'

    # clik mous in the window
    # mouse to center of the screen
    screen_x, screen_y = pyautogui.size()
    pyautogui.moveTo(screen_x / 2, screen_y / 2)

    # pyautogui.write(update_threshold)
    # pyautogui.press('enter')

    # time.sleep(2)
    # pyautogui.write(update_retry)
    # pyautogui.press('enter')

    # time.sleep(2)
    # pyautogui.write(update_head)
    # pyautogui.press('enter')

    time.sleep(4)
    pyautogui.write(update_head)
    pyautogui.press('enter')

    # time.sleep(4)
    # pyautogui.write(restart)
    # pyautogui.press('enter')



    ## check parameter
    val = 0
    if check:
        subprocess.call(["osascript", "-e", 'tell application "Terminal" to activate'])
        get_json = 'cat /opt/smartbeat/config/algo.json'
        pyautogui.write(get_json)
        pyautogui.press('enter')
        copy_terminal_output()
        val = extract_parameter_from_clipboard('"weak_breathing_threshold"')
        if float(val)==5:
            print('success')
        else:
            print('failure')
    temp_df = pd.DataFrame({'acc_id': [acc_id], 'MAC': [MAC], 'weak_breathing_threshold': [val], 'success': [float(val)==5]})
    process_df = pd.concat([process_df, temp_df])
    process_df.to_csv('process_df.csv', index=False)

    # subprocess.call(["osascript", "-e", 'tell application "Terminal" to activate'])
    # cmd = 'exit'
    # pyautogui.write(cmd)
    # pyautogui.press('enter')


    #  close all drivers and terminals
    # driver.quit()
    # closee terminal window
    terminal.windows[0].close()
    n +=1







