import json
import pandas as pd
import matplotlib
import os
import tarfile
import shutil
import os
import re

matplotlib.rcParams['interactive'] = True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tarfile


def extract_dfs_from_log_file(log_path: str):
    dict = {
        'rut_s': None,
        'cnf_baby_dtct': None,
        'med_breath_clsf': None,
        'nbc_breath_clsf': None,
        'pwr': None,
        'baby_dtct_cnf': None,
        'head_dtct_cnf': 0,
        'usr_frq': None,
    }
    event_df = pd.DataFrame(columns=dict.keys())
    with open(log_path, "r") as file:
        for line in file:
            try:
                if 'ALGO_SEC' in line:
                    json_str = line.split('ALGO_SEC: ')[1].strip()
                    json_str = json_str.replace(
                        '"ctx": "{', '"ctx": {'
                    ).replace('}"', '}')
                    json_str = json_str.replace('[(', '[[').replace(')]', ']]')
                    json_data = json.loads(json_str)
                    cnf_baby_dtct = json_data['breath_clsf']['cnf']
                    med_breath_clsf = json_data['breath_clsf']['med']
                    nbc_breath_clsf = json_data['breath_clsf']['nbc']
                    baby_dtct_cnf = json_data['baby_dtct']['cnf']
                    pwr = json_data['pwr']
                    route = json_data['rut']["s"]
                    usr_frq = json_data['usr_frq']

                    # set dict sing this values
                    dict_temp = {
                        'route': route,
                        'cnf_baby_dtct': cnf_baby_dtct,
                        'med_breath_clsf': med_breath_clsf,
                        'nbc_breath_clsf': nbc_breath_clsf,
                        'pwr': pwr,
                        'baby_dtct_cnf': baby_dtct_cnf,
                        'usr_frq': usr_frq,
                    }

                    # add dict_temp to event_df
                    temp_df = pd.DataFrame([dict_temp])
                    try:
                        temp_df["head_dtct_cnf"] = event_df[
                            "head_dtct_cnf"
                        ].iloc[-1]
                    except IndexError:
                        temp_df["head_dtct_cnf"] = 0
                    event_df = pd.concat([event_df, temp_df], ignore_index=True)
                elif "ALGO_EVENT" in line:
                    match = re.search(r'"conf": (\d+\.\d+)', line)
                    if match:
                        head_dtct_cnf = float(match.group(1))
                        event_df["head_dtct_cnf"].iloc[-1] = head_dtct_cnf
            except json.JSONDecodeError as e:
                print(f'Error parsing JSON: {e} in line: {line}')

    return event_df


def plot_data(df, indices, user_id):

    factor = 0.2
    # Create the plot

    fontsize = 50 * factor
    label_size = 50 * factor
    title_size = 60 * factor
    # Set the background color based on the 'route' column
    colors = {
        'no breathing': '#ff9999',  # light red
        'calibrating decision': '#99ccff',  # light blue
        'calibrating': '#99ff99',  # light green
        'small motion': '#ffff99',  # light yellow
        'large motion': '#ffcc99',  # light orange
        'breathing micro': '#cc99ff',  # light purple
        'micro motion': '#ff99cc',  # light pink
    }
    df['color'] = df['route'].map(colors)
    patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in colors.items()
    ]

    breathing_classifier_high_threshold = 0.5
    subject_threshold = 0.4
    pwr_threshold = 450
    head_dtct_cnf_threshold = 0.1
    columns = [
        'med_breath_clsf',
        'pwr',
        'baby_dtct_cnf',
        'usr_frq',
        'head_dtct_cnf',
    ]
    thresholds = [
        breathing_classifier_high_threshold,
        pwr_threshold,
        subject_threshold,
        0,
        head_dtct_cnf_threshold,
    ]
    fig, axs = plt.subplots(len(columns), figsize=(100 * factor, 50 * factor))
    for idx, column in enumerate(columns):
        axs[idx].plot(
            df.loc[indices].index, df.loc[indices, column], color='black'
        )

        for i in range(len(indices) - 1):
            axs[idx].fill_between(
                [df.index[indices[i]], df.index[indices[i + 1]]],
                [0, 0],
                [df.loc[indices][column].max(), df.loc[indices][column].max()],
                color=df['color'][indices[i]],
            )

        axs[idx].set_xlabel('Index', fontsize=fontsize)
        axs[idx].set_ylabel(column, fontsize=fontsize)
        axs[idx].tick_params(axis='both', which='major', labelsize=label_size)
        axs[idx].set_title(f'{user_id}_{column}', fontsize=title_size)
        axs[idx].autoscale_on = True
        axs[idx].grid(True)
        axs[idx].legend(handles=patches, loc='upper right', fontsize=fontsize)
        if thresholds[idx] != 0:
            axs[idx].axhline(
                y=thresholds[idx],
                color='r',
                linestyle='--',
                label=f'{column} Threshold',
                linewidth=3,
            )

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"/Users/shirmilstein/Code/algo-log-analyzer/data/{user_id}/{user_id}_plot.png"
    )
    plt.show(block=True)


def extract_syslogs_from_path(src_path, dst_path, user_id):

    # TODO - limited to one file!
    # iterate over tgz files in the directory
    for file in os.listdir(src_path):
        if file.endswith(".tgz") and file.startswith(user_id):
            # Open the tar.gz file
            with tarfile.open(f"{src_path}/{file}", "r:gz") as tar:
                # Iterate through the members to find the syslog file
                for member in tar.getmembers():
                    if (
                        member.name.endswith('syslog')
                        and 'prev/varlog/' in member.name
                    ):
                        # Extract the syslog file to the destination path
                        member.name = os.path.basename(
                            member.name
                        )  # To avoid extracting full path
                        tar.extract(member, f"{dst_path}/{user_id}")
                        print(
                            f"Extracted {member.name} to {dst_path}/{user_id}"
                        )
                        return
                # If syslog file is not found, raise an exception
                raise FileNotFoundError("syslog file not found in the archive")


user_id = 'E06290422B33'
src_path = '/Users/shirmilstein/Code/algo-log-analyzer/temp'
dst_path = f'/Users/shirmilstein/Code/algo-log-analyzer/data'

# extract_syslogs_from_path(src_path, dst_path, user_id)
df = extract_dfs_from_log_file(f"{dst_path}/{user_id}/syslog")
df.to_csv(f"{dst_path}/{user_id}/{user_id}_data.csv", index=False)
df = pd.read_csv(f"{dst_path}/{user_id}/{user_id}_data.csv")
# # run and choose indexes
plot_data(df, range(7750, 7850), user_id=user_id)
# print('done')
