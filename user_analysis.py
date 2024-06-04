import json
import pandas as pd
import matplotlib

matplotlib.rcParams['interactive'] = True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def extract_dfs_from_log_file(log_path: str, only_algo_sec=False):
    dict = {
        'rut_s': None,
        'cnf_baby_dtct': None,
        'med_breath_clsf': None,
        'nbc_breath_clsf': None,
        'pwr': None,
        'baby_dtct_cnf': None,
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
                    event_df = pd.concat([event_df, temp_df], ignore_index=True)

            except json.JSONDecodeError as e:
                print(f'Error parsing JSON: {e} in line: {line}')

    return event_df


def plot_data(df, indices, user_id):

    breathing_classifier_high_threshold = 0.5
    subject_threshold = 0.4
    pwr_threshold = 450
    factor = 0.2
    # Create the plot
    fig, axs = plt.subplots(3, figsize=(100 * factor, 50 * factor))
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

    # Plot med_breath_clsf
    axs[0].plot(
        df.loc[indices].index, df.loc[indices, 'med_breath_clsf'], color='black'
    )
    for i in range(len(indices) - 1):
        axs[0].fill_between(
            [df.index[indices[i]], df.index[indices[i + 1]]],
            [0, 0],
            [df['med_breath_clsf'].max(), df['med_breath_clsf'].max()],
            color=df['color'][indices[i]],
        )
    axs[0].set_xlabel('Index', fontsize=fontsize)
    axs[0].set_ylabel('med_breath_clsf', fontsize=fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=label_size)
    axs[0].set_title(f'{user_id}_med_breath_clsf', fontsize=title_size)
    axs[0].autoscale_on = True
    axs[0].grid(True)
    axs[0].legend(
        handles=patches, loc='upper right', fontsize=fontsize
    )  # Add the legend to the plot
    axs[0].axhline(
        y=breathing_classifier_high_threshold,
        color='r',
        linestyle='--',
        label='Breathing Classifier High Threshold',
        linewidth=5,
    )

    # Plot pwr
    axs[1].plot(df.loc[indices].index, df.loc[indices, 'pwr'], color='black')
    for i in range(len(indices) - 1):
        axs[1].fill_between(
            [df.index[indices[i]], df.index[indices[i + 1]]],
            [0, 0],
            [df.loc[indices]['pwr'].max(), df.loc[indices]['pwr'].max()],
            color=df['color'][indices[i]],
        )
    axs[1].set_xlabel('Index', fontsize=fontsize)
    axs[1].set_ylabel('pwr', fontsize=fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=label_size)
    axs[1].set_title(f'{user_id}_pwr', fontsize=title_size)
    axs[1].autoscale_on = True
    axs[1].grid(True)
    axs[1].axhline(
        y=pwr_threshold,
        color='r',
        linestyle='--',
        label='pwr_threshold',
        linewidth=5,
    )

    # Plot baby_dtct_cnf
    axs[2].plot(
        df.loc[indices].index, df.loc[indices, 'baby_dtct_cnf'], color='black'
    )
    for i in range(len(indices) - 1):
        axs[2].fill_between(
            [df.index[indices[i]], df.index[indices[i + 1]]],
            [0, 0],
            [df['baby_dtct_cnf'].max(), df['baby_dtct_cnf'].max()],
            color=df['color'][indices[i]],
        )
    axs[2].set_xlabel('Index', fontsize=fontsize)
    axs[2].set_ylabel('baby_dtct_cnf', fontsize=fontsize)
    axs[2].tick_params(axis='both', which='major', labelsize=label_size)
    axs[2].set_title(f'{user_id}_baby_dtct_cnf', fontsize=title_size)
    axs[2].autoscale_on = True
    axs[2].grid(True)
    axs[2].axhline(
        y=subject_threshold,
        color='r',
        linestyle='--',
        label='Subject Threshold',
        linewidth=5,
    )

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"/Users/shirmilstein/Code/algo-log-analyzer/data/{user_id}/{user_id}_plot.png"
    )
    plt.show(block=True)


user_id = 'E062906E3449'

# create a df #TODO update according to the data
# log_path = f"/Users/shirmilstein/Code/algo-log-analyzer/data/{user_id}/prev/varlog/syslog"
# event_df0 = extract_dfs_from_log_file(log_path, only_algo_sec=False)
# log_path = f"/Users/shirmilstein/Code/algo-log-analyzer/data/{user_id}/prev 2/varlog/syslog"
# event_df2 = extract_dfs_from_log_file(log_path, only_algo_sec=False)
# log_path = f"/Users/shirmilstein/Code/algo-log-analyzer/data/{user_id}/prev 3/varlog/syslog"
# event_df3 = extract_dfs_from_log_file(log_path, only_algo_sec=False)
# concat_df = pd.concat([event_df0, event_df2, event_df3])
# concat_df.to_csv('data.csv', index=False)


df = pd.read_csv('data.csv')
# run and choose indexes
plot_data(df, range(2000, 2200), user_id=user_id)
print('done')
