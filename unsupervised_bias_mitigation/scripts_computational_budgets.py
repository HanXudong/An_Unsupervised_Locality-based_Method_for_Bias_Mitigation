import pandas as pd
import os
from datetime import datetime


def estimate_computational_budget(file_path):

    with open(file_path) as f:
        file = f.readlines()

    n_repeat = 0
    for line in file:
        if line ==  '{}\n':
            n_repeat += 1

    start_time = file[0][:19]

    end_time = file[-1][:19]

    start_time = datetime.strptime(start_time.strip(), "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time.strip(), "%Y-%m-%d %H:%M:%S")

    (end_time-start_time).seconds/n_repeat

    return {
        "num_repeat":n_repeat,
        "average_time":(end_time-start_time).seconds/n_repeat
    }

exp_file_dir = r"replace_with_your_dir"

file_names = [f for f in os.listdir(exp_file_dir)]
output_dir = "results/dev_run_time.pkl"

_run_time_list = []
for f_name in file_names:
    file_path = os.path.join(exp_file_dir, f_name)
    if f_name[-4:] == ".out":
        pass
    else:
        continue
    try:
        _run_time = estimate_computational_budget(file_path)
    except:
        print(f_name)
        _run_time = {'num_repeat': 0, 'average_time': 0}
    _run_time["f_name"] = f_name
    _run_time_list.append(
        _run_time
    )
run_time_df = pd.DataFrame(_run_time_list)
run_time_df.to_pickle(output_dir)