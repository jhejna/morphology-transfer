import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from stable_baselines.results_plotter import window_func
from stable_baselines.bench.monitor import load_results
from .loader import BASE

EPISODES_WINDOW = 100

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]

def ts2xy(timesteps, xaxis, yaxis='r'):
    """
    Modified to let you get keyword values.
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
    y_var = timesteps[yaxis].values
    return x_var, y_var

def get_subdirs(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def generate_plots(dirs, xaxis=X_TIMESTEPS, yaxis='r', title=None, labels=None, num_timesteps=None, subsample=None, individual=True):
    for i in range(len(dirs)):
        if not dirs[i].startswith('/'):
            dirs[i] = os.path.join(BASE, dirs[i])

    # If pointing to a single folder and that folder has many results, use that as dir
    if len(dirs) == 1 and len(get_subdirs(dirs[0])) > 1:
        dirs = [os.path.join(dirs[0], subdir) for subdir in get_subdirs(dirs[0])]
    
    # Make everything reproducible by sorting. Can comment out later for organization.
    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in dirs]

    dirs, labels = zip(*sorted(zip(dirs, labels), key=lambda x: x[0]))

    print("Dirs", dirs)
    for i, folder in enumerate(dirs):
        # If directory contains 1 folder, and none of those folders have params.json, move down.
        while True:
            contents = get_subdirs(folder)
            if any(['params.json' in os.listdir(os.path.join(folder, c)) for c in contents]):
                break
            folder = os.path.join(folder, contents[0])
        sns.set_context(context="paper", font_scale=1.5)
        sns.set_style("darkgrid", {'font.family': 'serif'})
        runs = [os.path.join(folder, r) for r in get_subdirs(folder)]
        xlist, ylist = [], []
        for run in runs:
            timesteps = load_results(run)
            if num_timesteps is not None:
                timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
            x, y = ts2xy(timesteps, xaxis, yaxis=yaxis)
            if x.shape[0] >= EPISODES_WINDOW:
                x, y = window_func(x, y, EPISODES_WINDOW, np.mean)
            xlist.append(x)
            ylist.append(y)
        if individual:
            for i, (xs, ys) in enumerate(zip(xlist, ylist)):
                sns.lineplot(x=xs, y=ys, label='s' +str(i))
        else:
            # Zero-order hold to align the data for plotting
            joint_x = sorted(list(set(np.concatenate(xlist))))
            combined_x, combined_y = [], []
            for xs, ys in zip(xlist, ylist):
                cur_ind = 0
                zoh_y = []
                for x in joint_x:
                    # The next value matters
                    if cur_ind < len(ys) - 1 and x >= xs[cur_ind + 1]:
                        cur_ind += 1
                    zoh_y.append(ys[cur_ind])
                if subsample:
                    combined_x.extend(joint_x[::subsample])
                    combined_y.extend(zoh_y[::subsample])
                else:
                    combined_x.extend(joint_x)
                    combined_y.extend(zoh_y)
            data = pd.DataFrame({xaxis : combined_x, yaxis: combined_y})
            sns.lineplot(x=xaxis, y=yaxis, data=data, ci="sd", sort=True, label=labels[i])

        print("Completed folder", folder)

    if title:
        plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout(pad=0)
