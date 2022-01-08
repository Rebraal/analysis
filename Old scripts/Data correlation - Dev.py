import numpy as np
from scipy import sparse
# from scipy.signal import savgol_filter
from datetime import datetime
import math
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import dates
# import matplotlib.ticker as ticker
# from joblib import Parallel, delayed
from dateutil import parser
from datetime import timedelta

# source_folder = r'D:\Google Drive Sync\Programs\Python\Analysis\Results\Scikit analysis\2021-03-31 1400/'
# dump_folder = r'C:\Python dumps/'
source_folder = '../Source data/'
dump_folder = '../Python dumps/'
try:
    os.mkdir(dump_folder)
except FileExistsError:
    pass

time_steps = [0.25, 0.5, 1, 2, 4, 8, 12, 18, 24, 30, 36, 48, 60, 72]
rebinSteps = [1, 2, 4, 8, 16, 32, 48, 72, 96, 120, 144, 192, 240, 288]
# load data from file, arrays of length ~50k
all_data = np.load(source_folder + 'All data.npz', allow_pickle=True)['data'].tolist()
# all_inputs, all_outputs = sparse.vstack((all_data[0], all_data[1])), sparse.vstack((all_data[2], all_data[3]))
all_keys = np.load(source_folder + 'All data.npz', allow_pickle=True)['keys'].tolist()
all_rebins = np.load(source_folder + 'All data.npz', allow_pickle=True)['rebins'].tolist()
ipk, opk = np.array(all_keys[0] + all_keys[1]),  np.array(all_keys[2] + all_keys[3])

# strip Sn, Sd, Sa from opk.
opk_stripped = np.array([i for i in range(len(opk))
                         if ('Sn' not in opk[i]) and ('Sd' not in opk[i]) and ('Sa' not in opk[i])])

# note that all of these are for corrected values, so negative is better

interesting_data_names = [
    'Sy-00 10 01 Contentment',
    'Sy-00 10 02 Tranquility',
    'Sy-00 10 03 Enthusiasm',
    'Sy-00 20 01 Mental clarity',
    'Sy-00 20 02 Processing speed',
    'Sy-00 20 03 Focus',
    'Sy-00 30 01 Energy',
    'Sy-00 30 02 Motivation',
    'Sy-00 30 03 Fatigue',
    'Sy-00 30 04 Sleepiness',
    'Sy-00 50 01 Anxiety',
    'Sy-00 50 02 Depression',
    'Sy-00 60 01 Irritation',
    'Sy-00 60 02 Confrontational',
    'Da-Overview'
]
opk_l = opk.tolist()
interesting_data_keys = [opk_l.index(n) for n in interesting_data_names]
supplements = [i for i in range(len(ipk)) if 'Su-' in ipk[i]]


def rebin_output(ts_ind, data_ind):
    """
    returns all_data[data_ind] rebinned to time_steps[ts_ind], assuming data_ind belongs to all_data[2 or 3]
    """
    if ts_ind > 0:
        rebin = all_rebins[ts_ind]
        count = all_rebins[ts_ind][:, 0].sum()
    else:
        rebin, count = 1, 1
        
    if data_ind >= all_data[2].shape[0]:
        y_data = all_data[3][data_ind - all_data[2].shape[0]] * rebin / count
    else:
        y_data = all_data[2][data_ind] * rebin
    y_data.reshape(1, y_data.shape[1]).tocsr()
    return y_data


def rebin_input(ts_ind, data_ind):
    """
    returns all_data[data_ind] rebinned to time_steps[ts_ind], assuming data_ind belongs to all_data[2 or 3]
    """
    if ts_ind > 0:
        rebin = all_rebins[ts_ind]
        count = all_rebins[ts_ind][:, 0].sum()
    else:
        rebin, count = 1, 1

    if data_ind >= all_data[0].shape[0]:
        y_data = all_data[1][data_ind - all_data[1].shape[0]] * rebin / count
    else:
        y_data = all_data[0][data_ind] * rebin
    y_data.reshape(1, y_data.shape[1]).tocsr()
    return y_data


def rebin_array(data, binLen, axis=0, add=False):
    """
    takes an array and for each segment of binLen, either returns a sum or an average.
    rebins array, if add is False, calculates mean."""
    slices, step = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen),
                               endpoint=False, retstep=True, dtype=np.intp)
    den = 1 if add else step
    return (np.add.reduceat(data, slices, axis=axis) / den).astype('f8')


def plot_original_data(ts_ind=3, data_ind=8, close=True):
    """
    for data[data_ind], time_steps[ts_ind]
    plot data against time.
    """
    steps_per_day = int(24 / time_steps[ts_ind])
    y_data = rebin_output(ts_ind, data_ind).toarray()[0]
    # x_data = np.arange(0, y_data.shape[0] * time_steps[ts_ind], time_steps[ts_ind])  # as hours
    x_data = np.arange(0, y_data.shape[0] * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days
    x_label_locs = x_data[::steps_per_day]
    x_labels = np.arange(x_label_locs.shape[0])
    title = opk[data_ind]
    if close:
        plt.close()
        fig = plt.figure()
    fig = plt.gcf()
    plt.tight_layout()
    ax = plt.gca()
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set(title=title, xlabel='Time, days', ylabel='Value')
    plt.plot(x_data, running_mean(y_data, 30))
    # plt.plot(x_data, y_data, linewidth=0.5)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(12 * 24))
    # plt.xticks(x_label_locs, x_labels, rotation=90)
    ax.set_xlim(x_data.min(), x_data.max())
    plt.tight_layout()

    # return x_data, y_data
    return title
# x_data, y_data = plot_original_data()


def running_mean(array, N):
    cs = np.cumsum(np.insert(array, 0, np.zeros(N)))
    return (cs[N:] - cs[:-N]) / float(N)


def write_key_list(key_list, file_name):
    """
    write key_list to a csv with file_name
    """
    with open(dump_folder + file_name + '.txt', 'w+') as f:
        f.write('\n'.join([str(i) + ' ' + key_list[i] for i in range(len(key_list))]))


def plot_significant_original_data(ts_ind=13):
    """
    load, and plot to pdf, all significant scatter graphs
    """
    tst1 = datetime.now()
    print('plot_significant_original_data()', tst1)

    # plt.switch_backend('Agg')
    # plt.ioff()
    # for data_ind in to_scatter:
    # input_ts_ind, ts_ind = ts_ind, 0
    x_data_len = len(rebin_output(ts_ind, 0).toarray()[0])
    # x_data = np.arange(0, y_data.shape[0] * time_steps[ts_ind], time_steps[ts_ind])  # as hours
    x_data = np.arange(0, x_data_len * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days
    title = 'Significant markers'
    plt.tight_layout()
    ax = plt.gca()
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set(title=title, xlabel='Time, days', ylabel='Value - lower is better')
    ax.set_xlim(x_data.min(), x_data.max())
    matplotlib.rc('image', cmap='tab20')

    offsets = np.arange(len(significant_keys)+1) * 5
    offsets = np.flip(offsets)

    avg_data = np.zeros(x_data_len)
    for index in range(len(significant_keys)):
        data_ind = significant_keys[index]
        y_data = rebin_output(ts_ind, data_ind).toarray()[0]
        # check for corrected #P#
        if y_data.min() < 0:
            y_data += 5
            zero_fill = np.argmax(y_data < 5)
            y_data[:zero_fill] = 0
        avg_data += y_data
        # monthly running mean. Index offsets vertically.
        plt.plot(x_data, running_mean(y_data, 10)+offsets[index], label=str(data_ind) + ' ' + opk[data_ind])

    plt.hlines(offsets, x_data.min(), x_data.max(), color='black', linewidth=0.2)

    y_avgs = running_mean(2 * avg_data / len(significant_keys), 10)
    plt.plot(x_data, y_avgs, label='Average of outputs')

    pc = 10
    low_p, high_p, rm_avg = np.percentile(y_avgs, pc), np.percentile(y_avgs, 100-pc), np.mean(y_avgs)
    plt.hlines([low_p, rm_avg, high_p], 0, 1000, color='red', linewidth=0.2)
    # pos = y_avgs > np.percentile(y_avgs, 100-pc)
    # neg = y_avgs < np.percentile(y_avgs, pc)

    ax.set_ylim(0, offsets[0]+5)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), facecolor='white', framealpha=0.3)
    plt.pause(0.5)
    plt.tight_layout()

    print('finished', datetime.now() - tst1)


def zero_crossings(data):
    """
    return indices in array where following element has opposite sign, ie crosses zero
    """
    return np.where(np.diff(np.signbit(data)))[0]


def plot_all_rebins(data_ind=198):
    """
    for all rebins of data_ind, plot with running mean
    """

    fig, ax = plt.subplots()
    title = opk[data_ind]
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set(title=title, xlabel='Time, days', ylabel='Value')

    offsets = np.flip(np.arange(len(all_rebins)) * 5)
    # colors = plt.cm.tab20b(np.arange(20))
    colors = plt.cm.brg(np.linspace(0.1, 0.9, 14))

    ts_ind = 0
    
    for ts_ind in range(len(all_rebins)):
        # print(ts_ind)
        mean_length = math.ceil(30 * 24 / time_steps[ts_ind])
        y_data = rebin_output(ts_ind, data_ind).toarray()[0]
        # check for corrected #P#
        if y_data.min() < 0:
            y_data += 5
            zero_fill = np.argmax(y_data < 5)
            y_data[:zero_fill] = 0
        x_data = np.arange(0, y_data.shape[0] * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days

        plt.plot(x_data, y_data + offsets[ts_ind], c=colors[ts_ind], linewidth=0.2)
        plt.plot(x_data, running_mean(y_data, mean_length)*2 + offsets[ts_ind], c=colors[ts_ind], label=time_steps[ts_ind],
                 linewidth=0.5)

    plt.hlines(offsets, x_data.min(), x_data.max(), color='black', linewidth=0.1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), facecolor='white', framealpha=0.3)
    ax.set_xlim(x_data.min(), x_data.max())
    ax.set_ylim(0, offsets[0]+5)
    plt.pause(1)
    plt.tight_layout()


def isolate_stages(ts_ind=8, data_ind=199):
    """
    plot data, rolling mean of 30 days and a line from each peak / trough of rolling mean
    ts_ind = 8 => 24 hour bins
    """

    plt.close()
    fig, ax = plt.subplots()
    title = str(data_ind) + ' ' + opk[data_ind]
    title = title.replace('?', 'qm').replace('*', 'tx')
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set(title=title + ' stages', xlabel='Time, days', ylabel='Value')

    rm_days = 30
    mean_length = math.ceil(rm_days * 24 / time_steps[ts_ind])
    y_data = rebin_output(ts_ind, data_ind).toarray()[0]
    # check for corrected #P#
    if y_data.min() < 0:
        y_data += 5
        zero_fill = np.argmax(y_data < 5)
        y_data[:zero_fill] = 0
    x_data = np.arange(0, y_data.shape[0] * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days
    y_data_rm = running_mean(y_data, mean_length)
    y_data_rm_diffs = np.insert(np.diff(y_data_rm), 0, 0)
    y_data_rm_grad = (y_data_rm_diffs * 10) + 2.5
    crossings = zero_crossings(y_data_rm_diffs)
    y_data_phases = y_data_rm[crossings]
    y_data_phases_grad = np.insert(np.diff(y_data_phases), 0, 0)
    y_data_phases_grad *= (1 / np.max(np.abs(np.array(y_data_phases_grad.max(), y_data_phases_grad.min()))))
    y_data_phases_grad += 2.5
    x_data_masked = x_data[crossings]

    lines = {
        'data': plt.plot(x_data, y_data, c='grey', linewidth=0.2, label='Data'),
        'rm': plt.plot(x_data, y_data_rm, c='grey', linewidth=0.3, label='Running mean,\n30 days'),
        'phases': plt.plot(x_data[crossings], y_data_rm[crossings], c='blue', linewidth=0.5, label='Phases'),
        'grad': plt.plot(x_data_masked, y_data_phases_grad, c='red', linewidth=0.5, label='Running mean gradient'),
        'hlines': [plt.hlines(2.5, 0, 1000, color='black', linewidth=0.2)]
    }

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), facecolor='white', framealpha=0.3)
    ax.set_xlim(x_data.min(), x_data.max())
    ax.set_ylim(0, 5)
    plt.pause(1)
    plt.tight_layout()

    # acquire list of dates to investigate
    # for positives and negatives
    #   start date = end date - 30
    #   end date

    pc = 10
    pos = y_data_phases_grad > np.percentile(y_data_phases_grad, 100-pc)
    neg = y_data_phases_grad < np.percentile(y_data_phases_grad, pc)
    start_date = parser.parse('2019-09-01')
    op_pos, op_neg = 'Positives\n', '\n\nNegatives\n'
    x_pos, x_neg = x_data_masked[pos], x_data_masked[neg]
    op_pos += '\n'.join([(start_date + timedelta(days=x_pos[i]-rm_days)).strftime('%Y-%m-%d') + '\t' +
                       (start_date + timedelta(days=x_pos[i])).strftime('%Y-%m-%d') + '\t' +
                       '%.3f' % y_data_phases[pos][i]
                       for i in range(len(x_pos))])
    op_neg += '\n'.join([(start_date + timedelta(days=x_neg[i]-rm_days)).strftime('%Y-%m-%d') + '\t' +
                       (start_date + timedelta(days=x_neg[i])).strftime('%Y-%m-%d') + '\t' +
                       '%.3f' % y_data_phases[neg][i]
                       for i in range(len(x_neg))])

    with open(dump_folder + 'to check ' + title + '.csv', 'w+') as f:
        f.write(op_pos + op_neg)


def calculate_average_data(ts_ind):
    """
    for a ts_ind, return x_data and avg_data, where avg_data is the mean of the interesting data
    """

    x_data_len = len(rebin_output(ts_ind, 0).toarray()[0])
    x_data = np.arange(0, x_data_len * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days
    avg_data = np.zeros(x_data_len)
    for index in range(len(interesting_data_keys)):
        data_ind = interesting_data_keys[index]
        y_data = rebin_output(ts_ind, data_ind).toarray()[0]
        # check for corrected #P#
        if y_data.min() < 0:
            y_data += 5
            zero_fill = np.argmax(y_data < 5)
            y_data[:zero_fill] = 0
        if opk[data_ind] in ['Da-Overview', 'Da-Overview adjusted **3']:
            y_data /= y_data.max()
        avg_data += y_data

    return x_data, avg_data / len(interesting_data_keys)


def plot_original_data_averages():
    """
    load, and plot to pdf, all significant scatter graphs
    """
    tst1 = datetime.now()
    print('plot_significant_original_data_averages()', tst1)

    title = 'Significant data, averages'
    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set(title=title, xlabel='Time, days', ylabel='Value - lower is better')
    seg_len = 30

    x_data_0, y_data_0 = calculate_average_data(0)
    y_data_0_rm = running_mean(y_data_0, int((24 / 0.25) * seg_len))
    # x_data_72, y_data_72 = calculate_average_data(13)
    # y_data_72_rm = running_mean(y_data_72, int((24 / 72) * seg_len))

    raw_line_0 = plt.plot(x_data_0, y_data_0, label='0.25h bins, raw', color='grey', linewidth=0.1)
    rm_line_0 = plt.plot(x_data_0, y_data_0_rm, label='0.25h bins,\n30d running mean', color='black', linewidth=0.5)
    # plt.plot(x_data_0[::1000], y_data_0_rm[::1000], label='0.25h bins,\n30d running mean, [::1000]', color='red', linewidth=1)
    # plt.plot(x_data_72, y_data_72_rm, label='72h bins,\n30d running mean', color='black', linewidth=0.5)
    x_min, x_max = x_data_0.min(), x_data_0.max()

    # create violin plots
    seg_elements = int(30 * (24 / time_steps[0]))  # 30 day segment lengths
    chunks = np.arange(seg_elements, len(y_data_0), seg_elements)
    y_data_0_v = np.split(y_data_0, chunks)
    x_data_0_v = np.hstack((np.arange(seg_len, x_max, seg_len), x_max))
    v_plots = plt.violinplot(y_data_0_v, positions=x_data_0_v-(seg_len / 2), widths=seg_len*0.95, showmeans=True)
    v_means_y = np.array([e[0, 1] for e in v_plots['cmeans'].get_segments()])
    av_line = plt.plot(x_data_0_v, v_means_y, label='0.25h bins,\n30d averages', color='red', linewidth=1)
    plt.vlines(x_data_0_v, 0, 5, color='grey', linewidth=0.1)

    # set line properties
    for obj in ['cmins', 'cmaxes', 'cbars']:
        v_plots[obj].set(colors='black', linewidth=0.3)

    pc = 10
    low_p, high_p, rm_avg = np.percentile(y_data_0_rm, pc), np.percentile(y_data_0_rm, 100 - pc), np.mean(y_data_0_rm)
    pc_lines = plt.hlines([low_p, rm_avg, high_p], x_min, x_max, color='black', linewidth=0.2)

    ax.set_xlim(30, x_max)
    ax.set_ylim(0.5, 2.5)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), facecolor='white', framealpha=0.3)
    plt.pause(0.5)
    plt.tight_layout()

    print('finished', datetime.now() - tst1)


def plot_daily_supplement_quantities_over_data_averages():
    """
    plot a line for each supplement over all time, normalised magnitude
    """

    title = 'Daily supplement intake over interesting average data'
    plt.close()
    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set(title=title, xlabel='Time, days', ylabel='Quantity')

    ts_ind = 8
    x_data_len = len(rebin_input(ts_ind, 0).toarray()[0])
    x_data = np.arange(0, x_data_len * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days

    # 'Sl-' prefix data seems to be broken. No Solgar folate taken since London days, pretty sure.
    # supplements = [i for i in range(len(ipk)) if ('Su-' in ipk[i]) or ('Sl-' in ipk[i])]
    supplements = [i for i in range(len(ipk)) if 'Su-' in ipk[i]]
    # supplements = supplements[:5]
    offsets = np.flip(np.arange(len(supplements)))
    colors = plt.cm.spring(np.linspace(0.1, 0.9, len(supplements)))
    # colors = plt.cm.tab20b(np.arange(20))

    # set up plot and background of imshow
    seg_len = 1
    bins = 20
    x_data_0, y_data_0 = calculate_average_data(0)
    x_min, x_max = x_data_0.min(), x_data_0.max()
    seg_elements = int(seg_len * (24 / time_steps[0]))  # 30 day segment lengths
    chunks = np.arange(seg_elements, len(y_data_0), seg_elements)
    y_data_0_v = np.split(y_data_0, chunks)
    x_data_0_v = np.hstack((np.arange(seg_len, x_max, seg_len), x_max))
    y_chunks_means = np.array([np.mean(l) for l in y_data_0_v])
    # plt.imshow(np.tile(y_chunks_means, (2, 1)), cmap='binary', aspect='auto', extent=[0, x_max, 0, offsets[0]+1])
    y_d = np.digitize(y_chunks_means, np.linspace(y_chunks_means.min(), y_chunks_means.max(), bins), right=True)
    bg_col = plt.cm.binary(np.linspace(0.1, 0.9, bins))
    y_chunks_mean_normalized = (y_chunks_means - y_chunks_means.min()) / (y_chunks_means.max() - y_chunks_means.min()) * offsets[0]
    y_m_norm = plt.plot(x_data_0_v, y_chunks_mean_normalized, color='black', linewidth=1, visible=False)
    bg_rect = [ax.add_patch(Rectangle((x_data_0_v[i], 0),
                                       x_data_0_v[i+1] - x_data_0_v[i],
                                       offsets[0]+1,
                                       facecolor=bg_col[y_d[i]]
                                      )) for i in range(0, len(x_data_0_v)-1)]

    line_data, label_data = [], []
    for i, k in enumerate(supplements):
        y_data = rebin_input(ts_ind=8, data_ind=k).toarray()[0]
        y_max = y_data.max()
        y_data /= y_max
        line_data.append(plt.plot(x_data, y_data + offsets[i], label=str(y_max) + ' ' + ipk[k], linewidth=1, c=colors[i]))
        label_data.append(plt.text(x_max+1, offsets[i]+0.3, ipk[k], fontsize=6))

    # plt.vlines(x_data_0_v, 0, offsets[0]+1, color='black', linewidth=0.1)
    plt.hlines(offsets+1, 0, x_max, color='white', linewidth=0.2)
    ax.set_xlim(seg_len, x_data.max())
    ax.set_ylim(0, offsets[0] + 1)
    ax.set_position([0.05, 0.1, 0.75, 0.85])
    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), facecolor='white', framealpha=0.3)
    plt.pause(0.5)
    # plt.tight_layout()


def check_supplement_data():
    """
    plot a line for each supplement over all time over a background of average interesting data
    """
    title = 'Daily supplement intake over interesting average data'
    plt.close()
    # plt.switch_backend('Qt5Agg')
    fig, ax = plt.subplots(1,1)
    # fig, ax = plt.gcf(), plt.gca()
    # plt.get_current_fig_manager().window.state("zoomed")
    fig.set_size_inches([13.66, 3.32])
    ax.set(title=title, xlabel='Time, days', ylabel='Quantity')

    ts_ind = 8
    x_data_len = len(rebin_input(ts_ind, 0).toarray()[0])
    x_data = np.arange(0, x_data_len * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days

    # 'Sl-' prefix data seems to be broken. No Solgar folate taken since London days, pretty sure.
    # supplements = [i for i in range(len(ipk)) if ('Su-' in ipk[i]) or ('Sl-' in ipk[i])]

    supplements = supplements[:2]

    # set up plot and background of imshow
    seg_len = 1
    bins = 20
    x_data_0, y_data_0 = calculate_average_data(0)
    x_min, x_max = x_data_0.min(), x_data_0.max()
    seg_elements = int(seg_len * (24 / time_steps[0]))  # 30 day segment lengths
    chunks = np.arange(seg_elements, len(y_data_0), seg_elements)
    y_data_0_v = np.split(y_data_0, chunks)
    x_data_0_v = np.hstack((np.arange(seg_len, x_max, seg_len), x_max))
    y_chunks_means = np.array([np.mean(l) for l in y_data_0_v])
    y_d = np.digitize(y_chunks_means, np.linspace(y_chunks_means.min(), y_chunks_means.max(), bins), right=True)
    bg_col = plt.cm.binary(np.linspace(0.1, 0.9, bins))
    # y_chunks_mean_normalized = (y_chunks_means - y_chunks_means.min()) / (y_chunks_means.max() - y_chunks_means.min()) * y_data.max()
    # y_m_norm = plt.plot(x_data_0_v, y_chunks_mean_normalized, color='black', linewidth=1, visible=False)
    bg_rect = [ax.add_patch(Rectangle((x_data_0_v[i], 0),
                                       x_data_0_v[i+1] - x_data_0_v[i],
                                       1000,
                                       facecolor=bg_col[y_d[i]]
                                      )) for i in range(0, len(x_data_0_v)-1)]

    ax.set_xlim(seg_len, x_data.max())

    plt.ion()
    plt.draw()
    plt.pause(5)
    line_data, label_data, supp_line = [], [], None
    for i, k in enumerate(supplements):
        if supp_line is not None:
            supp_line[0].remove()
        y_data = rebin_input(ts_ind=8, data_ind=k).toarray()[0]
        supp_line = ax.plot(x_data, y_data, linewidth=1, c='black')
        ax.set_ylim(0, y_data.max())
        subtitle = ipk[k].replace('?', 'qm').replace('*', 'tx')
        ax.set(title=title + '\n' + subtitle)
        plt.tight_layout()
        plt.savefig(dump_folder + 'Supplement graphs/' + subtitle + '.pdf', format='pdf')
        plt.draw()
        while True:
            try:
                plt.pause(0.1)
            except KeyboardInterrupt:
                break


def multiple_graph_display():
    """
    display multiple plots consequetively, pausing for input. This works, but does not allow user interaction with
    the plot
    """
    # plt.switch_backend('Qt5Agg')
    plt.close()
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    plt.pause(1)
    for n in [10,20]:
        plt.plot(np.arange(n))
        ax.set(xlim=(0,n), ylim=(0,n))
        plt.tight_layout()
        plt.draw()
        while True:
            try:
                plt.pause(0.1)
            except KeyboardInterrupt:
                break
        # ip = input('(k)eep or (l)ose? ')
        ax.cla()



    plt.close()
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    plt.pause(1)
    for n in [10,20]:
        plt.plot(np.arange(n))
        ax.set(xlim=(0,n), ylim=(0,n))
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        ip = input('(k)eep or (l)ose? ')
        ax.cla()

# main()
# analyse_all()
# plot_all_distributions_scatter()
# plot_all_rebins(data_ind=199)
# isolate_stages()