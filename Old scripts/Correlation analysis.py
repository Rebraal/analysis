from datetime import datetime
import numpy as np
import json as json
import os
import re
from dateutil import parser
from datetime import timedelta
import math
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
from scipy import sparse
import tkinter
from tkinter.filedialog import askopenfilename



data, keys, dataObj = None, None, {}
dataSet, dataList = None, []
weatherDataFolder = r'C:\Memento analysis\Data sources\Visualcrossing weather data/'
weatherDataNullDays = []
secInHours = 60 * 60
dateRange = ['2019-09-01', '2021-03-02']
dateList = []
startDateTimeDT = parser.parse(dateRange[0] + ' 04:00 UTC')
startDateTimeInMin = math.floor(datetime.timestamp(startDateTimeDT) / 60)
endDateTimeDT = parser.parse(dateRange[1] + ' 03:00 UTC')
endDateTimeInMin = math.floor(datetime.timestamp(endDateTimeDT) / 60)
stepCount1Min = math.ceil((endDateTimeDT - startDateTimeDT).total_seconds() / 60)
allDataBase, all_keys, errors = None, None, []
# base_dir = r'D:\Memento analysis, unstored\Correlated data/'
base_dir = r'D:\Memento analysis, unstored\Correlated data\intakes and weather top 60pc/'
csr_dir = r'D:\Memento analysis, unstored\Correlated data, significant CSRs/'
overview_dir = r'D:\Google Drive Sync\Programs\Python\Analysis\Results\Scikit analysis\2021-03-31 1400/'
all_results, file_names, significance_m = None, None, None
timeSteps = ['0.25', '0.50', '1', '2', '4', '8', '12', '18', '24', '30', '36', '48', '60', '72']
rebinSteps = [1, 2, 4, 8, 16, 32, 48, 72, 96, 120, 144, 192, 240, 288]
lenAllTimeSteps = len(timeSteps)
ipk, opk, allRebins, resultsCut, allData = None, None, None, None, [0,0,0,0]
all_nans = []
duplicate_list = []
fig, axs = None, None
overview_info, significant_data = None, None

def main():
    global significant_data, overview_info
    load_stuff()
    # heatmap_for_file('000')
    # create_overview_data(1560, 7)
    # create_overview_graph()
    if not os.path.exists(base_dir + 'Significant data.npz'):
        create_significant_data(base_dir + 'Significant data.npz')
    else:
        significant_data = {}
        d = np.load(base_dir + 'Significant data.npz', allow_pickle=True)
        for file in d.files:
            significant_data[file] = d[file][()]
        del(d)
    # create_variable_profile_scatter(6)
    # overview_info = create_overview_data(92, 7)
    # create_overview_graph(overview_info)
    # check_normality()


def load_stuff():
    global allDataBase, all_keys, errors, rowLen, ipk, opk, allRebins, resultsCut, allKeys, inputs, outputs

    base_dir = overview_dir  # fudges for all data to be read.

    allDataBase = np.load(base_dir + 'All data.npz', allow_pickle=True)['data'].tolist()
    allRebins = np.load(base_dir + 'All data.npz', allow_pickle=True)['rebins'].tolist()
    resultsCut = allDataBase[0].shape[0] + allDataBase[1].shape[0]

    allKeys = np.load(base_dir + 'All data.npz', allow_pickle=True)['keys'].tolist()
    ipk, opk = allKeys[0] + allKeys[1], allKeys[2] + allKeys[3]
    for i in range(len(allKeys)):
        allKeys[i] = np.array(allKeys[i])

    all_keys = ipk + opk

    inputs, outputs = sparse.vstack((allDataBase[0], allDataBase[1])), sparse.vstack((allDataBase[2], allDataBase[3]))


def create_overview_data(ipi, opi):
    """
    Retrieves r value for each offset and timeStep for ipi, opi
    Populates overview_info
    :param ipi: input key index
    :param opi: output key index
    :return:
    ('ts', np.unicode_, 4),
    ('raw', np.object_),
    ('min', np.float16),
    ('max', np.float16),
    ('pc', np.float16),
    ('csr_raw', np.object_),
    ('csr_norm', np.object_)
    """
    global overview_info
    tst = datetime.now()
    print('create_overview_data()', tst)

    overview_info = {'raw': [], 'norm': [], 'offsets': []}
    find_offset = re.compile('.*(\d\d\d).*')

    # for file in sorted(os.listdir(base_dir)):
    for file in sorted(os.listdir(overview_dir)):
        if 'Offset' in file:
            print(file)
            d = np.load(overview_dir + file, allow_pickle=True)['arr_0']
            overview_info['pc'] = d[0]['pc']
            v = [ar['raw'][ipi, opi] for ar in d]
            overview_info['norm'].append(
                [v[i] / -ar['min'] if v[i] < 0 else v[i] / ar['max'] for i, ar in enumerate(d)])
            overview_info['raw'].append(v)
            overview_info['offsets'].append([float(find_offset.match(file).group(1)) / 4])

    for key in overview_info:
        overview_info[key] = np.array(overview_info[key], dtype='f8')
    # overview_info['pc'] = d[0]['pc']
    overview_info['graph_title'] = str(ipi) + ' ' + ipk[ipi] + ' vs ' + str(opi) + ' ' + opk[opi]

    np.savez_compressed(csr_dir + overview_info['graph_title'], **overview_info)
    print('To construct combination overview: ', datetime.now() - tst, '\nCompleted at: ', datetime.now())
    return overview_info


def create_overview_graph(ipi=None, opi=None, self_norm=False):
    """
    Creates a plot of overview results data for a pair of variables, showing change in r with offset
    :return:
    """
    global fig, ax, overview_info
    if (ipi is not None) & (opi is not None):
        create_overview_data(ipi, opi)
    if overview_info is None:
        root = tkinter.Tk()
        oif = askopenfilename(parent=root,  title='Select overview_info file',
                              initialdir=r'D:\Memento analysis, unstored\Correlated data, significant CSRs')
        overview_info = np.load(oif, allow_pickle=True)
        root.destroy()

    plt.close()
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', plt.cm.gist_ncar(np.linspace(0.1, 0.9, overview_info['raw'].shape[1])))
    # fig.set_size_inches(23.5, 17)
    plt.get_current_fig_manager().window.state("zoomed")
    ax.set_title(overview_info['graph_title'])
    ax.set_xlabel('offset, h')
    ylabel = 'r-value, normalized per line' if self_norm else 'r-value'
    ax.set_ylabel(ylabel)

    graph_y_data = overview_info['norm'] if self_norm else overview_info['raw']

    plt.plot(overview_info['offsets'], graph_y_data)
    plt.legend(timeSteps, loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
# create_overview_graph()
# 92 - Co-Brazil, 7 - Sc-Tired
# 720 - tC-Banana, 473 - Da-Overview


def heatmap_for_file(offset):
    """
    Produce a 7 wide by 2 deep set of heatmaps for raw data from file
        offset as str"""

    global fig, axs, d
    d = np.load(base_dir + 'Offset - ' + offset + '.npz', allow_pickle=True)['arr_0']
    plt.close()
    fig, axs = plt.subplots(2, 7,
                            sharex=True, sharey=True
                            )
    # fig.set_size_inches(23.5, 17)
    plt.get_current_fig_danager().window.state("zoomed")
    axs_flat = axs.flat
    plt.pause(2)
    for i in range(lenAllTimeSteps):
        graph_data = d['raw'][i].astype('f8')
        pos = axs_flat[i].imshow(graph_data, cmap='Spectral')
        axs_flat[i].set(title=d['ts'][i] + 'h')
    fig.colorbar(pos, ax=axs_flat[i])
    plt.tight_layout()


def create_significant_data():
    """
    Loop through the result npzs and populate a csr for each timeStep with significant data in.
    So for any offset file, and for any timeStep within it, if any input / output pair has a significant value
    as created by the Data correlation routine, store its info.
    :return:
    updates significant_data:
        'counts' = occurrences of significance, array of shape=(len(ipk), len(opk))
        'sums_raw' = sum of the standardised values that are significant.
        'sums_norm' = for positive / negative r-values, new_r = r / max(all positive / negative rs in timeStep)
        counts / sums should then give the average significance of each significant ip vs op, and be heatmap-able
    """
    global significant_data, ret_obj
    tst = datetime.now()
    print('create_significant_data()', tst)

    significant_data = {'counts': sparse.csr_matrix(np.zeros((len(ipk), len(opk)), dtype='f2')),
                        'sums_raw': sparse.csr_matrix(np.zeros((len(ipk), len(opk)), dtype='f2')),
                        'sums_norm': sparse.csr_matrix(np.zeros((len(ipk), len(opk)), dtype='f2')),
                        'pc': 0
                        }

    for file in sorted(os.listdir(base_dir)):
        # if its a useable file
        if 'Offset' in file:
            print(file, datetime.now())
            d = np.load(base_dir + file, allow_pickle=True)['arr_0']
            for ar in d:
                significant_data['counts'] += ar['csr_raw'] != 0
                significant_data['sums_raw'] += ar['csr_raw']
                significant_data['sums_norm'] += ar['csr_norm']
            significant_data['pc'] = ar['pc']


    # add file to zip
    # filepath = 'fp'
    # with zipfile.ZipFile(filepath, 'a') as zipf:
    #     zipf.write(source_file, destination_in_zip_name)

    # deletion requires decompressing and recompressing.

    for key in ['counts', 'sums_raw', 'sums_norm']:
        dt = significant_data[key].tocoo()
        to_write = '\n'.join(ipk[dt.row[i]] + '\t' + opk[dt.col[i]] + '\t' + str(dt.data[i]) for i in np.argsort(dt.data))
        with open(base_dir + key + '.csv', 'w+') as f:
            f.write(to_write)

    np.savez_compressed(base_dir + 'Significant data', **significant_data)

    print('To construct create_significant_data: ', datetime.now() - tst, '\nCompleted at: ', datetime.now())


def condense_significant_data():
    """
    remove all lines and columns that are full of zeros and create new key lists accordingly.
    filter columns to only contain interesting outputs. I seem to have lost this json? file.
    :return:
    """
    global significant_data

    # find rows and columns with sum != 0
    ipknz_mask = np.array(np.sum(significant_data['counts'], axis=1)).flatten()
    opknz_mask = np.array(np.sum(significant_data['counts'], axis=0)).flatten()
    opknz_mask = opknz_mask != 0
    ipknz_mask = ipknz_mask != 0

    # mask off 0 sum data, update ip & op names and convert to useable coo
    opknz, ipknz = np.array(opk)[opknz_mask], np.array(ipk)[ipknz_mask]

    new_sums = significant_data['sums_raw'][ipknz_mask]
    new_sums = new_sums[:, opknz_mask]
    new_counts = significant_data['counts'][ipknz_mask]
    new_counts = new_counts[:, opknz_mask]
    new_sums, new_counts = new_sums.tocoo(), new_counts.tocoo()

    new_data = sparse.coo_matrix((new_sums.data / new_counts.data, (new_sums.row, new_sums.col)), shape=new_sums.shape)
    sdc = significant_data['counts'].tocoo()
    sig_av = significant_data['sums_raw'].data / significant_data['counts'].data
    sig_av_args = np.argsort(sig_av)
    sig_key_pairs = np.array([[ipk[sdc.row[i]], opk[sdc.col[i]], el]
                     for i, el in enumerate(sig_av)])[sig_av_args]

    # plot new_data


def create_significant_data_graph():
    """
    Using significant_data created by create_significant_data(), create a faux heatmap of important results.
    Dividing csr matrices seems to create problems, and we can plot coo matrices as a scatter more easily than as a
    heatmap. Due to plot order, I want the least important plotted first, with the larger, darker points plotted on top
    last.
    :return:
    """
    # can use coo format to create scatter plot, with cmap
    significant_data['sums_raw'] = significant_data['sums_raw'].tocoo()
    significant_data['sums_norm'] = significant_data['sums_norm'].tocoo()
    significant_data['counts'] = significant_data['counts'].tocoo()

    significant_data['mean_raw'] = sparse.coo_matrix(
        (significant_data['sums_raw'].data / significant_data['counts'].data,
         (significant_data['counts'].row, significant_data['counts'].col)),
        shape=significant_data['counts'].shape)

    significant_data['mean_norm'] = sparse.coo_matrix(
        (significant_data['sums_norm'].data / significant_data['counts'].data,
         (significant_data['counts'].row, significant_data['counts'].col)),
        shape=significant_data['counts'].shape)

    plt.close()
    g_d = significant_data['mean_raw']
    # sort data into least important first
    g_d_sort = np.argsort(g_d.data ** 2)

    plt.scatter(g_d.col[g_d_sort],
                g_d.row[g_d_sort],
                c=g_d.data[g_d_sort],
                cmap='coolwarm',
                s=(g_d.data[g_d_sort] * 5) ** 4,
                vmin=-1, vmax=1
                )
    plt.colorbar()
    x_max, y_max = np.max(significant_data['mean_raw'].col), np.max(significant_data['mean_raw'].row)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.xticks(np.arange(x_max+1), opk, rotation=30)
    plt.yticks(np.arange(y_max+1), ipk)
    plt.gca().invert_yaxis()
    plt.tight_layout()


def create_variable_profile_scatter(op):
    """
    Creates a slice for a single output of r-values across all inputs, offsets and timeSteps
    This only works for creating a profile with a fixed OP, because I'm in a hurry.
    :param op: output to create profiles for
    :return:
    """
    global variable_profile_data, current_output
    tst = datetime.now()
    print('add_significant_csrs()', tst)
    file_list = [file for file in sorted(os.listdir(base_dir)) if 'Offset' in file]
    current_output = [op, opk[op]]

    # (14 matrices of (offsets rows, ipk cols))
    dtype = np.dtype([('ts', np.unicode_, 4), ('data', np.float32, (len(file_list), len(ipk)))])
    variable_profile_data = np.zeros(lenAllTimeSteps, dtype=dtype)

    for offset, file in enumerate(file_list):
        print(file, datetime.now())
        d = np.load(base_dir + file, allow_pickle=True)['arr_0']
        for i, ar in enumerate(d):
            variable_profile_data[i]['ts'] = ar['ts']
            variable_profile_data[i]['data'][offset] = ar['raw'][:, op]

    np.savez_compressed(csr_dir + 'Profile_data ' + str(op) + ' ' + opk[op], variable_profile_data)

    print('To complete create_variable_profile_scatter: ', datetime.now() - tst, '\nCompleted at: ', datetime.now())


def check_correlations(ipi, opi, offset, timeStep):
    """
    For ipi vs opi, at offset and timeStep:
    Create scatter plot of ipi vs opi, points sized according to frequency
    :param ipi: input index to check, ie Co - Brazil is index 92
    :param opi: output index to check
    :param offset: from graph in hours
    :param timeStep: from graph as a float
    :return:
    """
    # for dev
    # ipi, opi, offset, timeStep = 92, 7, 10, 6
    offset = int(offset / 0.25)
    timeStep = timeSteps.index(str(timeStep))
    x_label, y_label = ipk[ipi], opk[opi]
    title = ipk[ipi] = x_label + ' vs ' + y_label + ', ' + str(offset) + ' offset, ' + timeSteps[timeStep] + ' timeStep'
    g_data = rebin_data(offset, timeStep)

    # correct opi for rebinned data
    opi += len(ipk)
    x_data, y_data = g_data[ipi].toarray(), g_data[opi].toarray()
    g_stacked = np.vstack((x_data, y_data)).T

    g_bins, g_counts = np.unique(g_stacked, axis=0, return_counts=True)
    plt.scatter(g_bins[:, 0], g_bins[:, 1], c=g_counts, cmap='copper', s=g_counts**2)
    plt.colorbar()
    plt.get_current_fig_manager().window.state("zoomed")
    plt.gcf().suptitle(title)
    plt.gcf().canvas.set_window_title(title)
    plt.gca().set(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()


def get_significant_r_value(n, p):
    t = stats.t.ppf(1 - (p / 2), n)
    return t / np.sqrt(np.square(t) + (n - 2))


def check_normality():
    """
    Check normality of data for all inputs and outputs. Assumed all timeSteps are normal if the first is.
    Might check last.
    :return: list of True (normal) or False
    """

    alpha = 0.001  # if p is less than this, it's a normal distribution

    all_data_stacked = sparse.vstack((allDataBase[0], allDataBase[1], allDataBase[2], allDataBase[3]))
    nz_rows = np.array([True if len(row.data) > 10 else False for row in all_data_stacked])
    data = all_data_stacked[nz_rows]

    # ct = 0
    # for i, row in enumerate(data):
    #     print(i)
    #     if stats.normaltest(row.toarray()[0])[1] < alpha:
    #         ct += 1
    # print(ct, ' of ', data.shape[0], ' are normal for offset=0, ts=0')
    # ct = 0
    # for i, row in rebin_data(0, 13, use_data=all_data_stacked[nz_rows]):
    #     print(i)
    #     if stats.normaltest(row.toarray()[0])[1] < alpha:
    #         ct += 1
    # print(ct, ' of ', data.shape[0], ' are normal for offset=0, ts=0')

    normals = np.array([True if stats.normaltest(row.toarray()[0])[1] < alpha
                        else False for row in all_data_stacked[nz_rows]])
    print(np.sum(normals), ' of ', len(normals), ' are normal for offset=0, ts=0')

    for ts in range(1, 14):
        normals = np.array([True if stats.normaltest(row.toarray()[0])[1] < alpha
                            else False for row in rebin_data(0, ts, use_data=all_data_stacked[nz_rows])])
        print(np.sum(normals), ' of ', len(normals), ' are normal for offset=0, ts=', ts)


def multivariable_regression():
    global results
    time_now = datetime.now()
    print(time_now)
    # X = sparse.vstack((inputs, np.ones(inputs.shape[1]))).T
    X = inputs.T
    y = outputs[795].toarray()
    print('Started lsqr')
    il = 1000
    # make sure r and r[il] are created.
    r[il]['s'] = sparse.linalg.lsqr(X, y, iter_lim=il)[0]
    r[il]['f'] = r[il]['s'] * inputs  # fitted
    r[il]['r'] = (y - r[il]['f'])[0]  # residuals
    print(datetime.now() - time_now)


def spam():
    data = np.load(base_dir + 'Offset - 000.npz', allow_pickle=True)['arr_0']

    coo = np.load(base_dir + 'Offset - 000.npz', allow_pickle=True)['csrs'][()]['72'].tocoo()
    coo_data = np.vstack((coo.data, coo.row, coo.col))
    coo_data_sorted = coo_data[:, np.argsort(coo_data[0])]

    with open(r'D:\Memento analysis, unstored\Correlated data\Keys to check.json') as f:
        outputs = json.load(f)

    ipk, opk = allData['inputKeys'], allData['outputKeys']
    all_results_norm = (all_results - np.min(all_results, axis=1, keepdims=True)) / \
                       np.max(all_results, axis=1, keepdims=True) + \
                       np.arange(all_results.shape[1]) / 10


def rebin_data(offset, timeStep, use_data=None):
    """
    Rebins data to offset for visualising.
    :param offset: 0-287 - offsets of 0.25h from input / output @ t=0 to input @ t=0, output @ t=287
    :param timeStep: 0-13 - rebins of data. 0 = 0.25h bins, 13 = 72h
    :return:
    """

    use_data = allDataBase if use_data is None else use_data
    negRebinCuts = [-math.ceil(offset / rebinSteps[i]) for i in range(0, lenAllTimeSteps)]
    negOffset = -offset
    i = timeStep

    if offset > 0:
        allData[0] = use_data[0][:, :-offset]
        allData[1] = use_data[1][:, :-offset]
        allData[2] = use_data[2][:, offset:]
        allData[3] = use_data[3][:, offset:]

    return sparse.vstack((allData[0] * allRebins[i][:negOffset, :negRebinCuts[i]],
                         (allData[1] * allRebins[i][:negOffset, :negRebinCuts[i]]) / rebinSteps[i],
                         allData[2] * allRebins[i][:negOffset, :negRebinCuts[i]],
                         (allData[3] * allRebins[i][:negOffset, :negRebinCuts[i]]) / rebinSteps[i]
                         ))


if __name__ == '__main__':
    pass
    main()
