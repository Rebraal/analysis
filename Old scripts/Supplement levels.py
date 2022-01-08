import numpy as np
from scipy import sparse
from datetime import datetime
import math
import os
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from dateutil import parser
from datetime import timedelta

# source_folder = r'D:\Google Drive Sync\Programs\Python\Analysis\Results\Scikit analysis\2021-03-31 1400/'
# dump_folder = r'C:\Python dumps/'
source_folder = '../Source data/'
dump_folder = '../Python dumps/'

def hld():

    time_amt = [
        ['2021-04-07 08:25', 12],
        ['2021-04-07 13:25', 12],
        ['2021-04-07 17:25', 12],
        ['2021-04-08 08:30', 18],
        ['2021-04-08 17:35', 18],
        ['2021-04-09 08:38', 18],
        ['2021-04-09 18:42', 18],
        ['2021-04-10 08:36', 15],
        ['2021-04-10 17:55', 15],
        ['2021-04-11 08:17', 5],
        ['2021-04-11 13:10', 5],
        ['2021-04-11 19:10', 5],
        ['2021-04-12 08:27', 10],
        ['2021-04-12 13:57', 10],
        ['2021-04-12 18:30', 10],
        ['2021-04-13 08:24', 12],
        ['2021-04-13 13:43', 12],
        ['2021-04-13 17:49', 12],
        ['2021-04-14 09:35', 15],
        ['2021-04-14 13:16', 15],
        ['2021-04-14 18:38', 15],
        ['2021-04-15 09:35', 10],
        ['2021-04-15 11:35', 2],
        ['2021-04-15 13:16', 10],
        ['2021-04-15 15:16', 2],
        ['2021-04-15 18:38', 10],
        ['2021-04-15 21:00', 2],
        ['2021-04-15 04:00', 5],
        ['2021-04-16 09:35', 5],
        ['2021-04-16 11:35', 5],
        ['2021-04-16 13:16', 5],
        ['2021-04-16 15:16', 5],
        ['2021-04-16 18:38', 5],
        ['2021-04-16 21:00', 5],
    ]
    sec_in_hour = 60 * 60
    half_life = 2 * 60  # min

    y_data = np.array([0])
    for i in range(0, len(time_amt)-1):
        timespan = math.floor((parser.parse(time_amt[i+1][0]) - parser.parse(time_amt[i][0])).seconds / 60)
        time_data = np.arange(timespan, dtype='float64')
        n0 = y_data[-1] + time_amt[i][1]
        y_data = np.concatenate((y_data, (n0 * 0.5 ** (time_data / half_life))))
    y_data = y_data[1:]
    x_data = np.arange(len(y_data)) + 8.5 * 60

    plt.close()
    plt.plot(x_data, y_data, c='black')
    plt.hlines([0.1, 1.1, 8], 0, 250*60, color='black', linestyles='dashed')
    times = np.array([0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240]) * 60
    plt.vlines(times, 0, 20, colors='red')
    plt.vlines(times[1:] - 4 * 60, 0, 17, colors='green', linestyles='dashed')
    plt.vlines(times[1:]-6*60, 0, 15, colors='red', linestyles='dashed')
    plt.vlines(times[1:]-12*60, 0, 10, colors='red', linestyles='dashed')
    ax = plt.gca()
    ax.set_xlim(0, 250*60)
    plt.tight_layout()


    # nt = n0 * 0.5 ** (time / half_life)
