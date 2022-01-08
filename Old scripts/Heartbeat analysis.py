# 2021-03-20 15:41
# works to display 4 graphs of breakfast, lunch, dinner and combined for a single day.
# Couldn't get multiple days to work and display one after the other.

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
from matplotlib import dates
from matplotlib.ticker import MultipleLocator
import tkinter
from tkinter.filedialog import askopenfilename

heartrate_source_data = None
heartrate_source_file = None
mealtime_source_file =None
mealtime_source_data, mealtimes_info, new_file = None, None, None
first_time, last_time = None, None
fig, axs, lines = None, None, None
intervals, average_data = None, None
control_start, control_end = '2021-03-14 11:00', '2021-03-17 23:00'
save_graph_file = r'D:\Google Drive Sync\Programs\Python\Analysis\Heartrate analysis\Save/'
discard_graph_file = r'D:\Google Drive Sync\Programs\Python\Analysis\Heartrate analysis\Discard/'
intervals = ['start-30', 'start-15', 'start', 'end', 'end+15', 'end+30', 'end+60', 'end+90']
control_averages = {'Breakfast': [66.2, 69.1, 78.2, 76.5, 67, 68.8, 73.6],
                    'Lunch': [77, 74.6, 77.1, 72.6, 67.6, 68.2, 71],
                    'Dinner': [66.7, 69.2, 74.8, 76.2, 72.2, 67.6, 67.8],
                    'Average': [70, 71, 76.7, 75.1, 68.9, 68.2, 70.8]
                    }

def main():
    read_heartrate_data()
    read_mealtime_data()
    data_analysis()
    # plot_heartrate_data()
    # plot_averages()
    plot_day_averages()


def read_heartrate_data():
    """
    Take heartrate information from a file and stick it into a structured numpy array. Not sure I like these.
    :return:
    """
    global heartrate_source_file, heartrate_source_data

    if heartrate_source_file is None:
        # this enables the askopenfilename window to have focus above other stuff.
        root = tkinter.Tk()
        heartrate_source_file = askopenfilename(parent=root,  title='Select heartrate source file',
                                                initialdir=r'D:\Memento analysis, unstored')
        root.destroy()

    tst = datetime.now()
    print('read_heartrate_data()', tst)
    dtype = np.dtype([('date', np.unicode_, 10), ('time', np.unicode_, 5), ('bpm', np.int16)])
    heartrate_source_data = np.genfromtxt(heartrate_source_file, delimiter=',', skip_header=1, dtype=dtype)
    ts_dtype = np.dtype([('ts', np.float64)] +
                        heartrate_source_data.dtype.descr +
                        [('bpm60', np.float16)])
    tmp = np.empty(heartrate_source_data.shape, dtype=ts_dtype)
    tmp['date'] = heartrate_source_data['date']
    tmp['time'] = heartrate_source_data['time']
    tmp['bpm'] = heartrate_source_data['bpm']
    tmp['ts'] = np.array([dates.date2num(parser.parse(tmp['date'][i] + ' ' + tmp['time'][i] + ' UTC'))
                          for i in range(len(tmp))])
    tmp['bpm60'] = running_mean(tmp['bpm'], 60)
    heartrate_source_data = tmp
    del(tmp)
    print(datetime.now() - tst, ' to complete\nCompleted at: ', datetime.now())


def read_mealtime_data():
    """
    Read csv export of individual entries, determine times of meals by looking for meat, veg, fruit
    Create array of timestamps to add lines to graph from.
    :return:
    """
    global mealtime_source_file, mealtime_source_data, mealtimes_info, first_time, last_time, new_file

    if mealtime_source_file is None:
        # this enables the askopenfilename window to have focus above other stuff.
        root = tkinter.Tk()
        mealtime_source_file = askopenfilename(parent=root, title='Select mealtime source file',
                                               initialdir=r'D:\Google Drive Sync\Programs\Python\Analysis')
        root.destroy()

    # note that the raw Memento export needs to be opened in Calc, \n replaced with \n, regexp, and saved with
    # editting options, | as the delimiter.
    # This is now obsolete, as long as the reformat_mealtime_data() rountine keeps working...

    new_file = reformat_mealtime_data()
    # new_file = mealtime_source_file
    # Note that U100 will cut most fields sadly short. This is sufficient for the moment, I feel as it eases the burden
    mealtime_source_data = np.genfromtxt(new_file, delimiter='|', comments="###", filling_values='',
                                         dtype='U100', encoding=None)

    # strip out the useful columns
    headers = mealtime_source_data[0]
    useful_headers = {'Date stamp', 'Intake - vegetables', 'Intake - meat and fish', 'Intake - fruit'}
    # 'Intake - misc'}
    col_nums = [i for i, val in enumerate(headers) if val in useful_headers]

    # note that mealtimes are sorted in the opposite direction to heartrate data, so this corrects that
    mealtime_source_data = np.flip(mealtime_source_data[1:, tuple(col_nums)], axis=0)

    first_time = max(heartrate_source_data[0]['date'] + ' ' + heartrate_source_data[0]['time'],
                     mealtime_source_data[0, 0])
    last_time = min(heartrate_source_data[-1]['date'] + ' ' + heartrate_source_data[-1]['time'],
                    mealtime_source_data[-1, 0])


    # need to look at 30 minutes before meal starts, duration, 15, 30, 60 90 minutes afterward.
    dtype = np.dtype([('start', np.float64), ('end', np.float64),
                      ('start-30', np.float64), ('start-15', np.float64),
                      ('end+15', np.float64), ('end+30', np.float64), ('end+60', np.float64), ('end+90', np.float64),
                      ('start_dt', np.object_), ('end_dt', np.object_),
                      ('str', str, 16)])
    mealtimes_info = np.array([(None, None,
                                None, None,
                                None, None, None, None,
                                None, parser.parse(e[0] + ' UTC'),
                                e[0]
                                )
                               for e in mealtime_source_data
                               if (len(''.join(e)) > 16) & (first_time <= e[0] <= last_time)], dtype=dtype)
    mealtimes_info['start_dt'] = np.array([get_meal_start(e) for e in mealtimes_info['str']], dtype=np.object_)
    mealtimes_info['start'] = dates.date2num(mealtimes_info['start_dt'])
    mealtimes_info['end'] = dates.date2num(mealtimes_info['end_dt'])
    tmp = dates.date2num(np.array([[e['start_dt'] - timedelta(minutes=30),
                                    e['start_dt'] - timedelta(minutes=15),
                                    e['end_dt'] + timedelta(minutes=15),
                                    e['end_dt'] + timedelta(minutes=30),
                                    e['end_dt'] + timedelta(minutes=60),
                                    e['end_dt'] + timedelta(minutes=90)
                                  ] for e in mealtimes_info]))
    for i, e in enumerate(['start-30', 'start-15', 'end+15', 'end+30', 'end+60', 'end+90']):
        mealtimes_info[e] = tmp[:, i]


def reformat_mealtime_data():
    """
    Rather than open and resave in LibreOffice, this should:
    replace all \n with "\n", so rather than line breaks which mess up genfromtext, it should read.
    replace all "," with "|" so genfromtext can read it properly.
    :return:
    """

    with open(mealtime_source_file) as f:
        data = f.read().replace('\n', '\\n').replace('\\n"*start*"', '\n"*start*"')
        regex = re.compile('"(,+)"')
        # data = re.sub('"(,+)"', lambda m: '"' + ('|' * len(m.group(1))) + '"', data)
        data = re.sub('"(,+)"', lambda m: '|' * len(m.group(1)), data)
    new_file = os.path.dirname(mealtime_source_file) + '/'\
               + os.path.basename(mealtime_source_file)[:-4] + ' newlined.csv'
    with open(new_file, 'w+') as f:
        f.write(data)
    return new_file


def get_meal_start(date_time):
    """
    Takes a date time string, works out the start of the meal from it and returns a matplotlib date time float
    :param date_time: date time in string format
    :return: date time in matplotlib format
    """
    mealtime = date_time[-5:]
    if '07:45' < mealtime < '09:00':
        return parser.parse(date_time + ' UTC').replace(hour=7, minute=45)
    elif '12:30' < mealtime < '13:45':
        return parser.parse(date_time + ' UTC').replace(hour=12, minute=30)
    elif '18:00' < mealtime < '19:00':
        return parser.parse(date_time + ' UTC').replace(hour=18, minute=00)
    else:
        return parser.parse(date_time + ' UTC') - timedelta(minutes=30)


def data_analysis():
    """
    need to look at 30 minutes before meal starts, duration, 15, 30, 60 90 minutes afterward.
    :return:
    """
    global intervals, average_data, control_average, average_data
    intervals = ['start-30', 'start-15', 'start', 'end', 'end+15', 'end+30', 'end+60', 'end+90']

    # for each gap in intervals, calculate mean and plot.
    # s-30 to s-15 plotted at s-15.

    len_heartrate_source_data = len(heartrate_source_data)
    average_data = []
    for meal in mealtimes_info:
        # get indices of slices. This can fall over when there's no data - watch taken off.
        slices = np.searchsorted(heartrate_source_data['ts'], list(meal[intervals]), side='right')
        # check for no data errors - rectify by going to the next last available data. Not ideal, better than nowt.
        # print(meal['str'], '\n', slices)
        for i in range(1, len(slices)):
            if slices[i-1] == slices[i]:
                # print('o', slices)
                slices[i-1] -= 1
                # print('n', slices)
        # print(meal['str'], '\n', slices)
        # create slices
        slices = np.array([slice(slices[i-1], slices[i]) for i in range(1, len(slices))
                           if slices[i] < len_heartrate_source_data])

        average_tmp = np.zeros((len(intervals) - 1, 2))
        for i in range(len(slices)):
            slc_data = heartrate_source_data['bpm'][slices[i]]
            average_tmp[i] = [meal[intervals][i+1], np.sum(slc_data) / len(slc_data)]
        average_data.append({'str': meal['str'], 'data': average_tmp})

    return intervals, average_data


def plot_heartrate_data():
    """
    Plot heart rate data onto a line plot, adding in rolling averages
    :return:
    """
    global fig, ax
    tst = datetime.now()
    print('plot_heartrate_data()', tst)

    plt.close()
    fig, ax = plt.subplots()
    plt.get_current_fig_manager().window.state("zoomed")
    title = 'Heart rate over time'
    ax.set(title=title, xlabel='Time', ylabel='BPM')

    # last_x = input('how many days past to plot?')
    # last_x *= 24 * 60 * 60
    # prefer to use rebin rather than rolling average, as rebin reflects instantaneous data.
    ts30 = heartrate_source_data['ts'][::30]
    hr_min30 = rebin_min_max(heartrate_source_data['bpm'], 30, fn_max=False)
    hr_max30 = rebin_min_max(heartrate_source_data['bpm'], 30, fn_max=True)
    hr_avg30 = rebin_mean(heartrate_source_data['bpm'], 30)

    p_avg, = plt.plot(ts30, hr_avg30, linewidth=1, color='black')
    p_fill = plt.fill_between(ts30, hr_min30, hr_max30, color='#bbbbbb')
    plt.plot(heartrate_source_data['bpm'])

    for m in mealtimes_info:
        p_start = plt.axvline(m['start'], c='#00ff00', linewidth=0.75)
        p_end = plt.axvline(m['end'], c='#ff0000', linewidth=0.75)
    for m in average_data:
        p_analysis, = plt.plot(m['data'][:, 0], m['data'][:, 1], c='#0000ff', linewidth=2)

    ax.xaxis.set_major_locator(dates.HourLocator(interval=6))
    # ax.xaxis.set_minor_locator(dates.MinuteLocator(interval=15))
    dtfmt = dates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax.xaxis.set_major_formatter(dtfmt)
    fig.autofmt_xdate(rotation=0, ha='center', which='both')
    ax.tick_params(axis='x', size=0.5)
    ax.set_ylim(40, 120)
    plt.legend([p_avg, p_fill, p_start, p_end, p_analysis],
               ['Average,\n30min bins', 'Range', 'Meal start', 'Meal end', 'Meal\nanalysis'],
               loc='upper right', bbox_to_anchor=(1, 1), facecolor='white', framealpha=0.3)
    # plt.pause(2)
    plt.tight_layout()

    print(datetime.now() - tst, ' to complete\nCompleted at: ', datetime.now())
# plot_heartrate_data()


def plot_averages():
    global fig, ax, results
    plt.close()

    analysis_start_date = input('yyyy-mm-dd as start date or (a)ll ')
    if analysis_start_date != 'a':
        analysis_start_date += ' 04:00'
        print('Start date: ', analysis_start_date)
        data_to_analyse = [e for e in average_data if e['str'] >= analysis_start_date]
    else:
        data_to_analyse = average_data

    # set up for showing multiple consecutive plots
    plt.axis()
    plt.ion()
    plt.show()
    plt.plot()
    plt.draw()
    # plt.pause(5)
    plt.pause(0.5)

    # fig, ax = plt.subplots()

    # plt.get_current_fig_manager().window.state("zoomed")
    this_manager = plt.get_current_fig_manager()
    title = 'Average heart rates over mealtimes'

    fig = plt.gcf()
    this_manager.window.wm_geometry('681x696+676+0')
    results = {'save': [], 'discard': []}

    for m in data_to_analyse:
        ax = plt.gca()
        ax.set_ylim(60, 100)
        # ax.set_prop_cycle('color', plt.cm.binary(np.linspace(0.1, 0.9, control_average.shape[0])))
        ax.set(title=m['str'], xlabel='Stage', ylabel='BPM')
        plt.xticks(np.arange(len(intervals)-1), labels=intervals[1:])
        plt.grid(b=True, which='both')

        plt.plot(m['data'][:, 1], color='#aaaaaa')
        plt.plot(control_average, color='red', linewidth=2)

        plt.tight_layout()

        plt.draw()
        plt.pause(0.1)
        response = input('(s)ave, (d)iscard or (b)reak? ')
        if response == 'b':
            break
        elif response == 's':
            results['save'].append(m['str'])
            plt.savefig(save_graph_file + m['str'].replace(':', '') + '.pdf', format='pdf')
        else:
            results['discard'].append(m['str'])
            plt.savefig(discard_graph_file + m['str'].replace(':', '') + '.pdf', format='pdf')
        plt.clf()

    plt.close()
# plot_averages()


def plot_day_averages():
    """
    create an object with control averages,
    breakfast, lunch, dinner
    overall average

    Data will probably be incomplete most runs
    :return:
    """
    global fig, axs, days_to_analyse, lines
    plt.close()

    analysis_start_date = input('yyyy-mm-dd to plot or (t)oday ? No quotes needed around date. ')
    food_added = input('Food added today? No quotes needed. ')
    if analysis_start_date == 't':
        start = datetime.now().strftime('%Y-%m-%d') + ' 04:00'
        end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') + ' 04:00'
    else:
        start = analysis_start_date + ' 04:00'
        end = (parser.parse(start + ' UTC') + timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
    data_to_analyse = [e for e in average_data if start <= e['str'] < end]
    # can't get this to work for multiple dates, and probably not worth wasting much time on it, so, commenting out
    # elif analysis_start_date != 'a':
    #     analysis_start_date += ' 04:00'
    #     print('Start date: ', analysis_start_date)
    #     data_to_analyse = [e for e in average_data if e['str'] >= analysis_start_date]
    # else:
    #     data_to_analyse = average_data

    # cack-handed, but I can't think clearly
    days_to_analyse = [{'day': '', 'day_data': []}]
    for e in data_to_analyse:
        e['meal'] = 'Breakfast' if e['str'][11:] < '12:00' else 'Lunch' if e['str'][11:] < '18:00' else 'Dinner'
        if e['str'][:10] == days_to_analyse[-1]['day']:
            days_to_analyse[-1]['day_data'].append(e)
        else:
            days_to_analyse.append({'day': e['str'][0:10], 'day_data': [e]})
    days_to_analyse.pop(0)

    foods_added = [food_added]

    plt.axis()
    plt.ion()
    plt.show()
    # plt.plot()
    plt.draw()
    plt.pause(0.5)
    # fig, axs = plt.subplots(2, 2)
    axs = [plt.subplot(2, 2, i) for i in range(1, 5)]
    fig = plt.gcf()

                            # sharex=True, sharey=True)
    plt.get_current_fig_manager().window.state("zoomed")

    formatter = ['k:', 'k--', 'k']
    if len(days_to_analyse) == 0:
        raise("You've probably forgotten to get Zepp to update the files.")
    for index, day in enumerate(days_to_analyse):
        # this assumes the day data is sorted in ascending order, as it should be
        figure_title = parser.parse(day['day']).strftime('%Y-%m-%d %A')\
                       + ', ' + foods_added[index] + ' added'
        fig.suptitle(figure_title)
        plt.gcf().canvas.set_window_title(figure_title)

        for ax in axs:
            # ax.tick_params(axis='y', which='both')
            ax.yaxis.set_minor_locator(MultipleLocator(5))
            # ax.set_xticks(np.arange(len(intervals) - 1))
            ax.set_xticks(np.arange(len(intervals)) - 0.5)
            # ax.set_xticklabels(intervals[1:])
            ax.set_xticklabels(intervals)
            ax.grid(b=True, which='both')
            ax.set_ylim(50, 100)
            ax.set_xlim(-0.5, 7.5)

        for meal in day['day_data']:
            if meal['meal'] == 'Breakfast':
                ax = axs[0]
            elif meal['meal'] == 'Lunch':
                ax = axs[1]
            else:
                ax = axs[2]
            ax.plot(meal['data'], c='black')
            ax.plot(control_averages[meal['meal']], c='red')
            ax.set(title=meal['meal'] + ' ' + meal['str'][11:], xlabel='Stage', ylabel='BPM')

        ax = axs[3]
        lines = []
        for i, meal in enumerate(day['day_data']):
            l = ax.plot(meal['data'], formatter[i])
            lines.append([l[0], meal['meal']])
        lines = np.array(lines)
        ax.plot(control_averages['Average'], c='red', label='Overall\naverage')
        ax.legend(lines[:, 0], lines[:, 1], loc='upper right', bbox_to_anchor=(1, 1), facecolor='white', framealpha=0.3)
        ax.set(title='All meals against average', xlabel='Stage', ylabel='BPM')
        plt.tight_layout()
        plt.draw()
        plt.show()
        plt.pause(0.1)
        # response = input('(s)ave, or (d)iscard?')
        # if response == 's':
        #     plt.savefig(r'D:\Google Drive Sync\Programs\Python\Analysis\Heartrate analysis\Day views/'
        #                 + figure_title + '.pdf', format='pdf')
        # plt.clf()

    # plt.close()
# plot_day_averages()



def spam():
    # averages = [np.average(np.array([m['data'][i, 1] for m in average_data]))
    #             for i in range(len(average_data[0]['data']))]
    # plt.plot(averages)

    #ax.set_prop_cycle('color', plt.cm.gist_ncar(np.linspace(0.1, 0.9, how_many_lines_i_want)))
    pass


def running_mean(array, N):
    cs = np.cumsum(np.insert(array, 0, np.zeros(N)))
    return (cs[N:] - cs[:-N]) / float(N)


def rebin_mean(data, binLen):
    slices, step = np.linspace(0, data.shape[0], math.ceil(data.shape[0] / binLen),
                               endpoint=False, retstep=True, dtype=np.intp)
    return np.add.reduceat(data, slices) / step


def rebin_min_max(data, binLen, fn_max=True):
    return np.maximum.reduceat(data, np.r_[:len(data):binLen]) if fn_max \
        else np.minimum.reduceat(data, np.r_[:len(data):binLen])


if __name__ == '__main__':
    pass
    main()
