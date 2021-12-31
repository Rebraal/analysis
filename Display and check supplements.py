# 2021-05-26 13:21
# updating all_data directly, without messing around with individual or overview entries. Need then to tack new data
# onto the end of all_data. Don't think there's any point trying to correct entries before 2020-06 - too far back
#
# running this file creates a graph of a supplement and its intake
# double clicking an area on the graph zooms in, double right clicking zooms back out
# single clicking on a dot selects it to be edited with uad(new_value)
# use Display and check inputs.py with cd('YYYY-MM-DD') to look at what was going on before editting.
#
# 2021-05-28
# reworking to use a tk gui to display overview information and to use as an input.
#
# 2021-12-30 21:41
# this clearly isn't finished. Half works. Window sizing fails due to nonsense with Spyder using
# a separate backend. Has to be set to tkinter to fit in with this,but window stuff is written
# for Qt5

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib import dates
from dateutil import parser
from datetime import timedelta
from datetime import datetime
import pytz
import tkinter as tk
import json as json
import os

# plt.switch_backend('TkAgg')
# plt.switch_backend('Qt5Agg')

source_folder = '../Source data/'
dump_folder = '../Python dumps/'
source_folder_overview = '../Source data/Clean JSON overview entries/'
time_steps = [0.25, 0.5, 1, 2, 4, 8, 12, 18, 24, 30, 36, 48, 60, 72]
rebinSteps = [1, 2, 4, 8, 16, 32, 48, 72, 96, 120, 144, 192, 240, 288]
start_date = parser.parse('2019-09-01 04:00')
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
# could include 'Sl-' (sublinguals, with duration) if the data was working
# could include 'Sp-' (sprays, with location)
supplements = [i for i in range(len(ipk)) if 'Su-' in ipk[i]]
seg_len = 1  # length of interval to calculate interesting data for in days

graph_data, points_to_check = None, []
callback, bnext, bprev = None, None, None
fig, ax = None, None
x_data, bg_x, bg_y_norm, supp_line = None, None, None, None
text_data, bbox = {}, None
t, text_box = None, {}
x_data_len, x_dates_raw = None, None
tkgui, root = {}, {}


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
    if data_ind >= all_data[0].shape[0]:
        y_data = all_data[1][data_ind - all_data[1].shape[0]]
        if ts_ind > 0:
            rebin = all_rebins[ts_ind]
            count = rebin[:, 0].sum()
            y_data = y_data * (rebin / count)

    else:
        y_data = all_data[0][data_ind]
        if ts_ind > 0:
            rebin = all_rebins[ts_ind]
            y_data = y_data * rebin
    y_data.reshape(1, y_data.shape[1]).tocsr()
    return y_data


def running_mean(array, N):
    cs = np.cumsum(np.insert(array, 0, np.zeros(N)))
    return (cs[N:] - cs[:-N]) / float(N)


def create_graph_background(dx, dy):
    """Set up graph background"""
    # plt.switch_backend('TkAgg')
    # plt.switch_backend('Qt5Agg')
    # fig.set_size_inches([13.66, 3.32])
    # fig.set_size_inches([13.66, 6.6])
    fig_m = plt.get_current_fig_manager()
    # plt.get_current_fig_manager().window.state("zoomed")  # TkAgg
    plt.get_current_fig_manager().window.state("normal")  # TkAgg
    # fig_m.window.showMaximized()  # Qt5
    # x, y, dx, dy = fig_m.window.geometry().getRect()
    # fig_m.window.setGeometry(0, 35, dx, int(dy*0.5)-35)
    ax.set(xlabel='Time, days', ylabel='Quantity')

    ts_ind = 8
    x_data_len = len(rebin_input(ts_ind, 0).toarray()[0])
    x_data = np.arange(0, x_data_len * time_steps[ts_ind], time_steps[ts_ind]) / 24  # as days
    x_dates = dates.date2num([start_date + timedelta(days=e) for e in x_data])
    x_dates_raw = dates.date2num([start_date + timedelta(hours=e) for e in
                                  np.arange(0, len(rebin_input(0, 0).toarray()[0])) * 0.25])

    # set up plot and background of imshow
    bins = 20

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

    x_data_0, y_data_0 = calculate_average_data(0)
    x_min, x_max = x_data_0.min(), x_data_0.max()
    seg_elements = int(seg_len * (24 / time_steps[0]))  # 30 day segment lengths
    chunks = np.arange(seg_elements, len(y_data_0), seg_elements)
    y_data_0_v = np.split(y_data_0, chunks)
    x_data_0_v = np.hstack((np.arange(seg_len, x_max, seg_len), x_max))
    # x_dates_0_v = x_data_0_v
    x_dates_0_v = dates.date2num([start_date + timedelta(days=e) for e in x_data_0_v])
    y_chunks_means = np.array([np.mean(l) for l in y_data_0_v])
    y_chunks_mean_normalized = 1 - (y_chunks_means - y_chunks_means.min()) / (y_chunks_means.max() - y_chunks_means.min())

    def add_patches(ax, X, Y, bins):
        bg_col = plt.cm.binary(np.linspace(0.1, 0.9, bins))
        y_d = np.digitize(Y, np.linspace(Y.min(), Y.max(), bins), right=True)
        [ax.add_patch(Rectangle((X[i], 0),
                                X[i + 1] - X[i],
                                1000,
                                facecolor=bg_col[y_d[i]]
                                )) for i in range(0, len(X) - 1)]

    bg_rect = add_patches(ax, x_dates_0_v, y_chunks_means, bins)

    # ax.set_xlim(x_data.min(), x_data.max())
    ax.set_xlim(x_dates.min(), x_dates.max())
    # ax.set_position([0.055, 0.17, 0.92, 0.68])

    ax.xaxis.set_major_locator(dates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y\n%m-%d'))
    fig.autofmt_xdate(rotation=90, ha='center', which='both')
    ax.tick_params(axis='x', size=0.5)

    # return x_data, x_data_0_v, y_chunks_mean_normalized
    return x_dates, x_dates_0_v, x_dates_raw, x_data_len, y_chunks_mean_normalized


def save_all_data():
    """
    update all_data.npz with new values
    """
    all_data_file = np.load(source_folder + 'All data.npz', allow_pickle=True)
    ret_obj = {}
    for key in all_data_file.files:
        ret_obj[key] = all_data_file[key]
    if 'data' in ret_obj:
        ret_obj['data'] = all_data
    np.savez_compressed(source_folder + 'All data.npz', **ret_obj)


def get_title(i):
    """replaces daft characters"""
    return ipk[supplements[i]].replace('?', 'qm').replace('*', '')


def update_plot(i):
    """Display new supplement data"""
    global x_data, supp_line
    k = supplements[i]
    title = 'Click point, then run update_all_data()'
    subtitle = str(i+1) + ' of ' + str(len(supplements)) + ' ' + get_title(i)
    y_data = rebin_input(ts_ind=8, data_ind=k).toarray()[0]
    y_data_raw = rebin_input(ts_ind=0, data_ind=k).toarray()[0]
    entries = (all_data[2][np.argwhere(opk == 'En-Entry')[0][0]].toarray() != 0)[0]
    if supp_line is not None:
        # for l in supp_line:
        #     supp_line[l].remove()
        supp_line['data'].set_ydata(y_data)
        supp_line['bg'].set_ydata(bg_y_norm * y_data.max())
        supp_line['data_raw'].set_ydata(y_data_raw)
        supp_line['raw_scatter'].remove()
        supp_line['raw_entries'].remove()
        supp_line['raw_scatter'] = ax.scatter(x_dates_raw, y_data_raw, zorder=10, color='black', s=5)
        supp_line['raw_entries'] = ax.scatter(x_dates_raw[entries], y_data_raw[entries], zorder=11, color='red', s=1)
    else:
        supp_line = {'bg': ax.plot(bg_x, bg_y_norm * y_data.max(), color='black', linewidth=1)[0],
                     'data': ax.plot(x_data, y_data, linewidth=1, c='red')[0],
                     'data_raw': ax.plot(x_dates_raw, y_data_raw, linewidth=0.5, c='green', picker=2)[0],
                     'raw_scatter': ax.scatter(x_dates_raw, y_data_raw, zorder=10, color='black', s=10),
                     'raw_entries': ax.scatter(x_dates_raw[entries], y_data_raw[entries], zorder=11, color='red', s=10),
                     'y_data': y_data,
                     'y_data_raw': y_data_raw,
                     'ax': ax
                     }
    ax.set_ylim(-0.1, y_data.max()*1.1)
    ax.set(title=title + '\n' + subtitle)
    # plt.savefig(dump_folder + 'Check supplement graphs/' + subtitle + '.pdf', format='pdf')
    return x_data, supp_line


def create_tkinter_gui(ws, hs):
    """
    gui to display the text from the supplement overviews, hopefully a touch faster than matplotlib
    with buttons to:
        update text from selected point
        update all_data from text box
    text box and labels in corner.
    T   T   T
    IP  T   T
    IP  T   T
    IP  T   T
    IP  T   T
    as a layout, roughly; T = text , IP = input
    """

    global root, tkgui

    width, height = ws*1, hs*0.85,
    wpad, hpad = width*0.05, hs*0.1
    # root.geometry('%dx%d+%d+%d' % (width, height, ws-width-wpad, hs-height-hpad))  # width, height, x, y
    root.geometry('%dx%d+%d+%d' % (width, height, 0, 0))  # width, height, x, y, full screen display here now.


    class Tk_obj:
        pass

    tkgui = Tk_obj
    pad = 5  # px

    text_data['file_list'] = sorted(os.listdir(source_folder_overview))
    
    def get_overview_info(date):
        """
        return json from overview file
        """
        for file in text_data['file_list']:
            file_date = file[0:10]
            if file.endswith('.json') & (file_date == date):
                print(date)
                with open(source_folder_overview + file) as f:
                    return json.load(f)

    def update_all_data_tk(event):
        """
        update all_data for current supplement, then updates graph. Triggers when tk button is pressed
        """
        global all_data
        if tkgui.setup:
            return
        else:
            new_val = float(tkgui.new_val_Entry.get())
            x_ind = supp_line['ind']
            all_data_ind = supplements[callback.ind]
            if all_data_ind >= all_data[0].shape[0]:
                all_data[1][all_data_ind - all_data[1].shape[0], x_ind] = new_val
            else:
                all_data[0][all_data_ind, x_ind] = new_val
            update_plot(callback.ind)
            tkgui.new_val_label_SV.set('new_val=%f' % new_val)
            tkgui.new_val_entry_SV.set('')
            
    def update_overview_display(event):
        global tkgui
        date = supp_line['date']
        ov_json = get_overview_info(date)
        # print(ov_json['Header'], ov_json['Supplements - overview'])
        tkgui.title_SV.set(date)
        tkgui.c0_overviews_label_SV.set(ov_json['Supplements - overview'])
        tkgui.c1_ov_breakdowns_label_SV.set(ov_json['Supplements'])
        tkgui.c2_ov_comments_label_SV.set(ov_json['Symptoms - comments'])
        tkgui.c3_intakes_combined_label_SV.set(ov_json['Intake - combined'])

    def initialise_gridspec():
        root.rowconfigure(1, weight=1)
        # root.rowconfigure(1, weight=1)
        # root.rowconfigure(2, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)

    def create_input_frame():
        """
        input
          note pack() packs from the bottom as default
          click_info_Label
          update_ov_Button
          new_val_Entry
          update_new_val_Button
          update_new_val_Label
        """

        row_start = 1
        # tkgui.input_Frame = tk.Frame(root, padx=pad, pady=pad)
        tkgui.input_Frame = root  # should just replacethis
        # tkgui.input_Frame.grid(row=3, column=0, sticky='NSEW')
        # tkgui.input_Frame.pack(side='bottom')

        tkgui.pick_info_SV = tk.StringVar()
        tkgui.pick_info_Label = tk.Label(tkgui.input_Frame, textvariable=tkgui.pick_info_SV)
        # tkgui.pick_info_Label.pack(fill='x')
        tkgui.pick_info_Label.grid(row=row_start+1, rowspan=1, column=0, ipadx=pad, ipady=pad, sticky='EW')

        tkgui.update_ov_info_Button = tk.Button(tkgui.input_Frame, text='(u) to update overviews')
        # tkgui.update_ov_info_Button.pack(fill='x')
        tkgui.update_ov_info_Button.grid(row=row_start+2, rowspan=1, column=0, ipadx=pad, ipady=pad, sticky='EW')
        tkgui.update_ov_info_Button.bind('<Button-1>', update_overview_display)
        root.bind('<u>', update_overview_display)
    
        tkgui.new_val_entry_SV = tk.StringVar()
        tkgui.new_val_Entry = tk.Entry(tkgui.input_Frame, textvariable=tkgui.new_val_entry_SV)
        # tkgui.new_val_Entry.pack(fill='x')
        tkgui.new_val_Entry.grid(row=row_start+3, rowspan=1, column=0, ipadx=pad, ipady=pad, sticky='EW')

        tkgui.update_new_val_Button = tk.Button(tkgui.input_Frame, text='(Return) to update value')
        # tkgui.update_new_val_Button.pack(fill='x')
        tkgui.update_new_val_Button.grid(row=row_start+4, rowspan=1, column=0, ipadx=pad, ipady=pad, sticky='EW')
        tkgui.update_new_val_Button.bind('<Button-1>', update_all_data_tk)
        root.bind('<Return>', update_all_data_tk)
        
        tkgui.new_val_label_SV = tk.StringVar()
        tkgui.new_val_Label = tk.Label(tkgui.input_Frame, textvariable=tkgui.new_val_label_SV)
        # tkgui.new_val_Label.pack(fill='x')
        tkgui.new_val_Label.grid(row=row_start+5, rowspan=1, column=0, ipadx=pad, ipady=pad, sticky='EW')

    def create_information_display_columns():

        l = int(ws / 4)
        row = 1
        font = ('monospace', '7')

        # title
        tkgui.title_SV = tk.StringVar()
        tkgui.title_Label = tk.Label(root, textvariable=tkgui.title_SV,
                                            anchor='nw', justify='center', font=('monospace, 9'))
        tkgui.title_Label.grid(row=0, rowspan=1, column=0, columnspan=4, ipadx=pad, ipady=pad, sticky='n')

        # supp overviews
        tkgui.c0_overviews_label_SV = tk.StringVar()
        tkgui.c0_overviews_Label = tk.Label(root, textvariable=tkgui.c0_overviews_label_SV,
                                            anchor='nw', wraplength=l, justify='left', font=font)
        tkgui.c0_overviews_Label.grid(row=row, rowspan=1, column=0, ipadx=pad, ipady=pad, sticky='n')

        # supp breakdowns
        tkgui.c1_ov_breakdowns_label_SV = tk.StringVar()
        tkgui.c1_ov_breakdowns_Label = tk.Label(root, textvariable=tkgui.c1_ov_breakdowns_label_SV,
                                                anchor='nw', wraplength=l, justify='left', font=font)
        tkgui.c1_ov_breakdowns_Label.grid(row=row, rowspan=6, column=1, ipadx=pad, ipady=pad, sticky='n')

        # comments
        tkgui.c2_ov_comments_label_SV = tk.StringVar()
        tkgui.c2_ov_comments_Label = tk.Label(root, textvariable=tkgui.c2_ov_comments_label_SV,
                                              anchor='nw', wraplength=l, justify='left', font=font)
        tkgui.c2_ov_comments_Label.grid(row=row, rowspan=6, column=2, ipadx=pad, ipady=pad, sticky='n')

        # intakes - combined
        tkgui.c3_intakes_combined_label_SV = tk.StringVar()
        tkgui.c3_intakes_combined_Label = tk.Label(root, textvariable=tkgui.c3_intakes_combined_label_SV,
                                              anchor='nw', wraplength=l, justify='left', font=('monospace', '5'))
        tkgui.c3_intakes_combined_Label.grid(row=row, rowspan=6, column=3, ipadx=pad, ipady=pad, sticky='n')


    tkgui.setup = True
    # root.state('zoomed')
    root.state('normal')
    initialise_gridspec()
    create_input_frame()
    create_information_display_columns()
    update_overview_display(None)
    tkgui.setup = False


def main():
    """
    plot a line for each supplement over all time over a background of average interesting data
    """
    global callback, fig, ax, x_data, bg_x, bg_y_norm, x_dates_raw, x_data_len, root, tkgui

    fig, ax = plt.subplots()

    def is_dst(dt=None, tz='UTC'):
        if dt is None:
            dt = datetime.utcnow()
        tz = pytz.timezone(tz)
        try:
            tzad = tz.localize(dt, is_dst=None)
            return tzad.tzinfo._dst.seconds != 0
        except ValueError:
            return dt.tzinfo._dst.seconds != 0

    def on_pick(event):
        """
        when a specific point is picked on thisline
        """
        global supp_line
        # thisline = event.artist
        thisline = supp_line['data_raw']
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind[0]  # index along data
        # points = (xdata[ind], ydata[ind])
        dt = start_date + timedelta(hours=ind * 0.25)
        dst = is_dst(dt, 'Europe/London')
        dt = dt + timedelta(hours=0.25) if dst else dt + timedelta(hours=-0.75)
        supp_line['ind'] = ind
        supp_line['date'] = dt.strftime('%Y-%m-%d')
        # print(dt, dst, 'x_ind:', ind, 'y val:', ydata[ind], thisline)
        tkgui.pick_info_SV.set(dt.strftime('%Y-%m-%d %H:%M') + ' x_ind: ' + str(ind) + ' y_val: ' + str(ydata[ind]))

    def on_click(event):
        """
        on double left click, zoom in
        on double right click, zoom out
        """
        global supp_line
        if event.dblclick:
            print('button: %d, x=%d, xdata=%d' %
                  (event.button, event.x, event.xdata))
            if event.button == 1:
                supp_line['ax'].set_xlim(event.xdata - 1.0, event.xdata + 1.0)
                dt = dates.num2date(event.xdata)  # only need to be accurate to a day.probably dst messes around.
                supp_line['date'] = dt.strftime('%Y-%m-%d')
                tkgui.pick_info_SV.set(('button: %d, x=%d, xdata=%d\n' + dt.strftime('%Y-%m-%d')) %
                                       (event.button, event.x, event.xdata))
            elif event.button == 3:
                # supp_line['ax'].set_xlim(supp_line['xmin'], supp_line['xmax'])
                # 2020-06-01
                supp_line['ax'].set_xlim(18414, supp_line['xmax'])


    def on_key_press(event):
        global graph_data
        # print(' Key pressed ' + event.key)
        # toggle RectangleSelector on / off
        if event.key == 'b':
            callback.prev(None)
        elif event.key == 'n':
            callback.next(None)

    class Graph_data:
        ind = 0
        bnext, bprev = None, None
        x_data, supp_line = None, None

        def __init__(self, ind):
            self.ind = ind

        def next(self, event):
            self.ind += 1
            if self.ind < len(supplements):
                # print(get_title(self.ind))
                self.x_data, self.supp_line = update_plot(self.ind)
                save_all_data()
            else:
                self.ind = len(supplements) - 1

        def prev(self, event):
            self.ind -= 1
            if self.ind > -1:
                # print(get_title(self.ind))
                self.x_data, self.supp_line = update_plot(self.ind)
                save_all_data()
            else:
                self.ind = 0

        def create_buttons(self):
            self.bnext = Button(plt.axes([0.81, 0.91, 0.1, 0.075]), '(n) Next')
            self.bnext.on_clicked(callback.next)
            self.bprev = Button(plt.axes([0.7, 0.91, 0.1, 0.075]), '(b) Back')
            self.bprev.on_clicked(callback.prev)

    # argument is first supplement to graph
    callback = Graph_data(3)

    root = tk.Tk()
    ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()

    x_data, bg_x, x_dates_raw, x_data_len, bg_y_norm = create_graph_background(ws, hs)
    update_plot(callback.ind)
    supp_line['xmin'] = x_data.min()
    supp_line['xmax'] = x_data.max()
    supp_line['fig'] = plt.gcf()
    supp_line['date'] = '2020-06-01'
    plt.legend([supp_line['bg'], supp_line['data']],
               [str(seg_len) + ' day interesting data av. Hi=good', 'Supplement'],
               loc='lower left', bbox_to_anchor=(0., 1.), facecolor='white', framealpha=0.3)
    callback.create_buttons()
    ax.set_position([0.055, 0.2, 0.92, 0.62])

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()
    create_tkinter_gui(ws, hs)  # create this first so it's in the background
    # root.after(plt.get_current_fig_manager().window_raise_())  # should put graph to foreground
    root.after(0, root.lower)
    root.mainloop()  # tk main loop starts - console now can't be used

    # return fig, ax, callback, supp_line


def dev():
    dt = 76
    yd = supp_line['y_data'][dt:]  # 2020-06-01
    diff = np.insert(np.diff(yd), 0, 0)
    di = (np.arange(len(diff)) + dt)[diff != 0]
    splt = np.split(yd, di)
    lens = np.array([len(a) for a in splt])
    print(np.sum(lens < 4))


def write_key_list(key_list, file_name):
    """
    write key_list to a csv with file_name
    """
    with open(dump_folder + file_name + '.txt', 'w+') as f:
        f.write('\n'.join([str(i) + ' ' + key_list[i] for i in range(len(key_list))]))



def export_data_to_numpy():
    """
    exports contents of all_data to individual npy files
    prefixes:
        ii, io, oo = input, input / output, output
        a, m = add, mean
        met = metadata - does not get rebinned
    """

    prefixes = ['a', 'm', 'a', 'm']
    for iotype in range(len(all_keys)):
        for ind in range(len(all_keys[iotype])):
            # if an input
            prefix = ''
            if iotype < 2:
                # if the key is both an input and an output
                prefix = 'io' if all_keys[iotype][ind] in all_keys[iotype + 2] else 'ii'
            else:
                prefix = 'io' if all_keys[iotype][ind] in all_keys[iotype - 2] else 'oo'
            prefix += prefixes[iotype] + '-'

            np.save('../../Data sources/Numpy entries/' +
                    prefix + all_keys[iotype][ind].replace('?', 'qm').replace('*', ''),
                    all_data[iotype][ind].toarray()[0])

main()