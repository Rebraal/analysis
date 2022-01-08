# 2021-05-15 14:46
# code functions. First date displayed displays incorrectly, don't know why. Following dates display ok, so I can't
# waste more time. Looks like there's only room for one set of data and comments
#
# 2021-05-17 15:00
# pretty sluggish graphs, so in terms of using this laptop, I'm going to spit out all comments to a single text file.

import numpy as np
import os
import json as json
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from dateutil import parser
from datetime import timedelta

# plt.switch_backend('TkAgg')
# plt.switch_backend('Qt5Agg')

source_folder = '../Source data/'
dump_folder = '../Python dumps/'
source_folder_overview = '../Clean JSON overview entries/'
# source_folder_overview = 'C:/Memento analysis/Clean JSON overview entries/'
source_folder_individual = 'C:/Memento analysis/Raw individual entries/'
# time_steps = [0.25, 0.5, 1, 2, 4, 8, 12, 18, 24, 30, 36, 48, 60, 72]
# rebinSteps = [1, 2, 4, 8, 16, 32, 48, 72, 96, 120, 144, 192, 240, 288]
# start_date = parser.parse('2019-09-01')
# # load data from file, arrays of length ~50k
# all_data = np.load(source_folder + 'All data.npz', allow_pickle=True)['data'].tolist()
# # all_inputs, all_outputs = sparse.vstack((all_data[0], all_data[1])), sparse.vstack((all_data[2], all_data[3]))
# all_keys = np.load(source_folder + 'All data.npz', allow_pickle=True)['keys'].tolist()
# all_rebins = np.load(source_folder + 'All data.npz', allow_pickle=True)['rebins'].tolist()
# ipk, opk = np.array(all_keys[0] + all_keys[1]),  np.array(all_keys[2] + all_keys[3])

callback, bnext, bprev = None, None, None
fig2, ax2 = None, None
text_data = {}


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


def add_text(txt, fld_name):
    """
    add text to graph and update next location
    """
    global text_data, bbox
    ax2 = text_data['ax_main']
    plt.sca(ax2)
    wrap = True if fld_name == 'Symptoms - comments' else False
    new_text = ax2.text(text_data['tx'], text_data['ty'],
                       txt,
                       ha='left', va='top',
                       fontfamily='monospace', fontsize=6,
                       wrap=wrap
                       )
    text_data['txt_objs'].append(new_text)
    # print('new_text.get_window_extent()', new_text.get_window_extent(), '\n',
    #       'ax2.transData.inverted()', ax2.transData.inverted(),
    #      )
    # plt.draw()
    bbox = new_text.get_window_extent().transformed(ax2.transData.inverted())
    return bbox


def delete_existing_text():
    """
    delete existing text on graph
    """
    # this might need tweaking, or a for loop deleting text boxes
    # for tb in text_data['txt_objs']:
    #     tb.remove()
    text_data['ax_main'].cla()
    text_data['txt_objs'] = []


def update_overview_display(date):
    """
    udpate display
    """
    global text_data
    delete_existing_text()
    ov_json = get_overview_info(date)
    initialise_text_coords()
    for col in text_data['key_list']:
        text_data['ty'] = text_data['tyo']
        for row in col:
            # print('init: tx:', text_data['tx'], 'dx:', text_data['dx'])
            bbox = add_text(ov_json[row], row)  # bbox == x0, x1, y0, y1
            # print('bbox: x0', bbox.x0, 'x1:', bbox.x1)
            text_data['ty'] = bbox.y0 - text_data['dy']  # add row spacing
            # print('erow: tx:', text_data['tx'], 'dx:', text_data['dx'])
            # plt.draw()
        # text_data['tx'] = (bbox.x1/2) + text_data['dx']  # add col spacing
        text_data['tx'] = bbox.x1 + text_data['dx']  # add col spacing
        # print('ecol: tx:', text_data['tx'], 'dx:', text_data['dx'])

    plt.draw()
    text_data['date'] = date


def initialise_text_coords():
    global text_data

    text_data['dx'], text_data['dy'] = 0.01, 0.01
    text_data['txo'], text_data['tyo'] = 0.02, text_data['input_bottom'] - text_data['dy']
    text_data['tx'], text_data['ty'] = text_data['txo'], text_data['tyo']


def new_date(i):
    """advance date by i"""
    return (parser.parse(text_data['date']) + timedelta(days=i)).strftime('%Y-%m-%d')


def display_overviews_check_key_press(event):
    if event.key == 'b':
        prev_date(None)
    elif event.key == 'n':
        next_date(None)


def prev_date(event):
    text_data['box'].set_val(new_date(-1))


def next_date(event):
    text_data['box'].set_val(new_date(1))


def cd(date):
    """change date to"""
    text_data['box'].set_val(date)


def initialise_display_input(init_date):
    global text_data
    # axes [left, bottom, width, height
    text_data['input_bottom'], text_data['input_height'] = 0.965, 0.025
    text_data['input_width'] = 0.1

    text_data['bprev'] = Button(plt.axes([0.35, text_data['input_bottom'],
                                          text_data['input_width'], text_data['input_height']]), 'B back')
    text_data['bprev'].on_clicked(prev_date)

    text_data['box'] = TextBox(plt.axes([0.45, text_data['input_bottom'],
                                         text_data['input_width'], text_data['input_height']]), '')
    text_data['box'].on_submit(update_overview_display)

    text_data['bnext'] = Button(plt.axes([0.55, text_data['input_bottom'],
                                          text_data['input_width'], text_data['input_height']]), 'N next')
    text_data['bnext'].on_clicked(next_date)

    fig2.canvas.mpl_connect('key_press_event', display_overviews_check_key_press)
    text_data['box'].set_val(init_date)  # Trigger `submit` with the initial string.


def initialise_display(init_date):
    global fig2, ax2, text_data
    plt.close()
    # print by columns, then rows
    text_data['key_list'] = [['Supplements - overview'], ['Supplements'], ['Symptoms - comments']]
    text_data['file_list'] = sorted(os.listdir(source_folder_overview))

    fig2, ax2 = plt.subplots()
    fig2.tight_layout()
    fig_m = plt.get_current_fig_manager()
    # plt.get_current_fig_manager().window.state("zoomed")  # TkAgg
    plt.get_current_fig_manager().window.showMaximized()  # Qt5
    # x, y, dx, dy = fig_m.window.geometry().getRect()
    ax2.set(position=[0.0, 0.0, 1, 1])
    # create text box
    text_data['ax_main'] = ax2
    text_data['txt_objs'] = []

    plt.draw()
    initialise_display_input(init_date)


initialise_display('2021-01-01')


def all_comments_to_text():
    """
    read all overview files
    write 'Symptoms - comments' to a single text file
    replace 'text. Stuff' with 'text.\nStuff' to avoid messing around with line break code.
    """
    ov_dir = r'D:\Memento analysis, unstored\Memento analysis\Clean JSON overview entries/'
    op_string = ''
    for file in sorted(os.listdir(ov_dir)):
        if file.endswith('.json'):
            print(file)
            file_date = file[0:10]
            with open(ov_dir + file) as f:
                op_string += file_date + '\n' + json.load(f)['Symptoms - comments'].replace('. ', '.\n') + '\n\n'

    with open(dump_folder + 'All symptoms comments.txt', 'w+') as f:
        f.write(op_string)
