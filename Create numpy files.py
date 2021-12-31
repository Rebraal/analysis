# This program is designed to run on my mobile and should update a set of numpy files for updating from the latest
# backup in the memento folder.
# This file should be located in the memento folder
#
# These files are in the format:
# prefixes:
#     ii, io, oo = input, input / output, output
#     a, m = add, mean
#     met = metadata - does not get rebinned
#
# so iia-Li-Tea.npy
# met files are:
#   Dt-Date strings: at DST go YYYY:MM:DD 01:45 to YYYY:MM:DD 01:00 or vice versa
#   Dt-Seconds elapsed: convertible into a datetime by datetime.fromtimestamp()
#   Lo-Locations: in format [[lat, lon], ...]
#   En-Entry timestamps: a list of all entry timestamps for picking up the last date processed.

import sqlite3 as lite
import numpy as np
import json as json
import os
import re
from dateutil import parser
from datetime import timedelta
from datetime import datetime
import pytz
import math
import platform
import shutil

# pc_path = '../Data sources'
# mobile_path = '/storage/emulated/0/memento'
# path_prefix = pc_path if platform.system() == 'Windows' else mobile_path
path_prefix = '..'
JSON_folder = path_prefix + '/Compiling JSONs/'
numpy_folder = path_prefix + '/Numpy entries/'

excludedOutputs = ['', 'undefined']
excludedIntakes = ['', 'undefined']

split_map_field_data = re.compile('.*/(-*\d+\.\d+)%2C(-*\d+\.\d+)')

key_prefix_decode = {
                     "Al": "iia",  # allergens
                     "Co": "iia",  # combined solid food
                     "Da": "oom",  # functions data
                     "Dt": "met",  # continuous date time array
                     "En": "met",  # timestamps only
                     "Ex": "ioa",  # exercise
                     "Li": "iia",  # liquid
                     "Lo": "met",  # locations
                     "Me": "iia",  # metadata, should really be met
                     "Pi": "met",  # pollen inputs
                     "Pl": "iia",  # procedures list, nasal rinse etc
                     "RE": "ioa",  # rest
                     "SA": "ioa",  # sleep - all
                     "SD": "ioa",  # sleep - during day
                     "SN": "ioa",  # sleep - during night
                     "Sc": "iom",  # symptoms conglomerate
                     "Sl": "iim",  # sublingual
                     "Sp": "iia",  # spray
                     "St": "ioa",  # step count
                     "Su": "iia",  # supplements
                     "Sy": "iom",  # symptoms
                     "We": "iim",  # weather
                     "tC": "iia",  # combined solid food conglomerate
                     "tL": "iia",  # combined liquid conglomerate
                     }
update_numpy_errors = []


def main(update_numpy_files=False):
    timing_start = datetime.now()
    print(timing_start)
    global Con, Cur
    global date_stamps, useful_indices, empty_row
    global start_date_time, end_date_time, all_date_times, utc, startDateTimeInMin


    # zip and backup current numpy files.
    if update_numpy_files:
        backup_numpy_files()
    # unzip database to /backup/ folder
    # unzip_database()

    # connect to database. Con and Cur are now the global variables that are used from this
    Con, Cur = open_database('../Epoch 1 databases/memento 2021-01-01 on.db')

    date_stamps = list_content(field='Date stamp')
    # get time stamp of last processed entry
    all_date_times = np.load('../Numpy entries old/met-Dt-Date strings.npy', allow_pickle=True)
    start_date_time = all_date_times[-1]

    # could be useful to check if there's a gap between the last numpy entry and the first database entry.

    # assume that this will be run frequently enough not to need the archive.
    # for the initial run, shift the archive data across
    # get last entry in Current status that we care about
    # end_date_time = get_end_date()
    end_date_time = '2021-07-02 03:45'

    # apply this to all results from list_content
    useful_indices = get_useful_indices(start_date_time, end_date_time)
    date_stamps = date_stamps[useful_indices]

    utc = pytz.UTC
    startDateTimeDT = parser.parse(start_date_time).astimezone(utc)
    startDateTimeInMin = math.floor(datetime.timestamp(startDateTimeDT) / 60)
    endDateTimeDT = parser.parse(end_date_time).astimezone(utc)
    endDateTimeInMin = math.floor(datetime.timestamp(endDateTimeDT) / 60)
    stepCount1Min = math.ceil((endDateTimeDT - startDateTimeDT).total_seconds() / 60)
    empty_row = np.zeros(shape=stepCount1Min, dtype='f2')

    compile_all_intakes()
    compile_metadata()
    compile_weather()
    compile_pollen()
    compile_exercise()
    compile_sleep()

    # might need to update numpy objects with results of this stuff first to 
    # clear memory.

    compile_all_outputs()

    if update_numpy_files:
        pass
        # update_numpy_files_with_new()

    # os.remove(path_prefix + '/backup/memento.db')  # deletes extracted database.
    print(timing_start, '\n', datetime.now() - timing_start, '\nEnd main')
    return Con, Cur


def backup_numpy_files():
    """
    for all numpy files in numpy_folder, send to zip
    """
    shutil.make_archive(path_prefix + '/Numpy backup ' + datetime.today().strftime('%Y-%m-%d %Hc%Mc%S'),
                        'zip',
                        numpy_folder)


def unzip_database():
    """
    find latest autobackup and unzip it
    """
    files = sorted([file for file in sorted(os.listdir(path_prefix + '/backup/')) if 'autobackup' in file])
    shutil.unpack_archive(path_prefix + '/backup/' + files[-1], path_prefix + '/backup/')


def compile_all_intakes():
    """
    For liquid, food and supplements, update Numpy files
    :return:
    """
    global inputs_fails, inputs_obj_add, inputs_obj_mean, sublingual_set

    with open(JSON_folder + 'Sublingual intact.json') as f:
        sublingual_left_intact = set(json.load(f))
    with open(JSON_folder + 'Rationalize combined search.json') as f:
        rationalize_search = np.array(json.load(f))
    with open(JSON_folder + 'Rationalize combined replace.json') as f:
        rationalize_replace = np.array(json.load(f))
    with open(JSON_folder + 'Allergens.json') as f:
        allergens = json.load(f)
    with open(JSON_folder + 'Conglomerate.json') as f:
        intakes_to_conglomerate = json.load(f)
    with open(JSON_folder + 'Intake decode.json') as f:
        intake_specific_decode = json.load(f)
    intake_general_decode = {'l': 5, 'y': 3, 'u': 3, 's': 1, 'a': 1, 't': 0.1, 'L': 5, 'Y': 3, 'U': 3, 'S': 1, 'A': 1,
                             'T': 0.1}
    intake_liquid_decode = {'l': 1.5, 'y': 1, 'u': 1, 's': 0.5, 'a': 0.5, 't': 0.125, 'L': 1.5, 'Y': 1, 'U': 1,
                            'S': 0.5, 'A': 0.5, 'T': 0.125}
    intake_suffixes = {'prefix': " - ", 'l': "cooled", 's': "soaked", 'c': "cooked", 'r': "raw"}
    let_to_num = [[',', '.'], [' ', '.'],
                  ['q', '1'], ['w', '2'], ['e', '3'], ['r', '4'], ['t', '5'],
                  ['y', '6'], ['u', '7'], ['i', '8'], ['o', '9'], ['p', '0'],
                  ['a', '1'], ['s', '2'], ['d', '3'], ['f', '4'], ['g', '5'],
                  ['h', '6'], ['j', '7'], ['k', '8'], ['l', '9']
                  ]

    inputs_fails = []
    inputs_obj_add = {}
    inputs_obj_mean = {}
    supplements_set = set()
    sublingual_set = set()
    spray_set = set()

    get_intake = re.compile('(?:(?:\d\d#)|(?:\d\d\.\d#))? *(.*\D): (.*)')
    check_split_qty = re.compile('(.), (.)')
    strip_spaces = re.compile(' +')

    def process_liquid_line(line, index):
        """
        processes a line and stores it to inputs_obj_mean
        :param line: line to process
        :param index: index within empty_row to store new info
        :return: None
        """
        # decode line into intake and numeric qty
        is_intake = get_intake.match(line)
        if is_intake is not None:
            intake = 'Li-' + is_intake.group(1)
            qty = is_intake.group(2)
            try:  # to convert the qty to a number
                qty = float(qty)
            except ValueError:
                try:  # to decode the qty
                    qty = intake_liquid_decode[qty]
                except KeyError:
                    inputs_fails.append(['liquid', intake, str(qty), line])

        # update inputs_obj_add
        try:
            inputs_obj_add[intake][index] = qty
        except KeyError:
            inputs_obj_add[intake] = empty_row.copy()
            inputs_obj_add[intake][index] = qty

    def process_combined_line(line, index):
        """
        processes a line and stores it to inputs_obj_mean
        :param line: line to process
        :param index: index within empty_row to store new info
        :return: None
        """
        # decode line into intake and numeric qty
        is_intake = get_intake.match(line)
        if is_intake is not None:
            intake = 'Co-' + is_intake.group(1)
            qty = is_intake.group(2)
            if qty != '':
                try:  # to convert the qty to a number
                    qty = float(qty)
                except ValueError:  # not a plain number
                    qty = qty.lower()
                    if intake in intake_specific_decode:
                        split_qty = check_split_qty.match(qty)
                        if split_qty is not None:
                            qty = split_qty.group(1)
                            try:
                                intake += intake_suffixes['prefix'] + intake_suffixes[split_qty.group(2)]
                            except KeyError:
                                inputs_fails.append(['combined, split qty', intake, str(split_qty.group(2)), line])
                        else:
                            try:
                                qty = intake_specific_decode[intake][qty]
                            except KeyError:
                                inputs_fails.append(['combined, specific', intake, str(qty), line])
                                qty = 0
                    else:
                        split_qty = check_split_qty.match(qty)
                        if split_qty is not None:
                            qty = split_qty.group(1)
                            try:
                                intake += intake_suffixes['prefix'] + intake_suffixes[split_qty.group(2)]
                                qty = float(qty)
                            except Exception:
                                inputs_fails.append(['combined, split qty', intake, str(split_qty.group(2)), line])
                        try:  # to decode the qty
                            qty = intake_general_decode[qty]
                        except KeyError:
                            inputs_fails.append(['combined, general', intake, str(qty), line])
                            qty = 0

                # update inputs_obj_add
                try:
                    inputs_obj_add[intake][index] = qty
                except KeyError:
                    inputs_obj_add[intake] = empty_row.copy()
                    inputs_obj_add[intake][index] = qty

    def process_supplement_line(line, index):
        """
        processes a line and stores it to inputs_obj_mean
        store any new sublingual as a terminus
        :param line: line to process
        :param index: index within empty_row to store new info
        :return: None
        """
        # decode line into intake and numeric qty
        is_intake = get_intake.match(line)
        if is_intake is not None:
            intake = 'Su-' + strip_spaces.sub(' ', is_intake.group(1).replace('*', ''))
            qty = is_intake.group(2)
            # print(intake, qty)
            if qty != '':
                try:  # to convert the qty to a number
                    qty = float(qty)
                except ValueError:
                    qty = qty.lower()
                    # replace any stray letters with numbers
                    for el in let_to_num:
                        qty = qty.replace(el[0], el[1])
                        if qty.replace('.', '', 1).isdigit():
                            qty = float(qty)
                            break

            # update inputs_obj_add
            try:
                inputs_obj_add[intake][index] = qty
            except KeyError:
                inputs_obj_add[intake] = empty_row.copy()
                inputs_obj_add[intake][index] = qty

    def intakes_to_numpy(ar, type):
        """
        takes an array of intake strings
        processes quantities into numeric form
        stores in inputs_obj_add
        :param ar: array of intake strings, each element a list of \n separated intakes
        :param type: governs which decoders to use
        :return: None
        """

        ar = np.char.splitlines(ar)
        if type == 'liquid':
            for i, entry in enumerate(ar):
                # skip zero length
                if len(entry) > 0:
                    index = get_timestamp_in_min(date_stamps[i])
                    for line in entry:
                        process_liquid_line(line, index)
        elif type == 'combined':
            for i, entry in enumerate(ar):
                # skip zero length
                if len(entry) > 0:
                    index = get_timestamp_in_min(date_stamps[i])
                    for line in entry:
                        process_combined_line(line, index)
        else:
            for i, entry in enumerate(ar):
                # skip zero length
                if len(entry) > 0:
                    index = get_timestamp_in_min(date_stamps[i])
                    for line in entry:
                        # print(date_stamps[i], line)
                        process_supplement_line(line, index)

    def rationalize_intakes():
        """
        replace key names with consistent ones
        at 2021-06-22, there are no liquids in this, so the ordering of liquids after works
        """
        for oki, old_key in enumerate(rationalize_search):
            if old_key in inputs_obj_add:
                inputs_obj_add[rationalize_replace[oki]] = inputs_obj_add.pop(old_key)

    def compile_sublingual_terminii():
        """
        for each sublingual terminated and each food intake that isn't listed, populate array
        at this point, only rationalized combined intakes are present.
        """
        # total all intakes, then subtract the ones that need to be ignored
        sub_term = info_to_numpy(list_content(field='Sublingual terminated')[useful_indices])

        for key in inputs_obj_add:
            sub_term += inputs_obj_add[key]

        for ignore in sublingual_left_intact:
            try:
                sub_term -= inputs_obj_add[ignore]
            except KeyError:
                pass

        inputs_obj_add['Me-Sublingual terminated'] = (sub_term != 0).astype('f2')

        # for all times where sublingual is intact, create a list of indices where this is the case
        # then set those indices to 0 in the terminated list.
        sub_intact_ind = np.argwhere(info_to_numpy(list_content(field='Sublingual intact')[useful_indices]) != 0)[0]
        inputs_obj_add['Me-Sublingual terminated'][sub_intact_ind] = 0

        # get a list of all sublinguals taken, add them to termnii
        for key in inputs_obj_add:
            if ' sl*' in key:
                # sublingual_set.add(intake)  # should 'intake' be 'key'?
                sublingual_set.add(key)
                inputs_obj_add['Me-Sublingual terminated'] += inputs_obj_add[key].nonzero()[0]

        # convert to 1 or 0
        inputs_obj_add['Me-Sublingual terminated'] = inputs_obj_add['Me-Sublingual terminated'].nonzero()[0]

    def compile_sublinguals():
        """
        find all sublinguals ' sl '
        create new empty row and fill with value between start and next sublingual terminus
        """
        print('compile_sublinguals() ', datetime.now())

        terminii = inputs_obj_add['Me-Sublingual terminated']
        terminii[-1] = 1  # make sure the thing ends at the end of the array
        for sl in sublingual_set:

            sl_name = 'Sl' + sl[2:]
            sl_data = inputs_obj_add[sl]
            last_sl_data = 0

            # check for key existence in Numpy entries, if exists, read last value.
            try:
                last_sl_data = np.load(path_prefix + '/Numpy entries/iim-' + sl_name + '.npy', allow_pickle=True)[-1]
            except FileNotFoundError:
                pass

            new_sl = empty_row.copy()
            sl_data = np.insert(sl_data, 0, last_sl_data)
            sl_terminii = np.insert(terminii, 0, last_sl_data != 0)
            sl_ends = np.where(sl_terminii != 0)[0]
            starts = np.where(sl_data != 0)[0]

            # every start will also be an end, so ends should always be longer
            end_indices = np.searchsorted(sl_ends, starts, 'right')
            ends = sl_ends[end_indices]

            for start, end in zip(starts, ends):
                new_sl[start:end] = np.full((end-start), sl_data[start])
                inputs_obj_mean[sl_name] = new_sl

        print('compile_sublinguals() ended', datetime.now())

    def compile_intakes_conglomerates():
        """
        for each conglomerate in file,
            check for either an absolute key, or a partial match
            store data
        """
        for cong in intakes_to_conglomerate:
            cong_ar = empty_row.copy()
            # if we're looking for general strings, starting with...
            if cong['t'] == 'g':
                for frag_to_find in cong['d']:
                    for key in inputs_obj_add:
                        if frag_to_find in key:
                            cong_ar += inputs_obj_add[key]
            # else find exact strings
            else:
                for frag_to_find in cong['d']:
                    if frag_to_find in inputs_obj_add:
                        cong_ar += inputs_obj_add[frag_to_find]
            # if there's any data in the new array
            if np.sum(cong_ar) > 0:
                inputs_obj_add[cong['k']] = cong_ar

    def compile_allergens():
        """
        for every intake marked as an allergen in the file, update and store
        """
        print('compile_allergens() ', datetime.now())
        allergens_dict = {
            "c": {"n": "Al-Caffeine", "v": empty_row.copy()},
            "d": {"n": "Al-Dairy", "v": empty_row.copy()},
            "g": {"n": "Al-Gluten", "v": empty_row.copy()},
            "f": {"n": "Al-FODMAP", "v": empty_row.copy()},
            "l": {"n": "Al-Legumes", "v": empty_row.copy()},
            "n": {"n": "Al-Nuts", "v": empty_row.copy()},
            "r3": {"n": "Al-Omega 3", "v": empty_row.copy()},
            "r6": {"n": "Al-Omega 6", "v": empty_row.copy()},
            "p": {"n": "Al-Preservatives", "v": empty_row.copy()},
            "rs": {"n": "Al-Resistant starch", "v": empty_row.copy()},
            "Na": {"n": "Al-Sodium", "v": empty_row.copy()},
            "sp": {"n": "Al-Spices", "v": empty_row.copy()},
            "s": {"n": "Al-Sugar", "v": empty_row.copy()},
            "u": {"n": "Al-Unknown", "v": empty_row.copy()},
            "y": {"n": "Al-Yeast", "v": empty_row.copy()},
        }
        for alg_obj in allergens:
            if alg_obj['key'] in inputs_obj_add:
                for alg in alg_obj['a']:
                    allergens_dict[alg]['v'] += inputs_obj_add[alg_obj['key']]

        for key in allergens_dict:
            inputs_obj_add[allergens_dict[key]['n']] = allergens_dict[key]['v'].astype('f2')

        print('compile_allergens() end ', datetime.now())

    # combined
    combined = list_content(field='Intake - vegetables')[useful_indices]
    for fld in ['Intake - meat and fish', 'Intake - fruit', 'Intake - nuts', 'Intake - misc']:
        combined = np.char.add(combined, '\n')
        combined = np.char.add(combined, list_content(field=fld)[useful_indices])
    combined = np.char.strip(combined)  # need to strip '\n\n\n\n' entries
    intakes_to_numpy(combined, 'combined')

    # liquids and supplements shouldn't need rationalizing, and this speeds up things.
    rationalize_intakes()
    compile_sublingual_terminii()

    # liquids
    intakes_to_numpy(list_content(field='Intake - liquid')[useful_indices], 'liquid')
    compile_intakes_conglomerates()
    compile_allergens()

    # supplements
    intakes_to_numpy(list_content(field='Supplements - taken')[useful_indices], 'supplements')
    compile_sublinguals()

    inputs_obj_add['Me-Rushed intake'] = info_to_numpy(list_content(field='Rushed intake')[useful_indices])
    for fld in ['Sp-B12 oils location', 'Sp-Magnesium oil location', 'Pl-Procedures list']:
        inputs_obj_add[fld] = info_to_numpy(list_content(field=fld[3:])[useful_indices], ar_type='list')

    print('len(inputs_obj_add) = ', str(len(inputs_obj_add)))


def compile_all_outputs():
    """
    symptoms
    """

    global outputs_obj_mean, outputs_fails

    outputs_obj_mean = {}
    outputs_fails = []

    def compile_symptoms():
        """
        for each symptom, start at last value from numpy file or 0, and fill until next new value, or if missing, 0
        """
        print('compile_symptoms() ', datetime.now())
        symptoms = list_content(field='Symptoms - current')[useful_indices]
        unique_symptoms = np.unique(np.array([sym[:-3] for sym in '\n'.join(symptoms).split('\n')]))
        sym_ended = 99  # 99 used as marker for where a symptom has ended because no more value given
        # recreate symptoms for each individual symptom by using find sym in symptoms[i] for sym in unique symptoms
        
        # check for simple errors in symptom values.
        symptoms_decode = {'@': 1, '#': 2, '£': 3, '_': 4, '&': 5}

        for symptom in unique_symptoms:
            outputs_obj_mean[symptom] = empty_row.copy()
            for i, sym_data in enumerate(symptoms):
                get_sym = re.compile(symptom + ' i(.)\n')
                sym_results = get_sym.search(sym_data)
                index = get_timestamp_in_min(date_stamps[i])
                if sym_results is not None:
                    try:
                        val = float(sym_results.group(1))
                        outputs_obj_mean[symptom][index] = val
                    except ValueError:
                        try:
                            val = symptoms_decode[sym_results.group(1)]
                        except KeyError:
                            outputs_fails.append([index, sym_data])
                else:  # if symptom not found, it must have ended.
                    outputs_obj_mean[symptom][index] = sym_ended

            # this yields [v, 0, 0, 0,..., v,...] at 1 minute intervals
            # split for [v], [0, 0, 0, v]
            # check for key existence in Numpy entries, if exists, read last value.

            filename = numpy_folder + 'iom-Sy-' + symptom + '.npy'
            last_value = get_last_value(filename)
            # prepend last old value before doing anything else.
            outputs_obj_mean[symptom] = np.hstack((last_value, outputs_obj_mean[symptom]))
            splits = np.argwhere(np.diff(outputs_obj_mean[symptom]) < 0).T[0]

            # append first and last indices of array
            for start, end in zip(np.hstack(([0], splits)),
                                  np.hstack((splits, len(outputs_obj_mean[symptom])))):
                # if value at start is marking symptom end, then fill with a 0.
                outputs_obj_mean[symptom][start:end] = \
                    0 if outputs_obj_mean[symptom][start] == sym_ended else outputs_obj_mean[symptom][start]
            outputs_obj_mean[symptom] = outputs_obj_mean[symptom][1:]  # chop off old initial value

    def correct_positives():
        """
        For symptoms marked #P#, i5 is good, positive. For all other symptoms, this isn't so.
        Invert 5 - i value, then replace what would be the initial set of 5s with 0s
        """
        p_keys = [key for key in outputs_obj_mean if '#P#' in key]
        for key in p_keys:
            new_key = key.replace('#P# '), ''
            outputs_obj_mean[new_key] = outputs_obj_mean.pop(key)
            # replace initial 0s with 5s
            outputs_obj_mean[new_key][:np.argmax(outputs_obj_mean[new_key] > 0)] = 5
            outputs_obj_mean[new_key] = 5 - outputs_obj_mean[new_key]

    def compile_symptoms_conglomerates():
        """
        Create a conglomerate symptom for each conglomerate in file. Data in form {name1: [sym1, sym2,...], name2:..}
        """
        with open(JSON_folder + 'Symptoms conglomerates.json') as f:
            symptoms_conglomerates = json.load(f)

        for conglomerate in symptoms_conglomerates:
            outputs_obj_mean[conglomerate] = empty_row.copy()
            count = 0
            for symptom in symptoms_conglomerates[conglomerate]:
                try:
                    outputs_obj_mean[conglomerate] += outputs_obj_mean[symptom]
                    count += 1
                except KeyError:
                    pass
            outputs_obj_mean[conglomerate] /= count

    def compile_functions():
        """
        Urination, motion and accompanying notes.
        """
        pass

    def compile_data():
        """
        All numerical stuff like Schulte etc
        """
        pass


    compile_symptoms()
    correct_positives()
    compile_symptoms_conglomerates()
    compile_functions()
    compile_data()

    print('compile_all_outputs()', datetime.now())


def compile_metadata():
    """
    must be compiled before weather and pollen!
    met-Pi-Pollen inputs.npy
    met-Pi-Pollen inputs 0200 start.npy
    met-We-Weather inputs.npy
    met-En-Entry timestamps.npy
    met-Dt-Date strings.npy
    met-Dt-Seconds elapsed.npy
    met-Lo-Locations.npy
    'ooa-En-Entry'  # 1 at time where there is an entry, else 0
    """
    pass

    # met-Dt-Seconds elapsed.npy


def compile_weather():
    """

    """
    pass


def compile_pollen():
    """

    """
    pass


def compile_exercise():
    """

    """
    pass


def compile_sleep():
    """

    """
    pass


def get_last_value(filename):
    """
    get the last value of the numpy array from filename
    """
    try:
        print(filename)
        return np.load(filename, allow_pickle=True)[-1]
    except FileNotFoundError:
        return 0


def info_to_numpy(ar, ar_type=None):
    """
    take an array, ar, of numbers, bool or strings and convert from date - value to numpy array.
    date_stamps is pre-existing array from database.
    ar_type is list, string or None
    """

    if ar_type in ['list', 'string']:
        # creates an array of empty lists
        if ar_type == 'list':
            ret = np.empty(len(empty_row), dtype='object')
            ret[...] = [[] for _ in range(len(empty_row))]
        else:
            ret = np.full(len(empty_row), '', dtype='object')

        for i, entry in enumerate(ar):
            if len(entry) > 0:
                ret[get_timestamp_in_min(date_stamps[i])] = entry
    # numeric
    else:
        ret = empty_row.copy()
        for i, entry in enumerate(ar):
            if entry > 0:
                ret[get_timestamp_in_min(date_stamps[i])] = entry

    return ret


def update_numpy_file_with_new(new_data, key_name):
    """
    for key_name, update existing array with new_data appended or create new array
    """
    global update_numpy_errors

    try:  # to get prefix for input / output
        prefix = key_prefix_decode[key_name[:2]]
    except KeyError:
        print('update numpy error: prefix not found ', key_name)
        prefix = 'unk'
    filename = prefix + '-' + key_name + '.npy'

    try:  # to get old data
        old_data = np.load(numpy_folder + filename, allow_pickle=True)
    except FileNotFoundError:
        old_data = np.empty(len(all_date_times), dtype=new_data.dtype)

    if 'a' in prefix:
        new_data = rebin_add(new_data, 15)
    else:
        new_data = rebin_mean(new_data, 15)

    try:  # to save file
        np.save(numpy_folder + filename, np.hstack((old_data, new_data)))
    except Exception:
        print('update numpy error: hstack not possible ', key_name,
              '\nold dtype: ', old_data.dtype, ' new dtype: ', new_data.dtype)


def rebin_mean(data, binLen, axis=0):
    slices, step = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen),
                               endpoint=False, retstep=True, dtype=np.intp)
    return (np.add.reduceat(data, slices, axis=axis) / step).astype('f8')


def rebin_add(data, binLen, axis=0):
    slices = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen), endpoint=False, dtype=np.intp)
    return np.add.reduceat(data, slices, axis=axis).astype('f8')


def get_end_date():
    """
    returns the date stamp of the last entry for the last full day
    """
    sorted_date_stamps = np.sort(date_stamps)
    ld = sorted_date_stamps[-1]
    # 03:00 in Compile variables. May be problematic.
    if ld[-1][11:] < '04:00':  # if less than 4am, potential for more data to be added, so drop back 1 day
        return (parser.parse(ld[:10] + ' 03:59') - timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
    else:
        return ld[:10] + ' 03:59'  # if after 4am, drop back to 4am


def get_useful_indices(start_date_time, end_date_time):
    """
    returns an array of indices for selecting, in order, all entries within the range (inclusive)
    """
    
    # needs to be reworked to use np.unique to get rid of duplicates, and use one of the returns
    # to reconstruct within range. Or, apparently, it doesn't need to be reworked for this use 0.o
    within_range = np.argwhere((date_stamps >= start_date_time) & (date_stamps <= end_date_time)).T[0]
    return within_range[np.argsort(date_stamps[within_range])]


def get_timestamp_in_min(date_time):
    return math.floor(datetime.timestamp(parser.parse(date_time).astimezone(utc)) / 60) - startDateTimeInMin


# Database manipulation functions
def open_database(dbfile):
    global Con, Cur
    try:
        Con = lite.connect(dbfile)
        Cur = Con.cursor()
        return Con, Cur
    except lite.Error as e:
        print("Error %s:" % e.args[0])


def get_field(lib, field):
    libuid = get_lib(lib)
    Cur.execute('''SELECT uuid, type_code from tbl_flex_template where
        lib_uuid = ? and lower(title) = lower(?)''', (libuid, field))
    results = Cur.fetchone()
    if not results:
        print('Unknown field of library %s:' % lib, field)
    # print('%16s %s' % (results[1], results[0]))
    return results[0], results[1]


def get_lib(lib):
    Cur.execute('SELECT uuid from tbl_library where lower(title) = lower(?)', (lib,))
    results = Cur.fetchone()
    if not results:
        print('Unknown library:', lib)
    return results[0]

# reworked to provide all necessary data, including some duplicates
def list_content(lib='Dump', field='', distinct=False, to_print=False, archive=True, recursive=False):
# def list_content(lib='Current status', field='', distinct=False, to_print=False, archive=False, recursive=False):
    """
    :param lib: library to get data from
    :param field: field to get data from
    :param distinct: if True, returns only unique values
    :param to_print: if True, prints results to console
    :param archive: if True, pulls results in from archive specified
    :param recursive: if True, this function is being called recursively
    """
    global results, main_results
    print(lib, field, archive, recursive)
    archive_results = []
    if archive:
        archive_results = list_content(lib='TAS 2021-06', field=field, archive=False, recursive=True)
    fielduid, fieldtype = get_field(lib, field)
    distinct = ' DISTINCT ' if distinct else ' '
    Cur.execute('''SELECT''' + distinct +
                '''stringcontent,realcontent,intcontent from tbl_flex_content2 where templateuuid = ?''',
                (fielduid,))
    main_results = Cur.fetchall()

    # keep the header info from the most recent data, put the older stuff first, then the recent data
    if fieldtype != 'ft_multy_str_list':
        results = main_results[:1] + archive_results + main_results[1:]

    if to_print:
        for i in results:
            for j, t in zip(i, ['s', 'r', 'i']):
                if j:
                    print(t, j)
    
    # if this is ft_multy_str_list, we have problems, as changes in the list header can occur between
    # libraries
    if recursive:  # return raw output so it concatenates properly
        return main_results
    
    # if archive is used, we end up with 2 header rows, not 1.
    s = 2 if archive == True else 1
    if fieldtype == 'ft_string':
        return np.char.strip(np.array([e[0] for e in results[s:]]))
    elif fieldtype in ['ft_real']:
        return np.array([e[1] for e in results[s:]], dtype='f2')
    elif fieldtype in ['ft_rating', 'ft_boolean']:
        return np.array([e[2] for e in results[s:]], dtype='f2')
    elif fieldtype == 'ft_map':
        return np.array([[round(float(o[0]), 3), round(float(o[1]), 3)] for o in (split_map_field_data.match(e[0]).groups() for e in results[s:])])
    elif fieldtype == 'ft_multy_str_list':
        print('\tft_multy_str_list')
        # if we've not got archive results.
        # note s is not used here, as archive results are handled separately.
        if len(archive_results) == 0:
            header = {}
            for o in json.loads(results[0][0])['sl']:
                header[str(o['c'])] = o['t']
            # I reckon object will result in a smaller array size
            return np.array([[header[e] for e in e[0].split(',')] if len(e[0]) > 0 else '' for e in results[1:]],
                            dtype=object)
        else:
            archive_header, main_header = {}, {}
            for o in json.loads(archive_results[0][0])['sl']:
                archive_header[str(o['c'])] = o['t']
            archive_ret = [[archive_header[e] for e in e[0].split(',')] if len(e[0]) > 0 else '' for e in archive_results[1:]]
            for o in json.loads(main_results[0][0])['sl']:
                main_header[str(o['c'])] = o['t']
            main_ret = [[main_header[e] for e in e[0].split(',')] if len(e[0]) > 0 else '' for e in main_results[1:]]
            
            return np.array(archive_ret + main_ret, dtype=object)                              
    else:
        print('unknown field type', fieldtype, field)
        return results


def plot_against_time(data, start_date_time, interval_in_minutes):
    """
    from start_date_time in format YYYY-MM-DD HH:MM, create a numpy array length long
    that matplotlib will read as dates.
    interval_in_minutes is time between each element, so for interval_in_minutes == 15
    [0, 15,...]
    """
    global plt
    from matplotlib import pyplot as plt
    from matplotlib import dates

    x = (np.arange(len(data)) * (interval_in_minutes / (24 * 60))) + dates.date2num([parser.parse(start_date_time)])[0]
    plt.plot(x, data)

    # to format dates on axis:
    plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M'))
    return x


def compile_weather_data_input():
    """
    hack together something to get the weather data inputs that I want.
    """
    locs = np.round(list_content(field='Location'), decimals=1)
    date_stamps = np.array([np.char.ljust(list_content(field='Date stamp'), 10).astype('datetime64[D]').astype('f8')]).T
    inputs = np.unique(np.hstack((date_stamps, locs)), axis=0)
    # reconstruct as structured array for readability
    dt = np.dtype([('date', 'U10'), ('lat', 'f8'), ('lon', 'f8')])
    new_inputs = np.full(inputs.shape[0], None, dtype=dt)
    new_inputs['date'] = np.datetime_as_string(inputs[:, 0].astype('datetime64[D]'), unit='D')
    new_inputs['lat'] = inputs[:, 1]
    new_inputs['lon'] = inputs[:, 2]


def compile_spray_locations_from_existing():
    """
    Take the 100 odd files in format iia-Sp-H&B Magnesium oil - sprays 20mg-Biceps, left.npy: [0,..., 2,...]
    and convert to Sp-H&B Magnesium oil - sprays 20mg: [['Neck, Arms'], [],...]
    2021-06-25 I'm unlikely to actually need this data in the near future, and it's more manageable in this format
    """
    folder = r'C:\Users\ashby\Documents\tims docs\Python\Data sources\Spray locations/'
    length = len(np.load(folder + os.listdir(folder)[0], allow_pickle=True))

    # create and set up basic return object
    ret_obj = {'B12 oils location': np.empty(length, dtype='object'),
               'Magnesium oil location': np.empty(length, dtype='object')}

    for key in ret_obj:
        ret_obj[key][...] = [[] for _ in range(length)]

    # populate return object
    for file in sorted(os.listdir(folder)):
        names = file.split('-')
        key = 'B12 oils location' if 'B12' in file else 'Magnesium oil location'
        data = np.load(folder + file, allow_pickle=True)
        # ret_obj[key][data.nonzero()[0]] += names[-1][:-4] + '\n'
        for ind in data.nonzero()[0]:
            ret_obj[key][ind].append(names[-1][:-4])

    for key in ret_obj:
        np.save(numpy_folder + 'iia-' + key, ret_obj[key])

    
    
Con, Cur = main()
