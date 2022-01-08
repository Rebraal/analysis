import zipfile
import mmap
import sqlite3 as lite
import sys
import uuid
import os
import time
import numpy as np
from matplotlib import pyplot as plt

# source_folder = '../Source data/'
# db_filename = 'memento-autobackup-15062021-210423'
# db_file = source_folder + db_filename + '.zip'
# reformatting for linux
source_folder = '/media/lubuntu/7A56EB2A56EAE5BB/Memento unsynced/'



def list_all_tables():
    Cur.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
    results = Cur.fetchall()
    print('\n'.join([r[0] for r in results]))
    return results


def list_content(lib, field, distinct=False):
    fielduid = get_field(lib, field)
    distinct = ' DISTINCT ' if distinct else ' '
    Cur.execute('''SELECT''' + distinct +
                '''stringcontent,realcontent,intcontent from tbl_flex_content2 where templateuuid = ?''',
                (fielduid,))
    results = Cur.fetchall()
    for i in results:
        for j, t in zip(i, ['s', 'r', 'i']):
            if j:
                print(t, j)
    return results


def list_fields(lib):
    libuid = get_lib(lib)
    Cur.execute('SELECT title, type_code, uuid from tbl_flex_template where lib_uuid = ?', (libuid,))
    for i in Cur.fetchall():
        print('%s %16s %s' % (i[2], i[1], i[0]))


def list_libs():
    Cur.execute('SELECT uuid, title from tbl_library')
    for i in Cur.fetchall():
        print(i[0], i[1])


def get_field(lib, field):
    libuid = get_lib(lib)
    Cur.execute('''SELECT uuid, type_code from tbl_flex_template where
        lib_uuid = ? and lower(title) = lower(?)''', (libuid, field))
    results = Cur.fetchone()
    if not results:
        print('Unknown field of library %s:' % lib, field)
    print('%16s %s' % (results[1], results[0]))
    return results[0]


def get_lib(lib):
    Cur.execute('SELECT uuid from tbl_library where lower(title) = lower(?)', (lib,))
    results = Cur.fetchone()
    if not results:
        print('Unknown library:', lib)
    return results[0]


def opendb(dbfile):
    global Con, Cur
    try:
        Con = lite.connect(dbfile)
        Cur = Con.cursor()
        return Con, Cur
    except lite.Error as e:
        print("Error %s:" % e.args[0])


Con, Cur = opendb(source_folder + 'memento.db')

# Temporary functions for working out last sensible entry
# reviewing analysis 2021-12-20 11:00

# Strip out data from lists
def db2np(lst):
    """"Converts a result from list_content into a numpy array able to be
    sliced by npslicer. Otherwise, ascontiguousarray isn't necessary"""
    return np.ascontiguousarray(np.array(lst, dtype=str)[:, 0])

def npslicer(a, start, end):
    """"For each element in a numpy array of strings, a, returns the value between
        start and end
        eg nslicer(np.array(['hello', 'world]), 1, 3) == ['el', 'or']"""
    b = a.view((str, 1)).reshape(len(a), -1)[:, start:end]
    return np.frombuffer(b.tobytes(), dtype=(str,end-start))
    
    
# set up vars

# libs:
#   Current status
#   TAS 2021-06
#   Dump

# lib = 'Current status'
# contains partial daily entries from 2021-10-01 to 2021-11-01
# counts.min() = 1, counts.max() = 10

# lib = 'Dump'
# contains entries from 2021-06-01 to 2021-07-05 as old epoch. Staffs preps started here.
# contains entries from 2021-07-05 to 2021-09-30 as partial daily entries. 

# lib = 'Current status Copy'
# contains nothing of use

# lib = 'TAS 2021-06'
# contains entries from 2021-01-01 to 2021-05-31

# met-Dt-Date strings.npy contains dates up to 2021-04-17 02:45
# Create numpy files needs to be reworked to use TAS 2021-06, and then Dump
# This will create a solid block of data from 2019-09-01 to 2021-07-01


Comments = db2np(list_content(lib, 'Symptoms - comments'))
Supps = db2np(list_content(lib, 'Supplements - taken'))
IntakeM = db2np(list_content(lib, 'Intake - misc'))
IntakeL = db2np(list_content(lib, 'Intake - liquid'))
Date = db2np(list_content(lib, 'Date stamp'))
dates = npslicer(Date, 0, 10)

bins, counts = np.unique(dates, return_counts=True)

npsources = '/media/lubuntu/DAE43516E434F5FB/Google Drive Sync/Programs/Python/Data sources/Numpy entries'
curr_dates = np.load(npsources + '/met-Dt-Date strings.npy', allow_pickle=True)

