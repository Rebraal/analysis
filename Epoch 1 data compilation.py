print('Start')
import sqlite3 as lite
import sys
import uuid
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math
import pytz
from dateutil import parser
from datetime import timedelta
from datetime import datetime
import json as json
import urllib.request
import requests

source_folder = '/media/lubuntu/7A56EB2A56EAE5BB/Memento unsynced/'
JSONFolder = '../Compiling JSONs/'

# epoch setings
# need to take into account day ends at 03:59 the FOLLOWING day
utc = pytz.UTC
dateRange = ['2019-09-01', '2021-07-01']
dateList = []
startDateTimeDT = parser.parse(dateRange[0] + ' 04:00').astimezone(utc)
startDateTimeInMin = math.floor(datetime.timestamp(startDateTimeDT) / 60)
endDateTimeDT = parser.parse(dateRange[1] + ' 03:59').astimezone(utc)
endDateTimeInMin = math.floor(datetime.timestamp(endDateTimeDT) / 60)
total_time_elapsed_in_sec = (endDateTimeDT - startDateTimeDT).total_seconds()
stepCount1Min = math.ceil(total_time_elapsed_in_sec / 60)
secs_in_day = 24*60*60
standard_interval_in_min = 15
total_intervals_elapsed = math.ceil(stepCount1Min / standard_interval_in_min)

emptyRow = np.zeros(shape=stepCount1Min, dtype='f2')


# Use Create numpy files as a basis for stuff. 
# To do:
#   Run create numpy files on TAS 2021-06, then Dump, selecting appropriate entries.
#   Probably the simplest way to do this is to use Create numpy files directly and to alter the database
#   funtions to access all three databases and return all the info at once. 
#   Going to have to do something with the useful_indices problem. Put it inside the list_content?
#   Complete create numpy files, then run on all databases to complete epoch data.
#   Then clean everything.


# DataBase FUNctionS
class Dbfuns:

    def list_all_tables():
        Cur.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
        results = Cur.fetchall()
        print('\n'.join([r[0] for r in results]))
        return results
    
    
    def list_content(lib, field, distinct=False):
        fielduid = Dbfuns.get_field(lib, field)
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
        libuid = Dbfuns.get_lib(lib)
        Cur.execute('SELECT title, type_code, uuid from tbl_flex_template where lib_uuid = ?', (libuid,))
        for i in Cur.fetchall():
            print('%s %16s %s' % (i[2], i[1], i[0]))
    
    
    def list_libs():
        Cur.execute('SELECT uuid, title from tbl_library')
        for i in Cur.fetchall():
            print(i[0], i[1])
    
    
    def get_field(lib, field):
        libuid = Dbfuns.get_lib(lib)
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
	
	# Strip out data from lists
    def db2np(lst):
        """"Converts a result from list_content into a numpy array able to be
        sliced by Tools.npslicer. Otherwise, ascontiguousarray isn't necessary"""
        return np.ascontiguousarray(np.array(lst, dtype=str)[:, 0])
        

# Tools
class Tools:
    
    def npslicer(a, start, end):
        """"For each element in a numpy array of strings, a, returns the value between
            start and end
            eg nslicer(np.array(['hello', 'world]), 1, 3) == ['el', 'or']"""
        b = a.view((str, 1)).reshape(len(a), -1)[:, start:end]
        return np.frombuffer(b.tobytes(), dtype=(str,end-start))


# Date time functions
class Dates:
    
    def create_fixed_date_arrays(self):
        """
        from startDateTimeDT and endDateTimeDT, create:
            array of minutes elapsed, to create date strings from.
            array of date strings in 15m intervals (standard_interval_in_min)
        storing times in minutes, as I seem to be working in minutes for most things.
        save as, respectively:
            met-Dt-Minutes elapsed.npy
            met-Dt-Date strings.npy
        Returns
        -------
        array of minutes elapsed
        array of date strings
        """
        print('create_fixed_date_arrays()')
        max_intervals = math.ceil((endDateTimeInMin - startDateTimeInMin) / standard_interval_in_min)
        minutes_elapsed = np.arange(max_intervals, dtype='f8') * standard_interval_in_min
        date_strings = Dates.np_dstr_from_ts(minutes_elapsed, '%Y-%m-%d %H:%M')
        np.save('../Numpy entries/met-Dt-Minutes elapsed.npy', minutes_elapsed)
        np.save('../Numpy entries/met-Dt-Date strings.npy', date_strings)
        return minutes_elapsed, date_strings
        
    def ts_in_min_from_google(od):
        d = int(od)
        d = d / 1000 if d > 900000000 else d
        return math.floor((d / 60) - startDateTimeInMin)
    
    def dt_from_ts(ts):
        return datetime.fromtimestamp(60 * (startDateTimeInMin + ts))
    
    # inefficient as is, but converting between numpy datetime64 and datetime is not trivial?!
    # Date STRing from TimeStamp in min
    def np_dstr_from_ts(ar, fmt):
        """ ar: array of timestamps in min
            fmt: string like '%Y-%m-%d %H:%M'
        """
        return np.array([datetime.fromtimestamp(60 * (startDateTimeInMin + ts)).strftime(fmt) for 
                         ts in ar])
    
    # can't get this to work
    def NAnp_str_from_ts(ar):
        k = (np.datetime64(startDateTimeDT) - np.datetime64('1970-01-01T00:00:00Z'))
        return np.datetime_as_string(((ar + k) / np.timedelta64(1, 's')).astype('datetime64[D]'), unit='D')
    
    def __init__(self):
        self.minutes_elapsed, self.date_strings = self.create_fixed_date_arrays()
    

class Locations:
    
    # returns:
    #   [times in min]
    #   [[lat, lon]]
    #   [errors]   
    def createAllLocationsList(self):
        """ Loop through and rename each google file from 2020_Jan to 2020-01
            The above function has been removed from this version. Look at Create
            numpy files or Compile variables. This is otherwise necessary to make 
            sure that the files load in the right order. 2020_Dec loads before
            2020_Jan, otherwise.
            
            loop through each Google file
                summarize locations
                self.locations is in form [[lat, lon], [...],...]
                locationsTimessArray is in minutes from start.
            save both, respectively as:
                met-Dt-Locations raw.npy
                met-Dt-Minutes elapsed.npy
        """
        print('createAllLocationsList()')
        global namedLocations
        endTime = endDateTimeInMin - startDateTimeInMin
        locationsList, locationsErrors = [], []
        googleSourceFolder = '../Google location data/'
        with open(JSONFolder + 'Named locations.json') as f:
            namedLocations = json.load(f)
        
    
        for file in sorted(os.listdir(googleSourceFolder)):
            # create list of all location changes in file
            if file.endswith('.json'):
                events, errors = Locations.populateFromGoogleLocations(googleSourceFolder + file, endTime)
                locationsList += events
                locationsErrors += errors
    
        e = locationsList.pop(0)
        print(len(locationsList))
        timestamps, locations = [e[0]], [e[1]]
    
        for event in locationsList:
            if event[1] != locations[-1]:
                timestamps.append(event[0])
                locations.append(event[1])
    
    
        # add false entries at start and end to prevent getLocation returning an error.
        timestamps = [0] + timestamps + \
                              [endDateTimeInMin - startDateTimeInMin - 1]
    
        locations = [locations[0]] + locations + [locations[-1]]
        timestamps, locations = np.array(timestamps, dtype='i8'), \
                                                       np.array(locations, dtype='f2')
    
    
        # Chop off previous entries, as google files are in month chunks
        locationsMask = timestamps >= 0
        timestamps, locations = timestamps[locationsMask], \
                                                       locations[locationsMask]
    
        namedLocations = None
        np.save('../Numpy entries/met-Lo-Location raw times.npy', timestamps)
        np.save('../Numpy entries/met-Lo-Locations raw.npy', locations)
        return timestamps, locations, locationsErrors
    
   
    def populateFromGoogleLocations(file, endTime):
        locSums, locErrors = [], []
        # print('populateFromGoogleLocations:', file)
        with open(file) as f:
            timelineObjects = json.load(f)['timelineObjects']
        for tlo in timelineObjects:
            if 'activitySegment' in tlo:
                k1, k2 = 'activitySegment', 'startLocation'
            else:
                k1, k2 = 'placeVisit', 'location'
            ts = Dates.ts_in_min_from_google(tlo[k1]['duration']['startTimestampMs'])
    
            if (0 <= ts) & (ts <= endTime):
                if 'latitudeE7' in tlo[k1][k2]:
                    lat = Locations.convertE7(tlo[k1][k2]['latitudeE7'])
                    lon = Locations.convertE7(tlo[k1][k2]['longitudeE7'])
                else:
                    try:
                        lat = namedLocations[tlo[k1][k2]['placeId']]['lat']
                        lon = namedLocations[tlo[k1][k2]['placeId']]['lon']
                    except Exception as e:
                        locErrors.append(tlo)
                locSums.append([ts, [lat, lon]])
    
        return locSums, locErrors
    
    def convertE7(num):
        num = int(num)
        num = num - 4294967296 if (num > 900000000) else num
        return round(num / 10000000, 3)
    
    def create_expanded_locations_array(self):
        """
        from timestamps and locs, creates return
        currently pasted in from another file and needs reworking
        Returns
        -------
        array in form [[lat, lon]] with an entry for each 15m interval in range

        """
        print('create_expanded_locations_array()')
        lats_ar = np.empty(shape=total_intervals_elapsed, dtype='f2')
        lons_ar = np.empty(shape=total_intervals_elapsed, dtype='f2')
        
        start, end = 0, math.floor(self.timestamps[0] / standard_interval_in_min)
        np.put(lats_ar, np.arange(start, end), [self.locations[0, 0]])
        np.put(lons_ar, np.arange(start, end), [self.locations[0, 1]])
        for i in range(len(self.timestamps) - 1):
            start = math.floor(self.timestamps[i] / standard_interval_in_min)
            end = math.floor(self.timestamps[i+1] / standard_interval_in_min)
            np.put(lats_ar, np.arange(start, end), [self.locations[i, 0]])
            np.put(lons_ar, np.arange(start, end), [self.locations[i, 1]])
        np.put(lats_ar, np.arange(end, len(lats_ar)), [self.locations[-1, 0]])
        np.put(lons_ar, np.arange(end, len(lats_ar)), [self.locations[-1, 1]])
        expanded_locations = np.vstack((lats_ar, lons_ar)).T
        np.save('../Numpy entries/met-Lo-Expanded locations.npy', expanded_locations)
        return expanded_locations

    def __init__(self):
        self.timestamps, self.locations, self.errors = self.createAllLocationsList()
        self.expanded_locations = self.create_expanded_locations_array()


class Weather:
    
    def create_weather_inputs_from_google_locations(times, locs):
        """"
        takes an array of timestamps and an array of [[lat], [lon]] saves, and returns and array in the format
        [['YYYY-MM-DD', lat, lon]]
        NOTE THAT IF LOCATION HAS NOT CHANGED, THERE WILL BE NO ENTRY FOR THAT DATE.
        """
        print('create_weather_inputs_from_google_locations()')
        dates = Dates.np_dstr_from_ts(times, '%Y-%m-%d').T
        locs = np.round(locs, decimals=1)
        weather_inputs = np.unique(np.hstack((np.array([Dates.np_dstr_from_ts(times, '%Y-%m-%d')]).T, 
                                              np.round(locs, decimals=1))), 
                           axis=0)
        weather_inputs = np.flip(weather_inputs, axis=0)  # reverse list to do last dates first.
        np.save('../Numpy entries/met-We-Weather inputs', weather_inputs)
        return weather_inputs
    
    def get_all_weather_data():
        """Downloads and stores a bunch of data for the next 41 entries in the list of locations and days"""
        global weather_inputs, weatherListCopy
        print('get_all_weather_data()')
        weatherListCopy = weather_inputs.copy()
        weather_inputs = np.load('../Numpy entries/met-We-Weather inputs.npy', allow_pickle=True)
        for i in range(0, 41):
            if Weather.get_weather_info(weather_inputs[0]):
                weather_inputs = weather_inputs[1:]
        np.save('../Numpy entries/met-We-Weather inputs', weather_inputs)
        return True
        
    def get_weather_info(info):
        """Downloads weather data from visualcrossing.com"""
        # global weatherData

        latitude, longitude = info[1], info[2]
        timeStart, timeEnd = info[0], info[0]
        print(info, latitude, longitude, timeStart, timeEnd)
        query = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/' +\
                 str(latitude) + '%2C%20' + str(longitude) + '/' +\
                 timeStart + '/' + timeEnd +\
                 '?unitGroup=metric&key=Q5DSSG9L78QJ6C898QRJVX32X&' +\
                 'include=obs%2Cfcst%2Cstats%2Chours%2Ccurrent'

        try:
            response = urllib.request.urlopen(query)
            data = response.read()
            weatherData = json.loads(data.decode('utf-8'))
        except Exception:
            print("Error reading from {}"
              .format(query))
            return False

        errorCode = weatherData["errorCode"] if 'errorCode' in weatherData else 0

        if errorCode > 0:
            print("Error reading from errorCode {}, error={}"
                  .format(query, errorCode))
            return False
        else:
            with open('../Visualcrossing weather data/' + timeStart + ' - ' +\
                      str(latitude) + ', ' + str(longitude) + '.json', 'w+') as f:
                f.write(json.dumps(weatherData))
            return True
        
    def each_day_has_weather_info():
        """
        Creates a list of files in Visualcrossing weather data, and compares to a list of dates from start to end of epoch

        Returns
        -------
        True or False

        """
        weather_dates = np.unique(Tools.npslicer(np.array(os.listdir('../Visualcrossing weather data')), 0, 10))
        total_days = math.ceil((endDateTimeDT - startDateTimeDT).total_seconds() / secs_in_day)
        return total_days == weather_dates.shape[0]
        

class Pollen:

    def create_pollen_inputs():
        """
        creates an array of [timestamps, lat, lon]
        slices at appropriate intervals
        """
        dates = np.load('../Numpy entries/met-Dt-Date strings.npy', allow_pickle=True)
        locs = np.load('../Numpy entries/met-Lo-Expanded locations.npy', allow_pickle=True)
        dates = np.array([dates]).T
        # 16 = 4h intervals / 0.25h segments, starting at 04:00
        data_interval_in_hours = 4 
        steps = math.floor(data_interval_in_hours * 60 / standard_interval_in_min)
        one_hour_in_intervals = math.floor(60 / standard_interval_in_min)
        offset = steps - one_hour_in_intervals
        # might need to bodge the start dates with a [:-1] at the end, if there are a silly number of entries?
        # 12 = 16 - 4 (1h before)
        start_dates = np.array([np.append(['2019-09-01 03:00'], dates[offset::steps])]).T
        # need to have a time interval that is more than exactly 1h, so 23:00 to 00:15 gives 00:00 data.
        # this seems to cut off the last entry for some reason? 2021-07-01 03:45 results in 2021-06-30 23:00
        pollen_inputs = np.flip(np.hstack((start_dates[:-1], np.hstack((dates, locs))[1::steps])), axis=0)
        np.save('../Numpy entries/met-Pi-Pollen inputs.npy', pollen_inputs)
        
    def populate_pollen_data(num_records=100, key=0):
        """
        at 4h intervals
        Work backwards storing pollen data.
        :num_records: number of records to create from list.
        """

        def get_pollen_data(input_data, key=0, url="https://api.ambeedata.com/history/pollen/by-lat-lng"):
            """
            return pollen data for lat, lng from start_stamp to end_stamp
                "lat": "12.9889055",
                "lng": "77.574044",
                "from": "2020-07-13 12:16:44",  # note seconds on here.
                "to": "2020-07-14 12:16:44"  # note seconds on here.

                getambee logins
                bobashby7@aol.com get@mb3E
            """
            start_stamp, end_stamp, lat, lng = input_data  # want to know what's happened in hour leading up to time shown.
            start_stamp += ':00'
            end_stamp += ':00'
            querystring = {"lat": lat,
                           "lng": lng,
                           "from": start_stamp,
                           "to": end_stamp
                           }
            key = "QMHiXHhzaN3D9pScEpq2a1cGS82Xkccn2muBtC3Z" if key == 0 \
                else '688976de17b96025ac1554585fe0d3f657ee2a60dda7d8caba06e748592e73b1'
                # else "4c6a68263625a70f72eedd67c6eb99849a61d9f697e3c2c14ee21ae42ce86bf0"
            headers = {
                # 'x-api-key': "QMHiXHhzaN3D9pScEpq2a1cGS82Xkccn2muBtC3Z",  # Tim
                # 'x-api-key': "4c6a68263625a70f72eedd67c6eb99849a61d9f697e3c2c14ee21ae42ce86bf0",  # Bob
                # 'x-api-key': "688976de17b96025ac1554585fe0d3f657ee2a60dda7d8caba06e748592e73b1",  # T
                'x-api-key': key,
                'Content-type': "application/json"
                }
            response = requests.request("GET", url, headers=headers, params=querystring)
            # print(response.text)
            return response
    
        # get input data
        pollen_inputs = np.load('../Numpy entries/met-Pi-Pollen inputs.npy', allow_pickle=True)

        # get pollen data and store
        for r in range(num_records):
            file_name = ' '.join(pollen_inputs[0]).replace(':', 'c')
            print('starting: ', r, file_name)
            data = get_pollen_data(pollen_inputs[0], key)
            if data.status_code == 200:
                with open('../Ambee pollen data/' + file_name + '.json', 'w+') as f:
                    f.write(json.dumps(data.text))
                pollen_inputs = pollen_inputs[1:]
                # update input data
                np.save('../Numpy entries/met-Pi-Pollen inputs.npy', pollen_inputs)
            else:
                print('Broken at ', file_name, '\n', data.status_code)
                break

        print('Completed')    

# print(source_folder + 'memento.db')    
# Con, Cur = Dbfuns.opendb(source_folder + 'memento.db')
# dates = Dates()
# locations = Locations()
# weather_inputs = Weather.create_weather_inputs_from_google_locations(locations.timestamps, locations.locations)
# got_weather = Weather.get_all_weather_data()


print('Loaded')