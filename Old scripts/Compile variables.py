# 2021-02-13 20:00 compileAllInputs() works with no fails


import numpy as np
import json as json
import os
import re
from dateutil import parser
from datetime import timedelta
from datetime import datetime
import pytz
import math
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from scipy import sparse
import urllib.request


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

timeSteps = [0.25, 0.5, 1, 2, 4, 8, 12, 18, 24, 30, 36, 48, 60, 72]
sourceFolder = '../../Data sources/Clean JSON overview entries/'
sourceFolderIndividual = '../../Data sources/Raw individual entries/'
googleSourceFolder = '../../Data sources/Google location data/'
destinationFolder = '../Python dumps/Compiled data/'
JSONFolder = '../../Data sources/Compiling JSONs/'
excludedOutputs = ['', 'undefined']
globalFileList = sorted(os.listdir(sourceFolder))
secInHours = 60 * 60

# need to take into account day ends at 03:59 the FOLLOWING day
utc = pytz.UTC
dateRange = ['2019-09-01', '2021-04-17']
dateList = []
startDateTimeDT = parser.parse(dateRange[0] + ' 04:00').astimezone(utc)
startDateTimeInMin = math.floor(datetime.timestamp(startDateTimeDT) / 60)
endDateTimeDT = parser.parse(dateRange[1] + ' 03:00').astimezone(utc)
endDateTimeInMin = math.floor(datetime.timestamp(endDateTimeDT) / 60)
stepCount1Min = math.ceil((endDateTimeDT - startDateTimeDT).total_seconds() / 60)

emptyRow = np.zeros(shape=stepCount1Min, dtype='f2')

# Co - combined food
# Li - liquid intake
# tC - total combined conglomerate
# tL - total liquid conglomerate
# Su - supps
# Sp - supplement sprays
# Sl - sublingual
# Me - metadata
# We - weather
# Sn - sleep night, single value
# SN - sleep night, duration
# Sd - sleep day, single value
# SD - sleep day, duration
# Sa - all sleep, single value
# SA - all sleep, duration
# Re - rest, single value
# RE - rest, duration
# Sy - symptom
# Sc - symptom conglomerate
# St - step count
# Da - simple data
# Ex - exercise

# inputs
excludedIntakes = ['', 'undefined']
findIntkEntry = re.compile('(\d\d:\d\d)(.*?)\d\d:\d\d', re.DOTALL)
findSprayLocations = re.compile('(.*) - ({.*})')
findMgSpray = re.compile('(H&B \*Magnesium oil - sprays\* 20mg): (.*)\n|$')
findB12Spray = re.compile('(B12Oils.*): (.*)\n|$')
findRushedIntake = re.compile('Rushed intake - (\d\d:\d\d)', re.DOTALL)

sublingualLeftIntact, rationalizeSearch, rationalizeReplace, allergens, intakesToConglomerate = {}, {}, {}, {}, {}

# locations
getMonth = re.compile('(\d\d\d\d_)(.*)\.json')
locationsTimesArray, locationsLocationsArray, locationsDaysArray, locationsErrors, namedLocations = [], [], [], [], {}

# sleep
getSleep = re.compile('(Rest|Sleep) @ (\d\d:\d\d), (\d\.*\d*)')
majorSleeps = []

# symptoms
individualDateList = []
findSymData = re.compile('^(.*) i(.)$', re.MULTILINE)
symptomsFails, oldEntrySymptoms, newEntrySymptoms = [], [], []

# weather
weatherFails = []
rndLocs = 4
locArray, locData = [], {}

# exercise
exList = []

inputsObjAdd, inputsObjMean, inputsFails = {}, {}, []
outputsObjAdd, outputsObjMean, outputsFails = {}, {}, []
onlyZeros, subtracts = [], []
global_sparsify = True



def compileAllVariables(bCompileAllIntakes=True, bCompileAllSleep=True, bCompileAllSymptoms=True,
                        bCompileAllSteps=True, bCompileAllWeather=True, bGetAltitude=False, bGetSimpleData=True,
                        bGetExerciseData=True, clean=True, bDebug=False, sparsify=True):
    """Compile all variables into """
    # global inputsObjAdd, inputsObjMean, inputsFails, dateList
    #		For all entry overview files within dateRange
    #            Create the following in 1 minute time bins. This will be huge, but is the simplest form of rebinning
    #            POSSIBLY this could be reduced to 5 minute increments. Truth, data probably isn't accurate enough to worry
    #            about, and should just bite the bullet and go for 15 minute steps. This adds a layer of division.
    #            Summary - rebinning / rebinning is trivial, so do 1m timesteps.
    #

    constructDateListFromDateRange()
    constructIndividualDateList()
    createAllLocationsList()
    global global_sparsify
    global_sparsify = sparsify
    # stuff needs to be split out into minimum chunks to prevent memory overload.
    # need to initialise inputs & outputs objects here for each group
    if bCompileAllIntakes:
        compileAllIntakes(clean=clean, bDebug=bDebug)

    if bCompileAllSleep:
        compileAllSleep(clean=clean, bDebug=bDebug)

    if bCompileAllSymptoms:
        compileAllSymptoms(clean=clean, bDebug=bDebug)

    if bCompileAllSteps:
        compileAllSteps(clean=clean, bDebug=bDebug)

    if bCompileAllWeather:
        compileAllWeather(clean=clean, bGetAltitude=bGetAltitude, bDebug=bDebug)

    if bGetSimpleData:
        # No point in writing code to parse from overviews, seeing as individual entries are being looped through
        # already. For compartmentalisation, they'll have to be looped through a couple more times, maybe.
        compileSimpleData(clean=clean, bDebug=bDebug)

    if bGetExerciseData:
        compileExerciseData(clean=clean, bDebug=bDebug)

    np.save(destinationFolder + 'Subtracts', np.array(subtracts))



def constructDateListFromDateRange():
    print('constructDateListFromDateRange()')
    global dateList
    dateList = []
    for file in globalFileList:
        if file.endswith('.json') & checkDateInRange(file[0:10]):
            dateList.append(file)
    dateList.sort()


def checkDateInRange(file):
    return (dateRange[0] <= file) & (file <= dateRange[1])


def constructIndividualDateList():
    global individualDateList
    print('Building individual entry source list')
    individualDateList = []
    minFileDate, maxFileDate = dateRange[0] + ' 0400', dateRange[1] + ' 0359'
    for file in sorted(os.listdir(sourceFolderIndividual)):
        if file.endswith('.json') & (minFileDate <= file[:15]) & (file[:15] <= maxFileDate):
            individualDateList.append(file)
    individualDateList.sort()


def compileAllIntakes(clean=True, bDebug=False):
    """For each file in list, extract liquids, combined and supplements, including oil locations from supps."""
    tscAI = datetime.now()
    print('compileAllIntakes()', tscAI)
    global inputsObjAdd, inputsFails, sublingualLeftIntact, rationalizeSearch, rationalizeReplace, \
        allergens, intakesToConglomerate, inputsObjMean

    inputsFails = []
    inputsObjAdd = {
        'Me-Sublingual terminated': emptyRow.copy(),
        'Me-Rushed intake': emptyRow.copy(),
    }
    inputsObjMean = {}

    with open(JSONFolder + 'Sublingual intact.json') as f:
        sublingualLeftIntact = set(json.load(f))
    with open(JSONFolder + 'Rationalize combined search.json') as f:
        rationalizeSearch = np.array(json.load(f))
    with open(JSONFolder + 'Rationalize combined replace.json') as f:
        rationalizeReplace = np.array(json.load(f))
    with open(JSONFolder + 'Allergens.json') as f:
        allergens = json.load(f)
    with open(JSONFolder + 'Conglomerate.json') as f:
        intakesToConglomerate = json.load(f)

    endTime = endDateTimeInMin - startDateTimeInMin
    for file in dateList:
        print(file)
        fn = sourceFolder + file
        if file.endswith('.json'):
            try:
                with open(fn) as f:
                    entry = json.load(f)
            except Exception as ex:
                print(file)
                raise ex
            date = entry['Header'][0:-4]
            populateIntakesData(date, entry, endTime, 'Intake - liquid', 'Li-')
            populateIntakesData(date, entry, endTime, 'Intake - combined', 'Co-')
            populateIntakesData(date, entry, endTime, 'Supplements', 'Su-')

    compileTotalIntakes()
    # find all sublingual entries and calculate duration from them, updating intakesKeys and Values
    # this could be expanded to include single value sublinguals, ie at index = start, value = 5mg * 300min duration
    compileSublingualDurations()
    # create conglomerates for "Apple", "Apple - cooked" etc. This includes allergens too.
    compileConglomerateIntakes()
    compileAllergens()

    # rebin, sort, save and clear var.
    inputsObjAdd = cleanup('I-05 Intakes add', inputsObjAdd, 'add', clean=clean, bDebug=bDebug)
    inputsObjMean = cleanup('I-05 Intakes mean', inputsObjMean, 'mean', clean=clean, bDebug=bDebug)
    sublingualLeftIntact, rationalizeSearch, rationalizeReplace, allergens, intakesToConglomerate = \
        None, None, None, None, None

    print(datetime.now() - tscAI)


def compileAllSleep(clean=True, bDebug=False):
    tscAS = datetime.now()
    print('compileAllSleep()', tscAS)
    global inputsObjAdd, outputsObjAdd, majorSleeps
    majorSleeps = []
    inputsObjAdd = {
        'Re-Rest': emptyRow.copy(),
        'RE-Rest': emptyRow.copy(),
        'Sd-Sleep, day': emptyRow.copy(),
        'SD-Sleep, day': emptyRow.copy()
    }
    outputsObjAdd = {
        'Re-Rest': emptyRow.copy(),
        'RE-Rest': emptyRow.copy(),
        'Sd-Sleep, day': emptyRow.copy(),
        'SD-Sleep, day': emptyRow.copy()
    }
    endTime = endDateTimeInMin - startDateTimeInMin
    for file in dateList:
        print(file)
        fn = sourceFolder + file
        if file.endswith('.json'):
            try:
                with open(fn) as f:
                    entry = json.load(f)
            except Exception as ex:
                print(file)
                raise ex
            date = entry['Header'][0:-4]
            populateSleepData(date, entry, endTime)

    completeSleepNightData()

    inputsObjAdd = cleanup('I-01 Sleep add', inputsObjAdd, 'add', clean=clean, bDebug=bDebug)
    outputsObjAdd = cleanup('O-01 Sleep add', outputsObjAdd, 'add', clean=clean, bDebug=bDebug)

    print(datetime.now() - tscAS)


def populateIntakesData(date, entry, endTime, fld, prefix):
    global inputsObjAdd, inputsFails
    # Check this - does this grab the last entry in the field?
    sprayPrefix = 'Sp-'
    if prefix == 'Co-':
        sublingualTerminated = True  # pinch this conditional to stick this here
        rushedIntake = findRushedIntake.search(entry[fld])
        while rushedIntake is not None:
            inputsObjAdd['Me-Rushed intake'][getTimeStampInMinFromOverview(date, rushedIntake.group(1))] = 1
            rushedIntake = findRushedIntake.search(entry[fld], rushedIntake.span()[0] + 1)

    # look for a time, followed by stuff, then a time. Store first time and stuff. Add timeStamp to end of string, this
    # saves messing around debugging a new regexp
    fieldToSearch = entry[fld] + '\n03:59'
    intkEntry = findIntkEntry.search(fieldToSearch)
    # look for all time entries
    while intkEntry is not None:
        intakes = np.char.split(np.array(intkEntry.group(2).split('\n'), dtype='U200'), sep=': ')
        valuesIndex = getTimeStampInMinFromOverview(date, intkEntry.group(1))
        if (0 <= valuesIndex) & (valuesIndex <= endTime):

            for intk in intakes:
                # check if not a spray location
                if intk[0].find('{') == -1:
                    # and is a decent entry
                    if intk[0] != '':
                        if len(intk) == 2:
                            try:
                                val = float(intk[1])
                            except ValueError:
                                inputsFails.append(date + ': ' + str(intk))
                            else:
                                key = prefix + intk[0]

                                if prefix == 'Co-':
                                    # this could be done post array construction, but doing it here pays the price of a few
                                    # seconds for a little more free memory at the end.
                                    # if the key needs to be rationalized
                                    replaceKeyIndex = findNext(key, rationalizeSearch)
                                    key = key if replaceKeyIndex == -1 else rationalizeReplace[replaceKeyIndex]

                                    # if the key isn't in the list of stuff that leaves the sl intact, it's terminated
                                    sublingualTerminated = False if key in sublingualLeftIntact else True

                                if key in inputsObjAdd:
                                    inputsObjAdd[key][valuesIndex] = val
                                # else add key and data
                                else:
                                    inputsObjAdd[key] = emptyRow.copy()
                                    inputsObjAdd[key][valuesIndex] = val
                        # not enough items in list fail
                        else:
                            inputsFails.append(date + ' ' + intkEntry.group(1) + ' len ' + str(intk))
                # this must be a spray location
                else:
                    sprayInfo = findSprayLocations.search(intk[0])
                    # search for the qty attached to the spray
                    if sprayInfo is not None:
                        sprayQtyInfo = findMgSpray.search(intkEntry.group(2)) if sprayInfo.group(1) == 'Mg' \
                            else findB12Spray.search(intkEntry.group(2))
                        try:
                            sprayQty = float(sprayQtyInfo.group(2))
                        except (ValueError, TypeError):
                            inputsFails.append(date + ' ' + intkEntry.group(1) + ': ' +
                                                sprayInfo.group(1) + ' ' + str(sprayQtyInfo))
                        else:
                            # stored as spray qty / locations spray used on
                            # if spray location exists, store it,
                            sprayLocations = json.loads(sprayInfo.group(2))
                            div = len(sprayLocations.keys())
                            for loc in sprayLocations:
                                key = sprayPrefix + sprayQtyInfo.group(1) + '-' + sprayLocations[loc]
                                if key in inputsObjAdd:
                                    inputsObjAdd[key][valuesIndex] = sprayQty / div
                                # else add key and data
                                else:
                                    inputsObjAdd[key] = emptyRow.copy()
                                    inputsObjAdd[key][valuesIndex] = sprayQty / div
                    # spray location fail
                    else:
                        inputsFails.append(date + ' ' + intkEntry.group(1) + ' spr ' + str(intk))
            if prefix == 'Co-':
                if sublingualTerminated:
                    inputsObjAdd['Me-Sublingual terminated'][valuesIndex] = 1

        intkEntry = findIntkEntry.search(fieldToSearch, intkEntry.span()[0] + 1)


def compileTotalIntakes():
    global inputsObjAdd
    lit, cotq, cotc = emptyRow.copy(), emptyRow.copy(), emptyRow.copy()
    for key in inputsObjAdd:
        if 'Co-' in key:
            cotq += inputsObjAdd[key]
            cotc += inputsObjAdd[key] != 0
        elif 'Li-' in key:
            lit += inputsObjAdd[key]
    inputsObjAdd['Li-Total intake'] = lit
    inputsObjAdd['Co-Total intake, quantity'] = cotq
    inputsObjAdd['Co-Total intake, count'] = cotc


def compileSublingualDurations():
    """Find all sublinguals, for each sublingual start, pad 0s, (start), pad values until stop"""
    tscSL = datetime.now()
    global inputsObjMean
    print('compileSublingualDurations()', tscSL)
    slStr, slList = ' sl*', []
    slTerminated = (inputsObjAdd['Me-Sublingual terminated'] != 0).nonzero()[0]
    # if we find a suitable target, store the data to build up a list of sublinguals
    for key in list(inputsObjAdd):
        if slStr in key:
            print(slStr)
            newSl, putStart, putEnd, newKey = emptyRow.copy(), 0, 0, 'Sl-' + key[3:]
            slIndices = (inputsObjAdd[key] != 0).nonzero()[0]
            for i in range(len(slIndices)):
                # fill up to first index with 0s
                np.put(newSl, np.arange(putStart, slIndices[i]), [0])
                # then fill with value of taken supp
                putEnd = findNextNumberGreater(slIndices[i], slTerminated, start=putEnd)
                np.put(newSl, np.arange(slIndices[i] + 1, slTerminated[putEnd] + 1), inputsObjAdd[key][slIndices[i]])
                putStart = slTerminated[putEnd] + 1

            np.put(newSl, np.arange(putStart, len(newSl)), [0])
            inputsObjMean[newKey] = newSl

    print(datetime.now() - tscSL)


def test_put():
    """
    test fastest method
    np.put(array, np.arange(start, end), val)
    array[start:end] = np.full((end-start), val)
    """

    length = 10000
    rpts = 1000
    val = 9
    new_row = np.zeros(length)
    put_time = datetime.now()
    print(put_time)
    starts = np.random.randint(0, length / 2, rpts)
    ends = np.random.randint(length / 2, length-1, rpts)

    for i in range(rpts):
        np.put(new_row, np.arange(starts[i], ends[i]), val)
    print('put time = ', datetime.now() - put_time)

    new_row = np.zeros(length)
    put_time = datetime.now()
    print(put_time)
    for i in range(rpts):
        new_row[start:end] = np.full((starts[i] - ends[i]), val)
    print('slice time = ', datetime.now() - put_time)


def compileConglomerateIntakes():
    """groups a bunch of intake data together"""
    tscCI = datetime.now()
    global inputsObjAdd
    print('compileConglomerateIntakes()', tscCI)
    keysLen = len(inputsObjAdd)
    retKeys, retValues = [], []
    for congObj in intakesToConglomerate:
        tempCongAr = emptyRow.copy()
        # if we're looking for general strings, starting with...
        if congObj['t'] == 'g':
            for intFragToFind in congObj['d']:
                for key in list(inputsObjAdd):
                    if intFragToFind in key:
                        tempCongAr += inputsObjAdd[key]

        else:
            for intFragToFind in congObj['d']:
                if intFragToFind in inputsObjAdd:
                    tempCongAr += inputsObjAdd[intFragToFind]
        # if there's any data in the new array
        if np.sum(tempCongAr) > 0:
            inputsObjAdd[congObj['k']] = tempCongAr
        print(congObj)

    print(datetime.now() - tscCI)


def compileAllergens():
    """looks for allergens and populates appropriate arrays"""
    tscA = datetime.now()
    global inputsObjAdd
    print('compileAllergens()', tscA)
    allergensDict = {
        "c": {"n": "Al-Caffeine", "v": emptyRow.copy()},
        "d": {"n": "Al-Dairy", "v": emptyRow.copy()},
        "g": {"n": "Al-Gluten", "v": emptyRow.copy()},
        "f": {"n": "Al-FODMAP", "v": emptyRow.copy()},
        "l": {"n": "Al-Legumes", "v": emptyRow.copy()},
        "n": {"n": "Al-Nuts", "v": emptyRow.copy()},
        "r3": {"n": "Al-Omega 3", "v": emptyRow.copy()},
        "r6": {"n": "Al-Omega 6", "v": emptyRow.copy()},
        "p": {"n": "Al-Preservatives", "v": emptyRow.copy()},
        "rs": {"n": "Al-Resistant starch", "v": emptyRow.copy()},
        "Na": {"n": "Al-Sodium", "v": emptyRow.copy()},
        "sp": {"n": "Al-Spices", "v": emptyRow.copy()},
        "s": {"n": "Al-Sugar", "v": emptyRow.copy()},
        "u": {"n": "Al-Unknown", "v": emptyRow.copy()},
        "y": {"n": "Al-Yeast", "v": emptyRow.copy()},
    }
    for algObj in allergens:
        print(algObj)
        if algObj['key'] in inputsObjAdd:
            for alg in algObj['a']:
                allergensDict[alg]['v'] += inputsObjAdd[algObj['key']]

    for key in allergensDict:
        inputsObjAdd[allergensDict[key]['n']] = allergensDict[key]['v'].astype('f2')

    print(datetime.now() - tscA)


def populateSleepData(date, entry, endTime):
    """format = Sleep | Rest @ \d\d:\d\d, (dur)h q:(num), r:(num)"""
    global inputsObjAdd, outputsObjAdd, majorSleeps
    # print('populateSleepData()', tsPSND)

    # parse sleep data into numbers
    sleepData = entry['Sleep - night'].split('\n')
    initialiseSleepData('Sa-', 'SA-', 'Sleep')
    # for each sleep instance
    for sleepStr in sleepData:
        # check for major sleeps - starts between 18:00 and 06:00, longer than 3h
        # getSleep = re.comile('(Rest|Sleep) @ (\d\d:\d\d), (\d\.*\d*)')
        # locationsList in form [[time, lat, long],...]
        sleep = getSleep.search(sleepStr)
        if sleep is not None:
            durH = float(sleep.group(3))
            duration = int(durH * 60)
            timeIndex = getTimeStampInMinFromSleep(date, sleep.group(2))
            if True:
            # if (0 <= timeIndex) & (timeIndex <= endTime):
                if (sleep.group(1) == 'Sleep') &\
                   ((sleep.group(2) <= '06:00') | (sleep.group(2) >= '18:00')) &\
                   (durH >= 3.):

                    # find location
                    locationKey = str(getLocationFromTimestamp(timeIndex))
                    timeKey = int(sleep.group(2)[0:2])
                    # this enables averaging
                    timeKey = timeKey if timeKey > 6 else timeKey + 24
                    majorSleeps.append([timeIndex, durH, timeKey, date])

                elif sleep.group(1) == 'Rest':
                    if durH > 3:
                        raise Exception(date + ' ' + sleepStr + ' ' + str(duration))
                    writeSleepData('Re-', 'RE-', 'Rest', timeIndex, durH, duration)

                else:
                    writeSleepData('Sd-', 'SD-', 'Sleep, day', timeIndex, durH, duration)
                    writeSleepData('Sa-', 'SA-', 'Sleep', timeIndex, durH, duration)

    sleepData = entry['Sleep - day'].split('\n')
    for sleepStr in sleepData:
        sleep = getSleep.search(sleepStr)
        if sleep is not None:
            durH = float(sleep.group(3))
            duration = int(durH * 60)
            timeIndex = getTimeStampInMinFromSleep(date, sleep.group(2))
            writeSleepData('Sd-', 'SD-', 'Sleep, day', timeIndex, durH, duration)
            writeSleepData('Sa-', 'SA-', 'Sleep', timeIndex, durH, duration)


def getLocationFromTimestamp(timeStamp):
    locationIndex = np.searchsorted(locationsTimesArray, timeStamp)
    locationKey = locationsLocationsArray[locationIndex]
    return locationKey


def initialiseSleepData(k1, k2, key):
    global inputsObjAdd, outputsObjAdd
    if k1 + key not in inputsObjAdd:
        # inputsObjAdd[k1 + key] = emptyRow.copy()
        # outputsObjAdd[k1 + key] = emptyRow.copy()
        inputsObjAdd[k2 + key] = emptyRow.copy()
        outputsObjAdd[k2 + key] = emptyRow.copy()


def writeSleepData(k1, k2, key, timeIndex, durH, duration):
    global inputsObjAdd, outputsObjAdd
    # print(k1, key, timeIndex, duration)
    if (timeIndex < stepCount1Min) & (timeIndex + duration < stepCount1Min):
        # inputsObjAdd[k1 + key][timeIndex] = durH
        # outputsObjAdd[k1 + key][timeIndex] = durH
        np.put(inputsObjAdd[k2 + key], np.arange(timeIndex, timeIndex + duration), 1)
        np.put(outputsObjAdd[k2 + key], np.arange(timeIndex, timeIndex + duration), 1)


def completeSleepNightData():
    global inputsObjAdd, outputsObjAdd, majorSleeps
    print('completeSleepNightData()')
    # Take first 7 entries from major sleeps, average data and fill blanks
    majorSleeps = np.array(majorSleeps)
    msM, msDu, msH, msDay = majorSleeps[:, 0].astype('i4'), majorSleeps[:, 1].astype('f2'), \
                            majorSleeps[:, 2].astype('i2'), majorSleeps[:, 3]
    majorSleepsFilled = []
    for dayFile in dateList:
        day = dayFile[:10]
        if day not in msDay:
            hour = int(np.sum(msH[:7]) / 7)
            hour = hour - 24 if hour > 23 else hour
            majorSleepsFilled.append([getTimeStampInMinFromOverview(day, str(hour) + ':00'),
                                      round(np.sum(msDu[:7]) / 7, 2),
                                      hour,
                                      day])
        else:
            majorSleepsFilled.append([msM[0], msDu[0], msH[0] - 24 if msH[0] > 23 else msH[0], msDay[0]])
            msM, msDu, msH, msDay = msM[1:], msDu[1:], msH[1:], msDay[1:]

    majorSleeps = majorSleepsFilled
    for sleep in majorSleeps:
        # find location
        timeIndex = sleep[0]
        durH = sleep[1]
        duration = int(sleep[1] * 60)
        timeKey = str(sleep[2]).zfill(2)
        locationKey = str(getLocationFromTimestamp(timeIndex))
        # create rows if not created. Assumes if Sn created, SN created.
        initialiseSleepData('Sn-', 'SN-', locationKey)
        initialiseSleepData('Sn-', 'SN-', timeKey)

        writeSleepData('Sn-', 'SN-', locationKey, timeIndex, durH, duration)
        writeSleepData('Sn-', 'SN-', timeKey, timeIndex, durH, duration)
        writeSleepData('Sa-', 'SA-', 'Sleep', timeIndex, durH, duration)


def compileAllSymptoms(clean=True, bDebug=False):
    """Due to lack of thought in constructing the overview code, this has to be done from the raw individual entries."""
    tscAS = datetime.now()
    print('compileAllSymptoms()', tscAS)
    global inputsObjMean, outputsObjMean, symptomsFails, oldEntrySymptoms, newEntrySymptoms
    inputsObjMean, outputsObjMean, symptomsFails, oldEntrySymptoms, newEntrySymptoms = {}, {}, [], [], []

    print('Building symptoms object')
    for file in individualDateList:
        # print(file)
        fn = sourceFolderIndividual + file
        if file.endswith('.json'):
            try:
                with open(fn) as f:
                    entry = json.load(f)
            except Exception as ex:
                print(file)
                raise ex
            populateSymptomData(entry)

    parseDateTimeIntoRow()
    conglomerateSymptoms()

    # for each symptom, find the mode, and subtract it from all elements of that symptom
    # this increases sparsity, which may speed up things
    # commenting out sparsify, as it might just end up being a REAL nuisance the stuff appended to keys.

    # need to invert #P# symptoms here - subtract each from 0. Really bad score + good positive = 0

    tmpObj = {}
    for key in outputsObjMean:
        if '#P#' in key:
            tmpObj[key.replace(' #P#', '')] = 5 - outputsObjMean[key]

    for key in tmpObj:
        outputsObjMean[key] = tmpObj[key]

    # it's possible that there could be a link between symptoms - this might show that.
    for key in outputsObjMean:
        inputsObjMean[key] = outputsObjMean[key]

    inputsObjMean = cleanup('I-50 Symptoms mean', inputsObjMean, 'mean', clean=clean, bDebug=bDebug)
    outputsObjMean = cleanup('O-50 Symptoms mean', outputsObjMean, 'mean', clean=clean, bDebug=bDebug)

    print(datetime.now() - tscAS)


def populateSymptomData(entry):
    """Populate object with a bunch of timestamped values"""
    global outputsObjMean, symptomsFails, oldEntrySymptoms, newEntrySymptoms
    timeStamp = int(datetime.timestamp(parser.parse(entry['Date stamp']).astimezone(utc)) / 60) - startDateTimeInMin
    newEntrySymptoms = []
    for symData in findSymData.findall(entry['Symptoms - current']):
        # remove all feet symptoms from data
        symptom = 'Sy-' + symData[0]
        if 'Feet' in symptom:
            pass
        else:
            try:
                val = int(symData[1])
            except ValueError:
                symptomsFails.append((entry['Date stamp'], symData))
            else:
                if symptom not in outputsObjMean:
                    outputsObjMean[symptom] = []
                outputsObjMean[symptom].append((timeStamp, val))
                newEntrySymptoms.append(symptom)

    # check for symptom deletion implying termination
    for sym in oldEntrySymptoms:
        if sym not in newEntrySymptoms:
            outputsObjMean[sym].append((timeStamp, 0))

    oldEntrySymptoms = newEntrySymptoms


def parseDateTimeIntoRow():
    """Convert a list of [timeStamp, val] to a numpy array of continuous values"""
    global outputsObjMean
    print('parseDateTimeIntoRow()')
    # pad data with 0 from 0 to first index, and val from last index to end of array
    for key in outputsObjMean:
        # print(data)
        if type(outputsObjMean[key]) == list:
            tmp = emptyRow.copy()
            startIndex, val = outputsObjMean[key].pop(0)
            # print(data, startIndex, val)
            np.put(tmp, np.arange(0, startIndex), [0])
            for data in outputsObjMean[key]:
                # print(data, startIndex, data)
                np.put(tmp, np.arange(startIndex, data[0]), [val])
                startIndex, val = data
            np.put(tmp, np.arange(startIndex, len(tmp)), [val])
            outputsObjMean[key] = tmp


def conglomerateSymptoms():
    """Run through and conglomerate stuff like dry skin on hands"""
    print('conglomerateSymptoms()')
    global outputsObjMean
    # in form {'Sc-Dry skin': {'hands, dry skin', 'face, dry skin'}}
    with open(JSONFolder + 'Symptoms conglomerates.json') as f:
        conglomerates = json.load(f)

    for congKey in conglomerates:
        tmp = emptyRow.copy()
        for key in conglomerates[congKey]:
            try:
                tmp += outputsObjMean['Sy-' + key]
            except KeyError:
                pass
        outputsObjMean[congKey] = tmp / len(conglomerates[congKey])


def compileAllSteps(clean=True, bDebug=False):
    """Reads all files in folder and populates step data
        Google files start at 00:15 and end at 00:00"""
    global inputsObjAdd, outputsObjAdd
    print('compileAllSteps()')
    googleSourceFolder = 'C:/Memento analysis/Data sources/Google step data/'
    inputsObjAdd = {
        'St-Steps': emptyRow.copy(),
    }
    offset = 60 * 14
    for file in sorted(os.listdir(googleSourceFolder)):
        if file.endswith('.csv'):
            print(file)
            steps = np.genfromtxt(googleSourceFolder + file, filling_values=0, delimiter=',', comments="&&&",
                                  skip_header=1, dtype='str', usecols=(0, 11))
            for i in range(steps.shape[0]):
                timeStamp = int(
                    datetime.timestamp(parser.parse(file[:10] + ' ' + steps[i][0][:5]).astimezone(utc) +
                                       timedelta(seconds=offset)) / 60)
                if (startDateTimeInMin <= timeStamp) & (timeStamp <= endDateTimeInMin):
                    if steps[i][1] != '':
                        inputsObjAdd['St-Steps'][timeStamp - startDateTimeInMin] = int(steps[i][1])

    outputsObjAdd['St-Steps'] = inputsObjAdd['St-Steps']
    inputsObjAdd = cleanup('I-02 Steps add', inputsObjAdd, 'add', clean=clean, bDebug=bDebug)
    outputsObjAdd = cleanup('O-02 Steps add', outputsObjAdd, 'add', clean=clean, bDebug=bDebug)


def compileAllWeather(clean=True, bGetAltitude=False, bDebug=False):
    """From visualcrossing data and google location data compile a list of all weather conditions
        given I only have data to 60 minute resolution, populate array and then rebin to 0.25(!)"""
    global inputsObjMean, inputsObjAdd, weatherFails, elevationsLookup, locData
    print('compileAllWeather()')
    googleSourceFolder = 'C:/Memento analysis/Data sources/Visualcrossing weather data/'
    weatherDataSimple = ["temp",  #: 7.9,
                         "feelslike",  #: 7.9,
                         "humidity",  #: 78.43,
                         "dew",  #: 4.9,
                         "precip",  #: null,
                         "windgust",  #: null, set nulls to 0
                         "windspeed",  #: 5.4,
                         "winddir",  #: 90.0,
                         "pressure",  #: 1029.4,
                         "visibility",  #: 8.0,
                         "cloudcover",  #: 0.0,
                         ]
    inputsObjMean = {}
    newRow = rebinAdd(emptyRow.copy(), 60).astype('f2')
    for key in weatherDataSimple:
        inputsObjMean['We-' + key] = newRow.copy()

    print('\tCreating locations')
    locsLat, locsLon = newRow.copy(), newRow.copy()

    start, end = 0, math.floor(locationsTimesArray[0] / 60)
    np.put(locsLat, np.arange(start, end), [locationsLocationsArray[0, 0]])
    np.put(locsLon, np.arange(start, end), [locationsLocationsArray[0, 1]])
    for i in range(len(locationsTimesArray) - 1):
        start, end = math.floor(locationsTimesArray[i] / 60), math.floor(locationsTimesArray[i+1] / 60)
        np.put(locsLat, np.arange(start, end), [locationsLocationsArray[i, 0]])
        np.put(locsLon, np.arange(start, end), [locationsLocationsArray[i, 1]])
        np.put(locsLat, np.arange(end, len(locsLat)), [locationsLocationsArray[-1, 0]])
        np.put(locsLon, np.arange(end, len(locsLat)), [locationsLocationsArray[-1, 1]])
    locData = np.vstack((locsLat, locsLon)).T
    if bGetAltitude:
        elevationsLookup = getAltitude(locData)
    else:
        with open('C:/Memento analysis/Data sources/Elevation lookup.json') as f:
            elevationsLookup = json.load(f)

    print('\tImporting weather')
    inputsObjMean['We-altitude'] = newRow.copy()
    inputsObjMean['We-adjusted pressure'] = newRow.copy()
    for i in range(len(locData)):
        dayHour = datetime.fromtimestamp(((i * 60) + startDateTimeInMin) * 60).astimezone(utc).strftime('%Y-%m-%d %H')
        day, hour = dayHour[:10], int(dayHour[11:13])
        file = googleSourceFolder + day + ' - ' + str(round(locData[i, 0], 1)) + ', ' + str(round(locData[i, 1], 1))\
               + '.json'
        try:
            with open(file) as f:
                weatherInfo = json.load(f)
        except Exception as e:
            weatherFails.append([file, dayHour])
        else:
            try:
                for fld in weatherDataSimple:
                    inputsObjMean['We-' + fld][i] = weatherInfo['days'][0]['hours'][hour][fld]
            # this is likely to trigger with daylight savings stupidity, so just pinch the next hour's data
            except Exception:
                dayHour = datetime.fromtimestamp((((i + 1) * 60) + startDateTimeInMin) * 60).astimezone(utc).strftime(
                    '%Y-%m-%d %H')
                day, hour = dayHour[:10], int(dayHour[11:13])
                file = googleSourceFolder + day + ' - ' + str(round(locData[i, 0], 1)) + ', ' + str(
                    round(locData[i, 1], 1)) \
                       + '.json'
                try:
                    with open(file) as f:
                        weatherInfo = json.load(f)
                    for fld in weatherDataSimple:
                        inputsObjMean['We-' + fld][i] = weatherInfo['days'][0]['hours'][hour][fld]
                except Exception as e:
                    weatherFails.append([file, dayHour])
                    # print(file)
        altKey = '{:2.3f}'.format(locData[i, 0]) + ', ' + \
                 '{:2.3f}'.format(locData[i, 1])
        # print(i, altKey)
        inputsObjMean['We-altitude'][i] = elevationsLookup[altKey]


    # fill in missing values with averages, windgust, nan == 0. A bit dubious about putting precip to 0.

    print('\tCleaning data')
    inputsObjMean['We-windgust'] = np.nan_to_num(inputsObjMean['We-windgust'])
    inputsObjMean['We-precip'] = np.nan_to_num(inputsObjMean['We-precip'])

    for fld in ["temp", "feelslike", "humidity", "dew", "windspeed", "winddir", "pressure", "visibility", "cloudcover"]:
        # print(fld)
        mask = np.isnan(inputsObjMean['We-' + fld]) | (inputsObjMean['We-' + fld] == 0)
        inputsObjMean['We-' + fld][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                                     inputsObjMean['We-' + fld][~mask])

    inputsObjMean['We-adjusted pressure'] = adjustedPressure(np.arange(len(newRow)))
    inputsObjAdd['We-adjusted pressure difference'] = np.append(np.diff(inputsObjMean['We-adjusted pressure']), 0)
    inputsObjAdd['We-pressure difference'] = np.append(np.diff(inputsObjMean['We-pressure']), 0)

    # winddir needs to be split into 360 / 32? NSEW, etc. 4 would probably work, so go for 4, 8, 16, 32. 'Cos I can.
    print('\tRebinning winddir')
    for bins in [4, 8, 16, 32]:
        binArray = np.append(np.arange(0, 360, 360 / bins), 360)
        for i in range(len(binArray) - 1):
            key = 'We-winddir ' + str(bins) + ' bins, ' + str(binArray[i+1])
            inputsObjMean[key] = (binArray[i] < inputsObjMean['We-winddir']) & \
                                 (inputsObjMean['We-winddir'] <= binArray[i+1])

    # import moon phases from json
    print('\tImporting moon phases')
    googleSourceFolder = 'C:/Memento analysis/Data sources/Moon phases/'
    moonPhaseList = []
    inputsObjMean['We-moon phases'] = newRow.copy()
    for file in sorted(os.listdir(googleSourceFolder)):
        if file.endswith('.json'):
            with open(googleSourceFolder + file) as f:
                moonPhaseList += json.load(f)

    startTime = datetime.fromtimestamp(startDateTimeInMin/ 60).astimezone(utc)
    endTime = datetime.fromtimestamp(endDateTimeInMin / 60).astimezone(utc)
    for entry in moonPhaseList:
        time = parser.parse(entry['time'] + 'C')
        if (startTime <= time) & (time <= endTime):
            index = math.floor(((datetime.timestamp(time) / 60) - startDateTimeInMin) / 60)
            inputsObjMean['We-moon phases'][index] = entry['phase']

    # rebin from 1hr intervals to standard 1min intervals. Note the Add, preserves the initial value
    print('\tCleaning up')
    for key in inputsObjMean:
       inputsObjMean[key] = rebinAdd(inputsObjMean[key], 1/60)
    for key in inputsObjAdd:
       inputsObjAdd[key] = rebinAdd(inputsObjAdd[key], 1/60)

    inputsObjMean = cleanup('I-51 Weather mean', inputsObjMean, 'mean', clean=clean, bDebug=bDebug)
    inputsObjAdd = cleanup('I-51 Weather add', inputsObjAdd, 'add', clean=clean, bDebug=bDebug)


def getAltitude(locAr):
    """From open elevation, get all altitudes."""
    print('getAltitude()')
    global locArray

    locArray = np.unique(locAr, axis=0)
    location = {"locations": [{"latitude": float(i[0]), "longitude": float(i[1])} for i in locArray]}
    json_data = json.dumps(location, skipkeys=int).encode('utf8')

    url = "https://api.open-elevation.com/api/v1/lookup"
    response = urllib.request.Request(url, json_data, headers={"Content-Type": "application/json"})
    attempts, maxAttempts = 0, 1000
    while attempts < maxAttempts:
        try:
            fp = urllib.request.urlopen(response)
            res_byte = fp.read()
            res_str = res_byte.decode("utf8")
            elevations = json.loads(res_str)['results']
            fp.close()
        except Exception as e:
            print(attempts, e)
        else:
            break

    # construct a usable object from results
    elevationsLookup = {}
    for res in elevations:
        elevationsLookup['{:2.3f}'.format(res['latitude']) + ', ' +
                         '{:2.3f}'.format(res['longitude'])] = res['elevation']
    with open('C:/Memento analysis/Data sources/Elevation lookup.json', 'w+') as f:
        f.write(json.dumps(elevationsLookup))
    return elevationsLookup


def adjustedPressure(i):
    alt = 0.0065 * inputsObjMean['We-altitude'][i]
    return inputsObjMean['We-pressure'][i] * (1 - (alt / (inputsObjMean['We-temp'][i] + alt + 273.15))) ** 5.257


def compileSimpleData(clean=True, bDebug=False):
    """Loop through all entries, create appropriate I / O rows"""
    print('compileSimpleData()')
    global outputsObjMean, outputsObjAdd, outputsFails, inputsObjMean
    outputsObjMean, outputsObjAdd, outputsFails, tmpObj, inputsObjMean = {}, {}, [], {}, {}
    simpleData = ['Overview',
                  '6x6 Schulte',
                  'Weight',
                  'New Timespan'
                  ]
    listData = ['Urination notes',
                'Motion notes'
                ]
    # initialise simpleData
    for fld in simpleData:
        outputsObjMean['Da-' + fld] = []
    outputsObjMean['Da-Urination'], outputsObjMean['Da-Motion'] = emptyRow.copy(), emptyRow.copy()
    outputsObjAdd['Da-Urination counts'], outputsObjAdd['Da-Motion counts'] = emptyRow.copy(), emptyRow.copy()
    outputsObjMean['Da-Pulse'] = []
    outputsObjAdd['En-Entry'] = emptyRow.copy()  # 1 at the time there is an entry, 0 else.

    # for files
    for file in individualDateList:
        # print(file)
        fn = sourceFolderIndividual + file
        if file.endswith('.json'):
            try:
                with open(fn) as f:
                    entry = json.load(f)
            except Exception as ex:
                print(file)
                raise ex
            else:
                timeStamp = int(
                    datetime.timestamp(parser.parse(entry['Date stamp']).astimezone(utc)) / 60) - startDateTimeInMin
                outputsObjAdd['En-Entry'][timeStamp] = 1
                # simpleData
                for fld in simpleData:
                    if entry[fld] is not None:
                        try:
                            outputsObjMean['Da-' + fld].append((timeStamp, float(entry[fld])))
                        except Exception as e:
                            outputsFails.append((timeStamp, entry[fld]))

                if len(entry['Exercise - comments']) > 0:
                    if entry['Pulse'] is not None:
                        try:
                            outputsObjMean['Da-Pulse'].append((timeStamp, float(entry['Pulse'])))
                        except Exception as e:
                            outputsFails.append((timeStamp, entry['Pulse']))

                # Elimination
                if entry['Urination'] != 'NA':
                    if entry['Urination'] == 'Unknown':
                        outputsObjMean['Da-Urination'][timeStamp] = 2
                    else:
                        outputsObjMean['Da-Urination'][timeStamp] = int(entry['Urination'][:1])
                    outputsObjAdd['Da-Urination counts'][timeStamp] = 1
                if entry['Motion'] != 'NA':
                    if entry['Motion'] == 'Unknown':
                        outputsObjMean['Da-Motion'][timeStamp] = 3
                    else:
                        outputsObjMean['Da-Motion'][timeStamp] = int(entry['Motion'][:1])
                    outputsObjAdd['Da-Motion counts'][timeStamp] = 1

                # unknown elements of lists
                for fld in listData:
                    # obj = json.loads(entry[fld])
                    obj = entry[fld]
                    for key in obj:
                        if 'Da-' + obj[key] not in outputsObjMean:
                            outputsObjMean['Da-' + obj[key]] = emptyRow.copy()
                        outputsObjMean['Da-' + obj[key]][timeStamp] = 1

                obj = entry['Procedures list']
                for key in obj:
                    if 'Pl-' + obj[key] not in inputsObjMean:
                        inputsObjMean['Pl-' + obj[key]] = emptyRow.copy()
                    inputsObjMean['Pl-' + obj[key]][timeStamp] = 1

    print('\tPopulating simple data')
    parseDateTimeIntoRow()

    # accentuate deviation from the norm
    outputsObjMean['Da-Overview adjusted **3'] = ((outputsObjMean['Da-Overview'] - 4.5) * 2) ** 3

    outputsObjMean = cleanup('O-52 Simple data mean', outputsObjMean, 'mean', clean=clean, bDebug=bDebug)
    outputsObjAdd = cleanup('O-02 Simple data add', outputsObjAdd, 'add', clean=clean, bDebug=bDebug)
    inputsObjMean = cleanup('I-53 Procedures list mean', inputsObjMean, 'mean', clean=clean, bDebug=bDebug)


def compileExerciseData(clean=True, bDebug=False):
    print('Building exercise object')
    global inputsObjAdd, outputsObjAdd, exList
    exercises = [
        'km walk',
        'glutes 1', 'glutes 2', 'press ups', 'squats',
        'burpees', 'lunges', 'sit ups', 'triceps dips'
    ]
    exerciseRegexps = []
    for ex in exercises:
        inputsObjAdd['Ex-' + ex] = emptyRow.copy()
        exerciseRegexps.append(re.compile('(\d+\.*\d*) *(' + ex + ')'))

    for file in individualDateList:
        # print(file)
        fn = sourceFolderIndividual + file
        if file.endswith('.json'):
            try:
                with open(fn) as f:
                    entry = json.load(f)
                    if len(entry['Exercise - comments']) > 0:
                        for rx in exerciseRegexps:
                            res = rx.match(entry['Exercise - comments'])
                            if res is not None:
                                timeStamp = int(
                                    datetime.timestamp(
                                        parser.parse(entry['Date stamp']).astimezone(utc)) / 60) - startDateTimeInMin
                                inputsObjAdd['Ex-' + res.group(2)][timeStamp] = float(res.group(1))
            except Exception as ex:
                print(file)
                raise ex

    outputsObjAdd = inputsObjAdd.copy()
    cleanup('I-00 Exercise add', inputsObjAdd, 'add', clean=clean, bDebug=bDebug)
    cleanup('O-00 Exercise add', outputsObjAdd, 'add', clean=clean, bDebug=bDebug)
    if clean is True:
        inputsObjAdd, outputsObjAdd = {}, {}


@jit(nopython=True)
def findNext(item, array1D, start=0):
    """return index of first occurence of item in array1D, or -1, starting from i = start"""
    # note numba doesn't support f2 / float16 dtype
    for i in range(start, len(array1D)):
        if item == array1D[i]:
            return i
    return -1


@jit(nopython=True)
def findNextInexactString(string, array1D, start=0):
    """return index of first occurence of item in array1D, or -1, starting from i = start"""
    for i in range(start, len(array1D)):
        if string in array1D[i]:
            return i
    return -1


@jit(nopython=True)
def findNextNumberGreater(num, array1D, start=0):
    """return index of first occurrence of number that is greater than num in array1D, starting at start, or -1"""
    # note numba doesn't support f2 / float16 dtype
    for i in range(start, len(array1D)):
        if array1D[i] > num:
            return i
    return -1


def getTimeStampInMinSimple(date, time):
    return math.floor(datetime.timestamp(parser.parse(date + ' ' + time).astimezone(utc)) / 60) - startDateTimeInMin


def getTimeStampInMinFromOverview(date, time):
    """for times after 00:00 and before 03:59, the date needs to be advanced by 1 day
    returns timeStamp in seconds
    """
    if time > '03:59':
        return math.floor(datetime.timestamp(parser.parse(date + ' ' + time).astimezone(utc)) / 60) - startDateTimeInMin
    else:
        return math.floor(datetime.timestamp(parser.parse(date + ' ' + time).astimezone(utc) + timedelta(days=1)) / 60)\
               - startDateTimeInMin


def getTimeStampInMinFromSleep(date, time):
    """for times after 18:00 and before 00:00, the date needs to be retreated by 1 day
    returns timeStamp in seconds
    """
    if (time >= '18:00') & (time <= '23:59'):
        return math.floor(datetime.timestamp(parser.parse(date + ' ' + time).astimezone(utc) - timedelta(days=1)) / 60)\
               - startDateTimeInMin
    else:
        return math.floor(datetime.timestamp(parser.parse(date + ' ' + time).astimezone(utc)) / 60)\
               - startDateTimeInMin


def createAllLocationsList():
    """ Loop through and rename each google file from 2020_Jan to 2020-01
        loop through each Google file
            summarize locations
            locationsLocationsArray is in form [[lat, lon], [...],...]
            locationsTimessArray is in minutes from start.
    """
    global locationsTimesArray, locationsLocationsArray, locationsDaysArray, locationsErrors, namedLocations
    endTime = endDateTimeInMin - startDateTimeInMin
    locationsList = []
    with open(JSONFolder + 'Named locations.json') as f:
        namedLocations = json.load(f)
    print('importLocationsFromGoogleToOverviews()')
    for file in os.listdir(googleSourceFolder):
        newName = renameFile(file)
        if newName is not None:
            os.rename(googleSourceFolder + file, googleSourceFolder + newName)

    for file in sorted(os.listdir(googleSourceFolder)):
        # create list of all location changes in file
        if file.endswith('.json'):
            events, errors = populateFromGoogleLocations(googleSourceFolder + file, endTime)
            locationsList += events
            locationsErrors += errors

    e = locationsList.pop(0)
    locationsTimesArray, locationsLocationsArray = [e[0]], [e[1]]

    for event in locationsList:
        if event[1] != locationsLocationsArray[-1]:
            locationsTimesArray.append(event[0])
            locationsLocationsArray.append(event[1])


    # add false entries at start and end to prevent getLocation returning an error.
    locationsTimesArray = [0] + locationsTimesArray + \
                          [endDateTimeInMin - startDateTimeInMin - 1]

    locationsLocationsArray = [locationsLocationsArray[0]] + locationsLocationsArray + [locationsLocationsArray[-1]]
    locationsTimesArray, locationsLocationsArray = np.array(locationsTimesArray, dtype='i8'), \
                                                   np.array(locationsLocationsArray, dtype='f2')


    # Chop off previous entries, as google files are in month chunks
    locationsMask = locationsTimesArray >= 0
    locationsTimesArray, locationsLocationsArray = locationsTimesArray[locationsMask], \
                                                   locationsLocationsArray[locationsMask]

    namedLocations = None


def renameFile(file):
    renameDict = {'JANUARY': '01',
                  'FEBRUARY': '02',
                  'MARCH': '03',
                  'APRIL': '04',
                  'MAY': '05',
                  'JUNE': '06',
                  'JULY': '07',
                  'AUGUST': '08',
                  'SEPTEMBER': '09',
                  'OCTOBER': '10',
                  'NOVEMBER': '11',
                  'DECEMBER': '12'
                  }
    month = getMonth.search(file)
    if month is not None:
        try:
            return month.group(1) + renameDict[month.group(2)] + '.json'
        except Exception:
            return None


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
        ts = getTimeStampInMinFromGoogle(tlo[k1]['duration']['startTimestampMs'])

        if (0 <= ts) & (ts <= endTime):
            if 'latitudeE7' in tlo[k1][k2]:
                lat = convertE7(tlo[k1][k2]['latitudeE7'])
                lon = convertE7(tlo[k1][k2]['longitudeE7'])
            else:
                try:
                    lat = namedLocations[tlo[k1][k2]['placeId']]['lat']
                    lon = namedLocations[tlo[k1][k2]['placeId']]['lon']
                except Exception as e:
                    locErrors.append(tlo)
            locSums.append([ts, [lat, lon]])

    return locSums, locErrors


def getTimeStampInMinFromGoogle(od):
    d = int(od)
    d = d / 1000 if d > 900000000 else d
    return math.floor((d / 60) - startDateTimeInMin)


def convertE7(num):
    num = int(num)
    num = num - 4294967296 if (num > 900000000) else num
    return round(num / 10000000, 3)


def sliceArray(a, start, end):
    """Slices each string in array a of strings, from start index of string to end index."""
    b = a.view((str, 1)).reshape(len(a), -1)[:,start:end]
    return np.frombuffer(b.tobytes(), dtype=(str, end-start))


def rebinMean(data, binLen, axis=0):
    slices, step = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen),
                               endpoint=False, retstep=True, dtype=np.intp)
    return (np.add.reduceat(data, slices, axis=axis) / step).astype('f8')


def rebinAdd(data, binLen, axis=0):
    slices = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen), endpoint=False, dtype=np.intp)
    return np.add.reduceat(data, slices, axis=axis).astype('f8')


def cleanup(fn, obj, cleanType, clean=False, bDebug=False):
    """Rebins to 15min, sorts and saves data. Optionally deletes it from memory"""
    global onlyZeros, subtracts
    print('Cleaning up', fn, 'deleting', clean, 'global_sparsify', global_sparsify)
    # sparsifies all data by finding the most frequent value, then subtracting it from the array
    # Due to the correlation routine cutting out any data that only partially fills a timeStep, we * 2 here
    chop = 288 * 15 * 2
    if global_sparsify:
        for key in obj:
            bins, counts = np.unique(obj[key], return_counts=True)
            subtract = bins[counts == counts.max()][0]
            if subtract != 0:
                obj[key] = obj[key] - subtract
                subtracts.append([key, subtract])

    keys = [i for i in obj.keys()]
    if cleanType == 'add':
        for key in keys:
            if np.sum(obj[key][chop:-chop] != 0) > 0:
                obj[key] = sparse.csr_matrix(rebinAdd(obj[key], 15))
            else:
                obj.pop(key)
                onlyZeros.append((fn, key))
    else:
        for key in keys:
            if np.sum(obj[key][chop:-chop] != 0) > 0:
                obj[key] = sparse.csr_matrix(rebinMean(obj[key], 15))
            else:
                obj.pop(key)
                onlyZeros.append((fn, key))
    if not bDebug:
        fp = destinationFolder + fn + '.npz'
        np.savez_compressed(fp, **obj)
        obj = {}
        objZ = np.load(fp, allow_pickle=True)
        for key in sorted(objZ.files):
            obj[key] = objZ[key]
        np.savez_compressed(fp, **obj)

    return obj if clean is False else {}


def rationalizeRemainingLocations():
    global otherLocations
    otherLocations = []
    for e in locationsErrors:
        o = {}
        if 'placeVisit' in e:
            o['lk'] = e['placeVisit']['location']['placeId']
            o['t'] = datetime.fromtimestamp((getTimeStampInMinFromGoogle(e['placeVisit']['duration']['startTimestampMs']) + startDateTimeInMin) * 60).strftime('%Y-%m-%d %H:%M')
            if 'otherCandidateLocations' in e['placeVisit']:
                o['ocl'] = {'lat': convertE7(e['placeVisit']['otherCandidateLocations'][1]['latitudeE7']),
                            'lon': convertE7(e['placeVisit']['otherCandidateLocations'][1]['longitudeE7'])}
        else:
            o['lk'] = e['activitySegment']['startLocation']['placeId']
            o['t'] = datetime.fromtimestamp((getTimeStampInMinFromGoogle(e['activitySegment']['duration']['startTimestampMs']) + startDateTimeInMin) * 60).strftime('%Y-%m-%d %H:%M')
        otherLocations.append(o)
    with open(r'C:\Memento analysis\Locations.json', 'w+') as f:
        f.write(json.dumps(otherLocations))


if __name__ == '__main__':
    tst = datetime.now()
    print(tst)
    compileAllVariables(bCompileAllIntakes=False, bCompileAllSleep=False, bCompileAllSymptoms=False,
                        bCompileAllSteps=False, bCompileAllWeather=False, bGetAltitude=False, bGetSimpleData=False,
                        bGetExerciseData=False, clean=False, bDebug=False, sparsify=False)
    print('Total run time: ', datetime.now() - tst)
