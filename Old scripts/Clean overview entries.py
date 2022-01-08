from datetime import datetime
from datetime import timedelta
from dateutil import parser
import json as json
import os
import re
import numpy as np

arGoogleLocationSummaries, arGoogleLocationsErrors, summaries = [], [], []
cleaningErrors = []

stripSpaces = re.compile(' +')
findIntake = re.compile('(.*\D): (.*)')
splitQty = re.compile('(.)(?=, (.))?')
getMonth = re.compile('(\d\d\d\d_)(.*)\.json')

toConglomerate = []
with open(r'D:\Google Drive Sync\Programs\Python\Analysis\JSONs\Intake decode.json') as f:
         intakeSpecificDecode = json.load(f)
intakeGeneralDecode = {'l': 5, 'y': 3, 'u': 3, 's': 1, 'a': 1, 't': 0.1, 'L': 5, 'Y': 3, 'U': 3, 'S': 1, 'A': 1, 'T': 0.1}
intakeLiquidDecode = {'l': 1.5, 'y': 1, 'u': 1, 's': 0.5, 'a': 0.5, 't': 0.125, 'L': 1.5, 'Y': 1, 'U': 1, 'S': 0.5, 'A': 0.5, 'T': 0.125}
intakeSuffixes = {'prefix': " - ", 'l': "cooled", 's': "soaked", 'c': "cooked", 'r': "raw"}

sourceFolder = 'C:/Memento analysis/NowClean/'
stepCountSourceFolder = r'C:\Memento analysis\CleaningData\Stepcount data/'
importedLocationsDestination = r'C:\Memento analysis\Clean JSON overview entries/'
googleSourceFolder = r'C:\Memento analysis\CleaningData/'


datesList = []

def main(clean=False, steps=False, locations=False, weather=False):
    """Put raw overview exports into rawFolder
        Put google takeout data in appropriate folders
        Run
        Find new entries in importedLocationsDestination
        TAKE CARE THAT THE LAST FEW ENTRIES OF THE LOCATION SUMMARIES EXIST WHEN STAYING IN THE SAME PLACE FOR LONG
        PERIODS OF TIME"""
    if clean:
        cleanJSONs(rawFolder='C:/Memento analysis/ToClean',
                   destinationFolder='C:/Memento analysis/NowClean')
    if steps:
        importStepCount()
    if locations:
        importLocationsFromGoogleToOverviews(importedLocationsDestination=\
                                             r'C:\Memento analysis\Clean JSON overview entries/')



def cleanJSONs(rawFolder=None, destinationFolder=None):
    """ remove excess spaces in intakes
        parse all quantities to numeric form"""
    t = datetime.now()
    print('cleanJSONs()')
    global stripSpaces

    for file in os.listdir(rawFolder):
        # print(rawFolder + '/' + file)
        fn = rawFolder + '/' + file
        if fn.endswith('.json'):
            with open(fn) as f:
                entry = json.load(f)
            entry['Supplements - overview'] = cleanOverview(entry, 'Supplements - overview')
            entry['Liquid - overview'] = cleanOverview(entry, 'Liquid - overview')
            entry['Combined - overview'] = cleanOverview(entry, 'Combined - overview')
            entry['Supplements'] = cleanBreakdown(entry, 'Supplements')
            entry['Intake - liquid'] = cleanBreakdown(entry, 'Intake - liquid')
            entry['Intake - combined'] = cleanBreakdown(entry, 'Intake - combined')
            with open(destinationFolder + '/' + file[0:10] + '.json', 'w+') as f:
                f.write(json.dumps(entry))
    print(datetime.now() - t)


def cleanBreakdown(entry, fldName):
    global splitQty, findIntake
    date = entry['Header'][0:-4]
    fldList = entry[fldName].split('\n')
    retList = []
    for l in fldList:
        line = stripSpaces.sub(' ', l).strip()
        isIntake = findIntake.search(line)
        if isIntake is not None:
            intake = isIntake.group(1)
            qty = isIntake.group(2)
            if fldName == 'Supplements':
                # check if line does not contain oil locations
                if re.search('{', line) is None:
                    try:
                        qty = float(qty)
                    except ValueError:
                        cleaningErrors.append({'p': 1, 'd': date, 'f': fldName, 'v': str(qty), 'l': line})
            elif fldName == 'Intake - liquid':
                try:
                    qty = float(qty)
                except ValueError:
                    try:
                        qty = intakeLiquidDecode[qty]
                    except KeyError:
                        cleaningErrors.append({'p': 2, 'd': date, 'f': fldName, 'v': str(qty), 'l': line})
            elif fldName == 'Intake - combined':
                try:
                    qty = float(qty)
                except ValueError:
                # check for intake specific decode
                    splitRes = splitQty.search(qty)
                    if splitRes is not None:
                        qty, suff = splitRes.groups()
                    if suff is None:
                        suff = ''
                    qty = decodeIntake(intake, qty)
                    if qty is None:
                        cleaningErrors.append({'p': 3, 'd': date, 'f': fldName, 'v': str(qty) + ', ' + str(suff), 'l': line})
                        qty = isIntake.group(2)
                    else:
                        if suff != '':
                            try:
                                suff = intakeSuffixes[suff]
                            except KeyError:
                                cleaningErrors.append({'p': 4, 'd': date, 'f': fldName, 'v': str(qty) + ', ' + str(suff), 'l': line})
                            else:
                                intake = intake + intakeSuffixes['prefix'] + suff

            # line always needs appending, either to be checked later manually, or it's ok
            retList.append(intake + ': ' + str(qty))
        else:
            retList.append(line)

    return '\n'.join(retList)


def cleanOverview(entry, fldName):
    global stripSpaces
    retList = []
    for line in entry[fldName].split('\n'):
        retList.append(stripSpaces.sub(' ', line).strip())
    return '\n'.join(retList)


def decodeIntake(intake, qty):
    """decode to float or None"""
    # print(intake, qty)
    try:
        qty = float(qty)
    except ValueError:
        for dec in intakeSpecificDecode:
            if dec['n'] == intake:
                try:
                    qty = dec[qty]
                except KeyError:
                    return None
                else:
                    return qty
        try:
            qty = intakeGeneralDecode[qty]
        except KeyError:
            return None
        else:
            return qty
    else:
        return qty


def importStepCount():
    print('importStepCount()')
    for file in sorted(os.listdir(stepCountSourceFolder)):
        # print(file)
        fn = stepCountSourceFolder + file
        if fn.endswith('.csv'):
            targetFile = sourceFolder + file[:-3] + 'json'
            try:
                with open(targetFile) as f:
                    entry = json.load(f)
                entry['Step count'] = readSteps(fn, ',')
                with open(targetFile, 'w') as f:
                    f.write(json.dumps(entry))
            except Exception:
                pass


def readSteps(file, delimiter):
    return np.sum(np.genfromtxt(file,
                  filling_values=0, delimiter=delimiter, comments="&&&",
                  skip_header=1, dtype='float', usecols=(11)))


def importLocationsFromGoogleToOverviews(importedLocationsDestination=None):
    """ Loop through and rename each google file from 2020_Jan to 2020-01
        loop through each Google file
            loop through each day
                summarize locations
                find Overview entry
                update data
    """
    global arGoogleLocationSummaries, arGoogleLocationsErrors, summaries
    print('importLocationsFromGoogleToOverviews()')
    folder = r'D:\Memento analysis, unstored\Memento analysis\Data sources\Google location data/'
    for file in os.listdir(folder):
        newName = renameFile(file)
        if newName is not None:
            os.rename(folder + file, folder + newName)

    for file in sorted(os.listdir(folder)):
        # create list of all location changes in file
        if file.endswith('.json'):
            events, errors = createGoogleLocationSummary(folder + file)
            arGoogleLocationSummaries += events
            arGoogleLocationsErrors += errors

    # for all location changes, group into days
    print('Google data loaded')
    summaries, day = [], {},
    day['start'], day['end'] = getDayLimits(arGoogleLocationSummaries[0]['dt'])
    event0 = arGoogleLocationSummaries.pop(0)
    event0['dt'] = event0['dt'][11:]
    day['data'] = [event0]
    for event in arGoogleLocationSummaries:
        if eventInDay(event, day):
            if (round(event['la'], 3) != round(day['data'][-1]['la'])) | \
                    (round(event['lo'], 3) != round(day['data'][-1]['lo'])):
                event['dt'] = event['dt'][11:]
                day['data'].append(event)
        else:
            summaries.append(day.copy())
            day['start'], day['end'] = getDayLimits(event['dt'])
            event['dt'] = event['dt'][11:]
            day['data'] = [event]

    # summaries now contains a list of days, with a ['data'] property.
    # for each day, check if existing information, and store
    print('Summaries created')
    for file in sorted(os.listdir(sourceFolder)):
        print(file)
        fn = sourceFolder + file
        if fn.endswith('.json'):
            with open(fn) as f:
                entry = json.load(f)
            entry['Location breakdown'] = json.dumps(getLocationData(file, summaries))
            with open(importedLocationsDestination + file, 'w+') as f:
                f.write(json.dumps(entry))


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


def getLocationData(file, summaries):
    file = file[0:10]
    for i, day in enumerate(summaries):
        if day['start'][0:10] == file:
            return day['data']
        elif (summaries[i-1]['start'][0:10] < file) & (file < day['start'][0:10]):
            start = summaries[i-1]['data'][-1]
            start['dt'] = file + ' 04:00'
            end = start.copy()
            end['dt'] = file + ' 03:59'
            return [start, end]
    return ''


def createGoogleLocationSummary(file):
    locSums, locErrors = [], []
    print('createGoogleLocationSummary:', file)
    with open(file) as f:
        timelineObjects = json.load(f)['timelineObjects']
    for tlo in timelineObjects:
        try:
            locSums.append(
                {'dt': getDate(tlo['activitySegment']['duration']['startTimestampMs']),
                 'la': convertE7(tlo['activitySegment']['startLocation']['latitudeE7']),
                 'lo': convertE7(tlo['activitySegment']['startLocation']['longitudeE7'])
                 })
        except KeyError:
            try:
                locSums.append(
                    {'dt': getDate(tlo['placeVisit']['duration']['startTimestampMs']),
                     'la': convertE7(tlo['placeVisit']['location']['latitudeE7']),
                     'lo': convertE7(tlo['placeVisit']['location']['longitudeE7'])
                     })
            except KeyError:
                locErrors.append(tlo)
    return locSums, locErrors


def getDate(od):
    global datesList
    datesList.append(od)
    d = int(od)
    # d = d - 4294967296 if d > 900000000 else d
    d = d / 1000 if d > 900000000 else d
    return datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M')


def convertE7(num):
    num = int(num)
    num = num - 4294967296 if (num > 900000000) else num
    return num / 10000000


def getDayLimits(ts):
    s = ts[0:10] + ' 04:00'
    e = (parser.parse(s) + timedelta(hours=23, minutes=59)).strftime('%Y-%m-%d %H:%M')
    return s, e


def eventInDay(event, day):
    return (day['start'] <= event['dt']) & (event['dt'] <= day['end'])


def createWeatherDataRequestFile():
    """ Loop through all overview entries
            create list of all location changes with associated timestamps
            simplify list to lat / lon rounded to 1dp
            conglomerate to days
            write to file
    """
    global sourceFolder, arDayLocations, entry
    arDayLocations = []
    # create list of days with locations
    for file in sorted(os.listdir(importedLocationsDestination)):
        # create list of all location changes in file
        if file.endswith('.json'):
            fn = importedLocationsDestination + file
            print(file)
            with open(fn) as f:
                entry = json.load(f)
            arDayLocations += parseLocations(entry)
    arDayLocations.reverse()
    with open('C:\Memento analysis\Data sources/' \
              '2021-01-01 to 2021-05-16 Day locations.json',
              'w+') as f:
        f.write(json.dumps(arDayLocations))


def parseLocations(entry):
    """return a day object with an array of locations"""

    arLocations = []
    day = entry['Day start'][0:10]
    locations = json.loads(entry['Location breakdown'])
    locations = [[round(o['la'], 1), round(o['lo'], 1)] for o in locations]

    # strip out unique locations and store
    while len(locations) > 0:
        loc = locations.pop(0)
        if loc not in locations:
            arLocations.append([day, loc])
    return arLocations


if __name__ == '__main__':
    ts = datetime.now()
    main(clean=False, steps=False, locations=True, weather=False)
    createWeatherDataRequestFile()
    print(datetime.now() - ts)