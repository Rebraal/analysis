import json as json
import urllib.request
import numpy as np


weatherImportDestination = '../../Data sources/Visualcrossing weather data/'
weather_inputs = np.load('../../Data sources/Numpy entries/met-We-Weather inputs.npy', allow_pickle=True)

weatherData, weatherListCopy = {}, []


def populateWeatherData():
    """Downloads and stores a bunch of data for the next 41 entries in the list of locations and days"""
    global weather_inputs, weatherListCopy
    weatherListCopy = weather_inputs.copy()
    # while len(weather_inputs) > 0:
    for i in range(0, 41):
        if getWeatherInfo(weather_inputs[0]):
            weather_inputs = weather_inputs[1:]
    np.save('../../Data sources/Numpy entries/met-We-Weather inputs', weather_inputs)


def getWeatherInfo(info):
    """Downloads weather data from visualcrossing.com"""
    global weatherData

    latitude, longitude = info[1], info[2]
    timeStart, timeEnd = info[0], info[0]
    print(info, latitude, longitude, timeStart, timeEnd)
    # baseFolder = 'D:/Google Drive Sync/Programs/Python/Analysis/Data/02 Clean data construction/Import weather data/'
    query = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/' +\
             str(latitude) + '%2C%20' + str(longitude) + '/' +\
             timeStart + '/' + timeEnd +\
             '?unitGroup=metric&key=Q5DSSG9L78QJ6C898QRJVX32X&' +\
             'include=obs%2Cfcst%2Cstats%2Chours%2Ccurrent'
    # should've added astronomy data in here. Maybe can be added retrospectively?

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
        with open(weatherImportDestination + timeStart + ' - ' +\
                  str(latitude) + ', ' + str(longitude) + '.json', 'w+') as f:
            f.write(json.dumps(weatherData))
        return True


def create_weather_info():
    """
    see Create numpy files. Loads from database.
    """

if __name__ == '__main__':
    populateWeatherData()
    pass