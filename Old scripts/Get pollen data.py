import requests
import json as json
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

url = "https://api.ambeedata.com/history/pollen/by-lat-lng"
source_folder = '../Source data/'
input_folder = '../../Data sources/Numpy entries/'
output_folder = '../../Data sources/Ambee pollen data/'


def get_data(input_data, key=0):
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
        else "4c6a68263625a70f72eedd67c6eb99849a61d9f697e3c2c14ee21ae42ce86bf0"

    headers = {
        # 'x-api-key': "QMHiXHhzaN3D9pScEpq2a1cGS82Xkccn2muBtC3Z",  # Tim
        # 'x-api-key': "4c6a68263625a70f72eedd67c6eb99849a61d9f697e3c2c14ee21ae42ce86bf0",  # Bob
        'x-api-key': key,
        'Content-type': "application/json"
        }
    response = requests.request("GET", url, headers=headers, params=querystring)
    # print(response.text)
    return response


def plot_pollen_data():
    """
    plot a graph of pollen data over time of day
    expand to show average of bunch of data over day
    multiple lines for each count, expand to each species
    uses variables home_data and yorkley_data, created by hand
    :return:
    """
    global home_data, pj_data
    with open(
            r'C:\Users\ashby\Documents\tims docs\Python\Data sources/'
            '52.1002083 1.3203464 2021-06-15 04c00c00 2021-06-17 04c00c00.json',
            'r') as f:
        home_data = json.loads(json.load(f))

    with open(
            r'C:\Users\ashby\Documents\tims docs\Python\Data sources/'
            '51.7514517 -2.5349726 2021-02-01 04c00c00 2021-02-03 04c00c00.json',
            'r') as f:
        yorkley_data = json.loads(json.load(f))

    pollen_types = ['grass_pollen', 'tree_pollen', 'weed_pollen']

    x_data = [a for a in range(4, 24)] + [0, 1, 2, 3]
    x_data_list = [str(a).zfill(2) for a in x_data]  # for simpler searching
    x_data = np.tile(np.array(x_data_list), (3, 1))

    data_source = home_data
    y_data = np.zeros(x_data.shape)
    y_data_count = np.zeros(x_data.shape)

    for line in data_source['data']:
        for i, type in enumerate(pollen_types):
            y_data[i, x_data_list.index(line['createdAt'][11:13])] += line['Count'][type]
            y_data_count[i, x_data_list.index(line['createdAt'][11:13])] += 1

    y_data /= y_data_count

    for i, type in enumerate(pollen_types):
        plt.plot(x_data[i], y_data[i], label='Home ' + type + ' 2021-06-15 to 17')

    data_source = yorkley_data
    y_data = np.zeros(x_data.shape)
    y_data_count = np.zeros(x_data.shape)

    for line in data_source['data']:
        for i, type in enumerate(pollen_types):
            y_data[i, x_data_list.index(line['createdAt'][11:13])] += line['Count'][type]
            y_data_count[i, x_data_list.index(line['createdAt'][11:13])] += 1

    y_data /= y_data_count

    for i, type in enumerate(pollen_types):
        plt.plot(x_data[i], y_data[i], label='Yorkley ' + type + ' 2021-02-01 to 03')

    plt.legend()
    plt.title('Pollen counts')


def create_pollen_inputs():
    """
    creates an array of [timestamps, lat, lon]
    slices at appropriate intervals
    """
    dates = np.load('../../Data sources/Numpy entries/met-Dt-Date strings.npy', allow_pickle=True)
    locs = np.load('../../Data sources/Numpy entries/met-Lo-Locations.npy', allow_pickle=True)
    dates = np.array([dates]).T
    # 16 = 4h intervals / 0.25h segments, starting at 04:00
    # might need to bodge the start dates with a [:-1] at the end, if there are a silly number of entries?
    # 12 = 16 - 4 (1h before)
    start_dates = np.array([np.append(['2019-09-01 03:00'], dates[12::16])]).T
    # need to have a time interval that is more than exactly 1h, so 23:00 to 00:15 gives 00:00 data.
    pollen_inputs = np.flip(np.hstack((start_dates, np.hstack((dates, locs))[1::16])), axis=0)
    np.save('../../Data sources/Numpy entries/met-Pi-Pollen inputs.npy', pollen_inputs)


def populate_pollen_data(num_records=100, key=0):
    """
    at 4h intervals
    Work backwards storing pollen data.
    :num_records: number of records to create from list.
    """

    # get input data
    pollen_inputs = np.load(input_folder + 'met-Pi-Pollen inputs.npy', allow_pickle=True)

    # get pollen data and store
    for r in range(num_records):
        file_name = ' '.join(pollen_inputs[0]).replace(':', 'c')
        print('starting: ', r, file_name)
        data = get_data(pollen_inputs[0], key)
        if data.status_code == 200:
            with open(output_folder + file_name + '.json', 'w+') as f:
                f.write(json.dumps(data.text))
            pollen_inputs = pollen_inputs[1:]
            # update input data
            np.save(input_folder + 'met-Pi-Pollen inputs.npy', pollen_inputs)
        else:
            print('Broken at ', file_name)
            break

    print('Completed')

def test_put(rpts):
    """
    test fastest method
    np.put(array, np.arange(start, end), val)
    array[start:end] = np.full((end-start), val)
    """

    length = 10000
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
        new_row[starts[i]:ends[i]] = np.full((ends[i] - starts[i]), val)
    print('slice time = ', datetime.now() - put_time)


# populate_pollen_data(key=0)
# populate_pollen_data(key=1)
