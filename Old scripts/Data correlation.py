import math
import numpy as np
from scipy import sparse
from datetime import datetime
import os

# 2021-03-05 18:45 ~210s per offset
# 2021-03-06 15:39 ~220s per offset
#   moving variables and functions inline, where possible
# 2021-03-08 13:40 ~ 170s per offset
#   removed conjugate - M.conjugate == M where M is real.
# 2021-03-11 15:33 ~ 50s per offset
#   found error where data wasn't being sparsified.
# 2021-03-12 00:15 ~85s per offset
#   Added in results filtering for duplicates, eg sN - [0, 0] == sn -[0, 0], in input and output
#   Added in results csrs. Might as well do it all now as have to re-read and re-write stuff.
def main(bGetAllData=True, bSkipZero=False, startOffset=1, offsetStep=1, pc = 0.1):
    # global correlations, usefulAnswers
    global allData, allRebins, allKeys, testPre, testPost, allDataBase

    # unused in this setup
    # dataSourceFolder = 'C:/Memento analysis/Compiled data/'
    dataSourceFolder = r'D:\Memento analysis, unstored\Memento analysis\Compiled data/'

    dataDestinationFolder = r'D:\Google Drive Sync\Programs\Python\Analysis\Source data/'
    # dataDestinationFolder = 'C:\Memento analysis\Correlated data/'
    # dataDestinationFolder = './'
    # initial data binned at 0.25h
    timeSteps = ['0.25', '0.50', '1', '2', '4', '8', '12', '18', '24', '30', '36', '48', '60', '72']
    rebinSteps = [1, 2, 4, 8, 16, 32, 48, 72, 96, 120, 144, 192, 240, 288]
    correlations, usefulAnswers = {}, {}
    allData = [np.array([]) for _ in range(4)]
    allRebins = []
    maxOffset = int(72 / 0.25)
    lenAllTimeSteps = len(timeSteps)
    returnList = [None for _ in range(lenAllTimeSteps)]
    # allKeys = {'inputKeys': [], 'outputKeys': []}
    allKeys = [[], [], [], []]
    coArray = []
    allRs = []

    tst = datetime.now()
    print('main()', tst)
    # create output object


    # for e in timeSteps:
    #     correlations[e] = []
    # correlations['min'] = {}
    # correlations['max'] = {}
    # correlations['csr_raw'] = {}
    # correlations['csr_norm'] = {}
    # correlations['pc'] = pc

    # load all inputs and outputs into memory (stored as csrs)
    if bGetAllData:
        allData, allKeys = getAllData(dataSourceFolder, allData, allKeys)

        allRebins.append([])
        colLen = allData[0].shape[1]
        data = np.ones(colLen)
        rows = np.arange(colLen, dtype='int')
        for rbs in range(1, lenAllTimeSteps):
            # rowLen = math.floor(colLen / rebinSteps[rbs])
            cols = np.floor(np.arange(colLen) / rebinSteps[rbs]).astype('int')
            rowLen = np.nanmax(cols) + 1
            allRebins.append(sparse.csr_matrix((data, (rows, cols)), shape=(colLen, rowLen), dtype='f8'))

        np.savez_compressed(dataDestinationFolder + 'All data.npz',
                            **{'data': np.array(allData, dtype='object'),
                               'rebins': np.array(allRebins, dtype='object'),
                               'keys': np.array(allKeys, dtype='object')
                               })

    else:
        allData = np.load(dataDestinationFolder + 'All data.npz', allow_pickle=True)['data'].tolist()
        allKeys = np.load(dataDestinationFolder + 'All data.npz', allow_pickle=True)['keys'].tolist()
        allRebins = np.load(dataDestinationFolder + 'All data.npz', allow_pickle=True)['rebins'].tolist()
        ipk, opk = allKeys[0] + allKeys[1], allKeys[2] + allKeys[3]

    # set up output object
    dt = np.dtype([('ts', np.unicode_, 4),
                   # ('raw', np.float16, (len(ipk), len(opk))),
                   ('raw', np.object_),
                   ('min', np.float16),
                   ('max', np.float16),
                   ('pc', np.float16),
                   ('csr_raw', np.object_),
                   ('csr_norm', np.object_)
                   ])
    correlations = np.array([(ts, None, 0, 0, pc, None, None) for ts in timeSteps], dtype=dt)

    # for use further down the line
    pc = 1 - pc

    print('Got data', datetime.now())

    # created here for repeated use elsewhere
    resultsCut = allData[0].shape[0] + allData[1].shape[0]
    allDataBase = allData.copy()
    # duplicate_list = zero_duplicate_data()

    # initial offset separated out to avoid an if statement in loop. [:-0] doesn't work, sadly.
    print('offset: 0', datetime.now())

    a = sparse.vstack((allData[0], allData[1], allData[2], allData[3]))
    n = allData[0].shape[1]
    rs = a.sum(1)

    if not bSkipZero:
        c = (a.dot(a.T) - (rs.dot(rs.T) / n)) / (n - 1)
        d = np.diag(c)
        ts = 0
        # correlations[ts] = ((c / np.sqrt(np.outer(d, d)))[:resultsCut, resultsCut:]).astype('f2')
        correlations[ts]['raw'] = ((c / np.sqrt(np.outer(d, d)))[:resultsCut, resultsCut:]).astype('f2')

        # post processing
        # sets entries of sym1 vs sym1, sym1 vs sym2 to 0 for one half of the triangle
        # correlations[ts]['raw'][duplicate_list] = 0

        correlations[ts]['raw'] = np.where(correlations[ts]['raw'] == 1., 0, correlations[ts]['raw'])
        correlations[ts]['min'] = np.nanmin(correlations[ts]['raw'])
        correlations[ts]['max'] = np.nanmax(correlations[ts]['raw'])

        # creates csr matrices for possibly significant data. Because I need to keep 0 at the centre, I have to
        # normalize positive and negatives separately, as min & max are not symmetric about 0
        pc_min, pc_max = correlations[ts]['min'] * pc, correlations[ts]['max'] * pc
        o_data_pos = sparse.csr_matrix(np.where(correlations[ts]['raw'] > pc_max, correlations[ts]['raw'], 0))
        o_data_neg = sparse.csr_matrix(np.where(correlations[ts]['raw'] < pc_min, correlations[ts]['raw'], 0))
        correlations[ts]['csr_raw'] = o_data_pos + o_data_neg
        correlations[ts]['csr_norm'] = (o_data_pos / correlations[ts]['max']) + (o_data_neg / -correlations[ts]['min'])



    # 13 loops
    for i in range(1, lenAllTimeSteps):
        a = sparse.vstack((allData[0] * allRebins[i],
                           (allData[1] * allRebins[i]) / rebinSteps[i],
                           allData[2] * allRebins[i],
                           (allData[3] * allRebins[i]) / rebinSteps[i]
                           ))
        rs = a.sum(1)

        if not bSkipZero:
            c = (a.dot(a.T) - (rs.dot(rs.T) / n)) / (n - 1)
            d = np.diag(c)
            # ts = timeSteps[i]
            ts = i
            correlations[ts]['raw'] = ((c / np.sqrt(np.outer(d, d)))[:resultsCut, resultsCut:]).astype('f2')
            # correlations[ts]['raw'][duplicate_list] = 0

            correlations[ts]['raw'] = np.where(correlations[ts]['raw'] == 1., 0, correlations[ts]['raw'])
            correlations[ts]['min'] = np.nanmin(correlations[ts]['raw'])
            correlations[ts]['max'] = np.nanmax(correlations[ts]['raw'])

            pc_min, pc_max = correlations[ts]['min'] * pc, correlations[ts]['max'] * pc
            o_data_pos = sparse.csr_matrix(np.where(correlations[ts]['raw'] > pc_max, correlations[ts]['raw'], 0))
            o_data_neg = sparse.csr_matrix(np.where(correlations[ts]['raw'] < pc_min, correlations[ts]['raw'], 0))
            correlations[ts]['csr_raw'] = o_data_pos + o_data_neg
            correlations[ts]['csr_norm'] = (o_data_pos / -correlations[ts]['max']) + (
                        o_data_neg / correlations[ts]['min'])


    if not bSkipZero:
        np.savez_compressed(dataDestinationFolder + 'Offset - 000.npz', correlations)

    for offset in range(startOffset, maxOffset, offsetStep):
        # offset all data appropriately
        print('offset:', offset, datetime.now())
        # t = datetime.now()
        negRebinCuts = [-math.ceil(offset / rebinSteps[i]) for i in range(0, lenAllTimeSteps)]
        negOffset = -offset

        allData[0] = allDataBase[0][:, :-offset]
        allData[1] = allDataBase[1][:, :-offset]
        allData[2] = allDataBase[2][:, offset:]
        allData[3] = allDataBase[3][:, offset:]

        # first timeStep doesn't need rebinning.
        a = sparse.vstack((allData[0], allData[1], allData[2], allData[3]))
        n = allData[0].shape[1]
        rs = a.sum(1)
        c = (a.dot(a.T) - (rs.dot(rs.T) / n)) / (n - 1)
        d = np.diag(c)
        # ts = timeSteps[0]
        ts = 0
        correlations[ts]['raw'] = ((c / np.sqrt(np.outer(d, d)))[:resultsCut, resultsCut:]).astype('f2')
        # correlations[ts]['raw'][duplicate_list] = 0

        correlations[ts]['raw'] = np.where(correlations[ts]['raw'] == 1., 0, correlations[ts]['raw'])
        correlations[ts]['min'] = np.nanmin(correlations[ts]['raw'])
        correlations[ts]['max'] = np.nanmax(correlations[ts]['raw'])

        pc_min, pc_max = correlations[ts]['min'] * pc, correlations[ts]['max'] * pc
        o_data_pos = sparse.csr_matrix(np.where(correlations[ts]['raw'] > pc_max, correlations[ts]['raw'], 0))
        o_data_neg = sparse.csr_matrix(np.where(correlations[ts]['raw'] < pc_min, correlations[ts]['raw'], 0))
        correlations[ts]['csr_raw'] = o_data_pos + o_data_neg
        correlations[ts]['csr_norm'] = (o_data_pos / correlations[ts]['max']) + (o_data_neg / -correlations[ts]['min'])

        # print(0, datetime.now() - t)

        # 13 loops
        for i in range(1, lenAllTimeSteps):
            # t = datetime.now()
            a = sparse.vstack((allData[0] * allRebins[i][:negOffset, :negRebinCuts[i]],
                               (allData[1] * allRebins[i][:negOffset, :negRebinCuts[i]]) / rebinSteps[i],
                               allData[2] * allRebins[i][:negOffset, :negRebinCuts[i]],
                               (allData[3] * allRebins[i][:negOffset, :negRebinCuts[i]]) / rebinSteps[i]
                               ))
            rs = a.sum(1)
            c = (a.dot(a.T) - (rs.dot(rs.T) / n)) / (n - 1)
            d = np.diag(c)
            # ts = timeSteps[i]
            ts = i
            correlations[ts]['raw'] = ((c / np.sqrt(np.outer(d, d)))[:resultsCut, resultsCut:]).astype('f2')
            # correlations[ts]['raw'][duplicate_list] = 0

            correlations[ts]['raw'] = np.where(correlations[ts]['raw'] == 1., 0, correlations[ts]['raw'])
            correlations[ts]['min'] = np.nanmin(correlations[ts]['raw'])
            correlations[ts]['max'] = np.nanmax(correlations[ts]['raw'])

            pc_min, pc_max = correlations[ts]['min'] * pc, correlations[ts]['max'] * pc
            o_data_pos = sparse.csr_matrix(np.where(correlations[ts]['raw'] > pc_max, correlations[ts]['raw'], 0))
            o_data_neg = sparse.csr_matrix(np.where(correlations[ts]['raw'] < pc_min, correlations[ts]['raw'], 0))
            correlations[ts]['csr_raw'] = o_data_pos + o_data_neg
            correlations[ts]['csr_norm'] = (o_data_pos / correlations[ts]['max']) + (
                        o_data_neg / -correlations[ts]['min'])

        # print('\tStart save', datetime.now())
        np.savez_compressed(dataDestinationFolder + 'Offset - ' + str(offset).zfill(3) + '.npz', correlations)

    print(datetime.now())
    print('Total run time: ', datetime.now() - tst)


def getAllData(dataSourceFolder, allData, allKeys):
    # global
    print('getAllData()')
    for file in sorted(os.listdir(dataSourceFolder)):
        if file.endswith('.npz'):
            print(file)
            if 'I' in file:
                if 'add' in file:
                    allData[0], keys = getData(allData[0], np.load(dataSourceFolder + file, allow_pickle=True))
                    allKeys[0] += keys
                else:
                    allData[1], keys = getData(allData[1], np.load(dataSourceFolder + file, allow_pickle=True))
                    allKeys[1] += keys
                # print(len(keys))
                # allKeys['inputKeys'] += keys
            else:
                if 'add' in file:
                    allData[2], keys = getData(allData[2], np.load(dataSourceFolder + file, allow_pickle=True))
                    allKeys[2] += keys
                else:
                    allData[3], keys = getData(allData[3], np.load(dataSourceFolder + file, allow_pickle=True))
                    allKeys[3] += keys
                # print(len(keys))
                # allKeys['outputKeys'] += keys
    print([len(key) for key in keys])
    return allData, allKeys


def getData(data, fileData):
    start, keys = 0, []
    print(len(fileData.files))
    if data.shape[0] == 0:
        data = fileData[fileData.files[0]][()]
        keys.append(fileData.files[0])
        start = 1
    for i in range(start, len(fileData.files)):
        data = sparse.vstack((data, fileData[fileData.files[i]][()]), format='csr')
        keys.append(fileData.files[i])
    print('len(fileData.files)', len(fileData.files), 'len(keys)', len(keys))
    return data, keys


def zero_duplicate_data():
    global duplicate_list
    duplicate_list = []
    set_zeros1 = np.array([[i, j] for j in range(len(allKeys[2])) for i in range(len(allKeys[0]))
                           if allKeys[0][i].lower() == allKeys[2][j].lower()])
    add_len_x, add_len_y = len(allKeys[0]), len(allKeys[2])
    set_zeros2 = np.array(
        [[i + add_len_x, j + add_len_y] for j in range(len(allKeys[3])) for i in range(len(allKeys[1]))
         if allKeys[1][i].lower() == allKeys[3][j].lower()])
    # assumes that all the pairs are contigious, ie 0, 1, 2, 3... with no gaps.
    # This should be the way the keys are set up

    duplicate_list = np.array([[el[0], col]
                              for mask in [set_zeros1, set_zeros2]
                              for el in mask
                              for col in range(el[1], mask[-1, 1])])

    return tuple([tuple(duplicate_list[:, 0]), tuple(duplicate_list[:, 1])])


if __name__ == '__main__':
    main(bGetAllData=True, bSkipZero=True, startOffset=288, offsetStep=1, pc=0.1)
    print('main evaluated')
