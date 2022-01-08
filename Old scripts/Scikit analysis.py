from datetime import datetime
import numpy as np
import platform
from scipy import sparse

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MaxAbsScaler
from joblib import dump, load

data_dir, data, inputs, outputs = None, None, None, None
rebins, results_cut, keys, ipk, opk = None, None, None, None, None
X, y, model, cv, scores, pipe = None, None, None, None, None, None
source_dir = r'D:\Google Drive Sync\Programs\Python\Analysis\Results\Scikit analysis\2021-03-31 1400/'
results_dir = r'D:\Google Drive Sync\Programs\Python\Analysis\Results\Scikit analysis\2021-03-31 1400/'


def main():
    """
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = LassoCV(alphas=np.arange(0, 1, 0.05), cv=cv, n_jobs=-1)
    150min
    :return:
    """

    global X, y, model, cv, scores, pipe, outputs_to_test

    load_stuff()

    tst = datetime.now()
    print('begin regression calculation:', tst)

    # need to think a bit about the intakes conglomerates - they probably need a seperate run through
    #outputs_to_test = [796]
    outputs_to_test = [
                    # these three have duplicates in the inputs, so need a little more care.
                       # 7,  # SA-Sleep
                       # 8,  # SD-Sleep, day
                       # 313,  # St-Steps
                    # these are good to go, I think
                       314,  # Sc-Folate
                       315,  # Sc-Happiness
                       316,  # Sc-Headache
                       317,  # Sc-Positives
                       318,  # Sc-Potassium
                       319,  # Sc-Thinking
                       320,  # Sc-Tired
                       323,  # Sy - 00 10 01 #P# Contentment
                       324,  # Sy - 00 10 02 #P# Tranquility
                       325,  # Sy - 00 10 03 #P# Enthusiasm
                       326,  # Sy - 00 10 04 #P# Feeling
                       331,  # Sy - 00 20 01 #P# Mental clarity
                       332,  # Sy - 00 20 02 #P# Processing speed
                       333,  # Sy - 00 20 03 #P# Focus
                       335,  # Sy - 00 20 05 Rumination
                       336,  # Sy - 00 20 06 Unhelpful rumination
                       339,  # Sy - 00 30 01 #P# Energy
                       340,  # Sy - 00 30 02 #P# Motivation
                       341,  # Sy - 00 30 03 Fatigue
                       342,  # Sy - 00 30 04 Sleepiness
                       344,  # Sy - 00 50 01 Anxiety
                       345,  # Sy - 00 50 02 Depression
                       346,  # Sy - 00 50 03 Paranoia
                       347,  # Sy - 00 60 01 Irritation
                       348,  # Sy - 00 60 02 Confrontational
                       349,  # Sy - 00 70 01 Testosterone
                       350,  # Sy - 00 70 02 Biting fingers
                       367,  # Sy - 00 90 01 #P# Vivid dreams
                       790,  # Da-Motion
                       791,  # Da-Motion counts
                       795,  # Da-Overview
                       796,  # Da-Overview adjusted **2
                       813,  # Da-Urination
                       814  # Da-Urination counts
                       ]
    alphas = np.array([0,
                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
                       2., 3., 4., 5., 10.])
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    pipe = Pipeline(steps=[
        ('maxabsscaler', MaxAbsScaler()),
        ('incrementalpca', IncrementalPCA(n_components=100))
    ])
    tol = 0.0001
    for output in outputs_to_test[12:]:
        tss = datetime.now()
        print(opk[output])
        X, y = inputs.T, outputs[output].toarray().ravel()
        model = LassoCV(pipe, alphas=alphas, cv=cv, n_jobs=-1, verbose=0)
        model.fit(X, y)
        model_title = str(output) + ' ' + opk[output]
        dump(model, results_dir + 'model ' + model_title + '.joblib')
        with open(results_dir + model_title + '.csv', 'w+') as f:
            f.write('alpha\t' + str(model.alpha_) + '\n' + '\n'.join([str(i) + '\t' + ipk[i] + '\t%.5f' % model.coef_[i] for i in range(len(model.coef_))
                    if abs(model.coef_[i]) > tol]))
        print('To run single: ', datetime.now() - tss, '\nCompleted at: ', datetime.now())

    print('To run main: ', datetime.now() - tst, '\nCompleted at: ', datetime.now())


def load_stuff():
    global data_dir, data, inputs, outputs, rebins, results_cut, keys, ipk, opk

    data_dir = source_dir if platform.system() == 'Windows' else ''

    data = np.load(data_dir + 'All data.npz', allow_pickle=True)['data']
    inputs, outputs = sparse.vstack((data[0], data[1])), sparse.vstack((data[2], data[3]))
    # rebins = np.load(data_dir + 'All data.npz', allow_pickle=True)['rebins'].tolist()
    # results_cut = data[0].shape[0] + data[1].shape[0]

    keys = np.load(data_dir + 'All data.npz', allow_pickle=True)['keys'].tolist()
    ipk, opk = keys[0] + keys[1], keys[2] + keys[3]
    keys = ipk + opk
    print('Stuff loaded')


main()