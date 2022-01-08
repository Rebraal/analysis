import numpy as np
from datetime import datetime
from numba import jit
from matplotlib import pyplot as plt
from matplotlib import dates
import math


# develop code to introduce an accumulation with some form of half-life
class Decay_dev:
    """
    eg, k=0.75, 75% of previous unit remains.
    10,    0,     8,    0,    6  becomes
    10,  7.5, 11.63,  8.7, 11.03
    """   
    
    global file, k
    # file = '../Numpy entries/' + 'iom-Sy-00 30 03 Fatigue' + '.npy'
    file = '../Numpy entries/' + 'iia-tL-Water' + '.npy'
    k = 0.75

    def simple_iteration_benchmark(file, k, runs, file_cut=0):
        """"
        Runs at about 0.08s for 64224 data file, so ~3 minutes for 2k files. Probably not worth spending
        any more time on optimising.
        Runs at about 0.03s for 16056, or 64224 rebinned to hours. 1m for 2k files.
        """
        ts = datetime.now()
        for _ in range(runs):
            data = np.load(file, allow_pickle=True)
            data = data[len(data)-file_cut:] if file_cut != 0 else data
            for i in range(len(data)-2):
                data[i+1] = data[i] * k
                
        print(datetime.now() - ts)
        return data

    @jit(nopython=True, nogil=True, cache=True)
    def decayk(data, k, docopy=True):
        """
        decays an array of inputs by applying a constant decay to each element, and acuumlating the previous one

        Parameters
        ----------
        data : numeric np.array
        k : float
            decay value of quantity remaining.
        docopy : bool, optional
            If true, will output a copy of the original array, rather than overwriting.
            The default is True.

        Returns
        -------
            decayed array, if docopy == Truedecay

        """
        if docopy:
            ret = np.empty(shape=data.shape, dtype=data.dtype)
            ret[0] = data[0]
            for i in range(len(data)-1):
                ret[i+1] = (ret[i] * k) + data[i+1]
            return ret  # this actually updates the existing array... Don't really wany a copy?
        else:
            for i in range(len(data)-1):
                data[i+1] += data[i] * k            

    # ONLY GOING TO WORK ON NUMERIC DATA        
    def numba_iteration_benchmark(file, k, runs, file_cut=0):
        """"
        0.015s per.
        Runs at about 5s for 2000 x 64224 data files.
        """
                    
        ts = datetime.now()
        for _ in range(runs):
            data = np.load(file, allow_pickle=True)
            data = data[len(data)-file_cut:] if file_cut != 0 else data
            data = Decay_dev.decayk(data, k)
        
        print(datetime.now() - ts)
        # return data
        
    def testing():
        global data, dd
        data = np.load(file, allow_pickle=True)
        dd = Decay_dev.decayk(data, k)
        plt.plot(data)
        plt.plot(dd, linewidth=0.5, color='red')

class Rebin:
    
    def rebin_add(data, binLen, axis=0):
        slices = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen), endpoint=False, dtype=np.intp)
        return np.add.reduceat(data, slices, axis=axis).astype('f8')
    
    def rebin_mean(data, binLen, axis=0):
        slices, step = np.linspace(0, data.shape[axis], math.ceil(data.shape[axis] / binLen),
                                   endpoint=False, retstep=True, dtype=np.intp)
        return (np.add.reduceat(data, slices, axis=axis) / step).astype('f8')
    
# Visualisation
class Vis:
    
    global plt
    
    def __init__(self):
        self.plot_vs_dates_run = False  # Logs if function run once, ie graph set up
        self.numpy_dates = np.load('../Numpy entries/met-Dt-Numpy dates.npy', allow_pickle=True)
    
    def plot_vs_dates(self, data, numpy_dates=None, has_run=None, **kwargs):
        if numpy_dates is None:
            numpy_dates = self.numpy_dates
            
        ret = plt.plot(numpy_dates, data, **kwargs)

        if not self.plot_vs_dates_run:
            plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=1))
            # plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.plot_vs_dates_run = True
        
        if has_run is not None:
            self.plot_vs_dates_run = has_run
        
        return ret
    
class Tools:
    
    def load_data(data_name):
        return np.load('../Numpy entries/' + data_name.replace('#P# ', '') + '.npy', 
                       allow_pickle=True)
        

class Analysis1:
    """
    Purpose:
        Identify if there are any intakes which build up over time, and only affect me after building up
        
    Assumptions:
        Not much benefit in looking at data in 15m chunks - 1h chunks seems sufficient
        
    Focii:
        Dairy, nuts, chocolate, caffeine, FODMAP, sugar
        Symptoms relating to thought, fatigue and bloating
        
    Notes:
        Rebinning decayed data must be done with rebin_mean
    """
    
    def __init__(self):    
        global vis, plt
    
        
        self.inputs = [
            'iia-Al-Caffeine',
            'iia-Al-Dairy',
            'iia-Al-FODMAP',
            'iia-Al-Nuts',
            'iia-Al-Sugar',
            'iia-tC-Chocolate - dark'
            ]
        
        self.outputs = [
            'iom-Sy-00 20 01 #P# Mental clarity',
            'iom-Sy-00 20 02 #P# Processing speed',
            'iom-Sy-00 20 03 #P# Focus',
            'iom-Sy-00 20 06 Unhelpful rumination',
            'iom-Sy-00 30 01 #P# Energy',
            'iom-Sy-00 30 02 #P# Motivation',
            'iom-Sy-00 30 03 Fatigue',
            'iom-Sy-00 30 04 Sleepiness',
            'iom-Sy-00 50 01 Anxiety',
            'iom-Sy-00 50 02 Depression',
            'iom-Sy-00 60 01 Irritation',
            'iom-Sy-00 60 02 Confrontational',
            'iom-Sy-00 70 01 Testosterone',
            'iom-Sy-03 32 02 Digestion, bloating',
            'iom-Sy-03 31 01 Digestion, burping',
            'iom-Sy-03 31 02 Digestion, flatus',
            'iom-Sy-03 31 03 Digestion, flatus, foul smelling',
            'iom-Sy-03 32 03 Digestion, bowel, full'
            ]
    
        self.ip = Tools.load_data(self.inputs[0])
        self.op = Tools.load_data(self.outputs[0])
        self.vis = Vis()
        
    # basic plots
    def bp1(self):
        
        # rebin to hours
        bin_len = 4
        self.ip4 = Rebin.rebin_add(self.ip, bin_len)
        self.op4 = Rebin.rebin_mean(self.op, bin_len)
        numpy_dates = vis.numpy_dates[::bin_len]
        

a1 = Analysis1()


    
    