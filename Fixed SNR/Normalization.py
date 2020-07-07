# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:23:47 2020

@author: phg17
"""


import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
import scipy.io as scio
import scipy.io.wavfile as sio
import scipy.signal as sig
import time
import os
from scipy.signal.signaltools import hilbert, filtfilt
from scipy.signal.filter_design import butter
from time import sleep
import random
from tdt import DSPProject
import pandas as pd
from TDT_function import OnOffSet, RMS, Partial_RMS

path = os.path.join(r'//icnas2.cc.ic.ac.uk/phg17/GitHub/2AFCexp/Fixed SNR/','Syllables')

Two_Hz = np.load(os.path.join(path,'syllables_2Hz_39062_shifted.npy'))
Six_Hz = np.load(os.path.join(path,'syllables_6Hz_39062_shifted.npy'))

Two_Hz_norm = np.zeros(Two_Hz.shape)
Six_Hz_norm = np.zeros(Six_Hz.shape)

for i in range(len(Two_Hz)):
    data = Two_Hz[i]
    l = OnOffSet(data, 39062.5)
    Two_Hz_norm[i] = Partial_RMS(data, l)
    
for i in range(len(Six_Hz)):
    data = Six_Hz[i]
    l = OnOffSet(data, 39062.5)
    Six_Hz_norm[i] = Partial_RMS(data, l)