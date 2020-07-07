# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:07:46 2020

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
from scipy.stats import norm

Fs = 39062.5

wide = int(0.0075 * 39062.5)
file = 'Tactile/tactile_random_39062_shift.npy'
random_pulses = np.load(file)

for k in random_pulses:
    k[-1000:] = np.zeros(1000)

final = np.zeros(random_pulses.shape)
for k in range(len(random_pulses)):
    pulse = random_pulses[k]
    x = np.arange(len(pulse))
    y = np.zeros(len(pulse))
    dirac = pulse
    timing = []
    for i in range(len(pulse)):
        if pulse[i] == 1:
            timing.append(i)

    for j in timing:
        loc = j
        scale = wide
        y += norm.pdf(x,loc,scale)

   
#%% Generate Carrier Tone
   
    carry_tone = np.sin(np.arange(wide*12)/Fs*80*2*np.pi)
    a = np.argmax(carry_tone[750:])
    a+=750
    b = np.argmax(carry_tone[a:])
    b+=a

#%% Replace the Gaussians by the Carrier Tones
    win = 1000
    y2 = np.zeros(len(y))
    y2 += y
    for i in timing:
        y[i-win:i+win] = y[i-win:i+win] * carry_tone[b-win:b+win]
    final[k] = y

#%% Save
np.save('Tactile/pulse_random_39062.npy',final)