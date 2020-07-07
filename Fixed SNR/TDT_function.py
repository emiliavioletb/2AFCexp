# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:03:10 2020

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

#%% Some useful function 


def RMS(signal):
    return np.sqrt(np.mean(np.power(signal, 2)))

def Circuit_Setup(circuit_name):
    project = DSPProject()
    circuit = project.load_circuit(circuit_name, 'RX8')
    fs_circuit=circuit.fs
    return project, circuit, fs_circuit

def AddNoisePostNorm(Target, Noise, SNR, spacing):
    l_Target = len(Target)
    rmsS = 1
    rmsN = rmsS*(10**(-SNR/20.))
    insert = 0
    Noise = Noise[insert:insert + 2 * spacing + l_Target]
    Noise = Noise * rmsN
    Target_Noise = Noise
    Target_Noise[spacing:spacing + l_Target] += Target
    Target_Noise = Target_Noise / RMS(Target_Noise)
    
    return Target_Noise


def OnOffSet(data,Fs,pause=0.4):
    """return the index at which there are sentences"""
    n=len(data)
    i=0
    thres1 = np.mean(data) - np.std(data)/5
    thres2 = np.mean(data) + np.std(data)/5

    while thres1 < data[i] < thres2:
        i += 1
    idx1 = i
    
    i= n -1
    
    while thres1 < data[i] < thres2:
        i -= 1
    idx2 = i
    
    return [idx1,idx2]

def Partial_RMS(data,l):
    if len(l) != 2:
        raise Exception('Need one start and one end')
    data_norm = data
    data_norm[l[0]:l[1]] = data[l[0]:l[1]] / RMS(data[l[0]:l[1]])
    return data_norm

#%% Setup dictionary of conditions

conditions = dict()
conditions[0] = {'type':'2Hz','frequency':2,'phase':1,'syl_list':list(range(0, 12, 1)), 'reaction time':97653} #2 Hz at phase 1
conditions[1] = {'type':'2Hz','frequency':2,'phase':2,'syl_list':list(range(12, 24, 1)),'reaction time':102535} #2 Hz at phase 2
conditions[2] = {'type':'2Hz','frequency':2,'phase':3,'syl_list':list(range(24, 36, 1)),'reaction time':107420} #2 Hz at phase 3
conditions[3] = {'type':'2Hz','frequency':2,'phase':4,'syl_list':list(range(36, 48, 1)),'reaction time':112302} #2 Hz at phase 4

conditions[4] = {'type':'6Hz','frequency':6,'phase':1,'syl_list':list(range(0, 12, 1)),'reaction time':26039} #6 Hz at phase 1
conditions[5] = {'type':'6Hz','frequency':6,'phase':2,'syl_list':list(range(12, 24, 1)),'reaction time':27668} #6 Hz at phase 2
conditions[6] = {'type':'6Hz','frequency':6,'phase':3,'syl_list':list(range(24, 36, 1)),'reaction time':29295} #6 Hz at phase 3
conditions[7] = {'type':'6Hz','frequency':6,'phase':4,'syl_list':list(range(36, 48, 1)),'reaction time':30922} #6 Hz at phase 4

conditions[8] = {'type':'nostim','frequency':0,'phase':0, 'syl_list':None, 'reaction time':97653} #audio only
conditions[9] = {'type':'random','frequency':0,'phase':0, 'syl_list':None, 'reaction time':97653} #audio only


#%% Import necessary information

trials_list = pd.read_csv('Trials.csv', header=0)
Pair_list = trials_list['Pair']
Pair_to_show = trials_list['Pair_screen']
Gender_list = trials_list['Gender']
Syllable_list = trials_list['Syllable']
answer = trials_list['corrAns']
rt2 = pd.read_csv('rt2.csv')['reaction time']
rt6 = pd.read_csv('rt6.csv')['reaction time']


#%% Define class to handle stimuli

class Stimuli_Emilia():
    def __init__(self, condition,snr, circuit):
        start = time.time()
        
        #Handle General Info
        
        
        self.circuit = circuit
        self.type = conditions[condition]['type']
        self.frequency = conditions[condition]['frequency']
        self.phase = conditions[condition]['phase']
        self.Fs = 39062.5
        self.snr = snr
        

        
        path_to_syllables = os.path.join(r'//icnas2.cc.ic.ac.uk/phg17/GitHub/2AFCexp/Fixed SNR/','Syllables')
        path_to_tactile = os.path.join(r'//icnas2.cc.ic.ac.uk/phg17/GitHub/2AFCexp/Fixed SNR/','Tactile')
        
        
        #Generate Tactile Signal
        if condition == 9:
            random_shift = random.randint(0,4000)
            Tactile = np.load(os.path.join(path_to_tactile,'pulse_' + str(self.type) + '_39062.npy'))[random.randint(0,19)][random_shift:random_shift+len(np.load(os.path.join(path_to_tactile,'pulse_2Hz_39062.npy')) )]
        else:
            Tactile = np.load(os.path.join(path_to_tactile,'pulse_' + str(self.type) + '_39062.npy')) 
        self.tactile = Tactile
        
        
        #Generate Syllables
        self.syllables = conditions[condition]['syl_list']
        if self.syllables == None:
            self.syllable_index = np.random.choice(48, replace=False)
        else:
                self.syllable_index = np.random.choice(self.syllables)
             
        if condition == 9 or condition == 8:
            self.syllable = np.load(os.path.join(path_to_syllables,'syllables_2Hz_39062_shifted_norm.npy'))[self.syllable_index]
        
        else:
            self.syllable = np.load(os.path.join(path_to_syllables,'syllables_' + str(self.type) + '_39062_shifted_norm.npy'))[self.syllable_index]
        
        self.syllable = self.syllable/max(self.syllable)*6
        
        #Add Delay
        self.delay_length = random.randint(22049, 33074) 
        delay = np.zeros(self.delay_length)
        self.tactile = np.concatenate([delay,self.tactile])
        self.syllable = np.concatenate([delay,self.syllable])
        self.length = len(self.syllable)
        

        #Add Noise
        filename = (str(2) + '.mat')
        noise = scio.loadmat('SNR_list/' + filename, appendmat=False)
        noise = noise['newSNR']
        noise = noise.ravel()
        noise = np.concatenate([noise, noise, noise, noise]) # Make the noise long so it can then be cut.
        noise = noise[:self.length] #Make the noise the same length as the syllable (which now has delay at the beginning)
        noise /= RMS(noise)
        self.noise = noise
        self.syllable_in_noise = AddNoisePostNorm(self.syllable,self.noise,snr,spacing=0) 
        
        #Level for compatibility with the setup
        self.syllable /= 500
        self.syllable_in_noise /= 500
        if condition != 8:
            self.tactile /= max(self.tactile)
        
        #Generate Trigger
        Trigger = np.ones(self.length) * (condition + 1) # the trigger dictionary is shifted by one
        Trigger[0:50] *= 0 
        Trigger[-50:] *= 0
        self.trigger = Trigger
        
        #Information Relative to Syllable Played
        self.pair = Pair_list[self.syllable_index]
        self.pair_show = Pair_to_show[self.syllable_index]
        self.gender = Gender_list[self.syllable_index]
        self.played_syllable = Syllable_list[self.syllable_index]
        self.corrAns = answer[self.syllable_index]
        
        #Reaction Time 
        self.reaction_time = (conditions[condition]['reaction time'] + self.delay_length)/self.Fs 

        print('stimuli generated in ', str(time.time()-start), ' seconds')
        self.timescale = np.arange(self.length) / self.Fs
        
    def check(self):
        if (len(self.syllable_in_noise) == len(self.tactile) == len(self.trigger) == self.length) :
            print('All Stimuli have the same length')
        else:
            raise Exception('All Stimuli do not have the same length')
        if (self.Fs == self.circuit.fs):
            print('Consistent sampling frequency: ', str(self.Fs))
        else:
            raise Exception('Inconsistent Sampling Frequency')
        
    def plot(self, *stim, new_window = True):
        start = time.time()
        if new_window:
            plt.figure()
        if len(stim) == 0:
            plt.subplot(311)
            plt.plot(self.timescale,self.syllable)
            plt.subplot(312)
            plt.plot(self.timescale,self.tactile)
            plt.subplot(313)
            plt.plot(self.timescale,self.trigger)
        else:
            for i in stim:
                if i == 'syllable':
                    plt.plot(self.timescale,self.audio)
                elif i == 'tactile':
                    plt.plot(self.timescale,self.tactile/500)
                elif i == 'trigger':
                    plt.plot(self.timescale,self.trigger/500)
                else:
                    raise Exception('Invalid Condition')
        
        
        print('Graph generated in ', str(round(time.time() - start,2)), ' seconds')
        
    def load_into_buffer(self):
        start = time.time()
        
        self.check()
        
        audio_in_buffer = self.circuit.get_buffer('audio_in', 'w')
        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')
        stimtrack_in_buffer = self.circuit.get_buffer('stimtrack_in', 'w')
        trigger_in_buffer = self.circuit.get_buffer('trigger_in', 'w')


        self.circuit.set_tag('size_audio', self.length)
        self.circuit.set_tag('size_tactile', self.length)
        self.circuit.set_tag('size_stimtrack', self.length)
        self.circuit.set_tag('size_trigger', self.length)
        audio_in_buffer.write(self.syllable_in_noise)
        tactile_in_buffer.write(self.tactile)
        stimtrack_in_buffer.write(self.syllable)
        trigger_in_buffer.write(self.trigger)
    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
    def start(self):
        self.start_time = time.time()
        self.circuit.start()
    
    def stop(self):
        
        self.circuit.stop()
        self.end_time = time.time()
        print('Stimuli was sent for: ',str(self.end_time - self.start_time),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)
        self.circuit.set_tag('stimtrack_in_i', 0)
        self.circuit.set_tag('trigger_in_i', 0)
       