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

conditions = dict()
conditions[0] = {'type':'normal','frequency':2,'phase':1,'syl_list':list(range(0, 12, 1))} #2 Hz at phase 1
conditions[1] = {'type':'normal','frequency':2,'phase':2,'syl_list':list(range(12, 24, 1))} #2 Hz at phase 2
conditions[2] = {'type':'normal','frequency':2,'phase':3,'syl_list':list(range(24, 36, 1))} #2 Hz at phase 3
conditions[3] = {'type':'normal','frequency':2,'phase':4,'syl_list':list(range(36, 48, 1))} #2 Hz at phase 4

conditions[4] = {'type':'normal','frequency':6,'phase':1,'syl_list':list(range(0, 12, 1))} #6 Hz at phase 1
conditions[5] = {'type':'normal','frequency':6,'phase':2,'syl_list':list(range(12, 24, 1))} #6 Hz at phase 2
conditions[6] = {'type':'normal','frequency':6,'phase':3,'syl_list':list(range(24, 36, 1))} #6 Hz at phase 3
conditions[7] = {'type':'normal','frequency':6,'phase':4,'syl_list':list(range(36, 48, 1))} #6 Hz at phase 4

conditions[8] = {'type':'audio','frequency':0,'phase':0, 'syl_list':None } #audio only
conditions[9] = {'type':'random','frequency':0,'phase':0, 'syl_list':None} #audio only

class Stimuli_Emilia():
    def __init__(self, file, condition,snr, circuit):
        start = time.time()
        self.circuit = circuit
        self.type = conditions[condition]['type']
        self.frequency = conditions[condition]['frequency']
        self.phase = conditions[condition]['phase']
        self.Fs = 39062.5
        self.snr = snr
        
        path_to_syllables = os.path.join(r'//icnas2.cc.ic.ac.uk/phg17/GitHub/2AFCexp/Fixed SNR/','Syllables')
        path_to_tactile = os.path.join(r'//icnas2.cc.ic.ac.uk/phg17/GitHub/2AFCexp/Fixed SNR/','Tactile')
        
        #syllables = 
        
        
        #load noise
        filename = (str(2) + '.mat')
        noise = scio.loadmat('SNR_list/' + filename, appendmat=False)
        noise = noise['newSNR']
        noise = noise.ravel()
        noise = np.concatenate([noise, noise, noise]) # Make the noise long so it can then be cut.
        noise = noise[:length] #Make the noise the same length as the syllable (which now has delay at the beginning)
        
        
        '''

        targetAUDIO = os.path.join(path,str(file),'audio.npy')
        targetTACTILE = os.path.join(path,str(file),'tactile.npy')
        targetNOISE = os.path.join(path_to_noise,'ssn_1.npy')
        Audio = np.load(targetAUDIO)
        Tactile = np.load(targetTACTILE)
        Noise = np.load(targetNOISE)
        Audio = Audio / RMS(Audio)
        Tactile = Tactile / max(Tactile)
        Noise = Noise / RMS(Noise)
        Noise = np.asarray(list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise))[0:len(Audio)]
        Noise_Audio = AddNoisePostNorm(Audio,Noise,snr,spacing=0) 
        self.length = len(Audio)
        self.clean_audio = np.load(targetAUDIO) / 500
        self.duration = round(self.length/39062.5,2)
        
        if self.type == 'tactile': 
            self.audio = np.zeros(self.length)
            self.stimtrack = Tactile / 500
        else:
            self.audio = Noise_Audio / 500
            self.stimtrack = self.clean_audio
        
        if self.type == 'audio':
            self.tactile = np.zeros(self.length)
        elif not self.correlated:
            self.tactile = np.roll(Tactile, int(self.length/4))
        else:
            self.tactile = np.roll(Tactile, int(self.delay / 1000 * self.Fs))
        
        self.clean_tactile = Tactile
        
        Trigger = np.ones(self.length) * (condition + 1)
        Trigger[0:50] *= 0 
        Trigger[-50:] *= 0
        self.trigger = Trigger
        
        self.timescale = np.arange(self.length) / self.Fs
        
        if self.type == 'tactile':
            self.task = [ random.randint(10000,int(self.length / 3)) , random.randint(int(self.length / 3),int(self.length * 2 / 3)) , random.randint(int(self.length * 2 / 3),int(self.length-10000))]
            for i in self.task:
                self.tactile[i - 2000 : i + 2000] = np.cos(2 * np.pi * 40 * np.arange(0,4000) /39062.5)
                
        else:
            self.task = None
        
        
        self.info = dict()
        self.info['Fs'] = self.Fs
        self.info['type'] = self.type
        self.info['delay'] = self.delay
        self.info['correlated'] = self.correlated
        self.info['file'] = file
        self.info['task'] = self.task
        
        print('Stimuli generated in ', str(round(time.time() - start,2)), ' seconds')
        
        
    def check(self):
        if (len(self.audio) == len(self.tactile) == len(self.trigger) == self.length) :
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
            plt.plot(self.timescale,self.audio)
            plt.subplot(312)
            plt.plot(self.timescale,self.tactile)
            plt.subplot(313)
            plt.plot(self.timescale,self.trigger)
        else:
            for i in stim:
                if i == 'audio':
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
        audio_in_buffer.write(self.audio)
        tactile_in_buffer.write(self.tactile)
        stimtrack_in_buffer.write(self.stimtrack)
        trigger_in_buffer.write(self.trigger)
    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
    def send(self):
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            if (time.time()-t0) + 0.3 > self.length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)
        self.circuit.set_tag('stimtrack_in_i', 0)
        self.circuit.set_tag('trigger_in_i', 0)
'''

'''
            if syl_list == 'None':
                x = 48
                S_index = np.random.choice(x, replace=False)
            else:
                S_index = np.random.choice(syl_list) # Selects a random index (corresponds to a syllable) from the syl_list (phase) specified earlier.
                
            Syllable = stimuli_array[S_index,:] # Selects the syllable using this index.
            Syllable = Syllable.ravel() # Flattens syllable 
        
            # SELECTING RANDOM DELAY.
            # Add random delay between onset of noise & onset of tactile stimulation & stimuli.
            # Delay is between 1-1.5 seconds
            delay_length = random.randint(22049, 33074) 
            delay = np.zeros((delay_length, 1))
            trig = np.concatenate([delay, trig]) # Add delay to the beginning of the trigger array.
            delay = delay.ravel()
            Syllable = np.concatenate([delay, Syllable]) # Add the same delay to the beginning of the speech stimulus.
            length = len(Syllable)
            stim = np.concatenate([delay, stim]) # Add delay to the beginning of the stimulation array.
            
            # LOADING NOISE
            # Load noise file.
            filename = (str(2) + '.mat')
            noise = scipy.io.loadmat('SNR_list/' + filename, appendmat=False)
            noise = noise['newSNR']
            noise = noise.ravel()
            noise = np.concatenate([noise, noise, noise]) # Make the noise long so it can then be cut.
            noise = noise[:length] #Make the noise the same length as the syllable (which now has delay at the beginning)
                
            # Mix noise with speech stimulus.
            Stimulus = noise + Syllable # Mix noise & syllable
            
            # FINDING CONDITIONS FROM CSV FILE & ADDING EEG TRIGGERS.
            # Find conditions associated with the stimulus.
            Pair = Pair_list[S_index]
            Pair_show = Pair_to_show[S_index]
            Gender = Gender_list[S_index]
            Syllable_played = Syllable_list[S_index]
            corrAns = answer[S_index]
            
            Fs = 22050 # CHANGE IF NECESSARY
            reaction_time = (rt[S_index]) # Find when the actual sylable started so this can be subtracted from overall reaction time so that RT starts when syllable does not at the beginning of the trial.
            reaction_time = int(reaction_time[1:])
            reaction_time = (reaction_time+ len(delay))/Fs
                
            trial_text.setText(Pair_show)
            #trial_sound.setSound(Stimulus, hamming=True)
            #trial_sound.setVolume(1, log=False)
            
            # keep track of which components have finished
            trial_Components = [trial_text, trial_key_resp]
            for thisComponent in trial_Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            length = (len(Syllable)/Fs)+1




'''



'''
            if T_index in [0, 1, 2, 3]:
                tactile_freq = '2'
                frequency_array = frequency_two_array # This selects the corresponding sitmulation array.
                stimuli_array = two_array # This selects the corresponding auditory stimuli array.
                trig = two_trig # This selects the corresponding trigger array.
                rt = rt2
                if T_index == 0:
                    syl_list = list(range(0, 12, 1)) #syl list specifies indexes of the stimuli_array which correspond to the phase chosen.
                    cond = 'Phase 1 2Hz' # This is to store in the csv
                elif T_index == 1 :
                    syl_list = list(range(12, 24, 1))
                    cond = 'Phase 2 2Hz'
                elif T_index == 2:
                    syl_list = list(range(24, 36, 1))
                    cond = 'Phase 3 2Hz'
                elif T_index == 3:
                    syl_list = list(range(36, 48, 1))
                    cond = 'Phase 4 2Hz'
                
            elif T_index in [4, 5, 6, 7]:
                tactile_freq = '6'
                frequency_array = frequency_six_array
                stimuli_array = six_array
                trig = six_trig
                rt = rt6
                if T_index == 4:
                    syl_list = list(range(0, 12, 1))
                    cond = 'Phase 1 6Hz'
                elif T_index == 5 :
                    syl_list = list(range(12, 24, 1))
                    cond = 'Phase 2 6Hz'
                elif T_index == 6:
                    syl_list = list(range(24, 36, 1))
                    cond = 'Phase 3 6Hz'
                elif T_index == 7:
                    syl_list = list(range(36, 48, 1))
                    cond = 'Phase 4 6Hz'
                
            elif T_index == 8: 
                tactile_freq = 'None'
                frequency_array = np.zeros(len(six_array))
                cond = 'No stim'
                syl_list = 'None'
                stimuli_array = two_array
                trig = rand_two
                rt = rt2
           
            elif T_index == 9:
                tactile_freq = 'Random'
                cond = 'Rand'
                syl_list = 'None'
                stimuli_array = two_array
                trig = none_two
                rt = rt2
            
            #TThis is in order to select a random section of the random stimulation array.
            Ltwo = len(two_array[0])
            if tactile_freq == 'Random': 
                n = 20
                R_index = np.random.choice(n)
                #rand = random_stim_array[R_index] 
                rand =  np.zeros(len(six_array))
                for i in range(0, Ltwo):
                    frequency_array = rand[i:i+Ltwo]
                    
'''