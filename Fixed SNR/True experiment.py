#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.1.1),
    on Sun Apr 19 15:26:28 2020
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import gui, visual, core, data, event, logging, clock, sound
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding
import pandas as pd
import scipy.io
import random
from TDT_function import conditions, Stimuli_Emilia, Circuit_Setup
from tdt import DSPProject

from psychopy.hardware import keyboard

project = DSPProject()
circuit = project.load_circuit('circuit_long_trigger.rcx', 'RX8')
fs_circuit=circuit.fs

#%% Set Up Path and Information

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.1.1'
expName = 'True experiment'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/emilia/Desktop/Tobias/Psychopy/True experiment.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation


#%% Setup the Window

win = visual.Window(
    size=(2000, 1500), fullscr=False, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color='#000000', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')

# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
    
defaultKeyboard = keyboard.Keyboard()

#%% Initialize Components for screen 

# Initialize components for Routine "start_screen"
start_screenClock = core.Clock()
start_screen_key_resp = keyboard.Keyboard()
start_screen_text = visual.TextStim(win=win, name='start_screen_text',
    text='Please wait for the experimenter. ',
    font='Helvetica',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "blank_screen"
blank_screenClock = core.Clock()
blank_screen_text = visual.TextStim(win=win, name='blank_screen_text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
instructions1_text = visual.TextStim(win=win, name='instructions1_text',
    text='Thank you for taking part in this experiment. \n\nIt will last around 45 minutes.\n\nYou will be given regular breaks throughout. \n\nPlease contact the experimenter during these breaks if you need anything. \n\nPress the spacebar to continue to the next screen.',
    font='Helvetica',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
instructions1_key_resp = keyboard.Keyboard()

# Initialize components for Routine "instructions2"
instructions2Clock = core.Clock()
instructions2_text = visual.TextStim(win=win, name='instructions2_text',
    text='You will now hear a sequence of syllables. For each syllable please select which one you heard from the pair shown on screen using the right and left arrows.\n\nLeft arrow = to select the syllable on the left side of the screen.\n\nRight arrow = to select the syllable on the right side of the screen.\n\nPress the spacebar to move to the next screen.',
    font='Helvetica',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
instructions2_key_resp = keyboard.Keyboard()

# Initialize components for Routine "practice_instructions"
practice_instructionsClock = core.Clock()
practice_instructions_text = visual.TextStim(win=win, name='practice_instructions_text',
    text='You will now begin a practice trial to get aquainted with the stimuli. \n\nWhen you are ready, press the spacebar to begin.',
    font='Helvetica ',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
practice_instructions_key_resp = keyboard.Keyboard()

# Initialize components for Routine "cross"
crossClock = core.Clock()
cross_polygon = visual.ShapeStim(
    win=win, name='cross_polygon', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=0.3, lineColor='#FFFFFF', lineColorSpace='rgb',
    fillColor='#FFFFFF', fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)

# Initialize components for Routine "practice_trial"
practice_trialClock = core.Clock()
practice_trial_text = visual.TextStim(win=win, name='pratice_trial_text',
    text='default text',
    font='Helvetica',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
pratctice_trial_key_resp = keyboard.Keyboard()
#practice_sound = sound.Sound('A', secs=-1, stereo=True, sampleRate = 22050, hamming=True,
#    name='practice_sound')
#practice_sound.setVolume(1)

# Initialize components for Routine "blank_screen"
blank_screenClock = core.Clock()
blank_screen_text = visual.TextStim(win=win, name='blank_screen_text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "post_practice_instructions"
post_practice_instructionsClock = core.Clock()
Post_practice_instructions_text = visual.TextStim(win=win, name='Post_practice_instructions_text',
    text='To repeat the pratice instructions, press the "up" arrow.\n\nPlease contact the experimenter if you need any assistance.\n\nYou will have 1 SECOND to respond following each stimulus. \n\nPress the "down" arrow to begin the experiment.\n"',
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
post_practice_instructions_key_resp = keyboard.Keyboard()

# Initialize components for Routine "cross"
crossClock = core.Clock()
cross_polygon = visual.ShapeStim(
    win=win, name='cross_polygon', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=0.3, lineColor='#FFFFFF', lineColorSpace='rgb',
    fillColor='#FFFFFF', fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)

# Initialize components for Routine "blank_screen"
blank_screenClock = core.Clock()
blank_screen_text = visual.TextStim(win=win, name='blank_screen_text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "trial"
trialClock = core.Clock()
trial_text = visual.TextStim(win=win, name='trial_text',
    text='default text',
    font='Helvetica',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
trial_key_resp = keyboard.Keyboard()
trial_sound = sound.Sound('A', secs=-1, stereo=True, sampleRate = 22050, hamming=True,
    name='trial_sound')
trial_sound.setVolume(1)

# Initialize components for Routine "intrial_break"
intrial_breakClock = core.Clock()
intrial_break_text = visual.TextStim(win=win, name='intrial_break_text',
    text='Well done! \n\nTake a break and press any key when you are ready to continue. ',
    font='Helvetica',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
intrial_break_key_resp = keyboard.Keyboard()

# Initialize components for Routine "middle_break"
middle_breakClock = core.Clock()
middle_break_text = visual.TextStim(win=win, name='middle_break_text',
    text='Take some time to relax before continuing with the experiment. \n\nPress the spacebar to move to the next screen. ',
    font='Helvetica',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
middle_break_key_resp = keyboard.Keyboard()

# Initialize components for Routine "end_screen"
end_screenClock = core.Clock()
end_screen_text = visual.TextStim(win=win, name='end_screen_text',
    text='Well done! The experiment is finished.\n\nThank you for taking part!',
    font='Helvetica',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

#%%  Create some handy timers

globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 



#%% Prepare to start Routine "start_screen"

continueRoutine = True

# update component  for each repeat
start_screen_key_resp.keys = []
start_screen_key_resp.rt = []
_start_screen_key_resp_allKeys = []

# keep track of which components have finished
start_screenComponents = [start_screen_key_resp, start_screen_text]
for thisComponent in start_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
        
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
start_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

#%% Run Routine "start_screen"

while continueRoutine:
    # get current time
    t = start_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=start_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *start_screen_key_resp* updates
    waitOnFlip = False
    if start_screen_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        start_screen_key_resp.frameNStart = frameN  # exact frame index
        start_screen_key_resp.tStart = t  # local t and not account for scr refresh
        start_screen_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(start_screen_key_resp, 'tStartRefresh')  # time at next scr refresh
        start_screen_key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(start_screen_key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(start_screen_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if start_screen_key_resp.status == STARTED and not waitOnFlip:
        theseKeys = start_screen_key_resp.getKeys(keyList=['s'], waitRelease=False)
        _start_screen_key_resp_allKeys.extend(theseKeys)
        if len(_start_screen_key_resp_allKeys):
            start_screen_key_resp.keys = _start_screen_key_resp_allKeys[-1].name  # just the last key pressed
            start_screen_key_resp.rt = _start_screen_key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *start_screen_text* updates
    if start_screen_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        start_screen_text.frameNStart = frameN  # exact frame index
        start_screen_text.tStart = t  # local t and not account for scr refresh
        start_screen_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(start_screen_text, 'tStartRefresh')  # time at next scr refresh
        start_screen_text.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        win.close()
        
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in start_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#%% Ending Routine "start_screen"

for thisComponent in start_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
        
# check responses
if start_screen_key_resp.keys in ['', [], None]:  # No response was made
    start_screen_key_resp.keys = None
thisExp.addData('start_screen_key_resp.keys',start_screen_key_resp.keys)
if start_screen_key_resp.keys != None:  # we had a response
    thisExp.addData('start_screen_key_resp.rt', start_screen_key_resp.rt)
thisExp.addData('start_screen_key_resp.started', start_screen_key_resp.tStartRefresh)
thisExp.addData('start_screen_key_resp.stopped', start_screen_key_resp.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('start_screen_text.started', start_screen_text.tStartRefresh)
thisExp.addData('start_screen_text.stopped', start_screen_text.tStopRefresh)
# the Routine "start_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

#%% Prepare to start Routine "blank_screen"

continueRoutine = True
routineTimer.add(0.500000)

# update component parameters for each repeat
# keep track of which components have finished
blank_screenComponents = [blank_screen_text]
for thisComponent in blank_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
        
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
blank_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

#%% Run Routine "blank_screen"

while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = blank_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=blank_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *blank_screen_text* updates
    if blank_screen_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        blank_screen_text.frameNStart = frameN  # exact frame index
        blank_screen_text.tStart = t  # local t and not account for scr refresh
        blank_screen_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(blank_screen_text, 'tStartRefresh')  # time at next scr refresh
        blank_screen_text.setAutoDraw(True)
    if blank_screen_text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > blank_screen_text.tStartRefresh + 0.5-frameTolerance:
            # keep track of stop time/frame for later
            blank_screen_text.tStop = t  # not accounting for scr refresh
            blank_screen_text.frameNStop = frameN  # exact frame index
            win.timeOnFlip(blank_screen_text, 'tStopRefresh')  # time at next scr refresh
            blank_screen_text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in blank_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#%% Ending Routine "blank_screen"

for thisComponent in blank_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('blank_screen_text.started', blank_screen_text.tStartRefresh)
thisExp.addData('blank_screen_text.stopped', blank_screen_text.tStopRefresh)

#%% Prepare to start Routine "instructions"

continueRoutine = True

# update component parameters for each repeat
instructions1_key_resp.keys = []
instructions1_key_resp.rt = []
_instructions1_key_resp_allKeys = []

# keep track of which components have finished
instructionsComponents = [instructions1_text, instructions1_key_resp]
for thisComponent in instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
        
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

#%% Run Routine "instructions"

while continueRoutine:
    # get current time
    t = instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instructions1_text* updates
    if instructions1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions1_text.frameNStart = frameN  # exact frame index
        instructions1_text.tStart = t  # local t and not account for scr refresh
        instructions1_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions1_text, 'tStartRefresh')  # time at next scr refresh
        instructions1_text.setAutoDraw(True)
    
    # *instructions1_key_resp* updates
    waitOnFlip = False
    if instructions1_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions1_key_resp.frameNStart = frameN  # exact frame index
        instructions1_key_resp.tStart = t  # local t and not account for scr refresh
        instructions1_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions1_key_resp, 'tStartRefresh')  # time at next scr refresh
        instructions1_key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instructions1_key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instructions1_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instructions1_key_resp.status == STARTED and not waitOnFlip:
        theseKeys = instructions1_key_resp.getKeys(keyList=['space'], waitRelease=False)
        _instructions1_key_resp_allKeys.extend(theseKeys)
        if len(_instructions1_key_resp_allKeys):
            instructions1_key_resp.keys = _instructions1_key_resp_allKeys[-1].name  # just the last key pressed
            instructions1_key_resp.rt = _instructions1_key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#%% Ending Routine "instructions"

for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instructions1_text.started', instructions1_text.tStartRefresh)
thisExp.addData('instructions1_text.stopped', instructions1_text.tStopRefresh)
# check responses
if instructions1_key_resp.keys in ['', [], None]:  # No response was made
    instructions1_key_resp.keys = None
thisExp.addData('instructions1_key_resp.keys',instructions1_key_resp.keys)
if instructions1_key_resp.keys != None:  # we had a response
    thisExp.addData('instructions1_key_resp.rt', instructions1_key_resp.rt)
thisExp.addData('instructions1_key_resp.started', instructions1_key_resp.tStartRefresh)
thisExp.addData('instructions1_key_resp.stopped', instructions1_key_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

#%% Prepare to start Routine "instructions2"

continueRoutine = True
# update component parameters for each repeat
instructions2_key_resp.keys = []
instructions2_key_resp.rt = []
_instructions2_key_resp_allKeys = []
# keep track of which components have finished
instructions2Components = [instructions2_text, instructions2_key_resp]
for thisComponent in instructions2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructions2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

#%% Run Routine "instructions2"

while continueRoutine:
    # get current time
    t = instructions2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructions2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instructions2_text* updates
    if instructions2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions2_text.frameNStart = frameN  # exact frame index
        instructions2_text.tStart = t  # local t and not account for scr refresh
        instructions2_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions2_text, 'tStartRefresh')  # time at next scr refresh
        instructions2_text.setAutoDraw(True)
    
    # *instructions2_key_resp* updates
    waitOnFlip = False
    if instructions2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions2_key_resp.frameNStart = frameN  # exact frame index
        instructions2_key_resp.tStart = t  # local t and not account for scr refresh
        instructions2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions2_key_resp, 'tStartRefresh')  # time at next scr refresh
        instructions2_key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instructions2_key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instructions2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instructions2_key_resp.status == STARTED and not waitOnFlip:
        theseKeys = instructions2_key_resp.getKeys(keyList=['space'], waitRelease=False)
        _instructions2_key_resp_allKeys.extend(theseKeys)
        if len(_instructions2_key_resp_allKeys):
            instructions2_key_resp.keys = _instructions2_key_resp_allKeys[-1].name  # just the last key pressed
            instructions2_key_resp.rt = _instructions2_key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructions2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#%% Ending Routine "instructions2"

for thisComponent in instructions2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instructions2_text.started', instructions2_text.tStartRefresh)
thisExp.addData('instructions2_text.stopped', instructions2_text.tStopRefresh)
# check responses
if instructions2_key_resp.keys in ['', [], None]:  # No response was made
    instructions2_key_resp.keys = None
thisExp.addData('instructions2_key_resp.keys',instructions2_key_resp.keys)
if instructions2_key_resp.keys != None:  # we had a response
    thisExp.addData('instructions2_key_resp.rt', instructions2_key_resp.rt)
thisExp.addData('instructions2_key_resp.started', instructions2_key_resp.tStartRefresh)
thisExp.addData('instructions2_key_resp.stopped', instructions2_key_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "instructions2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

'''
#%% Prepare to start Routine "practice_instructions"

continueRoutine = True
# update component parameters for each repeat
practice_instructions_key_resp.keys = []
practice_instructions_key_resp.rt = []
_practice_instructions_key_resp_allKeys = []
# keep track of which components have finished
practice_instructionsComponents = [practice_instructions_text, practice_instructions_key_resp]
for thisComponent in practice_instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
practice_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

#%% Run Routine "practice_instructions"

while continueRoutine:
    # get current time
    t = practice_instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=practice_instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *practice_instructions_text* updates
    if practice_instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        practice_instructions_text.frameNStart = frameN  # exact frame index
        practice_instructions_text.tStart = t  # local t and not account for scr refresh
        practice_instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(practice_instructions_text, 'tStartRefresh')  # time at next scr refresh
        practice_instructions_text.setAutoDraw(True)
    
    # *practice_instructions_key_resp* updates
    waitOnFlip = False
    if practice_instructions_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        practice_instructions_key_resp.frameNStart = frameN  # exact frame index
        practice_instructions_key_resp.tStart = t  # local t and not account for scr refresh
        practice_instructions_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(practice_instructions_key_resp, 'tStartRefresh')  # time at next scr refresh
        practice_instructions_key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(practice_instructions_key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(practice_instructions_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if practice_instructions_key_resp.status == STARTED and not waitOnFlip:
        theseKeys = practice_instructions_key_resp.getKeys(keyList=['space'], waitRelease=False)
        _practice_instructions_key_resp_allKeys.extend(theseKeys)
        if len(_practice_instructions_key_resp_allKeys):
            practice_instructions_key_resp.keys = _practice_instructions_key_resp_allKeys[-1].name  # just the last key pressed
            practice_instructions_key_resp.rt = _practice_instructions_key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in practice_instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#%% Ending Routine "practice_instructions"

for thisComponent in practice_instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('practice_instructions_text.started', practice_instructions_text.tStartRefresh)
thisExp.addData('practice_instructions_text.stopped', practice_instructions_text.tStopRefresh)
# check responses
if practice_instructions_key_resp.keys in ['', [], None]:  # No response was made
    practice_instructions_key_resp.keys = None
thisExp.addData('practice_instructions_key_resp.keys',practice_instructions_key_resp.keys)
if practice_instructions_key_resp.keys != None:  # we had a response
    thisExp.addData('practice_instructions_key_resp.rt', practice_instructions_key_resp.rt)
thisExp.addData('practice_instructions_key_resp.started', practice_instructions_key_resp.tStartRefresh)
thisExp.addData('practice_instructions_key_resp.stopped', practice_instructions_key_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "practice_instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
practice_outer_loop = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='practice_outer_loop')
thisExp.addLoop(practice_outer_loop)  # add the loop to the experiment
thisPractice_outer_loop = practice_outer_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_outer_loop.rgb)
if thisPractice_outer_loop != None:
    for paramName in thisPractice_outer_loop:
        exec('{} = thisPractice_outer_loop[paramName]'.format(paramName))

for thisPractice_outer_loop in practice_outer_loop:
    currentLoop = practice_outer_loop
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_outer_loop.rgb)
    if thisPractice_outer_loop != None:
        for paramName in thisPractice_outer_loop:
            exec('{} = thisPractice_outer_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "cross"-------
    
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # keep track of which components have finished
    crossComponents = [cross_polygon]
    for thisComponent in crossComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "cross"-------
    
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = crossClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=crossClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_polygon* updates
        if cross_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_polygon.frameNStart = frameN  # exact frame index
            cross_polygon.tStart = t  # local t and not account for scr refresh
            cross_polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_polygon, 'tStartRefresh')  # time at next scr refresh
            cross_polygon.setAutoDraw(True)
        if cross_polygon.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross_polygon.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                cross_polygon.tStop = t  # not accounting for scr refresh
                cross_polygon.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cross_polygon, 'tStopRefresh')  # time at next scr refresh
                cross_polygon.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in crossComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "cross"-------
    
    for thisComponent in crossComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_outer_loop.addData('cross_polygon.started', cross_polygon.tStartRefresh)
    practice_outer_loop.addData('cross_polygon.stopped', cross_polygon.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    practice_inner_loop = data.TrialHandler(nReps=24, method='random', 
                               extraInfo=expInfo, originPath=-1, trialList = [None],
                               seed=None, name='practice_inner_loop')
    thisExp.addLoop(practice_inner_loop)  # add the loop to the experiment
    thisPractice_inner_loop = practice_inner_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock2.rgb)
    if thisPractice_inner_loop != None:
        for paramName in thisPractice_inner_loop:
            exec('{} = thisPractice_inner_loop[paramName]'.format(paramName))

    Parameters_file2 = pd.read_csv('Parameters_practice.csv', header=0)
    Pair_list2 = [n for n in Parameters_file2['Pair'].values]
    syllable_list2 = [n for n in Parameters_file2['Stimulus'].values]
    corr_Ans_list2 = [n for n in Parameters_file2['corrAns'].values]
    gender_list2 = [n for n in Parameters_file2['Gender'].values]

    for thispractice_inner_loop in practice_inner_loop:
        currentLoop = practice_inner_loop
        # abbreviate parameter names if possible (e.g. rgb = thisBlock2.rgb)
        if thispractice_inner_loop != None:
            for paramName in thispractice_inner_loop:
                exec('{} = thispractice_inner_loop[paramName]'.format(paramName))
    
        # ------Prepare to start Routine "Trial"-------
        
        continueRoutine = True
        # update component parameters for each repeat
        n = range(0,len(Pair_list2))
        index = random.choice(n)
                   
        pratctice_trial_key_resp.keys = []
        pratctice_trial_key_resp.rt = []
        _pratctice_trial_key_resp_allKeys = []
        
        index = random.choice(range(0,len(Pair_list2)))
        
        Pair = Pair_list2.pop(index)
        corrAns = corr_Ans_list2.pop(index)
        practice_trial_text.setText(Pair
        )
    
        gender = gender_list2.pop(index)
    
        shuffled_syllable_i = syllable_list2.pop(index)
        filename = (r'\\icnas2.cc.ic.ac.uk\phg17\GitHub\2AFCexp\Fixed SNR\Syllables/' + str(shuffled_syllable_i) +'.mat')
        syllable = scipy.io.loadmat(shuffled_syllable_i, appendmat=False)
        syllable = syllable['CV']
        syllable = syllable.ravel()
        
        filename2 = (r'\\icnas2.cc.ic.ac.uk\phg17\GitHub\2AFCexp\Fixed SNR/SNR_list/' + str(2) +'.mat')
        noise_file2 = scipy.io.loadmat(filename2, appendmat=False)
        noise2 = noise_file2['newSNR']
        noise2 = noise2.ravel()
        Stimulus = syllable + noise2
    
        #practice_sound.setSound(Stimulus, hamming=True)
        #practice_sound.setVolume(1, log=False)
        
        # keep track of which components have finished
        practice_trialComponents = [practice_trial_text, pratctice_trial_key_resp]
        for thisComponent in practice_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        practice_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
    
        # -------Run Routine "Trial"-------
        
        while continueRoutine:
            # get current time
            t = practice_trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=practice_trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
        
            # *key_resp* updates
            waitOnFlip = False
            if pratctice_trial_key_resp.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                pratctice_trial_key_resp.frameNStart = frameN  # exact frame index
                pratctice_trial_key_resp.tStart = t  # local t and not account for scr refresh
                pratctice_trial_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pratctice_trial_key_resp, 'tStartRefresh')  # time at next scr refresh
                pratctice_trial_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pratctice_trial_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(pratctice_trial_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if pratctice_trial_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > pratctice_trial_key_resp.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    pratctice_trial_key_resp.tStop = t  # not accounting for scr refresh
                    pratctice_trial_key_resp.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(pratctice_trial_key_resp, 'tStopRefresh')  # time at next scr refresh
                    pratctice_trial_key_resp.status = FINISHED
            if pratctice_trial_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = pratctice_trial_key_resp.getKeys(keyList=['left', 'right'], waitRelease=False)
                _pratctice_trial_key_resp_allKeys.extend(theseKeys)
                if len(_pratctice_trial_key_resp_allKeys):
                    pratctice_trial_key_resp.keys = _pratctice_trial_key_resp_allKeys[-1].name  # just the last key pressed
                    pratctice_trial_key_resp.rt = _pratctice_trial_key_resp_allKeys[-1].rt
                    # was this correct?
                    if (pratctice_trial_key_resp.keys == str(corrAns)) or (pratctice_trial_key_resp.keys == corrAns):
                        pratctice_trial_key_resp.corr = 1
                    else:
                        pratctice_trial_key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
        
            # *text* updates
            if practice_trial_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_text.frameNStart = frameN  # exact frame index
                practice_trial_text.tStart = t  # local t and not account for scr refresh
                practice_trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_text, 'tStartRefresh')  # time at next scr refresh
                practice_trial_text.setAutoDraw(True)
            if practice_trial_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_trial_text.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_trial_text.tStop = t  # not accounting for scr refresh
                    practice_trial_text.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(practice_trial_text, 'tStopRefresh')  # time at next scr refresh
                    practice_trial_text.setAutoDraw(False)
                    
            # start/stop sound_1
            #if practice_sound.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            #    # keep track of start time/frame for later
            #    practice_sound.frameNStart = frameN  # exact frame index
            #    practice_sound.tStart = t  # local t and not account for scr refresh
            #    practice_sound.tStartRefresh = tThisFlipGlobal  # on global time
            #    practice_sound.play(when=win)  # sync with win flip
        
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
        
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
        
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "practice_trial"-------
        
        for thisComponent in practice_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        practice_inner_loop.addData('practice_trial_text.started', practice_trial_text.tStartRefresh)
        practice_inner_loop.addData('practice_trial_text.stopped', practice_trial_text.tStopRefresh)
        #practice_inner_loop.addData('practice_sound.started', practice_sound.tStartRefresh)
        #practice_inner_loop.addData('practice_sound.stopped', practice_sound.tStopRefresh)
        practice_inner_loop.addData('Syllable played', syllable)
        practice_inner_loop.addData('Pair presented', Pair)
        # check responses
        if pratctice_trial_key_resp.keys in ['', [], None]:  # No response was made
            pratctice_trial_key_resp.keys = None
            # was no response the correct answer?!
            if str(corrAns).lower() == 'none':
               pratctice_trial_key_resp.corr = 1;  # correct non-response
            else:
               pratctice_trial_key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for practice_inner_loop (TrialHandler)
        practice_inner_loop.addData('pratctice_trial_key_resp.keys',pratctice_trial_key_resp.keys)
        practice_inner_loop.addData('pratctice_trial_key_resp.corr', pratctice_trial_key_resp.corr)
        if pratctice_trial_key_resp.keys != None:  # we had a response
            practice_inner_loop.addData('pratctice_trial_key_resp.rt', pratctice_trial_key_resp.rt)
        practice_inner_loop.addData('pratctice_trial_key_resp.started', pratctice_trial_key_resp.tStartRefresh)
        practice_inner_loop.addData('pratctice_trial_key_resp.stopped', pratctice_trial_key_resp.tStopRefresh)
        # the Routine "practice_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "blank_screen"-------
        
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        blank_screenComponents = [blank_screen_text]
        for thisComponent in blank_screenComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        blank_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "blank_screen"-------
        
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = blank_screenClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=blank_screenClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blank_screen_text* updates
            if blank_screen_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank_screen_text.frameNStart = frameN  # exact frame index
                blank_screen_text.tStart = t  # local t and not account for scr refresh
                blank_screen_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank_screen_text, 'tStartRefresh')  # time at next scr refresh
                blank_screen_text.setAutoDraw(True)
            if blank_screen_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_screen_text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_screen_text.tStop = t  # not accounting for scr refresh
                    blank_screen_text.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(blank_screen_text, 'tStopRefresh')  # time at next scr refresh
                    blank_screen_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank_screenComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "blank_screen"-------
        
        for thisComponent in blank_screenComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        practice_inner_loop.addData('blank_screen_text.started', blank_screen_text.tStartRefresh)
        practice_inner_loop.addData('blank_screen_text.stopped', blank_screen_text.tStopRefresh)
        thisExp.nextEntry()
        
    # completed 3 repeats of 'practice_inner_loop'
    
    # ------Prepare to start Routine "post instructions"-------
    
    continueRoutine = True
    # update component parameters for each repeat
    post_practice_instructions_key_resp.keys = []
    post_practice_instructions_key_resp.rt = []
    _key_resp_5_allKeys = []
    # keep track of which components have finished
    starttrialsComponents = [Post_practice_instructions_text, post_practice_instructions_key_resp]
    for thisComponent in starttrialsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    post_practice_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "starttrials"-------
    
    while continueRoutine:
        # get current time
        t = post_practice_instructionsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=post_practice_instructionsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Post_practice_instructions_text* updates
        if Post_practice_instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Post_practice_instructions_text.frameNStart = frameN  # exact frame index
            Post_practice_instructions_text.tStart = t  # local t and not account for scr refresh
            Post_practice_instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Post_practice_instructions_text, 'tStartRefresh')  # time at next scr refresh
            Post_practice_instructions_text.setAutoDraw(True)
        if Post_practice_instructions_text.status == STARTED:  # only update if drawing
            Post_practice_instructions_text.setText('To repeat the pratice instructions, press the "up" arrow.\n\nOtherwise, please take a break before beginning the experiment and contact the experimenter if you need any assistance.\n\n For the rest of the experiment, you will have 1 SECOND to respond following each syllable. \n\nPress the "down" arrow to begin the experiment.', log=False)
        
        # *post_practice_instructions_key_resp* updates
        waitOnFlip = False
        if post_practice_instructions_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            post_practice_instructions_key_resp.frameNStart = frameN  # exact frame index
            post_practice_instructions_key_resp.tStart = t  # local t and not account for scr refresh
            post_practice_instructions_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(post_practice_instructions_key_resp, 'tStartRefresh')  # time at next scr refresh
            post_practice_instructions_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(post_practice_instructions_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(post_practice_instructions_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if post_practice_instructions_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = post_practice_instructions_key_resp.getKeys(keyList=['down', 'up'], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                post_practice_instructions_key_resp.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                post_practice_instructions_key_resp.rt = _key_resp_5_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        if post_practice_instructions_key_resp.keys == 'down':
            practice_outer_loop.finished=1
        else:
            practice_outer_loop.finished=0
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in starttrialsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine ""-------
    for thisComponent in starttrialsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_outer_loop.addData('Post_practice_instructions_text.started', Post_practice_instructions_text.tStartRefresh)
    practice_outer_loop.addData('Post_practice_instructions_text.stopped', Post_practice_instructions_text.tStopRefresh)
    # check responses
    if post_practice_instructions_key_resp.keys in ['', [], None]:  # No response was made
        post_practice_instructions_key_resp.keys = None
    practice_outer_loop.addData('post_practice_instructions_key_resp.keys',post_practice_instructions_key_resp.keys)
    if post_practice_instructions_key_resp.keys != None:  # we had a response
        practice_outer_loop.addData('post_practice_instructions_key_resp.rt', post_practice_instructions_key_resp.rt)
    practice_outer_loop.addData('post_practice_instructions_key_resp.started', post_practice_instructions_key_resp.tStartRefresh)
    practice_outer_loop.addData('post_practice_instructions_key_resp.stopped', post_practice_instructions_key_resp.tStopRefresh)
    # the Routine "starttrials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 2 repeats of 'practice_outer_loop'
'''
# set up handler to look after randomisation of conditions etc
outer_loop = data.TrialHandler(nReps=2, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='outer_loop')
thisExp.addLoop(outer_loop)  # add the loop to the experiment
thisOuter_loop = outer_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisOuter_loop.rgb)
if thisOuter_loop != None:
    for paramName in thisOuter_loop:
        exec('{} = thisOuter_loop[paramName]'.format(paramName))
        
for thisOuter_loop in outer_loop:
    currentLoop = outer_loop
    # abbreviate parameter names if possible (e.g. rgb = thisOuter_loop.rgb)
    if thisOuter_loop != None:
        for paramName in thisOuter_loop:
            exec('{} = thisOuter_loop[paramName]'.format(paramName)) #######

    # set up handler to look after randomisation of conditions etc
    trials_outer_loop = data.TrialHandler(nReps=5, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_2Hz_outer_loop')
    thisExp.addLoop(trials_outer_loop)  # add the loop to the experiment
    thisTrials_outer_loop = trials_outer_loop.trialList[0]  # so we can initialise stimuli with some values


    # abbreviate parameter names if possible (e.g. rgb = thisTrials_2Hz_outer_loop.rgb)
    if thisTrials_outer_loop != None:
        for paramName in thisTrials_outer_loop:
            exec('{} = thisTrials_outer_loop[paramName]'.format(paramName))
    
    for thisTrials_outer_loop in trials_outer_loop:
        currentLoop = trials_outer_loop
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_2Hz_outer_loop.rgb)
        if thisTrials_outer_loop != None:
            for paramName in thisTrials_outer_loop:
                exec('{} = thisTrials_outer_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "cross"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        crossComponents = [cross_polygon]
        for thisComponent in crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "cross"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = crossClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=crossClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_polygon* updates
            if cross_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_polygon.frameNStart = frameN  # exact frame index
                cross_polygon.tStart = t  # local t and not account for scr refresh
                cross_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_polygon, 'tStartRefresh')  # time at next scr refresh
                cross_polygon.setAutoDraw(True)
            if cross_polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_polygon.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_polygon.tStop = t  # not accounting for scr refresh
                    cross_polygon.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(cross_polygon, 'tStopRefresh')  # time at next scr refresh
                    cross_polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "cross"-------
        for thisComponent in crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials_outer_loop.addData('cross_polygon.started', cross_polygon.tStartRefresh)
        trials_outer_loop.addData('cross_polygon.stopped', cross_polygon.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        trials_inner_loop = data.TrialHandler(nReps=50, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='trials_inner_loop')
        thisExp.addLoop(trials_inner_loop)  # add the loop to the experiment
        thisTrials_inner_loop = trials_inner_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_2Hz_inner_loop.rgb)
        if thisTrials_inner_loop != None:
            for paramName in thisTrials_inner_loop:
                exec('{} = thisTrials_inner_loop[paramName]'.format(paramName))
        
        for thisTrials_inner_loop in trials_inner_loop:
            currentLoop = trials_inner_loop
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_2Hz_inner_loop.rgb)
            if thisTrials_inner_loop != None:
                for paramName in thisTrials_inner_loop:
                    exec('{} = thisTrials_inner_loop[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "blank_screen"-------
            continueRoutine = True
            routineTimer.add(0.500000)
            # update component parameters for each repeat
            # keep track of which components have finished
            blank_screenComponents = [blank_screen_text]
            for thisComponent in blank_screenComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            blank_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "blank_screen"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = blank_screenClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=blank_screenClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_screen_text* updates
                if blank_screen_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_screen_text.frameNStart = frameN  # exact frame index
                    blank_screen_text.tStart = t  # local t and not account for scr refresh
                    blank_screen_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_screen_text, 'tStartRefresh')  # time at next scr refresh
                    blank_screen_text.setAutoDraw(True)
                if blank_screen_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_screen_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_screen_text.tStop = t  # not accounting for scr refresh
                        blank_screen_text.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(blank_screen_text, 'tStopRefresh')  # time at next scr refresh
                        blank_screen_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_screenComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "blank_screen"-------
            for thisComponent in blank_screenComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            trials_inner_loop.addData('blank_screen_text.started', blank_screen_text.tStartRefresh)
            trials_inner_loop.addData('blank_screen_text.stopped', blank_screen_text.tStopRefresh)
            
            # ------Prepare to start Routine "trial"-------
            #### START TRIAL ####
            
            continueRoutine = True
            # update component parameters for each repeat
            trial_key_resp.keys = []
            trial_key_resp.rt = []
            _trial_key_resp_allKeys = []
            
            # SELECTING STIMULATION
            
            # Select the stimulation condition between: 2Hz, 6Hz, no stim, random stim. 
            n = 10
            T_index = np.random.choice(n)
            stim = Stimuli_Emilia(T_index,-3,circuit)
                
            trial_text.setText(stim.pair_show)
            stim.load_into_buffer()
            k=0
            
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
            
            length = stim.length+1
            Started = False
            # -------Run Routine "trial"-------
            
            while continueRoutine:
                # get current time
                t = trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trial_key_resp* updates
                waitOnFlip = False
                if trial_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_key_resp.frameNStart = frameN  # exact frame index
                    trial_key_resp.tStart = t  # local t and not account for scr refresh
                    trial_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_key_resp, 'tStartRefresh')  # time at next scr refresh
                    trial_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(trial_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(trial_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                if trial_key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_key_resp.tStartRefresh + length-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_key_resp.tStop = t  # not accounting for scr refresh
                        trial_key_resp.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(trial_key_resp, 'tStopRefresh')  # time at next scr refresh
                        trial_key_resp.status = FINISHED
                        
                if trial_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = trial_key_resp.getKeys(keyList=['left', 'right'], waitRelease=False)
                    _trial_key_resp_allKeys.extend(theseKeys)
                    if len(_trial_key_resp_allKeys):
                        trial_key_resp.keys = _trial_key_resp_allKeys[-1].name  # just the last key pressed
                        trial_key_resp.rt = _trial_key_resp_allKeys[-1].rt
                        # was this correct?
                        if (trial_key_resp.keys == str(stim.corrAns)) or (trial_key_resp.keys == stim.corrAns):
                            trial_key_resp.corr = 1
                        else:
                            trial_key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False  
                        stim.stop()
                        
                # *trial_text* updates
                if trial_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_text.frameNStart = frameN  # exact frame index
                    trial_text.tStart = t  # local t and not account for scr refresh
                    trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_text, 'tStartRefresh')  # time at next scr refresh
                    trial_text.setAutoDraw(True)
                
                if trial_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_text.tStartRefresh + length-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_text.tStop = t  # not accounting for scr refresh
                        trial_text.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(trial_text, 'tStopRefresh')  # time at next scr refresh
                        trial_text.setAutoDraw(False)
                    
                 # start/stop trial_sound
                if not Started and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_sound.frameNStart = frameN  # exact frame index
                    #trial_sound.tStart = t  # local t and not account for scr refresh
                    #trial_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    #trial_sound.play(when=win)  # sync with win flip
                    stim.start()
                    Started = True
                    
                if Started:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlip > trial_sound.frameNStart  + stim.length/stim.Fs + 0.1 and k==0:
                        # keep track of stop time/frame for later
                        #trial_sound.tStop = t  # not accounting for scr refresh
                        #trial_sound.frameNStop = frameN  # exact frame index
                        #win.timeOnFlip(trial_sound, 'tStopRefresh')  # time at next scr refresh
                        k=1
                        stim.stop()
                        
                
                
                
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "trial_innerloop"-------
            #### END TRIAL ####
            stim.stop()
            for thisComponent in trial_Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            trials_inner_loop.addData('trial_text.started', trial_text.tStartRefresh)
            trials_inner_loop.addData('trial_text.stopped', trial_text.tStopRefresh)
            trials_inner_loop.addData('Stim freq', stim.frequency)
            trials_inner_loop.addData('Cond', stim.type)
            trials_inner_loop.addData('Syllable', stim.played_syllable)
            trials_inner_loop.addData('Pair', stim.pair)
            trials_inner_loop.addData('Pair_show', stim.pair_show)
            trials_inner_loop.addData('Gender', stim.gender)
            trials_inner_loop.addData('Sound started', stim.start_time)
            trials_inner_loop.addData('Sound stopped', stim.start_time + stim.length*stim.Fs)
            # check responses
            if trial_key_resp.keys in ['', [], None]:  # No response was made
                trial_key_resp.keys = None
                # was no response the correct answer?!
                if str(stim.corrAns).lower() == 'none':
                   trial_key_resp.corr = 1;  # correct non-response
                else:
                   trial_key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_2Hz_inner_loop (TrialHandler)
            trials_inner_loop.addData('trial_key_resp.keys',trial_key_resp.keys)
            trials_inner_loop.addData('trial_key_resp.corr', trial_key_resp.corr)
            
            if trial_key_resp.keys != None:  # we had a response
                trials_inner_loop.addData('trial_key_resp.rt', trial_key_resp.rt)
                final_reaction_time = trial_key_resp.rt - stim.reaction_time
                trials_inner_loop.addData('Reaction time', final_reaction_time)

                
            trials_inner_loop.addData('trial_key_resp.started', trial_key_resp.tStartRefresh)
            trials_inner_loop.addData('trial_key_resp.stopped', trial_key_resp.tStopRefresh)
            # the Routine "trial_2Hz" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 5 repeats of 'trials_inner_loop'
        
        
        # ------Prepare to start Routine "intrial_break"-------
        continueRoutine = True
        # update component parameters for each repeat
        intrial_break_key_resp.keys = []
        intrial_break_key_resp.rt = []
        _intrial_break_key_resp_allKeys = []
        # keep track of which components have finished
        intrial_breakComponents = [intrial_break_text, intrial_break_key_resp]
        for thisComponent in intrial_breakComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        intrial_breakClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "intrial_break"-------
        while continueRoutine:
            # get current time
            t = intrial_breakClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=intrial_breakClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *intrial_break_2hz_text* updates
            if intrial_break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                intrial_break_text.frameNStart = frameN  # exact frame index
                intrial_break_text.tStart = t  # local t and not account for scr refresh
                intrial_break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(intrial_break_text, 'tStartRefresh')  # time at next scr refresh
                intrial_break_text.setAutoDraw(True)
            
            # *intrial_break_2hz_key_resp* updates
            waitOnFlip = False
            if intrial_break_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                intrial_break_key_resp.frameNStart = frameN  # exact frame index
                intrial_break_key_resp.tStart = t  # local t and not account for scr refresh
                intrial_break_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(intrial_break_key_resp, 'tStartRefresh')  # time at next scr refresh
                intrial_break_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(intrial_break_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(intrial_break_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if intrial_break_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = intrial_break_key_resp.getKeys(keyList=None, waitRelease=False)
                _intrial_break_key_resp_allKeys.extend(theseKeys)
                if len(_intrial_break_key_resp_allKeys):
                    intrial_break_key_resp.keys = _intrial_break_key_resp_allKeys[-1].name  # just the last key pressed
                    intrial_break_key_resp.rt = _intrial_break_key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in intrial_breakComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "intrial_break"-------
        for thisComponent in intrial_breakComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials_outer_loop.addData('intrial_break_text.started', intrial_break_text.tStartRefresh)
        trials_outer_loop.addData('intrial_break_text.stopped', intrial_break_text.tStopRefresh)
        # check responses
        if intrial_break_key_resp.keys in ['', [], None]:  # No response was made
            intrial_break_key_resp.keys = None
        trials_outer_loop.addData('intrial_break_key_resp.keys',intrial_break_key_resp.keys)
        if intrial_break_key_resp.keys != None:  # we had a response
            trials_outer_loop.addData('intrial_break_key_resp.rt', intrial_break_key_resp.rt)
        trials_outer_loop.addData('intrial_break_key_resp.started', intrial_break_key_resp.tStartRefresh)
        trials_outer_loop.addData('intrial_break_key_resp.stopped', intrial_break_key_resp.tStopRefresh)
        # the Routine "intrial_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 5 repeats of 'trials_outer_loop'
    
    
    # ------Prepare to start Routine "middle_break"-------
    continueRoutine = True
    # update component parameters for each repeat
    middle_break_key_resp.keys = []
    middle_break_key_resp.rt = []
    _middle_break_key_resp_allKeys = []
    # keep track of which components have finished
    middle_breakComponents = [middle_break_text, middle_break_key_resp]
    for thisComponent in middle_breakComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    middle_breakClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "middle_break"-------
    while continueRoutine:
        # get current time
        t = middle_breakClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=middle_breakClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *middle_break_text* updates
        if middle_break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            middle_break_text.frameNStart = frameN  # exact frame index
            middle_break_text.tStart = t  # local t and not account for scr refresh
            middle_break_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(middle_break_text, 'tStartRefresh')  # time at next scr refresh
            middle_break_text.setAutoDraw(True)
        
        # *middle_break_key_resp* updates
        waitOnFlip = False
        if middle_break_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            middle_break_key_resp.frameNStart = frameN  # exact frame index
            middle_break_key_resp.tStart = t  # local t and not account for scr refresh
            middle_break_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(middle_break_key_resp, 'tStartRefresh')  # time at next scr refresh
            middle_break_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(middle_break_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(middle_break_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if middle_break_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = middle_break_key_resp.getKeys(keyList=['space'], waitRelease=False)
            _middle_break_key_resp_allKeys.extend(theseKeys)
            if len(_middle_break_key_resp_allKeys):
                middle_break_key_resp.keys = _middle_break_key_resp_allKeys[-1].name  # just the last key pressed
                middle_break_key_resp.rt = _middle_break_key_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in middle_breakComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "middle_break"-------
    for thisComponent in middle_breakComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('middle_break_text.started', middle_break_text.tStartRefresh)
    thisExp.addData('middle_break_text.stopped', middle_break_text.tStopRefresh)
    # check responses
    if middle_break_key_resp.keys in ['', [], None]:  # No response was made
        middle_break_key_resp.keys = None
    thisExp.addData('middle_break_key_resp.keys',middle_break_key_resp.keys)
    if middle_break_key_resp.keys != None:  # we had a response
        thisExp.addData('middle_break_key_resp.rt', middle_break_key_resp.rt)
    thisExp.addData('middle_break_key_resp.started', middle_break_key_resp.tStartRefresh)
    thisExp.addData('middle_break_key_resp.stopped', middle_break_key_resp.tStopRefresh)
    thisExp.nextEntry()
    # the Routine "middle_break" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
# completed 2 repeats of 'Outer_loop'


# ------Prepare to start Routine "end_screen"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
end_screenComponents = [end_screen_text]
for thisComponent in end_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
end_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end_screen"-------
while continueRoutine:
    # get current time
    t = end_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=end_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *end_screen_text* updates
    if end_screen_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_screen_text.frameNStart = frameN  # exact frame index
        end_screen_text.tStart = t  # local t and not account for scr refresh
        end_screen_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_screen_text, 'tStartRefresh')  # time at next scr refresh
        end_screen_text.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end_screen"-------
for thisComponent in end_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('end_screen_text.started', end_screen_text.tStartRefresh)
thisExp.addData('end_screen_text.stopped', end_screen_text.tStopRefresh)
# the Routine "end_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
