# 2AFCexp

STAIRCASE 

62 - originPath='/Users/emilia/Desktop/Staircase/staircase_trial.py'

1644 - filename = os.path.join('SNR_list', str(level) +'.mat') #This is the relative path as all the files & experiment are stored in the same folder

2580 - filename2 = ('/Users/emilia/Desktop/Staircase/SNR_list/' + str(level_next) +'.mat')

Both 'parameters.csv' & 'parameters.xlsx' use relative path Syllables/Ka.mat.

Line 240 & 336 = 'Fs = 22050'



FIXED SNR STUDY 

'Parameters practice' has relative path for sound files in the practice loop (/Syllables/Ka.mat)

56 - originPath='/Users/emilia/Desktop/Tobias/Psychopy/True experiment.py'

839 - filename = ('/Users/emilia/Desktop/FixedSNR/Syllables/' + str(shuffled_syllable_i) +'.mat') # This is entire path but can be changed to relative path

844 - filename2 = ('/Users/emilia/Desktop/FixedSNR/SNR_list/' + str(2) +'.mat')

1504 - noise = scipy.io.loadmat('SNR_list/' + filename, appendmat=False) # Relative path 

Sending the triggers: variable = 'trig' (line 1448)
Sending the stimulation: variable = 'stim' (line 1460)

Line 1504: Fs = 22050 

TO CHANGE THE SNR OF THE ENTIRE FIXED SNR STUDY: line 1486 (for trials) & line 844 (for practice trials) - SNR IS CURRENTLY SET TO +2 dB
