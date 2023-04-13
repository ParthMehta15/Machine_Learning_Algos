# Machine_Learning_Algos

**1. Adaboost**



**2. Independent Component Analysis (ICA)**
Can be used to seperate mixed signals (audio) into seperate signals.
Load the signls and stack them row-wise.
Eg. load 4 wav files using librosa, stack. The matrix will be of size (4, sampling_rate * audio_len(time)).
    Then pass the matrix through the function.
    The output will also have 4 rows. Each row is a seperated signal (audio).
    Can be saved as audio file using librosa.
