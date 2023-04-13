# Machine_Learning_Algos

**1. Adaboost**

--> Initialize AdaBoost class, pass number of desired weak learners as the argument:
Let us say we want 30 classifiers
```
adaboost = AdaBoost(T = 30)
```

--> To train the model pass training data and labels to the ababoost_train function.

Shape: X_tr -> (no. of data points, no of features) eg. (7000, 50)
        Y_tr -> (no. of data points, 1) eg. (7000, 1)
```
model = adaboost.adaboost_train(X_tr, Y_tr)
```

--> To get new prediciton
    Run this to perform prediction on the model after training.
    This function can be used for -
    
    X_train, X_val or X_test: (no. of data points, no of features) eg. (7000, 50)
```
predictions = adaboost.adaboost_predict(X_tr)
```



--> To save trained model:

```
adaboost.save_model()
```


--> To load a saved model:

```
adaboost.load_model(model_path):
```


**2. Independent Component Analysis (ICA)**
Can be used to seperate mixed signals (audio) into seperate signals.
Load the signls and stack them row-wise.
Eg. load 4 wav files using librosa, stack. The matrix will be of size (4, sampling_rate * audio_len(time)).
    Then pass the matrix through the function.
    The output will also have 4 rows. Each row is a seperated signal (audio).
    Can be saved as audio file using librosa.
