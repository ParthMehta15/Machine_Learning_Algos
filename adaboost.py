import numpy as np
from numba import jit


'''
This is a binary AdaBoost classifier. CLASSES +1, -1


The numba jit helps greatly speeds up the training process.
In case it throws errors due to software/hardware compatibality issues,
** Comment out the decoraters before find_threshold and adaboost_train functions **


'''


class AdaBoost:
    def __init__(self, T):
      
        '''
        T is the no. of desired weak classifiers
        '''
        self.T = T 

    @jit(parallel=True)
    def find_threshold(self, X_tr):
        X_tr = np.sort(X_tr.T)
        threshold_values = np.insert(X_tr, 0, np.min(X_tr,axis = 1) - 2, axis=1)
        threshold_values = np.sort(0.5*(threshold_values[:,1:] + threshold_values[:,:-1]))
        return threshold_values

    @jit(parallel=True)
    def adaboost_train(self, X_tr, Y_tr): #X->(data points, no of features)
      
        '''

        X_tr -> (no. of data points, no of features) eg. (7000, 50)
        Y_tr -> (no. of data points, 1) eg. (7000, 1)
        
        returns -> dictionary of weak classifiers

        '''

        print('X_tr ', X_tr)
        weights = [1/X_tr.shape[0]] * X_tr.shape[0]
        print('Initial weights ', weights)
        threshold_values = self.find_threshold(X_tr)
        print('Threshold values ',threshold_values)
        print(threshold_values.shape)
        eps = 1e-10

        best_classifiers = []
        for weak_classifier in range(self.T):

            print('############# CLASSIFIER NO ##########', weak_classifier)

            best_stumps = {"threshold":100, "polarity":1, "error": 100000000, "feature_no" : -1,"classifier_no":weak_classifier,"pred":0}
            for feature in range(X_tr.shape[1]):

                feature_threshold = np.sort(np.unique(threshold_values[feature,:]))
                X_feature = X_tr[:,feature]
                best_threshold = {"threshold":100, "polarity":1, "error": 100000000, "feature_no" : -1,"pred":0}
                for threshold in feature_threshold:

                    # If X_feature < threshold = -1 --> polarity is -1, else polarity = 1
                    polarity = -1
                    pred = np.ones((X_feature.shape))
                    pred[X_feature< threshold] = -1
                    error = np.sum(weights*(1-pred*Y_tr.T))*0.5

                    if error>0.5:
                        polarity = 1
                        error = 1- error
                        pred= pred*-1

                    #Selecting Best Threshold
                    if error<best_threshold['error']:
                        best_threshold['error'] = error
                        best_threshold['polarity']=polarity
                        best_threshold['threshold']=threshold
                        best_threshold['feature_no']= feature
                        best_threshold['pred'] = pred

                # Selecting Best Weak Classifier
                if best_threshold['error']<best_stumps['error']:
                    best_stumps['error'] = best_threshold['error']
                    best_stumps['polarity'] = best_threshold['polarity']
                    best_stumps['threshold'] = best_threshold['threshold']
                    best_stumps['feature_no'] = best_threshold['feature_no']
                    best_stumps['pred'] = best_threshold['pred']

            alpha_t = 0.5*np.log((1-best_stumps['error']+eps)/(best_stumps['error']+eps))
            best_stumps['alpha_t'] = alpha_t
            best_classifiers.append(best_stumps)

            weights = (weights * np.exp(-1*alpha_t*Y_tr.T*best_stumps['pred']))
            weights = weights/np.sum(weights)
            print(best_stumps)

        self.model = best_classifiers
        print(best_classifiers)
        return best_classifiers

    def adaboost_predict(self, X_tr): #

        '''
        Run this to perform prediction on the model after training.
        This function can be used for 
        X_tr -> X_train, X_val, X_test: (no. of data points, no of features) eg. (7000, 50)

        returns: predictions -> (no. of data points, 1) eg. (7000, 1)

        '''
        print(len(self.model))
        output = np.zeros((X_tr.shape[0]))
        for i in self.model:
            X_feature = X_tr[:,i['feature_no']]
            if i['polarity'] == -1:
                pred = np.ones((X_feature.shape))
                pred[X_feature< i['threshold']] = -1
            else:
                pred = np.ones((X_feature.shape))*-1
                pred[X_feature< i['threshold']] = 1
            output = output + pred*i['alpha_t']

        output = np.sign(output)
        return np.expand_dims(output,axis = 1)
    
    def save_model(self):
        '''
        Run this to save model after training.
        '''
        np.save('adaboost_model.npy', self.model)

    def load_model(self, model_path):
        
        '''
        Run this to load the trained model and then perform prediction.
        '''

        self.model = np.load(model_path, allow_pickle=True)