import os
import pickle as pkl
import numpy as np
from hyperopt import trials_from_docs
import tensorflow as tf

def write_params(params,path):
    with open(path,'a') as fid:
        for key in params.keys():
            if type(key) == dict:
                fid.write('{}: '.format(key))
                fid.write('\n')
                for inner_key in key.keys():
                    fid.write('{}:{}'.format(inner_key,key[inner_key]))
                    fid.write('\n')
            else:
                fid.write('{}:{}'.format(key,params[key]))
                fid.write('\n')
    return


def trial_loader(path,summariespath,foldnumber):

    trial_names = os.listdir(path)
    for i in range(len(trial_names)):
        with open('{}/{}'.format(path,trial_names[i]),'rb') as fid:
            random_trial = pkl.load(fid)
        if i==0:
            trial_container = random_trial
        else:
            trial_container = trials_from_docs(list(trial_container) + list(random_trial))

        try:
            with open('{}/{}'.format(summariespath,foldnumber),'rb') as fid:
                completed_trials = pkl.load(fid)
            trial_container = trials_from_docs(list(trial_container) + list(completed_trials))
        except:
            pass

    return trial_container

def create_dir(dirname):
    
  try:
    os.mkdir(dirname)
  except FileExistsError:
    print('The directory already exists')
    pass

  return

def splitIDs(trainID,testID,full_indicies):
    
    training = np.array([])
    for index in trainID:
        matching_indices = np.where(full_indicies==index)
        training = np.append(training,matching_indices)
    
    test = np.array([])
    for index in testID:
        matching_indices = np.where(full_indicies==index)
        test = np.append(test,matching_indices)
        
    training = np.array(training,dtype=int)
    test = np.array(test,dtype=int)
    
    return training,test


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if epoch == 1:
            self.best_weights = self.model.get_weights()
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
   

