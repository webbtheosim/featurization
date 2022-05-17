# -*- coding: utf-8 -*-
"""
Author(s):
                Roshan Patel <roshanp@princeton.edu>
Contributor(s):
                Michael Webb <mawebb@princeton.edu>
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modules
#~~~~~~~~~~~~~~~~~~~~~~~~~~

# general modules
import argparse
import time,os
import pickle as pkl
import numpy as np

#for paralellization
from joblib import Parallel,delayed
from functools import partial

# ML modules
import hyperopt.pyll.stochastic
import tensorflow as tf
from sklearn.model_selection    import KFold,StratifiedKFold
from sklearn.metrics            import r2_score, mean_absolute_error,mean_squared_error
from hyperopt import fmin,tpe,hp,space_eval,trials_from_docs,Trials
from hyperopt.pyll.base import scope
from training_utils import write_params,trial_loader, create_dir, splitIDs, EarlyStoppingAtMinLoss

def create_parser():

    parser = argparse.ArgumentParser(description='Training script for simple deep neural network.')

    parser.add_argument('-input'   ,default=None,help = 'Name of file with data inputs. (default: None)')
    parser.add_argument('-label'  ,default=None,help = 'Name of file with data labels. (default: None)')
    parser.add_argument('-job'  ,default=None,help = 'Enter the name of the job. This will be the name of the dir with all training outputs')
    parser.add_argument('-outerk'  ,default=None,help = 'Enter the total number of outer folds')
    parser.add_argument('-foldnumber'  ,default=None,help = 'Enter the number of the fold that this script will be handling')
    parser.add_argument('-innerk'  ,default=None,help = 'Enter the number of inner folds this script will use to tune hyperparameters')
    parser.add_argument('-dataID',default=None,help='These ids are used to keep track of data varients produced via data augmentation')

    return parser


class myModel(tf.keras.Model):
    def __init__(self,params):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(params['dense1'],activation = 'relu')
        self.drop1  = tf.keras.layers.Dropout(params['drop1'])
        if params['layer2']['state'] == 'present':
            self.dense2 = tf.keras.layers.Dense(params['layer2']['dense2'],activation = 'relu')
            self.drop2  = tf.keras.layers.Dropout(params['layer2']['drop2'])
            if params['layer2']['layer3']['state'] == 'present':
                self.dense3 = tf.keras.layers.Dense(params['layer2']['layer3']['dense3'],activation = 'relu')
                self.drop3  = tf.keras.layers.Dropout(params['layer2']['layer3']['drop3'])
            else:
                self.dense3 = None
                self.drop3  = None
        else:
            self.dense2 = None
            self.drop2  = None
            self.dense3 = None
            self.drop3  = None
        self.dense4 = tf.keras.layers.Dense(1)

    def call(self,x):

        x = self.dense1(x)
        x = self.drop1(x)
        if self.dense2 is not None:
            x = self.dense2(x)
            x = self.drop2(x)
            if self.dense3 is not None:
                x = self.dense3(x)
                x = self.drop3(x)

        x = self.dense4(x)

        return x

def return_paramspace():


    paramSpace = {

        'dense1':scope.int(hp.quniform('dense1', 10, 750, 20)),
        'drop1' :hp.quniform('drop1', 0, 0.8, 0.1),
        'layer2':hp.choice('layer2',options = [
            {
            'state':'present',
            'dense2':scope.int(hp.quniform('dense2', 10, 750, 20)),
            'drop2' :hp.quniform('drop2', 0, 0.8, 0.1),
            'layer3':hp.choice('layer3',options = [

                {
                'state':'present',
                'dense3':scope.int(hp.quniform('dense3', 10, 750, 20)),
                'drop3':hp.quniform('drop3', 0, 0.8, 0.1)
                },

                {
                'state':'absent'
                }
                ])
            },
            {
            'state':'absent'
            }
            ]),

        'lr':hp.choice('lr', [0.001, 0.005,0.01,0.05]),
        'bs':hp.choice('bs',[32,64,128,256])

        }

    return paramSpace

def build_model(params): ####

    model = myModel(params)
    lr = params['lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer = optimizer,loss='mse')

    return model

def objective_function(params,data,labels,ids,trainID,summariespath,foldnumber,innerfold,trials):

    temp_path = '{}/fold{}_hpsummary.txt'.format(summariespath,foldnumber)
    with open(temp_path,'a') as fid:
        fid.write('~~~~~~Trial {} Hyperparameters~~~~~~~'.format(len(trials)))
        fid.write('\n')
    write_params(params,path=temp_path)
    with open('{}/fold{}_trials.pkl'.format(summariespath,foldnumber),'wb') as fid:
        pkl.dump(trials,fid)

    kfold2 = KFold(n_splits=innerfold,shuffle=True,random_state=27)
    splits2 = [(trainID[train],trainID[test]) for (train,test) in kfold2.split(trainID)]
    mses = Parallel(n_jobs=innerfold)(delayed(parallel_k2)(split,params,data,labels,ids) for split in splits2)
    loss = np.mean(mses)

    with open(temp_path,'a') as fid:
        fid.write('Loss: {:.2f}'.format(loss) + '\n')

    return loss


def parallel_k2(split,params,data,labels,ids):

    (i_trainID, i_testID) = split
    i_train_index,i_test_index = splitIDs(i_trainID,i_testID,ids)

    i_xtrain = data[i_train_index,:]
    i_xtest  = data[i_test_index,:]
    i_ytrain = labels[i_train_index,:]
    i_ytest  = labels[i_test_index,:]

    model = build_model(params)
    batch_size = params['bs']
    model.fit(i_xtrain,i_ytrain,
                callbacks=EarlyStoppingAtMinLoss(50),
                validation_split=0.15,shuffle=True,
                epochs=400,batch_size=batch_size,verbose=0)

    y_predict = model.predict(i_xtest)
    mse = mean_squared_error(i_ytest,y_predict)

    return mse


def main():


    parser = create_parser()
    args   = parser.parse_args()

    datapath  = args.input
    labelpath = args.label
    idpath    = args.dataID
    resultspath = '{}/results'.format(args.job)
    summariespath = '{}/summaries'.format(args.job)
    create_dir(args.job)
    create_dir(resultspath)
    create_dir(summariespath)


    outerfold = int(args.outerk)
    foldnumber= int(args.foldnumber)
    innerfold = int(args.innerk)

    #load in the data
    with open(datapath,'rb') as fid:
        data = pkl.load(fid)
        data = np.array(data)

    #load in the labels
    with open(labelpath,'rb') as fid:
        labels = pkl.load(fid)
        labels = np.array(labels)

    #load in the dataids
    if idpath is not None:
        with open(idpath,'rb') as fid:
            ids = pkl.load(fid)
    else:
        ids = np.arange(data.shape[0])

    unique_ids = np.unique(ids)


    #this is to approximiately stratify the splits by DP
    counter = 0
    tags = []
    for i in range(int(len(data)/5)):
        for j in range(5):
            tags.append(counter)
        counter+=1
    tags = np.array(tags)

    #run k=5 fold to evaluate the model on different splits of the data

    kfold1 = StratifiedKFold(n_splits = 5,shuffle=True,random_state = 27)
    trainID = [unique_ids[train] for (train,test) in kfold1.split(X=unique_ids,y=tags)][foldnumber]
    testID  = [unique_ids[test] for (train,test) in kfold1.split(X=unique_ids,y=tags)][foldnumber]
    train_index,test_index = splitIDs(trainID, testID, ids)

    x_train = data[train_index,:]
    x_test = data[test_index,:]
    y_train = labels[train_index,:]
    y_test = labels[test_index,:]

    #determine the best hyperparameters
    try:
        trials = trial_loader('trials',summariespath,resultspath)
    except:
        trials = Trials()

    paramSpace = return_paramspace()

    fmin_objective = partial(objective_function,
                             data=data,
                             labels = labels,
                             ids = ids,
                             trainID = trainID,
                             summariespath = summariespath,
                             foldnumber = foldnumber,
                             innerfold = innerfold,
                             trials = trials,
                             )

    best = fmin(
          fn=fmin_objective,
          space=paramSpace,
          algo=tpe.suggest,
          max_evals=200,
          trials=trials)

    best_params = space_eval(paramSpace, best)

    #build the model, write out the best parameters and the performance
    model = build_model(best_params)
    batch_size = best_params['bs']
    model.fit(x_train,y_train,
                    callbacks=EarlyStoppingAtMinLoss(50),
                    validation_split=0.15,shuffle=True,
                    epochs=400,batch_size=batch_size,verbose=0)
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test,y_predict)
    mae = mean_absolute_error(y_test, y_predict)

    model.save('{}/model{}'.format(summariespath,foldnumber))
    #write out the results
    with open('{}/fold{}_predictions.txt'.format(summariespath,foldnumber),'a') as fid:
        for i in range(len(y_predict)):
            fid.write(str(y_predict[i]) + ' ' + str(y_test[i]))
            fid.write('\n')

    temp_path = '{}/Model_{}_Hyperparameters.txt'.format(resultspath,foldnumber)
    with open(temp_path,'a') as fid:
        fid.write('~~~~~~Fold {} Hyperparameters~~~~~~~'.format(foldnumber))
        fid.write('\n')
    write_params(best_params,path=temp_path)

    with open('{}/Model_{}_Performance.txt'.format(resultspath,foldnumber),'a') as fid:
        fid.write('~~~~~~Fold {} Performance~~~~~~~'.format(foldnumber))
        fid.write('\n')
        fid.write('MAE: {:.3f}'.format(mae)+'\n')
        fid.write('R2 : {:.3f}'.format(r2)+'\n')

    return

if __name__ == '__main__':
    main()


