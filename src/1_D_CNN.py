# -*- coding: utf-8 -*-
"""
Author(s):
                Roshan Patel <roshanp@princeton.edu>
Contributor(s):
                Michael Webb <mawebb@princeton.edu>
"""

import argparse
import time,os,sys,gc
import pickle as pkl
import numpy as np
from joblib import Parallel,delayed
from functools import partial
import hyperopt.pyll.stochastic
import tensorflow as tf
from sklearn.model_selection    import KFold,StratifiedKFold
from sklearn.metrics            import r2_score, mean_absolute_error,mean_squared_error
from hyperopt import fmin,tpe,hp,space_eval,Trials,trials_from_docs
from hyperopt.pyll.base import scope
from training_utils import write_params,trial_loader, create_dir, splitIDs, EarlyStoppingAtMinLoss


def create_parser():

    parser = argparse.ArgumentParser(description='Training implementation for convolutional neural network.')

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
        #conv1
        self.conv1 = tf.keras.layers.Conv1D(params['filter'],params['kernel'],activation='relu')
        #pool1
        if params['CNN_L1a']['state'] == 'present':
            if params['CNN_L1a']['type'] == 'avg':
                self.pool1 = tf.keras.layers.AveragePooling1D(params['CNN_L1a']['pool_size'])
            elif params['CNN_L1a']['type'] == 'max':
                self.pool1 = tf.keras.layers.MaxPool1D(params['CNN_L1a']['pool_size'])
            else:
                sys.exit('Error on pool2')
        elif params['CNN_L1a']['state'] == 'absent':
            self.pool1 = None
        else:
            sys.exit('Error on pool layer 1')

        #conv2 / pool2
        if params['CNN_L2']['state'] == 'present':
            self.conv2 = tf.keras.layers.Conv1D(params['CNN_L2']['filter'],params['CNN_L2']['kernel'],activation='relu')
            if params['CNN_L2']['pooling']['state'] == 'present':
                if params['CNN_L2']['pooling']['type'] == 'avg':
                    self.pool2 = tf.keras.layers.AveragePooling1D(params['CNN_L2']['pooling']['pool_size'])
                elif params['CNN_L2']['pooling']['type'] == 'max':
                    self.pool2 = tf.keras.layers.MaxPool1D(params['CNN_L2']['pooling']['pool_size'])
                else:
                    sys.exit('Error on pool2')
            elif params['CNN_L2']['pooling']['state'] == 'absent':
                self.pool2 = None
            else:
                sys.exit('Error on pool layer 2')

        elif params['CNN_L2']['state'] == 'absent':
            self.conv2 = None
            self.pool2 = None
        else:
            sys.exit('Error on conv layer 2')


        self.flatten = tf.keras.layers.Flatten()

        #dense1
        self.dense1 = tf.keras.layers.Dense(params['s1'],activation='relu')
        #drop1
        self.drop1  = tf.keras.layers.Dropout(params['d1'])

        #dense2
        if params['layer2']['state'] == 'present':
            self.dense2 = tf.keras.layers.Dense(params['layer2']['s2'],activation='relu')
            self.drop2  = tf.keras.layers.Dropout(params['layer2']['d2'])
        elif params['layer2']['state'] == 'absent':
            self.dense2 = None
            self.drop2  = None
        else:
            sys.exit('Problem on dense layer 2')

        self.dense3 = tf.keras.layers.Dense(1)

    def call(self,x):
        y = self.conv1(x)
        if self.pool1 is not None: y = self.pool1(y)
        if self.conv2 is not None: y = self.conv2(y)
        if self.pool2 is not None: y = self.pool2(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.drop1(y)
        if self.dense2 is not None: y = self.dense2(y)
        if self.drop2  is not None: y = self.drop2(y)
        y = self.dense3(y)
        return y


def return_paramspace():

    paramSpace = {

        'filter':scope.int(hp.quniform('filter1', 8, 64, 8)), ####
        'kernel':scope.int(hp.quniform('kernel1', 5, 25, 5)), ####

        'CNN_L1a': hp.choice('cnnl1a',[

            {
            'state':'present',
            'pool_size':scope.int(hp.quniform('pool1', 3, 9, 2)),
            'type':hp.choice('pt1',['max','avg']) ####
            },
            {
            'state':'absent'
            }
            ]),

        'CNN_L2': hp.choice('cnnl2',[

            {
            'state':'present',
            'filter':scope.int(hp.quniform('filter2', 8, 64, 8)),
            'kernel':scope.int(hp.quniform('kernel2', 5, 25, 5)),
            'pooling':hp.choice('pooling',options=[
                {
                'state':'present',
                'pool_size':scope.int(hp.quniform('pool2', 3, 9, 2)), ####
                'type':hp.choice('pt2',['max','avg'])

                },
                {
                'state':'absent'
                }])
            },

            {
            'state':'absent'
            }

            ]),


        #layer 1 options
        's1':scope.int(hp.quniform('s1', 10, 750, 20)),
        'd1':hp.quniform('d1', 0, 0.8, 0.1),

        #layer 2 options
        'layer2':hp.choice('layer2',[
            {
            'state':'present',
            's2':scope.int(hp.quniform('s2', 10, 750, 20)),
            'd2':hp.quniform('d2', 0, 0.8, 0.1)
            },
            {
            'state':'absent'
            },
            ]),

        'lr':hp.choice('lr', [0.001, 0.005,0.01]),
        'bs':hp.choice('bs',[32,64,128,256])

        }

    return paramSpace

def build_model(params):

    model = myModel(params)
    lr = params['lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

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

    model = build_model(params)
    batch_size = params['bs']
    model.fit(data[i_train_index,:],labels[i_train_index],
                callbacks=EarlyStoppingAtMinLoss(50),
                validation_split=0.15,shuffle=True,
                epochs=400,batch_size=batch_size,verbose=0)
    gc.collect()

    i_xtest  = data[i_test_index]
    i_ytest  = labels[i_test_index]
    y_predict = model.predict(i_xtest)

    try:
        mse = mean_squared_error(i_ytest,y_predict)
    except:
        mse = np.inf

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

    #create the dir that has a summary of the outputs
    if foldnumber == 0:
        create_dir(summariespath)

    #load in the data
    with open(datapath,'rb') as fid:
        data = pkl.load(fid)
        data = np.array(data)
        channels  = data.shape[-1]
        width      = data.shape[-2]
        height     = data.shape[-3]

    #load in the labels
    with open(labelpath,'rb') as fid:
        labels = pkl.load(fid)
        labels = np.array(labels)

    #load in the dataids
    if idpath is not None:
        with open(idpath,'rb') as fid:
            ids = pkl.load(fid).flatten()
    else:
        ids = np.arange(data.shape[0])

    unique_ids = np.unique(ids)

    #this is to approximiately stratify by DP
    counter = 0
    tags = []
    for i in range(int(len(unique_ids)/5)):
        for j in range(5):
            tags.append(counter)
        counter+=1
    tags = np.array(tags)

    #run k=5 fold to evaluate the model on different splits of the data

    kfold1 = StratifiedKFold(n_splits = 5,shuffle=True,random_state = 27)
    trainID = [unique_ids[train] for (train,test) in kfold1.split(X=unique_ids,y=tags)][foldnumber]
    testID  = [unique_ids[test] for (train,test) in kfold1.split(X=unique_ids,y=tags)][foldnumber]
    train_index,test_index = splitIDs(trainID, testID, ids)


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
    bs = best_params['bs']
    model.fit(data[train_index,:],labels[train_index],
                    callbacks=EarlyStoppingAtMinLoss(50),
                    validation_split=0.15,shuffle=True,
                    epochs=400,batch_size=bs,verbose=0)
    gc.collect()

    x_test = data[test_index,:]
    y_test = labels[test_index]
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








