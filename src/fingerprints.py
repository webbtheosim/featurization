# -*- coding: utf-8 -*-
"""
Author(s):
                Roshan Patel <roshanp@princeton.edu>
Contributor(s):

"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Module List
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
import random
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.keras.preprocessing import sequence
from spektral.data.graph import Graph
from spektral.data import Dataset
import tensorflow as tf

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUX Functions:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def process_fingerprint(fingerprint):

    number_array = []
    fingerprint = fingerprint[1:-1].strip().split()
    for i,string in enumerate(fingerprint):
        if i!=len(fingerprint)-1:
            number = float(string[:-1])
        else:
            number = float(string)
        number_array.append(number)

    return np.array(number_array,dtype='float32')

def write_out(data_to_write,dirname,name):

    with open('{}/{}.pkl'.format(dirname,name),'wb') as fid:
        pkl.dump(data_to_write,fid,protocol=4)

    return


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classes:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class sequences(object):
    """
    A class meant for the storing information about the sequence to be processes
    """


    #~~~~~~~~~~~~~~~~~~~~~~
    # Class Initialization
    #~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self,seq_path,database_path):

        #you're going to need to change this when you're using it on the cluster
        self.seq_dir = seq_path
        self.database_dir = database_path

        # return the main sequ
        with open('{}'.format(self.seq_dir),'r') as fid:
            lines = [line.strip().split() for line in fid if line]

        # find the max and minimum length of the sequences
        min_seq = None
        max_seq = None
        for line in lines:
            if min_seq is None or len(line) < min_seq:
                min_seq = len(line)
            if max_seq is None or len(line) > max_seq:
                max_seq = len(line)
        self.min_length = min_seq
        self.max_length = max_seq

        #turn the data into a n x max_len matrix, where every element in the line is the amino acid or is 'N/A'
        all_data = []
        for i in range(len(lines)):
            seq = []
            for j in range(self.max_length):
                if j < len(lines[i]):
                    monomer = int(lines[i][j])
                else:
                    monomer = 'N/A'
                seq.append(monomer)
            all_data.append(seq)
        self.seq = all_data

    def load_database(self):
        db = pd.read_csv('{}'.format(self.database_dir))
        return db

class linear_polymer(sequences):
    """
    A class meant for the manipulation of sequences to facilitate featurization
    of non-branching, linear polymers for machine learning models. Sublcass of
    sequences
    """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sequence Processing Methods
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #turning into an explicit sequence
    def to_long_sequence(self,database_col):

        database = self.load_database()
        fp_dictionary = pd.Series(database[database_col].values,
                                  index=database['Number Encoding']).to_dict()

        fp_dim = len(process_fingerprint(fp_dictionary[list(fp_dictionary.keys())[0]]))

        featurized = []
        for polymer in self.seq:
            data = []
            for monomer in polymer:
                if monomer!= 'N/A':
                    fingerprint = fp_dictionary[monomer]
                    p_fingerprint = process_fingerprint(fingerprint)
                else:
                    p_fingerprint = np.zeros(fp_dim)

                data.append(p_fingerprint)
            featurized.append(data)
        featurized = np.array(featurized,dtype=float)

        return featurized


    #scaled fingerprints, summation. It's like a scaled, global graph pooling operation on physically meaningful, not learned, embeddings
    def scaled_fingerprint(self,database_col,dp_rep = 'insert',fp_rep = 'compress'):

        database = self.load_database()
        fp_dictionary = pd.Series(database[database_col].values,
                                  index=database['Number Encoding']).to_dict()

        featurized = []
        for polymer in self.seq:
            freq = np.zeros(len(fp_dictionary))
            dp_counter = 0
            for monomer in polymer:
                if monomer != 'N/A':
                    freq[int(monomer)] += 1
                    dp_counter+=1
            composition = freq / np.sum(freq)

            poly_fp = []
            for i in range(len(composition)):
                fingerprint = fp_dictionary[i]
                p_fingerprint = process_fingerprint(fingerprint)
                scaled_fp = composition[i]*p_fingerprint
                poly_fp.append(scaled_fp)
            poly_fp = np.array(poly_fp)

            if fp_rep == 'flatten':
                poly_fp = poly_fp.flatten()
            elif fp_rep == 'compress':
                poly_fp = np.sum(poly_fp,axis=0)
            else:
                sys.exit('Fingerprint method not recognized')

            if dp_rep == 'insert':
                poly_fp = np.insert(poly_fp,0,dp_counter / self.max_length)
            elif dp_rep == 'scale':
                poly_fp *= (dp_counter / self.max_length)
            elif dp_rep == 'none':
                pass
            else:
                sys.exit('DP representation method not recognized')

            featurized.append(poly_fp)
        featurized = np.array(featurized)

        return featurized

    def linear_graph(self,labels,database_col,graph_dir):

        database = self.load_database()
        fp_dictionary = pd.Series(database[database_col].values,
                                  index=database['Number Encoding']).to_dict()
        with open('{}'.format(self.seq_dir),'r') as fid:
            seq = [line.strip().split() for line in fid if line]

        os.mkdir(graph_dir)
        featurized = []
        for i,polymer in enumerate(seq):
            x = []
            a = np.zeros((len(polymer),len(polymer)))
            for ii,monomer in enumerate(polymer):
                #add the feature to the feature matrix
                fingerprint = fp_dictionary[int(monomer)]
                p_fingerprint = process_fingerprint(fingerprint)
                x.append(p_fingerprint)
                #add the corresponding connections to the adjacency matrix
                if ii != len(polymer)-1:
                    a[ii,ii+1] = 1
                    a[ii+1,ii] = 1
                #we have no edge features to encode here
            #tag the graph with the corresponding global labels
            #(no node-level predictions are being done in this work)
            x = np.array(x,dtype='float32')
            y = np.array(labels[i])
            graph = Graph(x=x,a=a,y=y)
            with open('{}/graph{}.pkl'.format(graph_dir,i),'wb') as fid:
                pkl.dump(graph,fid)


        return

class polymer_graph(Dataset):

    def __init__(self, filename = None, dirname = None, ids = None, **kwargs):
        self.filename = filename
        self.dirname = dirname
        self.ids = ids
        super().__init__(**kwargs)

    def download(self):
        pass

    def read(self):
        if self.filename is not None:
            with open(self.filename,'rb') as fid:
                graphs = pkl.load(fid)
        elif self.dirname is not None:
            graphs = []
            for idx in self.ids:
                with open('{}/graph{}.pkl'.format(self.dirname,idx),'rb') as fid:
                    graph = pkl.load(fid)
                graphs.append(graph)

        return graphs
