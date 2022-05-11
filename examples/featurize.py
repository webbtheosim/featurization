#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: roshanpatel
"""

from feat_utils import linear_polymer,write_out
import pickle as pkl

idps = linear_polymer('metadata/seqs.list', 'metadata/DatasetA_metadata.csv')

#scaled OHE fingerprint for polymers
scaled_fp = idps.scaled_fingerprint(database_col='OHE')
write_out(scaled_fp, '.', 'scaled_ohe')

#sequence tensor of ohes
sequences = idps.to_long_sequence(database_col = 'OHE')
write_out(sequences, '.', 'seq_ohe')

#graph ohe node embeddings

#this featureization method already prints out a directory of graphs, so no 
#need to write anything out. Also, using data structures provided by spektral,
#we need to tag our graphs with labels.
with open('labels/rog.pkl','rb') as fid:
    labels = pkl.load(fid)
    
idps.linear_graph(labels, database_col = 'OHE', graph_dir = 'ohe_rog_graphs')
