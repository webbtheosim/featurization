#!/usr/bin/env python3
"""
Author(s):
                Roshan Patel <roshanp@princeton.edu>
Contributor(s):
                Michael Webb <mawebb@princeton.edu>
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modules
#~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import os.path
# the following assumes that the directory hierarchy is preserved (utils is in /src, featurization.py is in /src)
path0 = os.path.dirname(__file__)
sys.path.append(path0 + "/utils/")
from fingerprints import linear_polymer,write_out
import pickle as pkl
import argparse

exit()
#~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARSER
#~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_parser():

    parser = argparse.ArgumentParser(description='Processes sequences and encodings into fingerprints and converts selected labels into python readable .pkl.')

    parser.add_argument('-path'   ,default=None,help = 'Path to dataset directory (default: None)')

    return parser

#~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
#~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    parser      = create_parser()
    args        = parser.parse_args()
    datapath    = args.path

    # define linear polymers
    idps = linear_polymer(dataPath + 'sequences.txt', dataPath + 'encodings.csv')

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
    #with open('labels/rog.pkl','rb') as fid:
    #    labels = pkl.load(fid)

    #idps.linear_graph(labels, database_col = 'OHE', graph_dir = 'ohe_rog_graphs')
