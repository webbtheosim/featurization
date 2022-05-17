# SUMMARY
This repository contains data and code demonstrating the implementation of various featurization strategies for machine learning of polymers.

All content here is available under CC BY NC 4.0 License

Please consider citing references, as pertinent. 

# REFERENCES
Many of the featurization strategies are discussed and compared in the following article(s): 
1. Patel, R. A.; Borca, C. H. & Webb, M. A. "Featurization strategies for polymer sequence or composition design by machine learning"  Molecular Systems Design & Engineering, 2022, DOI: 10.1039/d1me00160d
2. Webb, M. A.; Jackson, N. E.; Gil, P. S. & de Pablo, J. J. "Targeted sequence design within the coarse-grained polymer genome" Science Advances, 2020, 6, eabc6216, DOI: 10.1126/sciadv.abc6216

Dataset A itself is archived/published here: http://arks.princeton.edu/ark:/88435/dsp01765374506 with DOI:
3. 10.34770/chzn-mj42

# PACKAGE REQUIREMENTS
The following packages are necessary to execute the code:
* numpy, scipy (general use)
* spektral (for graph neural networks)
* 
*

# Dataset A
This directory provides a duplicate of the data described in Refs. 1 and 3. 
The labels (labels.csv) are numerous physical properties for 2,585 intrinsically disordered proteins (IDPs) obtained by coarse-grained molecular dynamics simulation. 
The specific IDP sequences are sourced from version 9.0 of the DisProt database and are listed via simple numerical encoding in sequences.txt.
Corresponding metadata for the featurization of the sequences is provided in encodings.csv

# Code description and utilization
This directory contains scripts used to featurize IDP sequences and train models to predict their associated property labels. featurize.py has examples
of various featurization strategies and train.sh has commands that are used to train models to predict radius of gyration from the representations 
produced by featurize.py.

To run 'out-of-the-box', the relative directory hierarchy within /src/ should be preserved.

# HELP, SUGGESTIONS, CORRECTIONS?
If you need help, have suggestions, identify issues, or have corrections, please send your comments to Prof. Mike Webb at mawebb@princeton.edu
