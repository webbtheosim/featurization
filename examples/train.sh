python3 dnn.py -input scaled_ohe.pkl -label labels/rog.pkl -job dnn -outerk 5 -foldnumber 0 -innerk 5
python3 1_D_CNN.py -input seq_ohe.pkl -label /labels/rog.pkl -job seq -outerk 5 -foldnumber 0 -innerk 5
python3 split_gnn.py -graph_dir ohe_rog_graphdir -job graph_test -outerk 5 -foldnumber 0 -innerk 5
