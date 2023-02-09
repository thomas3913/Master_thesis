import os
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import sys
import time
import mir_eval
import madmom
import partitura as pt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
sys.path.append("../")
sys.path.append("../PM2S")
from helper_functions.helper_functions import *
from helper_functions.dataset_midi import *
from networks.my_madmom import my_madmom
#from PM2S.pm2s.features.beat import RNNJointBeatProcessor

import pytorch_lightning as pl

df2 = pd.read_csv("../PM2S/dev/metadata/metadata.csv")
        
all_datasets = get_midi_filelist(["ASAP","CPM","AMAPS"])

# DATALOADERS:

train_loader = DataLoader(
    get_pianoroll_dataset(all_datasets)[0],
    num_workers=4,
    shuffle=True,
    batch_size=1)

validation_loader = DataLoader(
    get_pianoroll_dataset(all_datasets)[1],
    shuffle=False,
    batch_size=1)

test_loader = DataLoader(
    get_pianoroll_dataset(all_datasets)[2],
    shuffle=False,
    batch_size=1)

print("Training set:",len(train_loader),"samples. Validation set:",len(validation_loader),"samples. Test set:",len(test_loader),"samples.\n")

# TESTING

#model_progress_info = "models/checkpoints_02b/model_iter_33600_progress_info.json"

#print_model_statistics(model_progress_info)

checkpoint_list = []

# Load test evaluation dictionary or create it:
if os.path.exists("evaluation/test_evaluation_results_02b.json"):
    with open("evaluation/test_evaluation_results_02b.json", 'r') as f:
        test_evaluation_results = json.load(f)
else:
    print("Test evaluation results dictionary file does not exist. Creating new one.")
    test_evaluation_results = {}

for checkpoint in checkpoint_list:
    try:
        model = torch.load(checkpoint,map_location = torch.device("cpu"))
    except Exception as e:
        print(checkpoint,"error:\n",e)
        continue
    model.eval()
    warnings.filterwarnings("ignore")

    fscore_list_test, fscore_list_test_db, cemgil_list_test, cemgil_list_test_db = list(), list(), list(), list()

    device = next(model.parameters()).device
    print("\nCheckpoint:",checkpoint,"--- Device:",device,"\n")

    if checkpoint not in test_evaluation_results.keys():
        with torch.no_grad():
            for j, datapoint_test in enumerate(test_loader):

                padded_array_test = cnn_pad(datapoint_test[0].float(),2)
                padded_array_test = torch.tensor(padded_array_test)
                
                if j % 10 == 0:
                    print("Test sample number",j+1,"/",len(test_loader),"---",datapoint_test[4][0],"--- Input shape:",datapoint_test[0].shape)
                outputs = model(padded_array_test)

                #Calculate F-Score:
                try:
                    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                    beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
                    evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, datapoint_test[1][0])
                    f_score_test = evaluate.fmeasure
                    fscore_list_test.append(f_score_test)
                    cemgil_list_test.append(evaluate.cemgil)
                except Exception as e:
                    print("Sample can not be processed correctly. Error in beat process:",e)
                    f_score_test = 0

                combined_0 = outputs[0].detach().cpu().numpy().squeeze()
                combined_1 = outputs[1].detach().cpu().numpy().squeeze()
                combined_act = np.vstack((np.maximum(combined_0 - combined_1, 0), combined_1)).T

                # Calculate F-Score db:
                try:
                    proc_db = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4],fps=100)
                    beat_times_db = proc_db(combined_act)
                    evaluate_db = madmom.evaluation.beats.BeatEvaluation(beat_times_db, datapoint_test[2][0], downbeats=True)
                    fscore_test_db = evaluate_db.fmeasure
                    fscore_list_test_db.append(fscore_test_db)
                    cemgil_list_test_db.append(evaluate_db.cemgil)
                except Exception as e:
                    print("Sample can not be processed correctly. Error in downbeat process:",e)
                    fscore_test_db = 0
                if j % 10 == 0:
                    print("F-Score (b):","%.4f"%f_score_test,"--- F-Score (db):","%.4f"%fscore_test_db,"--- Cemgil Score (b):","%.4f"%evaluate.cemgil,"--- Cemgil Score (db):","%.4f"%evaluate_db.cemgil,"\n")
    
        # Append test evaluation results to a dictionary and save it:
        average_fscore_b, average_fscore_db = np.sum(fscore_list_test)/len(fscore_list_test), np.sum(fscore_list_test_db)/len(fscore_list_test_db)
        average_cemgil_b, average_cemgil_db = np.sum(cemgil_list_test)/len(cemgil_list_test), np.sum(cemgil_list_test_db)/len(cemgil_list_test_db)

        test_evaluation_results[checkpoint] = [average_fscore_b,average_fscore_db,average_cemgil_b,average_cemgil_db]

        with open("evaluation/test_evaluation_results_02b.json", 'w') as f:
            json.dump(test_evaluation_results, f, indent=2)

        print("Summary for checkpoint:",checkpoint)
        print("Average test F-Score (b):",average_fscore_b,"--- Average test F-Score (db):",average_fscore_db,"--- Average test Cemgil score (b):",average_cemgil_b,"--- Average test Cemgil score (db):",average_cemgil_db)

    else:
        print("Checkpoint results are already stored.")
        
        
with open("models/checkpoints_02b/model_iter_6000_progress_info.json", 'r') as f:
    load_dict = json.load(f)

print(load_dict["learning_rates_list"])