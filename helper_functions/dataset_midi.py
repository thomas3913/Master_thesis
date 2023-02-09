import pandas as pd
from pathlib import Path
import os, sys
import numpy as np
from torch.utils.data import Dataset
from helper_functions.helper_functions import *
sys.path.append("../")
sys.path.append("../PM2S")
from PM2S.dev.data.data_utils import get_note_sequence_and_annotations_from_midi

df2 = pd.read_csv("../PM2S/dev/metadata/metadata.csv")

def get_midi_filelist(dataset_list):
    
    for elemnt in dataset_list:
        if elemnt not in ["ASAP","CPM","AMAPS"]:
            print("You can only enter ASAP, CPM or AMAPS")
            raise ValueError
    
    asap_dataset = "../../asap-dataset"
    AMAPS = "../../MIDI-Datasets/A-MAPS_1.2"
    CPM = "../../MIDI-Datasets/Piano-MIDI/midis"
    ACPAS = "../../MIDI-Datasets/ACPAS"

    df = pd.read_csv(Path(asap_dataset,"metadata.csv"))

    all_datasets = list()
    ASAP = list()
    for element in df["midi_performance"]:
        #midi_performances.append(str(Path(PROJECT_PATH,element)))
        ASAP.append(element)
        if "ASAP" in dataset_list:
            all_datasets.append(os.path.join(asap_dataset,element))
    
    AMAPS_files = list()
    for (dirpath, dirnames, filenames) in os.walk(AMAPS):
        AMAPS_files += [os.path.join(dirpath, file) for file in filenames if file[-3:] == "mid"]
    AMAPS_files = sorted(AMAPS_files)
    if "AMAPS" in dataset_list:
        all_datasets += [AMAPS_files[i] for i in range(len(AMAPS_files))]

    PIANO_MIDI_files = list()
    for (dirpath, dirnames, filenames) in os.walk(CPM):
        PIANO_MIDI_files += [os.path.join(dirpath, file) for file in filenames if file[-3:] in ["mid","MID"]]
    PIANO_MIDI_files = sorted(PIANO_MIDI_files)
    if "CPM" in dataset_list:
        all_datasets += [PIANO_MIDI_files[i] for i in range(len(PIANO_MIDI_files))]

    for _ in range(2):
        error_list = list()
        for i, element in enumerate(all_datasets):
            try:
                split = df2.loc[df2["midi_perfm"] == element]["split"].iloc[0]
            except Exception as e:
                error_list.append(element)
                all_datasets.remove(element)

    return all_datasets


def get_pianoroll_dataset(filelist):
    train_list = list()
    validation_list = list()
    test_list = list()
    not_used = list()

    for element in filelist:
        split = df2.loc[df2["midi_perfm"] == element]["split"].iloc[0]
        if split == "train":
            train_list.append(element)
        elif split == "valid":
            validation_list.append(element)
        elif split == "test":
            test_list.append(element)
        else:
            not_used.append(element)

    class Audio_Dataset(Dataset):
        def __init__(self,file_list):
            self.file_list = file_list
            

        def __getitem__(self, index):

            if df2.loc[df2["midi_perfm"] == self.file_list[index]]["source"].iloc[0] == "ASAP":  
                annotation_file = self.file_list[index][:-4]+"_annotations.txt"
                beats = beat_list_to_array(annotation_file,"annotations","beats")
                downbeats = beat_list_to_array(annotation_file,"annotations","downbeats")
            else:
                annot_from_midi = get_note_sequence_and_annotations_from_midi(self.file_list[index])
                beats = annot_from_midi[1]['beats']
                downbeats = annot_from_midi[1]['downbeats']

            pr = np.load(self.file_list[index][:-4]+"_pianoroll.npy")

            return pr, beats, downbeats, index, self.file_list[index]
        
        def __len__(self):
            
            return len(self.file_list)
        
    train_list_dataset = Audio_Dataset(file_list = train_list)
    validation_list_dataset = Audio_Dataset(file_list = validation_list)
    test_list_dataset = Audio_Dataset(file_list = test_list)
    
    return train_list_dataset, validation_list_dataset, test_list_dataset