import numpy as np
import csv
from scipy.ndimage import maximum_filter1d
import json
import matplotlib.pyplot as plt

def beat_list_to_array(filename, data_type, beat_type):
    beat_list = []
    with open(filename, newline='\n') as f:
        lines = csv.reader(f, delimiter='\t')
        for line in lines:
            # include downbeats to beats!
            if data_type == "detections":
                beat_list.append(np.float64(line[0]))

            elif data_type == "annotations":

                if beat_type == "beats":
                    beat_list.append(np.float64(line[0]))

                elif beat_type == "downbeats":
                    if line[2] == 'db':
                        beat_list.append(np.float64(line[0]))
                else:
                    print("Beat type not specified.")
                    raise TypeError
            else:
                print("Data type not specified.")
                raise TypeError

    beat_array = np.array(beat_list)

    return beat_array


def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:, :1, :], pad_frames, axis=1)
    pad_stop = np.repeat(data[:, -1:, :], pad_frames, axis=1)
    return np.concatenate((pad_start, data, pad_stop), axis=1)


def show_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return d, h, m, s


def widen_beat_targets(y, size=3, value=0.5):
    np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)


def print_model_statistics(model_checkpoint):
    with open(model_checkpoint, 'r') as f:
        load_dict = json.load(f)

    print("Checkpoint stats loaded:", model_checkpoint)
    d, h, m, s = show_time(load_dict["total_time"])
    print("Epochs trained:", load_dict["epoch"], "--- Iterations:", load_dict["iter"], "--- Total training time:", d,
          "days,", f"{h:02d}:{m:02d}:{s:02d}", "--- Last learning rate:", load_dict["learning_rate"])

    loss_list = load_dict["loss_list"]
    val_loss_list = load_dict["val_loss_list"]
    fscore_list_val_average = load_dict["fscore_list_val_average"]
    fscore_list_val_average_db = load_dict["fscore_list_val_average_db"]
    learning_rates_list = load_dict["learning_rates_list"]

    if len(loss_list) > 1:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 5))
        ax1.plot(np.arange(len(loss_list)), loss_list)
        ax1.set_title("Training loss")
        ax2.plot(np.arange(len(val_loss_list)), val_loss_list, color="green")
        ax2.set_title("Validation summary loss after " + str(len(val_loss_list)) + " validation runs")
        ax3.plot(np.arange(len(fscore_list_val_average)), fscore_list_val_average, color="red")
        ax3.set_title("Validation F-Score (beats) after " + str(len(fscore_list_val_average)) + " validation runs")
        ax3.set_xlabel("Last value: " + str("%.4f" % fscore_list_val_average[-1]) + " --- Max value: " + str(
            "%.4f" % np.max(fscore_list_val_average)) + " (Index " + str(np.argmax(fscore_list_val_average)) + ")")
        ax4.plot(np.arange(len(fscore_list_val_average_db)), fscore_list_val_average_db, color="orange")
        ax4.set_title(
            "Validation F-Score (downbeats) after " + str(len(fscore_list_val_average_db)) + " validation runs")
        ax4.set_xlabel("Last value: " + str("%.4f" % fscore_list_val_average_db[-1]) + " --- Max value: " + str(
            "%.4f" % np.max(fscore_list_val_average_db)) + " (Index " + str(
            np.argmax(fscore_list_val_average_db)) + ")")
        plt.show()

        plt.plot(np.arange(len(learning_rates_list)), learning_rates_list, color="purple")
        plt.title("Learning rate history")
        plt.show()

def plot_value_list(list,colour,title,save,save_name):
    if len(list) > 1:
        plt.clf()
        plt.plot(np.arange(len(list)), list, color=colour)
        plt.title(title)
        if save == False:
            plt.show()
        elif save == True:
            plt.savefig(save_name+".pdf")