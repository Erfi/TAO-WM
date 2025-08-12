"""
This script splits the dataset/data files, downloaded for tacorl, into two folders: train and validation.
the split is based on the dataset/data/split.json.
Other files also need to be split for consistency.
"""

import os
import json
import shutil
import tqdm
import numpy as np

# read the dataset_info/split.json file
with open(
    "dataset/data/split.json"
) as f:  # TODO: change this to dataset/data as per tacorl dataset when it is downloaded
    split_info = json.load(f)

# create train and validation directories
os.makedirs("dataset/calvin_data/training", exist_ok=True)
os.makedirs("dataset/calvin_data/validation", exist_ok=True)

train_files = split_info["train"]  # [[9415, 30396], [30397, 49306], ..]
val_files = split_info["validation"]  # [[49307, 68316], [68317, 87326], ..]


# ====== MOVE FILES ======
# split the files into train and validation directories
# each file in dataset/data is of form "episode_0553567.npz" with 7 digits.
# the split info tells the range of files to include in each split (need to pad the split with zeros on the left to get 7 digits)
# we should then copy the files from the original location to the new location based on the splits (start and end are includive)
def move_files_start_end(start, end, src, dst):
    for i in tqdm.tqdm(range(start, end + 1)):
        filename = f"episode_{i:07d}.npz"
        shutil.move(os.path.join(src, filename), os.path.join(dst, filename))


try:
    for train_split in train_files:
        move_files_start_end(train_split[0], train_split[1], "dataset/data", "dataset/training")
    print("Done moving training files.")
except FileNotFoundError as e:
    print(f"Error occurred while moving training files, are they in the right directory?")

try:
    for val_split in val_files:
        move_files_start_end(val_split[0], val_split[1], "dataset/data", "dataset/validation")
    print("Done moving validation files.")
except FileNotFoundError as e:
    print(f"Error occurred while moving validation files, are they in the right directory?")


# ====== Create ep_start_end_ids.npy and ep_lens.npy ======
# based on train_files and val_files we will create numpy arrays.
# ep_start_end_ids.npy format: [ [start1, end1], [start2, end2], ... ]
# ep_lens.npy format: [ len1, len2, ... ]
print("Creating ep_start_end_ids.npy and ep_lens.npy for training set...")
train_ep_start_end_ids = []
train_ep_lens = []
for start, end in train_files:
    train_ep_start_end_ids.append([start, end])
    train_ep_lens.append(end - start + 1)

np.save("dataset/calvin_data/training/ep_start_end_ids.npy", train_ep_start_end_ids)
np.save("dataset/calvin_data/training/ep_lens.npy", train_ep_lens)
print("Done!")

print("Creating ep_start_end_ids.npy and ep_lens.npy for validation set...")
val_ep_start_end_ids = []
val_ep_lens = []
for start, end in val_files:
    val_ep_start_end_ids.append([start, end])
    val_ep_lens.append(end - start + 1)

np.save("dataset/calvin_data/validation/ep_start_end_ids.npy", val_ep_start_end_ids)
np.save("dataset/calvin_data/validation/ep_lens.npy", val_ep_lens)
print("Done!")


# ====== Move Other files from dataset/data to dataset ========
try:
    print("Moving other files from dataset/data to dataset...")
    # copy nn_steps_from_step.json
    shutil.move(
        "dataset/data/nn_steps_from_step.json",
        "dataset/calvin_data/nn_steps_from_step.json",
    )

    # copy start_end_tasks.json
    shutil.move("dataset/data/start_end_tasks.json", "dataset/calvin_data/start_end_tasks.json")

    # copy hard_start_end_tasks.json
    shutil.move(
        "dataset/data/hard_start_end_tasks.json",
        "dataset/calvin_data/hard_start_end_tasks.json",
    )

    # copy statistics.yaml
    shutil.copy("dataset/data/statistics.yaml", "dataset/calvin_data/training/statistics.yaml")
    shutil.move("dataset/data/statistics.yaml", "dataset/calvin_data/validation/statistics.yaml")

    print("Done!")
except FileNotFoundError as e:
    print("Error occurred while moving other files, are they in the right directory?")

print("You can now delete the dataset/data directory.")
