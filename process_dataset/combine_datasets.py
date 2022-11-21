#!/usr/bin/env python
# coding: utf-8

# # Combine datasets to a single dataset in a pickle file

# ## Output format
# 
# ```py
# {
#     "train": [
#         {
#             # Note: no ID needed as the index can serve as an ID
#             dataset_name: "dataset_name",
#             filename: "dataset_name/images/0001.jpg",
#             width: 1280,
#             height: 720,
#             bboxes: ndarray([x1, y1, x2, y2], ...),
#             labels: array(1, ...)
#             },
#            ...
#     ],
#     "val": [...],
#     "test": [...],
#     "datasets": [
#         {
#             "name": "dataset_name",
#             "rel_path": "dataset_name"
#             },
#         ..
#     ]
# ```

# ### Import required libraries

# In[12]:


import os
import pickle
import random

import common


# ### Settings

# In[13]:


# Provide data split values for train, val, and testing data as percentage / 100
data_distribution = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
}
# Make sure it sums up to 1...
assert sum(data_distribution[key] for key in data_distribution.keys()) == 1

# Select if you want your splits to be continuous vs random
# Eg. of continuous: train will contain 0.jpg, 1.jpg, 2.jpg, ...
random_data_distribution = True

# Set random seed
random.seed(42)

data = []


# # Combine the datasets

# In[14]:


dataset = {
    "datasets": [],
    "train": [],
    "val": [],
    "test": []
}


# In[15]:


for name in list(common.datasets.keys()):
    print(f"Reading {name}")
    pickle_filepath = os.path.join(common.datasets_path, common.datasets[name]["path"], "gt.pickle")
    # imgs_path = common.datasets[name]["images"]["path"]

    
    # Open the dataset's pickle file and load and process data
    with open(pickle_filepath, "rb") as f:
        data += pickle.load(f)

        # Randomly reorder images if desired
        if random_data_distribution:
            random.shuffle(data)
        
        # Set dataset_name and update filename (relative filepath)
        # filename needs to be updated to be relative to the datasets folder
        for img in data:
            img["dataset_name"] = name
            # img["filename"] = os.path.join(imgs_path, img["filename"])

        # Split data into train/val/test
        data_len = len(data)
        train_val_split_index = int(data_len * data_distribution["train"])
        val_test_split_index = int(data_len * (data_distribution["train"] + data_distribution["val"]))
        dataset["train"] = data[:train_val_split_index]
        dataset["val"] =   data[train_val_split_index:val_test_split_index]
        dataset["test"] =  data[val_test_split_index:]

        dataset["datasets"].append({
            "name": name,
            "rel_dataset_path": common.datasets[name]["path"]
        })

print(f"Read {len(data)} image annotations")


# In[16]:


with open(common.dataset_pickle_filepath, 'wb') as f:
    # pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL) # Best storage-wise
    pickle.dump(dataset, f, protocol=0) # Said to be human-readable

