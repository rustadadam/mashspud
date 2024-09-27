"""
Helper functions for demonstration.

These functions are not intended to be used elsewhere.
"""

import pandas as pd
import numpy as np

def prepare_dataset(filepath):
    """
    Takes a filepath to one of the provided csv files (they are formatted in a specific way). Will return
    the labels and the data.
    """

    #Read in file
    data = pd.read_csv(filepath).to_numpy()

    #Transform labels to numeric data
    labels = pd.Categorical(data[: , 0]).codes

    #Double labels (for plotting later)
    labels = np.concatenate([labels, labels])

    features = data[:, 1:].astype(np.float32)

    return features, labels

def split_features(features, split = "distort"):
        """
        Split the features to create distinct domains.

        Try setting split to "distort", "random", or "rotation". 

        See more here: PAPER_DESCRIPTION
        """

        import random
        random.seed(42)

        if split == "random":

            # Generate column indices and shuffle them
            column_indices = np.arange(features.shape[1])
            np.random.shuffle(column_indices)

            # Choose a random index to split the shuffled column indices
            split_index = random.randint(1, len(column_indices) - 1)

            # Use the shuffled indices to split the features array into two parts
            split_a = features[:, column_indices[:split_index]]
            split_b = features[:, column_indices[split_index:]]

        elif split == "rotation":
            #Apply random rotation to q
            rng = np.random.default_rng(42)
            d = np.shape(features)[1]
            random_matrix = rng.random((d, d))
            q, _ = np.linalg.qr(random_matrix)

            split_a = features

            #Transform features by q
            split_b = features @ q
        
        elif split == "distort":
            #Split A remains the same
            split_a = features

            #Add noise to split B
            split_b = features + np.random.normal(scale = 0.05, size = np.shape(features))

        #Reshape if they only have one sample
        if split_a.shape[1] == 1:
            split_a = split_a.reshape(-1, 1)
        if split_b.shape[1] == 1:
            split_b = split_b.reshape(-1, 1)


        return split_a, split_b

def create_anchors(dataset_size):
    """Returns an array of anchors equal to the datset size."""

    import random

    random.seed(42)

    #Generate anchors that can be subsetted
    rand_ints = random.sample(range(dataset_size), dataset_size)

    return np.vstack([rand_ints, rand_ints]).T