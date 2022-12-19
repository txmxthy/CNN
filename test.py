# importing libraries
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import torch
import torchvision as tv
import splitfolders

import sklearn
from sklearn.metrics import classification_report, confusion_matrix


def load_model(model_name):
    """
    Load the model from the model_name.
    :param model_name:
    :return:
    """
    model = load_learner(path=".", file=model_name)
    return model


def get_data():
    """
    Load the test data for prediction
    """

    # Subdir of split is Test
    # Test has dirs for each label
    # Load each with label

    # Load the test data with normalize preprocessing


    test_dir = os.getcwd() + "/split/"
    name = "test"


    # This is a hack to avoid writing my own loader
    # noinspection PyNoneFunctionAssignment
    test_data = ImageDataBunch.from_folder(test_dir,
                                           valid=name,
                                           test=None,
                                           ds_tfms=get_transforms(),
                                           size=224,
                                           bs=32).normalize(imagenet_stats)

    # Check count of each label

    return test_data


def main():
    print("READ README!")
    model = load_model("exportA.pkl")
    data = get_data()

    # Add the data to the model
    model.data = data

    # Get the predictions for the validation set
    preds, y = model.get_preds(ds_type=DatasetType.Valid)

    # Get the labels
    labels = data.classes

    # Get the index of the highest probability
    pred_class = preds.argmax(dim=1)

    # Get the label of the highest probability
    pred_class = [labels[i] for i in pred_class]

    # Get the actual labels
    actual = [labels[i] for i in y]

    # Get the probabilities
    probs = preds.max(dim=1)

    # Create a dataframe of the results

    results = pd.DataFrame({"Actual": actual, "Predicted": pred_class, "Probability": probs[0]})

    # Save the results to a csv
    results.to_csv("results.csv", index=False)

    # Print the results
    print(results)

    # Calculate precision, recall, f1
    print(classification_report(actual, pred_class, target_names=labels))

    # Calculate TPR, FPR, TNR, FNR
    # Get the confusion matrix
    cm = confusion_matrix(actual, pred_class, labels=labels)

    # Plot the confusion matrix

    # Replace the labels with the actual labels


    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()






if __name__ == "__main__":
    main()
