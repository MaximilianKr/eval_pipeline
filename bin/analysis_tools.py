import json
import pandas as pd
from os import listdir, path
import matplotlib.pyplot as plt


def read_data_from_folder(folder_path):
    """
    Reads all JSON files from a specified folder and returns a single concatenated DataFrame.
    
    Args:
    folder_path (str): The path to the folder containing JSON files.

    Returns:
    pd.DataFrame: A DataFrame containing the concatenated data from all JSON files in the folder.
    """
    all_data = []

    files_to_read = [f for f in listdir(folder_path) if f.endswith(".json")]

    for file_name in files_to_read:
        file_path = path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
        results = pd.DataFrame(data["results"])
        for key, value in data["meta"].items():
            if not isinstance(value, list):
                results[key] = value
        all_data.append(results)

    df = pd.concat(all_data, ignore_index=True)
    
    return df


def compute_accuracy_metric(df):
    """
    Adds a column 'model_prefers_good_continuation' to the DataFrame
    indicating if the model prefers the good continuation.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the results.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    df["model_prefers_good_continuation"] = \
        (df["logprob_of_good_continuation"] > df["logprob_of_bad_continuation"])
    return df


def read_all_results(base_folder):
    """
    Iterates over all folders in the base results folder, reads JSON files,
    computes accuracy metrics, and returns a dictionary of DataFrames for each folder.
    
    Args:
    base_folder (str): The path to the base folder containing subfolders with JSON files.

    Returns:
    dict: A dictionary where keys are folder names and values are DataFrames.
    """
    folders = [f for f in listdir(base_folder) if path.isdir(path.join(base_folder, f))]
    
    all_dataframes = {}

    for folder in folders:
        folder_path = path.join(base_folder, folder)
        df = read_data_from_folder(folder_path)
        df = compute_accuracy_metric(df)
        all_dataframes[f"df_{folder}"] = df
    
    return all_dataframes


def plot_accuracy(all_dfs):
    """
    Plots the accuracy for each model and revision for each DataFrame separately.
    
    Args:
    all_dfs (dict): A dictionary where keys are DataFrame names and values are DataFrames.
    """
    for _, df in all_dfs.items():
        grouped = df.groupby(['model', 'revision'])
        accuracies = []
        model_revisions = []
        for (model_name, revision), group in grouped:
            accuracy = group["model_prefers_good_continuation"].mean()
            model_name = model_name.split('/')[-1]
            model_revision = f"{model_name}\n({revision})"
            accuracies.append(accuracy)
            model_revisions.append(model_revision)
        
        plt.figure(figsize=(12, 8))
        plt.bar(model_revisions, accuracies, color='skyblue')
        plt.xlabel('Model (Revision)')
        plt.ylabel('Accuracy')
        title = df["dataset"].unique()[0].capitalize()
        plt.title(f'Model Accuracy for {title}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
