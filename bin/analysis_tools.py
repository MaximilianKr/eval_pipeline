"""
Module for analyzing model evaluation results.

This module contains functions to read JSON data from folders,
compute accuracy metrics, and plot the results.
"""

import json
import os
from os import listdir

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data_from_folder(
        folder_path: str, final_chkpt_only: bool = False
        ) -> pd.DataFrame:
    """
    Reads all JSON files from a specified folder and returns a single
    concatenated DataFrame.

    Args:
        folder_path (str): The path to the folder containing JSON files.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all
        JSON files in the folder.
    """
    all_data = []

    if final_chkpt_only:
        files_to_read = [
            file for file in listdir(folder_path) if file.endswith("main.json")
        ]
    else:
        files_to_read = [
            file for file in listdir(folder_path) if file.endswith(".json")
        ]

    for file_name in files_to_read:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        results = pd.DataFrame(data["results"])
        for key, value in data["meta"].items():
            if not isinstance(value, list):
                results[key] = value
        all_data.append(results)

    df = pd.concat(all_data, ignore_index=True)

    # Extract model name after the last '/'
    df["model"] = df["model"].str.split("/").str[-1]

    return df


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes accuracy metric indicating if the model prefers good continuation.

    Args:
        df (pd.DataFrame): DataFrame with log probabilities.

    Returns:
        pd.DataFrame: Aggregated DataFrame with accuracy metric.
    """
    required_columns = [
        "logprob_of_good_continuation", "logprob_of_bad_continuation"
        ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the dataset.")

    df["model_prefers_good_continuation"] = (
        df["logprob_of_good_continuation"] > df["logprob_of_bad_continuation"]
    )

    df = df.groupby(
        ["model", "revision", "type", "relation"], as_index=False
        ).agg(accuracy=("model_prefers_good_continuation", "mean"))

    return df

def plot_bar_charts(df: pd.DataFrame, model_order: list) -> None:
    """
    Plots a bar chart of model accuracy using Seaborn.

    Args:
        df (pd.DataFrame): DataFrame containing the columns:
            - 'model': Model names.
            - 'accuracy': Accuracy values.
            - 'type': Evaluation type.
            - 'relation': Evaluation relation.
        model_order (list): List of model names in the desired order.

    The function combines 'type' and 'relation' for color distinction,
    orders models based on `model_order`, customizes axis labels,
    and adjusts the figure layout for readability.

    Example:
        plot_bar_charts(df)
    """
    df['type_relation'] = df['type'] + " - " + df['relation']

    plt.figure(figsize=(19, 8))  # adjust size

    sns.barplot(
        data=df,
        x="model",
        y="accuracy",
        hue="type_relation",
        palette="muted",
        dodge=True,
        errorbar=None,
        order=model_order
    )

    plt.xlabel("Model", fontsize=22)
    plt.ylabel("Accuracy", fontsize=22)

    plt.ylim(0.1, 1.00)  # adjust y-range

    plt.xticks(rotation=45, ha="right", fontsize=22)
    plt.yticks(fontsize=20)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.legend(
        title="Type - Relation",
        fontsize=22,
        title_fontsize=22,
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()
    plt.show()
