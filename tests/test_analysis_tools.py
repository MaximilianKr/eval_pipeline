"""
Test suite for the analysis_tools module.

This module contains tests for the functions in the analysis_tools module,
including read_data_from_folder, compute_accuracy_metric, read_all_results,
and plot_accuracy.
"""

import json
import pytest
import pandas as pd
from bin.analysis_tools import (
    read_data_from_folder, compute_accuracy, plot_bar_charts
)


@pytest.fixture
def sample_json_folder(tmp_path):
    """
    Creates a temporary folder with sample JSON files for testing.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.

    Returns:
        str: The path to the created temporary folder.
    """
    folder = tmp_path / "test_data"
    folder.mkdir()

    sample_data_1 = {
        "results": [
            {"logprob_of_good_continuation": 0.8, "logprob_of_bad_continuation": 0.2}
        ],
        "meta": {"model": "test_model/v1", "revision": "1", "type": "A", "relation": "X"}
    }
    sample_data_2 = {
        "results": [
            {"logprob_of_good_continuation": 0.6, "logprob_of_bad_continuation": 0.4}
        ],
        "meta": {"model": "test_model/v2", "revision": "2", "type": "B", "relation": "Y"}
    }

    with open(folder / "file1_main.json", "w", encoding="utf-8") as file:
        json.dump(sample_data_1, file)
    with open(folder / "file2.json", "w", encoding="utf-8") as file:
        json.dump(sample_data_2, file)

    return str(folder)


def test_read_data_from_folder(sample_folder_json):
    """
    Tests the read_data_from_folder function for correct data loading and
    processing.

    Args:
        sample_folder_json (str): Path to the temporary folder with JSON test data.
    """
    df = read_data_from_folder(sample_folder_json)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2  # Expecting two rows from two files
    assert "model" in df.columns
    assert df["model"].iloc[0] == "v1"

    df_final = read_data_from_folder(sample_json_folder, final_chkpt_only=True)
    assert df_final.shape[0] == 1  # Only one file with 'main' in the name
    assert df_final["model"].iloc[0] == "v1"


def test_compute_accuracy():
    """
    Tests the compute_accuracy function to ensure correct accuracy calculation
    and handling of missing required columns.
    """
    data = {
        "model": ["modelA", "modelA"],
        "revision": ["1", "1"],
        "type": ["A", "A"],
        "relation": ["X", "X"],
        "logprob_of_good_continuation": [0.8, 0.6],
        "logprob_of_bad_continuation": [0.2, 0.7]
    }
    df = pd.DataFrame(data)

    df_result = compute_accuracy(df)

    assert isinstance(df_result, pd.DataFrame)
    assert "accuracy" in df_result.columns
    assert df_result.loc[0, "accuracy"] == 1.0  # Both cases where good > bad

    df_missing = df.drop(columns=["logprob_of_good_continuation"])
    with pytest.raises(ValueError, match="Column 'logprob_of_good_continuation' is missing"):
        compute_accuracy(df_missing)


def test_plot_bar_charts(mocker):
    """
    Tests the plot_bar_charts function to ensure that it runs without errors
    and produces a valid plot.

    Args:
        mocker: Pytest mocker fixture to mock plt.show() during testing.
    """
    mocker.patch("matplotlib.pyplot.show")

    data = {
        "model": ["ModelA", "ModelB"],
        "accuracy": [0.85, 0.78],
        "type": ["Type1", "Type2"],
        "relation": ["Rel1", "Rel2"]
    }
    df = pd.DataFrame(data)
    model_order = ["ModelA", "ModelB"]

    plot_bar_charts(df, model_order)
    assert True  # If no exceptions occur, the test passes


def test_plot_bar_charts_invalid_order(mocker):
    """
    Tests plot_bar_charts function to check if it handles an invalid model
    order correctly.

    Args:
        mocker: Pytest mocker fixture to mock plt.show() during testing.
    """
    mocker.patch("matplotlib.pyplot.show")

    data = {
        "model": ["ModelA", "ModelB"],
        "accuracy": [0.85, 0.78],
        "type": ["Type1", "Type2"],
        "relation": ["Rel1", "Rel2"]
    }
    df = pd.DataFrame(data)
    model_order = ["ModelC", "ModelD"]  # Non-existent models in order

    with pytest.raises(ValueError):
        plot_bar_charts(df, model_order)
