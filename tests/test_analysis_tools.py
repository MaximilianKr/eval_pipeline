"""
Test suite for the analysis_tools module.

This module contains tests for the functions in the analysis_tools module,
including read_data_from_folder, compute_accuracy_metric, read_all_results,
and plot_accuracy.
"""

from unittest.mock import patch, mock_open
import pytest
import pandas as pd
from bin.analysis_tools import (
    read_data_from_folder,
    compute_accuracy_metric,
    read_all_results,
    plot_accuracy,
)


@pytest.fixture
def mock_json_data():
    return {
        "results": [
            {"logprob_of_good_continuation": -1.0, "logprob_of_bad_continuation": -2.0},
            {"logprob_of_good_continuation": -1.5, "logprob_of_bad_continuation": -1.0},
        ],
        "meta": {"model": "test_model", "revision": "v1", "dataset": "test_dataset"},
    }


@patch("bin.analysis_tools.listdir")
@patch(
    "bin.analysis_tools.open",
    new_callable=mock_open,
    read_data='{"results": [{"logprob_of_good_continuation": -1.0, \
        "logprob_of_bad_continuation": -2.0},{"logprob_of_good_continuation": \
            -1.5, "logprob_of_bad_continuation": -1.0}], "meta": {"model": \
                "test_model", "revision": "v1", "dataset": "test_dataset"}}',
)
@patch("bin.analysis_tools.path")
def test_read_data_from_folder(mock_path, mock_open_fn, mock_listdir):
    """
    Test reading data from a folder containing JSON files.
    """
    mock_listdir.return_value = ["file1.json", "file2.json"]
    mock_path.join.side_effect = lambda *args: "/".join(args)

    df = read_data_from_folder("mock_folder")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "logprob_of_good_continuation" in df.columns
    assert "logprob_of_bad_continuation" in df.columns
    assert "model" in df.columns
    assert "revision" in df.columns


def test_compute_accuracy_metric():
    """
    Test computing the accuracy metric for the DataFrame.
    """
    df = pd.DataFrame(
        {
            "logprob_of_good_continuation": [-1.0, -1.5],
            "logprob_of_bad_continuation": [-2.0, -1.0],
        }
    )

    df = compute_accuracy_metric(df)

    assert "model_prefers_good_continuation" in df.columns
    assert df["model_prefers_good_continuation"].tolist() == [True, False]


@patch("bin.analysis_tools.listdir")
@patch("bin.analysis_tools.path")
@patch("bin.analysis_tools.read_data_from_folder")
@patch("bin.analysis_tools.compute_accuracy_metric")
def test_read_all_results(
    mock_compute_accuracy_metric,
    mock_read_data_from_folder,
    mock_path,
    mock_listdir,
    mock_json_data,
):
    """
    Test reading all results from base folder and computing accuracy metrics.
    """
    mock_listdir.return_value = ["folder1", "folder2"]
    mock_path.isdir.return_value = True
    mock_path.join.side_effect = lambda *args: "/".join(args)
    mock_df = pd.DataFrame(mock_json_data["results"])
    mock_df["model"] = mock_json_data["meta"]["model"]
    mock_df["revision"] = mock_json_data["meta"]["revision"]
    mock_df["dataset"] = mock_json_data["meta"]["dataset"]
    mock_read_data_from_folder.return_value = mock_df
    mock_compute_accuracy_metric.return_value = mock_df

    all_results = read_all_results("base_folder")

    assert isinstance(all_results, dict)
    assert len(all_results) == 2
    assert "df_folder1" in all_results
    assert "df_folder2" in all_results
    assert isinstance(all_results["df_folder1"], pd.DataFrame)


@patch("bin.analysis_tools.plt.show")
def test_plot_accuracy(mock_plt_show):
    """
    Test plotting accuracy for each model and revision.
    """
    df = pd.DataFrame(
        {
            "logprob_of_good_continuation": [-1.0, -1.5, -1.0, -1.5],
            "logprob_of_bad_continuation": [-2.0, -1.0, -2.0, -1.0],
            "model": [
                "test_model/test1",
                "test_model/test1",
                "test_model/test2",
                "test_model/test2",
            ],
            "revision": ["v1", "v1", "v2", "v2"],
            "dataset": ["test_dataset", "test_dataset", "test_dataset", "test_dataset"],
        }
    )
    df = compute_accuracy_metric(df)
    all_dfs = {"df_test": df}

    plot_accuracy(all_dfs)

    assert mock_plt_show.called
