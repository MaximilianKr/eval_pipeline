"""
Test suite for the io module.

This module contains tests for the functions in the io module,
including timestamp, dict2json, and initialize_model.
"""

import json
import re
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytest
from bin.io import timestamp, dict2json, initialize_model


def test_timestamp():
    """
    Test the timestamp function to ensure it returns the current timestamp
    in the correct format.
    """
    with patch("bin.io.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 10, 1, 0, 0, 0)
        assert timestamp() == "2024-10-01 00:00:00"


def test_dict2json(tmp_path):
    """
    Test the dict2json function to ensure it correctly writes a dictionary
    to a JSON file.
    """
    test_dict = {"key": "value"}
    out_file = tmp_path / "test.json"
    dict2json(test_dict, out_file)
    with open(out_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    assert data == test_dict


@patch("bin.io.torch.cuda.is_available", return_value=True)
@patch("bin.io.torch.device")
@patch("bin.io.scorer.IncrementalLMScorer")
def test_initialize_model_pythia(mock_scorer, mock_device, _):
    """
    Test the initialize_model function when using a supported model with CUDA
    available.
    """
    mock_device.return_value = "cuda"
    model_name = "EleutherAI/pythia-6.7b"
    revision = "main"
    mock_model_instance = MagicMock()
    mock_scorer.return_value = mock_model_instance

    result = initialize_model(model_name, revision)

    mock_device.assert_called_with("cuda")
    mock_scorer.assert_called_with(model=model_name, device="cuda", revision=revision)
    assert result == mock_model_instance


@patch("bin.io.torch.cuda.is_available", return_value=False)
@patch("bin.io.torch.device")
@patch("bin.io.scorer.IncrementalLMScorer")
def test_initialize_model_cpu(mock_scorer, mock_device, _):
    """
    Test the initialize_model function when using a supported model with only
    CPU available.
    """
    mock_device.return_value = "cpu"
    model_name = "allenai/OLMo-1B-hf"
    revision = "main"
    mock_model_instance = MagicMock()
    mock_scorer.return_value = mock_model_instance

    result = initialize_model(model_name, revision)

    mock_device.assert_called_with("cpu")
    mock_scorer.assert_called_with(model=model_name, device="cpu", revision=revision)
    assert result == mock_model_instance


def test_initialize_model_unsupported_model():
    """
    Test the initialize_model function with an unsupported model to ensure it
    raises a ValueError.
    """
    model_name = "unsupported/model"
    revision = "main"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Model not (yet) supported! (Your model: {model_name})"),
    ):
        initialize_model(model_name, revision)
