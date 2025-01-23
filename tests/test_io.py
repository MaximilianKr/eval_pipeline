"""
Test suite for the io module.

This module contains tests for the functions in the io module,
including timestamp, dict2json, and initialize_model.
"""

from unittest.mock import patch
from datetime import datetime
import json
import pytest
import torch
from bin.io import dict2json, timestamp, initialize_model


def test_timestamp_format():
    """
    Test that the timestamp function returns a correctly formatted string.
    """
    time = timestamp()
    try:
        datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pytest.fail("Timestamp format is incorrect.")


def test_dict2json(tmp_path):
    """
    Test the dict2json function to ensure it correctly writes a dictionary to
    a JSON file.
    """
    test_dict = {"key": "value", "number": 42}
    test_file = tmp_path / "test_output.json"

    dict2json(test_dict, str(test_file))

    with open(test_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    ass = "The JSON file content does not match the original dictionary."
    assert data == test_dict, ass


def test_initialize_model():
    """
    Test the initialize_model function to ensure the correct model is
    initialized with expected parameters.
    """
    model_name = "gpt2"
    revision = "main"

    with patch("torch.cuda.is_available", return_value=False), \
         patch("minicons.scorer.IncrementalLMScorer") as mock_model:

        mock_model.return_value = "Mocked Model"
        model = initialize_model(model_name, revision)

        mock_model.assert_called_once_with(
            model=model_name, device=torch.device("cpu"), revision=revision,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        ass = "The initialized model does not match the mocked return value."
        assert model == "Mocked Model", ass


def test_initialize_model_gpu():
    """
    Test the initialize_model function for GPU availability and correct device
    assignment.
    """
    model_name = "gpt2"
    revision = "main"

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("minicons.scorer.IncrementalLMScorer") as mock_model:

        mock_model.return_value = "Mocked GPU Model"
        model = initialize_model(model_name, revision)

        mock_model.assert_called_once_with(
            model=model_name, device=torch.device("cuda"), revision=revision,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        ass = "The initialized model does not match the mocked return value."
        assert model == "Mocked GPU Model", ass


def test_initialize_model_multi_gpu():
    """
    Test the initialize_model function for multi-GPU scenarios and correct
    device mapping.
    """
    model_name = "gpt2"
    revision = "main"

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("minicons.scorer.IncrementalLMScorer") as mock_model:

        mock_model.return_value = "Mocked Multi-GPU Model"
        model = initialize_model(model_name, revision)

        mock_model.assert_called_once_with(
            model=model_name, device='auto', revision=revision,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        ass = "The initialized model does not match the mocked return value."
        assert model == "Mocked Multi-GPU Model", ass
