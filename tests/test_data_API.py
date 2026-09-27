import pytest
from unittest.mock import Mock
import perseval.data as data_module

from perseval.data import (available_datasets, download, download_and_split)

def test_available_datasets():
    datasets = available_datasets()

    assert isinstance(datasets, dict)

    assert "EPIC" in datasets
    assert "BREXIT" in datasets
    assert "DICES" in datasets
    assert "MHS" in datasets
    assert "MD" in datasets

def test_download_invalid_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        download("does_not_exist")
    
def test_download_requires_string():
    with pytest.raises(TypeError, match="dataset_name must be a string"):
        download(123)
        
def test_download_is_case_insensitive(monkeypatch):
    fake_dataset = Mock()

    monkeypatch.setitem(
        data_module._DATASETS,
        "epic",
        lambda: fake_dataset,
    )

    assert download("epic") is fake_dataset
    assert download("EPIC") is fake_dataset
    assert download("Epic") is fake_dataset
    
def test_download_epic(monkeypatch):
    fake_dataset = Mock()

    factory = Mock(return_value=fake_dataset)

    monkeypatch.setitem(
        data_module._DATASETS,
        "epic",
        factory,
    )

    result = download("epic")

    factory.assert_called_once_with()
    assert result is fake_dataset
    
def test_available_labels():
    dataset = data_module.PerspectivistDataset()

    dataset.labels = {
        "hs": set(),
        "offensiveness": set(),
        "aggressiveness": set(),
        "stereotype": set(),
    }

    assert dataset.available_labels() == [
        "hs",
        "offensiveness",
        "aggressiveness",
        "stereotype",
    ]
    
def test_download_and_split(monkeypatch):
    fake_dataset = Mock()

    download_mock = Mock(return_value=fake_dataset)

    monkeypatch.setattr(
        data_module,
        "download",
        download_mock,
    )

    result = download_and_split(
        "epic",
        user_adaptation="train",
        extended=True,
        named=True,
        baseline=False,
    )

    download_mock.assert_called_once_with("epic")

    fake_dataset.get_splits.assert_called_once_with(
        extended=True,
        user_adaptation="train",
        named=True,
        baseline=False,
    )

    assert result is fake_dataset
    
def test_available_labels_single_label():
    dataset = data_module.PerspectivistDataset()

    dataset.labels = {
        "irony": set(),
    }

    assert dataset.available_labels() == ["irony"]


def test_describe():
    dataset = data_module.PerspectivistDataset()

    dataset.name = "EPIC"
    dataset.label = "irony"
    dataset.key_user = "user"
    dataset.key_text = "text_id"
    
    dataset.labels = {
        "irony": set(),
    }

    dataset.dataset = [
        {
            "user": 1,
            "text_id": 10,
        },
        {
            "user": 1,
            "text_id": 11,
        },
        {
            "user": 2,
            "text_id": 10,
        },
    ]

    description = dataset.describe()

    assert isinstance(description, dict)

    assert description["name"] == "EPIC"
    assert description["instances"] == 3
    assert description["annotators"] == 2
    assert description["texts"] == 2

    assert description["annotations_per_text"] == 3 / 2
    assert description["annotations_per_annotator"] == 3 / 2

    assert description["default_label"] == "irony"
    assert "irony" in description["labels"]

def test_describe_verbose(capsys):
    dataset = data_module.PerspectivistDataset()

    dataset.name = "EPIC"
    dataset.label = "irony"
    dataset.key_user = "user"
    dataset.key_text = "text_id"

    dataset.labels = {
        "irony": set(),
    }

    dataset.dataset = [
        {"user": 1, "text_id": 10},
        {"user": 1, "text_id": 11},
        {"user": 2, "text_id": 10},
    ]

    description = dataset.describe(verbose=True)

    captured = capsys.readouterr()

    assert description["name"] == "EPIC"
    assert "EPIC" in captured.out
    assert "Task:" in captured.out
    assert "Annotators:" in captured.out
    assert "Texts:" in captured.out
    assert "Instances:" in captured.out
    assert "Annotations/text:" in captured.out
    assert "Annotations/annotator:" in captured.out
    assert "Labels:" in captured.out
    assert "Default label:" in captured.out
    assert "Metadata:" in captured.out


def test_describe_datasets(monkeypatch):
    fake_dataset = Mock()

    fake_dataset.describe.return_value = {
        "name": "EPIC",
        "task": "Irony",
        "annotators": 74,
        "texts": 3000,
        "instances": 14172,
        "annotations_per_text": 14172 / 3000,
        "annotations_per_annotator": 14172 / 74,
        "source": "Twitter, Reddit",
        "label_type": "Binary",
        "positive_class": "Irony",
        "metadata": ["Gender", "Nationality", "Age/Generation"],
        "labels": ["irony"],
        "default_label": "irony",
    }

    monkeypatch.setattr(
        data_module,
        "download",
        Mock(return_value=fake_dataset),
    )

    descriptions = data_module.describe_datasets()

    assert isinstance(descriptions, dict)

    for name in data_module.config.dataset_info:
        assert name in descriptions

    fake_dataset.describe.assert_called()


def test_describe_datasets_verbose(monkeypatch):
    fake_dataset = Mock()

    fake_dataset.describe.return_value = {
        "name": "EPIC",
    }

    monkeypatch.setattr(
        data_module,
        "download",
        Mock(return_value=fake_dataset),
    )

    descriptions = data_module.describe_datasets(verbose=True)

    assert isinstance(descriptions, dict)

    fake_dataset.describe.assert_called_with(verbose=True)
    
def test_public_api():
    import perseval

    assert callable(perseval.download)
    assert callable(perseval.download_and_split)
    assert callable(perseval.available_datasets)
    assert callable(perseval.describe_datasets)