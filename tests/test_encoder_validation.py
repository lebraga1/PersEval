"""Offline validation splitting and real checkpoint-restoration regression tests."""
import copy
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from perseval import config
from perseval.baselines import encoder
from perseval.data import PerspectivistSplit, User


def make_split(name, text_ids):
    split = PerspectivistSplit(type=name)
    for text_id in text_ids:
        split.texts[text_id] = {"post": f"text-{text_id}"}
        for label in range(2):
            user_id = f"{name}-u{label}"
            split.users.setdefault(user_id, User(user_id))
            annotation = {"user": user_id, "text": text_id, "label": {"irony": label}}
            split.annotation.append(annotation)
            split.annotation_by_text.setdefault(text_id, []).append(annotation)
            split._seen.add((annotation["user"], text_id))
    return split


def test_validation_is_reproducible_disjoint_and_holds_out_instances():
    # One text has 90 annotations; the other ten have one each.
    annotations = [{"user": f"u{i}", "text": 0 if i < 90 else i,
                    "label": {"irony": i % 2}} for i in range(100)]
    original = copy.deepcopy(annotations)
    train, validation = encoder._split_training_indices(annotations, 0.1, seed=42)
    repeated = encoder._split_training_indices(annotations, 0.1, seed=42)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])
    assert len(train) == 90 and len(validation) == 10
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(len(original)))
    assert 0 in {original[i]["text"] for i in train}
    assert 0 in {original[i]["text"] for i in validation}
    assert annotations == original


def test_instance_holdout_rounds_up_to_a_whole_annotation():
    train, validation = encoder._split_training_indices([{}] * 23, 0.1, seed=42)
    assert len(validation) == 3 and len(train) == 20


def test_validation_can_split_annotations_of_a_single_text():
    annotations = make_split("train", [1]).annotation
    train, validation = encoder._split_training_indices(annotations, 0.1, seed=42)
    assert len(train) == len(validation) == 1


@pytest.mark.parametrize("percentage", [-0.1, 0, 1, 1.1, float("nan")])
def test_invalid_validation_percentage_is_rejected(percentage):
    with pytest.raises(ValueError, match="validation_percentage"):
        encoder._split_training_indices(make_split("train", range(20)).annotation,
                                        percentage, seed=42)


@pytest.mark.parametrize("instance_count", [0, 1])
def test_validation_requires_multiple_instances(instance_count):
    with pytest.raises(ValueError, match="at least two training instances"):
        encoder._split_training_indices([{}] * instance_count, 0.1, seed=42)


class TinyTokenizer:
    def __len__(self):
        return 32

    def add_special_tokens(self, tokens):
        return 0

    def __call__(self, text, **kwargs):
        text_id = int(re.search(r"text-(\d+)", text).group(1))
        return {"input_ids": [1, 2 + text_id % 28, 3, 0],
                "attention_mask": [1, 1, 1, 0]}


@pytest.fixture
def tiny_encoder(monkeypatch, tmp_path):
    # Build a tiny local BERT: no pretrained weights, tokenizer or dataset downloads.
    bert_config = BertConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    )
    monkeypatch.setattr(encoder.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(encoder.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *args, **kwargs: BertForSequenceClassification(bert_config))
    model_config = copy.deepcopy(config.model_config)
    model_config["roberta-base"].update(
        output_dir=str(tmp_path / "checkpoints"), logging_dir=str(tmp_path / "logs"),
        num_train_epochs=3, learning_rate=0.01, per_device_train_batch_size=32,
        per_device_eval_batch_size=4,
        use_cpu=True, disable_tqdm=True,
    )
    monkeypatch.setattr(config, "model_config", model_config)
    monkeypatch.setattr(config, "prediction_dir", str(tmp_path / "predictions"))
    monkeypatch.setattr(config, "validation_percentage", 0.1)
    dataset = SimpleNamespace(
        name="EPIC", named=True, user_adaptation=False, extended=False, traits={},
        training_set=make_split("train", range(20)),
        adaptation_set=make_split("adaptation", []),
        test_set=make_split("test", range(100, 102)),
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield encoder.PerspectivistEncoder("roberta-base", dataset, "irony"), dataset
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("validation_percentage,validation_count,save_total_limit",
                         [(None, 4, 1), (0.2, 8, 3)])
def test_training_restores_earlier_best_checkpoint_and_predicts_on_test(
    tiny_encoder, monkeypatch, capsys, validation_percentage, validation_count, save_total_limit,
):
    model, dataset = tiny_encoder
    original = copy.deepcopy(dataset)
    config.model_config["roberta-base"]["save_total_limit"] = save_total_limit

    class ControlledValidationTrainer(encoder.CustomTrainer):
        def evaluate(self, *args, **kwargs):
            if round(self.state.epoch) == 3:
                self.last_epoch_weights = {
                    name: value.clone() for name, value in self.model.state_dict().items()
                }
            metrics = super().evaluate(*args, **kwargs)
            # Force an earlier best epoch to exercise restoration, not just final saving.
            metrics["eval_loss"] = [0.3, 0.1, 0.4][round(self.state.epoch) - 1]
            return metrics

    monkeypatch.setattr(encoder, "CustomTrainer", ControlledValidationTrainer)
    trainer = model.train(validation_percentage=validation_percentage)

    assert len(trainer.eval_dataset) == validation_count
    assert len(trainer.train_dataset) == 40 - validation_count
    assert trainer.args.eval_strategy.value == "epoch"
    assert trainer.args.load_best_model_at_end is True
    assert trainer.args.metric_for_best_model == "eval_loss"
    assert trainer.args.greater_is_better is False
    assert trainer.args.per_device_eval_batch_size == 4
    assert trainer.args.save_total_limit == save_total_limit
    assert trainer.state.epoch == 3
    assert trainer.state.best_metric == 0.1
    assert len([row for row in trainer.state.log_history if "eval_loss" in row]) == 3
    assert "Selected epoch: 2.0" in capsys.readouterr().out

    best = BertForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint)
    weights = trainer.model.state_dict()
    assert all(torch.equal(weights[name], value) for name, value in best.state_dict().items())
    assert any(not torch.equal(weights[name], value)
               for name, value in trainer.last_epoch_weights.items())
    assert model.model is trainer.model

    training_labels = [int(label) for label in trainer.train_dataset["labels"]]
    expected_weights = encoder._compute_balanced_class_weights(training_labels, 2)
    np.testing.assert_array_equal(trainer.class_weights, expected_weights)
    model.predict(trainer)
    predictions = pd.read_csv(f"{config.prediction_dir}/predictions_EPIC_True_False_False.csv")
    assert len(predictions) == len(dataset.test_set.annotation)
    assert set(predictions["text_id"]) == {100, 101}
    for name in ("training_set", "adaptation_set", "test_set"):
        before, after = getattr(original, name), getattr(dataset, name)
        assert before.annotation == after.annotation
        assert before.annotation_by_text == after.annotation_by_text
        assert before.texts == after.texts and before.users == after.users


def test_class_weights_are_calculated_only_from_retained_training_rows(tiny_encoder, monkeypatch):
    model, dataset = tiny_encoder
    train, validation = encoder._split_training_indices(dataset.training_set.annotation, 0.1, config.seed)
    # Make the validation distribution different, so including it changes the weights.
    for index in validation:
        dataset.training_set.annotation[index]["label"]["irony"] = 1
    captured = {}

    class CaptureTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state = SimpleNamespace(best_model_checkpoint=None)

        def set_class_weights(self, weights):
            captured["weights"] = weights

        def train(self):
            pass

    monkeypatch.setattr(encoder, "CustomTrainer", CaptureTrainer)
    model.train()
    np.testing.assert_array_equal(captured["weights"], [1.0, 1.0])
    assert len(captured["train_dataset"]) == len(train)
    assert len(captured["eval_dataset"]) == len(validation)


def test_missing_training_class_after_holdout_fails_instead_of_training(tiny_encoder):
    model, dataset = tiny_encoder
    train, _ = encoder._split_training_indices(dataset.training_set.annotation, 0.1, config.seed)
    for index in train:
        dataset.training_set.annotation[index]["label"]["irony"] = 0
    with pytest.raises(ValueError, match="does not contain model labels"):
        model.train()
