import os

import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, set_seed, TrainingArguments
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import numpy as np

from .. import config


def _split_training_indices(annotations, validation_percentage, seed):
    """Hold out a reproducible fraction of training annotation instances."""
    if not 0 < validation_percentage < 1:
        raise ValueError("validation_percentage must be between 0 and 1 (exclusive)")
    if len(annotations) < 2:
        raise ValueError("Validation requires at least two training instances")
    return train_test_split(
        np.arange(len(annotations)), test_size=validation_percentage, random_state=seed,
    )


def _compute_balanced_class_weights(labels, num_labels):
    """Return one balanced weight for every model label.

    Compute the weights from plain annotation values instead of a formatted
    Hugging Face Dataset column.
    """
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.arange(num_labels, dtype=np.int64)

    if labels.size == 0:
        raise ValueError("Cannot compute class weights from an empty training set")

    unexpected = np.setdiff1d(np.unique(labels), classes)
    if unexpected.size:
        raise ValueError(
            f"Training labels {unexpected.tolist()} are outside the model label range "
            f"0..{num_labels - 1}"
        )

    missing = np.setdiff1d(classes, np.unique(labels))
    if missing.size:
        raise ValueError(
            f"Training set does not contain model labels {missing.tolist()}"
        )

    return compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    ).astype("float32")


class PerspectivistEncoder():
    def __init__(self, model_identifier, persp_dataset, label):
        # Seed before loading the classification head
        set_seed(config.seed)

        self.model_id = model_identifier
        self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        self.label = label
        self.traits = persp_dataset.traits
        self.named = persp_dataset.named
        self.user_adaptation = persp_dataset.user_adaptation
        self.extended = persp_dataset.extended
        self.dataset = persp_dataset.name 
        self.output_path = config.prediction_dir
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        if self.dataset == "DICES-350":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=4)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)


    def train(self, *, validation_percentage=None):
        """Train with an instance-level validation holdout (10% by default).

        Validation selects the checkpoint; predict() still uses the test set.
        The original dataset's training, adaptation and test splits are unchanged.
        """
        if validation_percentage is None:
            validation_percentage = config.validation_percentage
        train_indices, validation_indices = _split_training_indices(
            self.training_split.annotation, validation_percentage, config.seed,
        )
        # Seed both that initialization and the subsequent training stage
        set_seed(config.seed)
        self.__add_special_tokens_to_tokenizer()
        set_seed(config.seed)
        tokenized = self.__generate_data(self.training_split)[0]
        train_dataset = tokenized.select(train_indices)
        validation_dataset = tokenized.select(validation_indices)

        labels = [
            self.training_split.annotation[index]["label"][self.label]
            for index in train_indices
        ]
        class_weights = _compute_balanced_class_weights(
            labels,
            num_labels=self.model.config.num_labels,
        )
        class_counts = np.bincount(
            np.asarray(labels, dtype=np.int64),
            minlength=self.model.config.num_labels,
        )
        print("Training class counts:", class_counts.tolist())
        print("Training class weights:", class_weights.tolist())
        print("Training annotations:", len(train_dataset),
              "| Validation annotations:", len(validation_dataset))

        training_options = {
            "seed": config.seed,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "per_device_eval_batch_size": 32,
            "save_total_limit": 1,
            **config.model_config[self.model_id],
        }
        training_args = TrainingArguments(**training_options)
        print('We will use the device:', training_args.device)

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )
        trainer.set_class_weights(class_weights)
        trainer.train()
        if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
            best_step = int(trainer.state.best_model_checkpoint.rsplit("-", 1)[-1])
            best_epoch = next(
                row["epoch"] for row in trainer.state.log_history
                if row.get("step") == best_step and "eval_loss" in row
            )
            print("Selected epoch:", best_epoch,
                  "| Validation loss:", trainer.state.best_metric)
            print("Selected checkpoint:", trainer.state.best_model_checkpoint)
        return trainer
    

    def predict(self, trainer):
        test_data, ids = self.__generate_data(self.test_split)
        predictions = trainer.predict(test_data)
        if not os.path.exists(config.prediction_dir): 
            os.makedirs(config.prediction_dir)  
        with open(config.prediction_dir+"/predictions_%s_%s_%s_%s.csv" % (self.dataset, self.named, self.user_adaptation, self.extended), "w") as fo:
            writer = csv.DictWriter(
                fo,
                fieldnames=[
                    "user_id",
                    "text_id",
                    "predictions"])
            writer.writeheader()
            for i, id in zip(enumerate(test_data), ids):
                pred = np.argmax(predictions.predictions[i[0]])
                writer.writerow({
                    "user_id": id[0],
                    "text_id": id[1],
                    "predictions": pred
                })


    def __generate_data(self, split):
        ids, texts, labels = [], [], []
        for ann in split.annotation:
            ids.append((ann["user"], ann["text"]))
            texts.append(
                self.__add_special_tokens_to_text(
                    split.users[ann["user"]],
                    split.texts[ann["text"]],
                )
            )
            labels.append(ann["label"][self.label])
        df = pd.DataFrame({"text":texts, "labels":labels})
        dt = Dataset.from_pandas(df)
        tokenized_dataset = dt.map(lambda x: self.__tokenize(x, self.tokenizer), remove_columns=['text'])
        tokenized_dataset = tokenized_dataset.with_format("torch")
        return tokenized_dataset, ids


    def __tokenize(self, x, tokenizer):
        return tokenizer(
            x["text"], 
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.max_length,
        )

    def __add_special_tokens_to_text(self, user, text):
        traits = sorted(list(self.traits.keys()))
        enriched_text = ""
        enriched_text +='<{}>'.format(user.id) + " "
        for trait in traits:
            if trait in user.traits:
                enriched_text +='<{}:{}>'.format(trait, user.traits[trait][0]) + " "
        for k in text:
            enriched_text += k + ": " + text[k] + " "
        return enriched_text
    
    
    def __add_special_tokens_to_tokenizer(self):
        special_tokens_dict = {"additional_special_tokens": []}
        new_tokens = set()
        for split in [self.training_split, self.adaptation_split, self.test_split]:
            for user in split.users:
                new_tokens.add('<{}>'.format(user))
                if self.named:
                    for dim in self.traits:
                        for trait in list(self.traits[dim]):
                            new_tokens.add('<{}:{}>'.format(dim, trait))
        special_tokens_dict['additional_special_tokens'] = list(new_tokens)
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer)) 


class CustomTrainer(Trainer):
    """ Custom Trainer class to implement a custom loss function
    """

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Focal Loss: https://arxiv.org/abs/1708.02002
        """
        gamma = 5.0
        alpha = .2
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        class_weights = torch.as_tensor(
            self.class_weights, device=logits.device, dtype=logits.dtype,
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        BCEloss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        # Focal Loss
        pt = torch.exp(-BCEloss)  # prevents nans when probability
        loss = alpha * (1 - pt) ** gamma * BCEloss
        return (loss, outputs) if return_outputs else loss
