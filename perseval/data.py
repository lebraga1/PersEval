import logging as log
from random import seed
from dataclasses import dataclass  
from functools import partial
import copy                                             

from tqdm import tqdm
import numpy as np
import yaml
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

from . import config
from .splitting import auto_split

import os
from pathlib import Path

def local_dataset_path(relative_path):
    path = Path(os.environ.get("PERSEVAL_DATA_DIR", "data")).expanduser() / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Set PERSEVAL_DATA_DIR to the folder with the local datasets.")
    return str(path)

log.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    encoding='utf-8', 
    level=log.INFO)

# Changing the random seed will change how the datasets are split
seed(config.seed)

UNKNOWN_TRAIT = "UNK"

@dataclass
class PerspectivistDataset:
    def __init__(self):
        self.name = None
        self.traits = {}
        self.labels = dict()
        self.training_set = None
        self.adaptation_set = None
        self.test_set = None
        self.user_adaptation = None
        self.named = None
        self.extended = None
        

        self.adaptation_test_user_ids = None
        self.train_user_ids = None
        self.adaptation_text_ids = None
        self.test_text_ids  = None
        # annotator column to stratify test users by (auto split)
        self.key_group = None
        self.split_report = None
        
    def available_labels(self):
        """Return the labels available for this dataset."""
        return list(self.labels.keys())

    def describe(self, verbose=False):
        """Return statistics and metadata about this dataset."""

        users = set(self.dataset[self.key_user])
        texts = set(self.dataset[self.key_text])

        instances = len(self.dataset)
        annotators = len(users)
        n_texts = len(texts)

        description = {
            "name": self.name,
            "annotators": annotators,
            "texts": n_texts,
            "instances": instances,
            "annotations_per_text": (
                instances / n_texts if n_texts else 0
            ),
            "annotations_per_annotator": (
                instances / annotators if annotators else 0
            ),
            "labels": self.available_labels(),
            "default_label": self.label,
        }

        if verbose:
            print(f"\n{description['name']}")
            print("-" * len(description["name"]))

            for key, value in description.items():
                if key != "name":
                    print(f"{key}: {value}")

        return description

    def get_splits(self, extended, user_adaptation, named, baseline=False):
        if not user_adaptation in [False, "train", "test"]:
            raise Exception(
                "Possible values are:\n \
                - False (bool): No adaptation is performed. The train and test splits are completly disjoint. The adaptation split is empty.\n \
                - 'train' (str): A small percentage (defined in the config) of the annotations by test users is contained in the training split. The adaptation split is empty. This mirrors a situation in which one can obtain a minimal amount of annotationd *before* training the system.\n \
                - 'test' (str): A small percentage (defined in the config) of the annotations by the test user is in the adapatation split. This mirrors a situation in which one has a trained system (trained on the training users, with no annotations from the test users) and want to adapt the system *after* training it.\n"
                )

        log.info("Generating Named: %s, User adaptation: %s, Extended: %s" % (named, user_adaptation, extended))

        self.user_adaptation = user_adaptation
        self.named = named
        self.extended = extended

        self.training_set = self.adaptation_set = self.test_set = None

        if (not user_adaptation and not named) and not baseline:
            raise Exception("Invalid parameter configuration (user_adaptation=False, named=False). \
                            You need to at least know the explicit user traits for test users if no annotations are available")

        self.train_user_ids, self.adaptation_test_user_ids, self.adaptation_text_ids, self.test_text_ids = self.sample_split_ids()
        train_split, adaptation_split, test_split = self._fill_splits(named)

        self._assign_splits(user_adaptation, train_split, adaptation_split, test_split)
        if not extended:
            self._remove_test_texts_from_training()

        self.check_splits(user_adaptation, extended, named)
        self.describe_splits()

    def sample_split_ids(self):
        user_groups = None
        if self.key_group:
            user_groups = dict(zip(self.dataset[self.key_user], self.dataset[self.key_group]))
        plan = auto_split(self.dataset[self.key_user], self.dataset[self.key_text],
                          user_groups=user_groups, seed=config.seed, **config.auto_split)
        self.split_report = plan.report
        log.info("Split: reserve %.0f%% of the texts for training, %d test users, test share %.0f%%",
                 100 * plan.report["reserve"], plan.report["test_users"], 100 * plan.report["test_share"])
        if not plan.report["target_met"]:
            log.warning("Split could not bring the test share down to %.0f%% (got %.0f%%)",
                        100 * plan.report["target_test_share"], 100 * plan.report["test_share"])
        return plan.train_user_ids, plan.test_user_ids, plan.adaptation_text_ids, plan.test_text_ids

    def read_text(self, row):
        raise NotImplementedError

    def read_labels(self, row):
        raise NotImplementedError

    def read_traits(self, row):
        return {}

    def _fill_splits(self, named):
        train_users, test_users = set(self.train_user_ids), set(self.adaptation_test_user_ids)
        adaptation_texts, test_texts = set(self.adaptation_text_ids), set(self.test_text_ids)
        train_split = PerspectivistSplit(type="train")
        adaptation_split = PerspectivistSplit(type="adaptation")
        test_split = PerspectivistSplit(type="test")

        for row in tqdm(self.dataset):
            user_id, text_id = row[self.key_user], row[self.key_text]

            memberships = []
            if user_id in train_users:
                memberships.append((train_split, True))
            if user_id in test_users:
                memberships.append((adaptation_split, text_id in adaptation_texts))
                memberships.append((test_split, text_id in test_texts))

            for split, annotated in memberships:
                # Read user
                if not user_id in split.users:
                    split.users[user_id] = User(user_id)
                user = split.users[user_id]
                # Read traits only if named
                if named:
                    self._record_traits(user, row)
                if annotated:
                    self._record_annotation(split, user, text_id, row)

        return train_split, adaptation_split, test_split

    def _record_traits(self, user, row):
        for dimension, value in self.read_traits(row).items():
            user.traits[dimension] = [value]
            if value != UNKNOWN_TRAIT:
                self.traits.setdefault(dimension, set()).add(value)

    def _record_annotation(self, split, user, text_id, row):
        # Read text
        split.texts[text_id] = self.read_text(row)
        # Read annotation
        labels = self.read_labels(row)
        split.record_annotation(self, row, labels)
        # Read labels by text
        if not text_id in split.annotation_by_text:
            split.annotation_by_text[text_id] = []
        split.annotation_by_text[text_id].append({"user": user, "label": dict(labels)})
        for label, value in labels.items():
            self.labels[label].add(value)

    def _assign_splits(self, user_adaptation, train_split, adaptation_split, test_split):
        if user_adaptation == False:
            # You know nothing about the new test users except their explicit traits
            # You cannot use their adaptation annotations
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

        elif user_adaptation == "train":
            # You can use a few annotations by test users at training time
            # These annotations are directly included in the training split,
            # the adaptation split is empty

            # Train + Adapt in the train set
            train_split.users = {**train_split.users, **adaptation_split.users}
            train_split.texts = {**train_split.texts, **adaptation_split.texts}

            train_split.merge_splits(adaptation_split)

            for t_id in adaptation_split.annotation_by_text.keys():
                if t_id in train_split.annotation_by_text:
                    # add the annotations
                    train_split.annotation_by_text[t_id] = train_split.annotation_by_text[t_id] + adaptation_split.annotation_by_text[t_id]
                else:
                    train_split.annotation_by_text[t_id] = adaptation_split.annotation_by_text[t_id]
            self.training_set = train_split
            self.adaptation_set = PerspectivistSplit(type="adaptation")
            self.test_set = test_split

        elif user_adaptation == "test":
            # You CANNOT use any test annotations at training time
            # However, you can use a few annotations to adapt your trained system to test users
            # These adaptation annotations from test users are in the adaptation split,
            self.training_set = train_split
            self.adaptation_set = adaptation_split
            self.test_set = test_split

    def _remove_test_texts_from_training(self):
        strict_train_split = self.training_set
        strict_train_split.annotation_by_text = {t:self.training_set.annotation_by_text[t] for t in self.training_set.annotation_by_text if t not in self.test_set.annotation_by_text}
        # Filter annotations
        strict_train_split.make_strict(self)

        # Filter texts
        strict_train_split.texts = {k:self.training_set.texts[k] for k in self.training_set.texts if not k in self.test_set.texts}
        self.training_set = strict_train_split

    def describe_splits(self):
        if not self.training_set.users:
            raise Exception("You need to first choose a task through "+self.name+".get_splits(extended, user_adaptation, named,baseline)")
        
        print("--- Unique users ---")
        print("Train set: %d" % len(self.training_set.users))
        if len(self.adaptation_set.users):
            print("Adaptation set: %d" % len(self.adaptation_set.users))
        print("Test set: %d" % len(self.test_set.users))
        print()
        print("--- Unique texts ---")
        print("Train set: %d" % len(self.training_set.annotation_by_text))
        if len(self.adaptation_set.annotation_by_text):
            print("Adaptation set: %d" % len(self.adaptation_set.annotation_by_text))
        print("Test set: %d" % len(self.test_set.annotation_by_text))
        print()
        print("--- Instances (text + user) ---")
        print("Train set: %d" % len(self.training_set.annotation))
        if len(self.adaptation_set.annotation):
            print("Adaptation set: %d" % len(self.adaptation_set.annotation))
        print("Test set: %d" % len(self.test_set.annotation))
        print()

        print("--- User-text train/adaptation/test distribution ---")
        number_user_adapt_texts, number_user_test_texts = [], []
        for u in self.test_set.users:
            user_adapt_texts, user_test_texts = 0, 0
            for i in self.adaptation_set.annotation:
                if i['user']==u:
                    user_adapt_texts+=1
            for i in self.test_set.annotation:
                if i['user']==u:
                    user_test_texts+=1
            number_user_adapt_texts.append(user_adapt_texts)
            number_user_test_texts.append(user_test_texts)
        print("The mean number of texts per users in the test set is %.3f" % np.mean(number_user_test_texts))
        
        if self.adaptation_set != PerspectivistSplit(type=="adaptation"):
            percentage_in_adapt = [d/t for d, t in zip(number_user_adapt_texts, number_user_test_texts)]
            print("The mean percentage of texts per users in the adaptation set is %.3f" % np.mean(percentage_in_adapt))
            print("The mean number of texts per users in the adaptation set is %.3f" % np.mean(number_user_adapt_texts))
            print("The min percentage of texts per users in the adaptation set is %.3f (i.e. %.0f instances)" % (np.min(percentage_in_adapt), np.min(number_user_adapt_texts)))
            print("The max percentage of texts per users in the adaptation set is %.3f (i.e. %.0f instances)" % (np.max(percentage_in_adapt), np.max(number_user_adapt_texts)))


    def check_splits(self, user_adaptation, extended, named):
        if user_adaptation == False:
            # The adaptation set is empty
            assert self.adaptation_set == PerspectivistSplit(type="adaptation")
        
        # Users
        if user_adaptation == False or user_adaptation == "test":
            # Train and adapt + test users have no overlap
            assert set(self.training_set.users).intersection(set(self.adaptation_set.users)) == set()
            assert set(self.training_set.users).intersection(set(self.test_set.users)) == set()
        if user_adaptation == "train":
            # All test users are also in the training set
            assert set(self.training_set.users).union(set(self.test_set.users)) == set(self.training_set.users) 
        
        # Texts
        # adapt and test texts have no overlap
        if user_adaptation == "test":
            assert set(self.adaptation_set.texts).intersection(set(self.test_set.texts)) == set()  

        for u in self.test_set.users:
            user_train_texts, user_adapt_texts, user_test_texts = 0, 0, 0
            for i in self.training_set.annotation:
                if i['user']==u:
                    user_train_texts+=1 
            for i in self.adaptation_set.annotation:
                if i['user']==u:
                    user_adapt_texts+=1
            for i in self.test_set.annotation:
                if i['user']==u:
                    user_test_texts+=1
            
        if user_adaptation == "train" and extended:
            # All test users and corresponding training users must have at least one annotation
            assert user_train_texts != 0
            assert user_test_texts != 0
        if user_adaptation == "test":
            # All test users and corresponding adapt users must have at least one annotation
            assert user_adapt_texts != 0
            assert user_test_texts != 0

        if not extended:
            # Train and test text have no overlap
            assert set(self.training_set.texts).intersection(set(self.test_set.texts)) == set()  
        log.info("All tests passed")


@dataclass
class Instance:
    def __init__(self, instance_id, instance_text, user, label):
        self.instance_id = instance_id
        self.instance_text = instance_text
        self.user = user
        self.label = label

    def __repr__(self):
        return f"{self.instance_id} {self.user} {self.label}"


@dataclass
class PerspectivistSplit:
    def __init__(self, type=None):
        self.type = type # Str, e.g., train, adaptation, test
        self.users = dict() 
        self.texts = dict()
        self.annotation = [] #user, text, label
        self.annotation_by_text = dict()
        self._seen = set()

    def __iter__(self):
        #for (user, instance_id), label in self.annotation.items()
        for row in self.annotation:
                yield Instance(
                    row['text'], 
                    self.texts[row['text']],
                    self.users[row['user']],
                    row['label'])

    def __len__(self):
        return len(self.annotation)

    def record_annotation(self, dataset, row, label):
        key = (row[dataset.key_user], row[dataset.key_text])
        if key in self._seen:
            return
        self._seen.add(key)
        self.annotation.append({'user': row[dataset.key_user], 'text': row[dataset.key_text], 'label' : label})

    def merge_splits(self,other_splits):
        assert not (self._seen & other_splits._seen)
        self.annotation = self.annotation + other_splits.annotation
        self._seen = self._seen | other_splits._seen

    def make_strict(self,dataset):
        test_text = dataset.test_set.annotation_by_text
        self.annotation = [
            a for a in dataset.training_set.annotation if a['text'] not in test_text
        ]
        self._seen = {(a['user'], a['text']) for a in self.annotation}

@dataclass
class User:
    def __init__(self, user):
        self.id = user
        self.traits = dict()

    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other):
        if self.id == other.id and self.traits == other.traits:
            return True
        else:
            return False


@dataclass
class ConfigDataset(PerspectivistDataset):
    REQUIRED = ["source", "annotator_column", "text_id_column", "text_columns", "labels"]
    LABEL_PREFIX = "__label__"

    def __init__(self, name, label=None):
        super(ConfigDataset, self).__init__()
        if name not in config.dataset_specs:
            raise ValueError(f"No dataset '{name}' in config.dataset_specs. Available: {', '.join(config.dataset_specs)}")
        self.name = name
        self.spec = spec = config.dataset_specs[name]
        self.check_spec(name, spec)
        self.label = label or next(iter(spec["labels"]))
        if self.label not in spec["labels"]:
            raise ValueError(f"{name} has no label '{self.label}'. Available labels: {', '.join(spec['labels'])}")
        self.missing_values = set(spec.get("missing_values", []))

        df = self._load(spec["source"])
        if "prepare" in spec:
            df = spec["prepare"](df)
        self._check_columns(df)
        df = df.dropna(subset=[c for rule in spec["labels"].values() for c in self._label_columns(rule)])
        text_columns = list(spec["text_columns"].values())
        df = df.dropna(subset=text_columns, how="all")
        df[text_columns] = df[text_columns].fillna("")

        self.key_user = spec["annotator_column"]
        self.key_text = spec["text_id_column"]
        if isinstance(self.key_text, list):
            df["text_id"] = df[self.key_text].astype(str).agg("_".join, axis=1)
            self.key_text = "text_id"
        if spec.get("same_text_columns"):
            df["text_id"] = df.groupby(spec["same_text_columns"], dropna=False)[self.key_text].transform("min")
            self.key_text = "text_id"
        self.key_group = spec.get("group_column")
        if spec.get("drop_duplicate_annotations"):
            df = df.drop_duplicates([self.key_user, self.key_text])

        for label_name, rule in spec["labels"].items():
            if "positive_if_any" in rule:
                values = (df[rule["positive_if_any"]] > 0).any(axis=1).astype(int).tolist()
            else:
                values = [self._label_value(label_name, rule, v) for v in df[rule["column"]]]
            df[self.LABEL_PREFIX + label_name] = values
            self.labels[label_name] = set()
        # empty cells become None, as in a Hugging Face dataset
        for column in df.columns[df.isna().any()]:
            df[column] = df[column].astype(object).where(df[column].notna(), None)
        self.dataset = Dataset.from_pandas(df, preserve_index=False)
        self._check_annotations(df)

    @classmethod
    def check_spec(cls, name, spec):
        if not isinstance(spec, dict):
            raise ValueError(f"{name} must be described by keys and values, e.g. 'source', 'labels'.")
        missing = [key for key in cls.REQUIRED if key not in spec]
        if missing:
            raise ValueError(f"{name} has no {', '.join(missing)}.")
        if not isinstance(spec["source"], dict):
            raise ValueError(f"The source of {name} must name its type, e.g. {{'csv': path}} or {{'huggingface': repository}}.")

    def _load(self, source):
        if "huggingface" in source:
            data = load_dataset(source["huggingface"])
            splits = source.get("split", "train")
            splits = [splits] if isinstance(splits, str) else splits
            return concatenate_datasets([data[split] for split in splits]).to_pandas()
        if "csv" in source:
            return load_dataset("csv", data_files=self._path(source["csv"]),
                                delimiter=source.get("separator", ","))["train"].to_pandas()
        if "disk" in source:
            return load_from_disk(self._path(source["disk"])).to_pandas()
        raise ValueError(f"The source of {self.name} needs a 'huggingface', 'csv' or 'disk' key.")

    @staticmethod
    def _path(path):
        return path if "://" in path or os.path.isabs(path) else local_dataset_path(path)

    def _check_columns(self, df):
        spec = self.spec
        text_id = spec["text_id_column"]
        needed = {"annotator_column": [spec["annotator_column"]],
                  "text_id_column": text_id if isinstance(text_id, list) else [text_id],
                  "same_text_columns": spec.get("same_text_columns", []),
                  "text_columns": list(spec["text_columns"].values()),
                  "labels": [c for rule in spec["labels"].values() for c in self._label_columns(rule)],
                  "traits": [rule["column"] for rule in spec.get("traits", {}).values()],
                  "group_column": [spec["group_column"]] if spec.get("group_column") else []}
        unknown = [f"'{column}' ({key})" for key, columns in needed.items() for column in columns if column not in df.columns]
        if unknown:
            raise ValueError(f"{self.name}: no column {', '.join(unknown)} in the data. "
                             f"Available columns: {', '.join(map(str, df.columns))}")

    def _check_annotations(self, df):
        pairs = list(zip(df[self.key_user], df[self.key_text]))
        repeated = len(pairs) - len(set(pairs))
        if repeated:
            log.warning("%s: %d annotations repeat an (annotator, text) pair. "
                        "Set drop_duplicate_annotations if this is expected.", self.name, repeated)
        if len({text for _, text in pairs}) == len(pairs):
            log.warning("%s: no text has more than one annotation. Does '%s' number the rows "
                        "instead of identifying the texts? If so, list the columns that identify "
                        "a text in same_text_columns.", self.name, self.key_text)
        for dimension, rule in self.spec.get("traits", {}).items():
            values = {self._trait_value(rule, v) for v in df[rule["column"]].unique()} - {None, UNKNOWN_TRAIT}
            by_case = {}
            for value in values:
                by_case.setdefault(str(value).strip().lower(), set()).add(value)
            clashes = [sorted(map(str, group)) for group in by_case.values() if len(group) > 1]
            if clashes:
                log.warning("%s: trait '%s' has values that differ only in case or spaces: %s. "
                            "Add \"lowercase\": True or a 'values' table.", self.name, dimension, clashes)

    @staticmethod
    def _label_columns(rule):
        return rule["positive_if_any"] if "positive_if_any" in rule else [rule["column"]]

    def _label_value(self, label_name, rule, value):
        if "values" in rule:
            if value not in rule["values"]:
                raise ValueError(f"{self.name}: value {value!r} of column '{rule['column']}' is not in "
                                 f"the 'values' of label '{label_name}'.")
            return rule["values"][value]
        if "positive_if_at_least" in rule:
            return int(value >= rule["positive_if_at_least"])
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: value {value!r} of label '{label_name}' is not a number. "
                             "Add a 'values' table or 'positive_if_at_least'.") from None
        if not number.is_integer():
            raise ValueError(f"{self.name}: value {value!r} of label '{label_name}' is not a whole number. "
                             "Labels are classes 0, 1, ...: add 'positive_if_at_least' or a 'values' table.")
        return int(number)

    def _trait_value(self, rule, value):
        if value is None or value in self.missing_values:
            return None if self.spec.get("missing_traits") == "skip" else UNKNOWN_TRAIT
        try:
            if rule.get("age_to_generation"):
                year = rule["age_to_generation"]
                return age_to_generation(int(value), 2022 if year is True else year)
            if rule.get("integer"):
                value = int(value)
        except ValueError:
            return UNKNOWN_TRAIT
        if rule.get("lowercase"):
            value = str(value).lower()
        return rule["values"].get(value, UNKNOWN_TRAIT) if "values" in rule else value

    def read_text(self, row):
        return {name: row[column] for name, column in self.spec["text_columns"].items()}

    def read_labels(self, row):
        return {name: row[self.LABEL_PREFIX + name] for name in self.labels}

    def read_traits(self, row):
        if not self.spec.get("traits"):
            raise ValueError(f"{self.name} has no annotator metadata, so named=True is not available.")
        traits = {}
        for dimension, rule in self.spec["traits"].items():
            value = self._trait_value(rule, row[rule["column"]])
            if value is not None:
                traits[dimension] = value
        return traits


def age_to_generation(age, year=2022):
    if age >= year - 1964:
        return "Boomer"
    if age >= year - 1980:
        return "GenX"
    if age >= year - 1996:
        return "GenY"
    return "GenZ"


_DATASETS = {}

_CLASS_NAMES = {"Epic": "EPIC", "Brexit": "BREXIT", "DICES": "DICES", "MHS": "MHS", "MD": "MD",
                "ConvAbuse": "CONVABUSE", "TAS": "TAS", "CSC": "CSC"}

def __getattr__(name):
    if name in _CLASS_NAMES:
        return partial(ConfigDataset, _CLASS_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def _all_datasets():
    specs = {name.lower(): partial(ConfigDataset, name) for name in config.dataset_specs}
    return {**specs, **_DATASETS}

def add_dataset(path):
    """Add the datasets described in a YAML file and return their names."""
    path = Path(path).expanduser().resolve()
    with open(path, encoding="utf-8") as f:
        specs = yaml.safe_load(f)
    if not isinstance(specs, dict) or "source" in specs:
        raise ValueError(f"{path} must describe each dataset under its name: a line 'MYDATA:' "
                         "followed by the indented keys.")
    for name, spec in specs.items():
        ConfigDataset.check_spec(name, spec)
        for key in ("csv", "disk"):
            if key in spec["source"] and "://" not in spec["source"][key]:
                spec["source"][key] = str(path.parent / Path(spec["source"][key]).expanduser())
    config.dataset_specs.update(specs)
    config.dataset_label.update({name: next(iter(spec["labels"])) for name, spec in specs.items()})
    return list(specs)

def available_datasets():
    """Return the names of all available datasets."""
    return list(_all_datasets())

def download(dataset_name, label=None):
    """
    Download and initialize a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Case-insensitive.

    label : str, optional
        Label to use. If omitted, the dataset's default label is used.

    Returns
    -------
    PerspectivistDataset
        Initialized dataset object.
    """
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string")

    dataset_name = dataset_name.lower()

    datasets = _all_datasets()

    if dataset_name not in datasets:
        available = ", ".join(datasets)
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    return datasets[dataset_name](label)

def download_and_split(
    dataset_name,
    label=None,
    user_adaptation=False,
    extended=False,
    named=False,
    baseline=False,
):
    """
    Download a dataset and generate its requested task split.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Case-insensitive.

    label : str, optional
        Label to use. If omitted, the dataset's default label is used.

    user_adaptation : bool or str
        False, "train", or "test".

    extended : bool
        Whether to allow training/test text overlap.

    named : bool
        Whether to use named user representations.

    baseline : bool
        Whether to allow the baseline configuration.

    Returns
    -------
    PerspectivistDataset
        Dataset with training, adaptation and test splits.
    """

    dataset = download(dataset_name, label=label)

    dataset.get_splits(
        extended=extended,
        user_adaptation=user_adaptation,
        named=named,
        baseline=baseline,
    )

    return dataset

def describe_datasets(verbose=False):
    """Describe all available datasets."""

    descriptions = {}

    for name in _all_datasets():
        dataset = download(name)
        descriptions[name] = dataset.describe(verbose=verbose)

    return descriptions