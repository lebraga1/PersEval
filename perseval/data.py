import logging as log
from random import seed, sample
from dataclasses import dataclass  
import copy                                             

from tqdm import tqdm
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from sklearn.model_selection import train_test_split

from . import config
from .splitting import auto_split

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
        # Optional annotator column used to stratify the test users (auto strategy)
        self.key_group = None
        self.split_strategy = None
        self.split_report = None
        
    def available_labels(self):
        """Return the labels available for this dataset."""
        return list(self.labels.keys())

    def describe(self, verbose=False):
        """Return statistics and metadata about this dataset.

        Parameters
        ----------
        verbose : bool
            If True, print a human-readable summary.

        Returns
        -------
        dict
            Dataset description and statistics.
        """

        users = set()
        texts = set()

        for row in self.dataset:
            users.add(row[self.key_user])
            texts.add(row[self.key_text])

        instances = len(self.dataset)
        annotators = len(users)
        n_texts = len(texts)

        description = {
            "name": self.name,
            "task": config.dataset_info[self.name]["task"],
            "annotators": annotators,
            "texts": n_texts,
            "instances": instances,
            "annotations_per_text": (
                instances / n_texts if n_texts else 0
            ),
            "annotations_per_annotator": (
                instances / annotators if annotators else 0
            ),
            "source": config.dataset_info[self.name]["source"],
            "label_type": config.dataset_info[self.name]["label_type"],
            "positive_class": config.dataset_info[self.name]["positive_class"],
            "metadata": config.dataset_info[self.name]["metadata"],
            "labels": self.available_labels(),
            "default_label": self.label,
        }

        if verbose:
            print(f"\n{description['name']}")
            print("-" * len(description["name"]))

            print(f"Task:                  {description['task']}")
            print(f"Annotators:            {description['annotators']}")
            print(f"Texts:                 {description['texts']}")
            print(f"Instances:             {description['instances']}")
            print(
                f"Annotations/text:      "
                f"{description['annotations_per_text']:.2f}"
            )
            print(
                f"Annotations/annotator: "
                f"{description['annotations_per_annotator']:.2f}"
            )
            print(f"Source:                {description['source']}")
            print(f"Label type:            {description['label_type']}")
            print(f"Positive class:        {description['positive_class']}")
            print(f"Labels:                {description['labels']}")
            print(f"Default label:         {description['default_label']}")
            print(f"Metadata:              {description['metadata']}")

        return description

    def get_splits(self, extended, user_adaptation, named, baseline=False, split_strategy=None):
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
        self.split_strategy = split_strategy or self.default_split_strategy()

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

    def default_split_strategy(self):
        # Datasets whose split is fixed in the config keep it, so that published
        # results stay reproducible. Any other dataset is split automatically.
        return config.dataset_specific_splits.get(self.name, {}).get("strategy", "auto")

    def sample_split_ids(self):
        if self.split_strategy == "legacy":
            return self._legacy_sample_split_ids()
        if self.split_strategy == "auto":
            return self._auto_sample_split_ids()
        raise ValueError("Unknown split strategy %r. Possible values are 'auto' and 'legacy'." % self.split_strategy)

    def _auto_sample_split_ids(self):
        user_groups = None
        if self.key_group:
            user_groups = dict(zip(self.dataset[self.key_user], self.dataset[self.key_group]))
        params = {**config.auto_split, **config.dataset_specific_splits.get(self.name, {}).get("auto_split", {})}
        plan = auto_split(self.dataset[self.key_user], self.dataset[self.key_text],
                          user_groups=user_groups, seed=config.seed, **params)
        self.split_report = plan.report
        log.info("Auto split: reserve %.0f%% of the texts for training, %d test users, test share %.0f%%",
                 100 * plan.report["reserve"], plan.report["test_users"], 100 * plan.report["test_share"])
        if not plan.report["target_met"]:
            log.warning("Auto split could not bring the test share down to %.0f%% (got %.0f%%)",
                        100 * plan.report["target_test_share"], 100 * plan.report["test_share"])
        return plan.train_user_ids, plan.test_user_ids, plan.adaptation_text_ids, plan.test_text_ids

    def _legacy_sample_split_ids(self):
        user_ids = set(list(self.dataset[self.key_user]))
        percentages = config.dataset_specific_splits[self.name]

        # Sample adapt+test users
        seed(config.seed)
        adaptation_test_user_ids = sample(sorted(user_ids), int(len(user_ids) * percentages["user_based_split_percentage"]))
        test_users = set(adaptation_test_user_ids)
        train_user_ids = [u for u in user_ids if not u in test_users]

        adapt_test_text_id = [t_id for t_id, user in zip(self.dataset[self.key_text], self.dataset[self.key_user]) if user in test_users]
        seed(config.seed)
        adaptation_text_ids = sample(sorted(adapt_test_text_id), int(len(adapt_test_text_id) * percentages["text_based_split_percentage"]))
        adaptation_texts = set(adaptation_text_ids)
        test_text_ids = [t_id for t_id in adapt_test_text_id if t_id not in adaptation_texts]
        return train_user_ids, adaptation_test_user_ids, adaptation_text_ids, test_text_ids

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
class Epic(PerspectivistDataset):
    def __init__(self, label):
        super(Epic, self).__init__()
        self.name = "EPIC"
        self.label = label
        dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
        self.dataset = dataset["train"]
        self.dataset = self.dataset.map(lambda x: {"label": config.label_map[label][x["label"]]})
        self.label = config.dataset_label[self.name]
        self.labels[label] = set()
        self.key_user = 'user'
        self.key_text = 'id_original'

    def read_text(self, row):
        return {"post": row['parent_text'], "reply": row['text']}

    def read_labels(self, row):
        return {self.label: row['label']}

    def read_traits(self, row):
        traits = {"Gender": row['Sex'], "Nationality": row['Nationality']}
        try:
            traits["Generation"] = self.__convert_age(int(row['Age']))
        except ValueError:
            traits["Generation"] = UNKNOWN_TRAIT
        return traits

    def __convert_age(self, age):
        """Function to convert the age, represented as an integer,
        into a label, according to Table 1 in the paper
        'EPIC: Multi-Perspective Annotation of a Corpus of Irony'
        https://aclanthology.org/2023.acl-long.774/
        """
        if age >= 58:
            return "Boomer"
        elif age >= 42:
            return "GenX"
        elif age >= 26:
            return "GenY"
        else:
            return "GenZ"


@dataclass
class Brexit(PerspectivistDataset):
    def __init__(self):
        super(Brexit, self).__init__()
        self.name = "BREXIT"
        dataset = load_dataset("silvia-casola/BREXIT")
        self.dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        labels = ["hs", "offensiveness", "aggressiveness", "stereotype"]
        self.label = config.dataset_label[self.name]
        for label in labels:
            self.labels[label] = set()
        self.key_user = 'annotator_id'
        self.key_text = 'instance_id'
        self.key_group = 'annotator_group'

    def _legacy_sample_split_ids(self):
        users_group_ids = self.dataset.to_pandas()[["annotator_id", "annotator_group"]].drop_duplicates()
        user_ids = list(users_group_ids['annotator_id'])
        user_group = list(users_group_ids['annotator_group']) 

        # Sample adapt+test users
        seed(config.seed)
        train_user_ids, adaptation_test_user_ids = train_test_split(user_ids,
                                                        test_size=config.dataset_specific_splits[self.name]["user_based_split_percentage"],
                                                        random_state=config.seed,
                                                        shuffle=True, stratify=user_group)
        seed(config.seed)
        all_text_ids = list(set(self.dataset["instance_id"]))
        train_text_ids = sample(sorted(all_text_ids), int(len(all_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_train"]))
        adaptation_test_text_ids = [t for t in all_text_ids if t not in train_text_ids]
        adaptation_text_ids = sample(sorted(adaptation_test_text_ids), int(len(adaptation_test_text_ids)*config.dataset_specific_splits[self.name]["text_based_split_percentage_dev"]))
        test_text_ids = [t for t in adaptation_test_text_ids if t not in adaptation_text_ids]
        return train_user_ids, adaptation_test_user_ids, adaptation_text_ids, test_text_ids

    def read_text(self, row):
        return {"tweet": row['tweet']}

    def read_labels(self, row):
        return {label: row[label] for label in self.labels}

    def read_traits(self, row):
        return {"Group": row['annotator_group']}


@dataclass
class DICES(PerspectivistDataset):
    def __init__(self, label):
        super(DICES, self).__init__()
        self.name = "DICES"
        self.label = label
        self.dataset = load_from_disk("data/diverse_safety_adversarial_dialog_350_enhanced")
        self.dataset = self.dataset.map(lambda x: {label: config.label_map[label][x[label]]})
        self.labels[label] = set()
        self.key_user = 'rater_id'
        self.key_text = 'text_id'

    def read_text(self, row):
        return {"context": row['context'], "reply": row['response']}

    def read_labels(self, row):
        return {self.label: row[self.label]}

    def read_traits(self, row):
        return {"Gender": row['rater_gender'],
                "Generation": row['rater_age'],
                "Race": row['rater_race'],
                "Education": row['rater_education']}

@dataclass
class MHS(PerspectivistDataset):
    EDUCATION = {"college_grad_aa":"educ-high","college_grad_ba":"educ-high","high_school_grad":"educ-low","masters":"educ-high","phd":"educ-high","professional_degree":"educ-low","some_college":"educ-low","some_high_school":"educ-low"}
    IDEOLOGY = {"conservative":"conservative","extremely_conservative":"conservative","extremely_liberal":"liberal","liberal":"liberal","neutral":"neutral","no_opinion":"neutral","slightly_conservative":"conservative","slightly_liberal":"liberal"}
    INCOME = {"100k-200k":"income-high","10k-50k":"income-low","<10k":"income-low",">200k":"income-high","50k-100k":"income-high"}

    def __init__(self, label):
        super(MHS, self).__init__()
        self.name = "MHS"
        self.label = label
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        self.dataset = dataset["train"]
        self.dataset = self.dataset.map(lambda x: {"hateful": 1 if x["hatespeech"] > 0 else 0})
        self.labels[label] = set()
        self.key_user = 'annotator_id'
        self.key_text = 'comment_id'

    def read_text(self, row):
        return {"post": row['text']}

    def read_labels(self, row):
        return {"hateful": 1 if row["hatespeech"] > 0 else 0}

    def read_traits(self, row):
        traits = {}
        # Education
        if row['annotator_educ'] is not None:
            traits["Education"] = self.EDUCATION[row['annotator_educ']]
        # Gender
        traits["Gender"] = row['annotator_gender']
        # Ideology
        if row['annotator_ideology'] is not None:
            traits["Ideology"] = self.IDEOLOGY[row['annotator_ideology']]
        # Income
        if row['annotator_income'] is not None:
            traits["Income"] = self.INCOME[row['annotator_income']]
        # Age
        if row['annotator_age'] is not None:
            traits["Age"] = self.__convert_age(int(row['annotator_age']))
        return traits

    def __convert_age(self, age):
        """Function to convert the age, represented as an integer,
        into a label. 
        The annotations were done in 2020, so the labels are based on 2020.
        """
        if age >= 56:
            return "Boomer"
        elif age >= 40:
            return "GenX"
        elif age >= 24:
            return "GenY"
        else:
            return "GenZ"


@dataclass
class MD(PerspectivistDataset):
    def __init__(self, label):
        super(MD, self).__init__()
        self.name = "MD"
        self.label = label
        dataset = load_dataset("csv", data_files="data/MD-Agreement_dataset/MD_agreement.csv")
        self.dataset = dataset["train"]
        self.labels[label] = set()
        self.key_user = 'annotators'
        self.key_text = 'text_id'

    def read_text(self, row):
        return {"text": row['text']}

    def read_labels(self, row):
        return {"offensiveness": row['annotations']}

    def read_traits(self, row):
        raise Exception("Invalid parameter configuration. \
                            This dataset does not contain any information about the annotators.")


@dataclass
class TAS(PerspectivistDataset):
    EDUCATION = {1: "educ-low", 2: "educ-low", 3: "educ-low", 4: "educ-high", 5: "educ-high", 6: "educ-high"}
    PARTY = {1: "Republican", 2: "Democrat", 3: "Independent"}

    def __init__(self, label="hate_speech"):
        super(TAS, self).__init__()
        self.name = "TAS"
        self.label = label
        dataset = load_dataset("soda-lmu/tweet-annotation-sensitivity-2")
        self.dataset = dataset["train"].filter(
            lambda x: x["hate_speech"] is not None and x["offensive_language"] is not None)
        for name in ["hate_speech", "offensive_language"]:
            self.labels[name] = set()
        self.key_user = 'annotator_id'
        self.key_text = 'tweet_id'

    def read_text(self, row):
        return {"tweet": row['tweet_hashed']}

    def read_labels(self, row):
        return {name: int(row[name]) for name in self.labels}

    def read_traits(self, row):
        traits = {"Condition": row['condition']}
        if row['age'] is not None:
            traits["Generation"] = self.__convert_age(int(row['age']))
        if row['education'] is not None:
            traits["Education"] = self.EDUCATION[int(row['education'])]
        if row['party_affiliation'] is not None:
            traits["Party"] = self.PARTY[int(row['party_affiliation'])]
        return traits

    def __convert_age(self, age):
        if age >= 58:
            return "Boomer"
        elif age >= 42:
            return "GenX"
        elif age >= 26:
            return "GenY"
        else:
            return "GenZ"


@dataclass
class ConvAbuse(PerspectivistDataset):
    URL = "https://raw.githubusercontent.com/amandacurry/convabuse/c0a9469f48956e868276c605d499010ea3c7c0d0/1_full/ConvAbuseEMNLPfull.csv"
    EXCHANGE = ["prev_agent", "prev_user", "agent", "user"]

    def __init__(self, label="abuse"):
        super(ConvAbuse, self).__init__()
        self.name = "CONVABUSE"
        self.label = label
        df = load_dataset("csv", data_files=self.URL)["train"].to_pandas()
        df[self.EXCHANGE] = df[self.EXCHANGE].fillna("")
        # example_no numbers the rows: the same exchange, labelled by several annotators, has several numbers
        df["text_id"] = df.groupby(["conv_id"] + self.EXCHANGE, dropna=False)["example_no"].transform("min")
        # Some exchanges were labelled twice by the same annotator to measure intra-annotator agreement
        df = df.drop_duplicates(["annotator_id", "text_id"])
        # Abusive is -1 (mildly) to -3 (strongly); 0 (ambiguous) and 1 (non-abusive) are not
        df[label] = (df[["is_abuse.-1", "is_abuse.-2", "is_abuse.-3"]].sum(axis=1) > 0).astype(int)
        self.dataset = Dataset.from_pandas(df, preserve_index=False)
        self.labels[label] = set()
        self.key_user = 'annotator_id'
        self.key_text = 'text_id'

    def read_text(self, row):
        return {name: row[name] for name in self.EXCHANGE}

    def read_labels(self, row):
        return {self.label: row[self.label]}

    def read_traits(self, row):
        raise ValueError("ConvAbuse only describes its annotators as a group, so named=True is not available.")


@dataclass
class CSC(PerspectivistDataset):
    URL = "https://raw.githubusercontent.com/CoPsyN/CSC/f11c92b8d54686f64cf1d92766cef07201efb986/data_full_w_demo.tsv"
    GENDER = {"female": "female", "male": "male", "non-binary": "non-binary"}
    MISSING = {None, "DATA_EXPIRED"}

    def __init__(self, label="sarcasm"):
        super(CSC, self).__init__()
        self.name = "CSC"
        self.label = label
        df = load_dataset("csv", data_files=self.URL, delimiter="\t")["train"].to_pandas()
        # A text is the response of one speaker to one context
        df["text_id"] = df["global_context_name"].astype(str) + "_" + df["global_speaker"].astype(str)
        df = df.drop_duplicates(["global_evaluator", "text_id"])
        # 1-6 Likert scale, binary-coded as in Jang and Frassinelli (2024): 4 (somewhat) or more is sarcastic
        df[label] = (df["sarcasm_score_by_evaluator"] >= 4).astype(int)
        df = df.astype(object).where(df.notna(), None)
        self.dataset = Dataset.from_pandas(df, preserve_index=False)
        self.labels[label] = set()
        self.key_user = 'global_evaluator'
        self.key_text = 'text_id'

    def read_text(self, row):
        return {"context": row['context_text'], "response": row['response_text']}

    def read_labels(self, row):
        return {self.label: row[self.label]}

    def read_traits(self, row):
        traits = {"Gender": self.GENDER.get(str(row['evaluator_sex']).lower(), UNKNOWN_TRAIT)}
        age = row['evaluator_age']
        traits["Generation"] = UNKNOWN_TRAIT if age in self.MISSING else self.__convert_age(int(age))
        ethnicity = row['evaluator_ethnicity']
        traits["Ethnicity"] = UNKNOWN_TRAIT if ethnicity in self.MISSING else ethnicity
        return traits

    def __convert_age(self, age):
        if age >= 58:
            return "Boomer"
        elif age >= 42:
            return "GenX"
        elif age >= 26:
            return "GenY"
        else:
            return "GenZ"


_DATASETS = {
    "epic": lambda: Epic(config.dataset_label["EPIC"]),
    "brexit": Brexit,
    "dices": lambda: DICES(config.dataset_label["DICES"]),
    "mhs": lambda: MHS(config.dataset_label["MHS"]),
    "md": lambda: MD(config.dataset_label["MD"]),
    "tas": lambda: TAS(config.dataset_label["TAS"]),
    "convabuse": lambda: ConvAbuse(config.dataset_label["CONVABUSE"]),
    "csc": lambda: CSC(config.dataset_label["CSC"]),
}

def available_datasets(verbose=False):
    """Return information about the available datasets.

    Parameters
    ----------
    verbose : bool
        If True, print the dataset information in a readable format.

    Returns
    -------
    dict
        Information about the available datasets.
    """

    datasets = {}

    for name, info in config.dataset_info.items():
        datasets[name] = {
            "name": name,
            **info,
            "default_label": config.dataset_label[name],
        }

    if verbose:
        for name, info in datasets.items():
            print(f"\n{'=' * 60}")
            print(f"Dataset: {name}")
            print(f"{'=' * 60}")

            for key, value in info.items():
                if key != "name":
                    print(f"{key}: {value}")

    return datasets

def download(dataset_name):
    """
    Download and initialize a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Case-insensitive.

    Returns
    -------
    PerspectivistDataset
        Initialized dataset object.
    """
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string")

    dataset_name = dataset_name.lower()

    if dataset_name not in _DATASETS:
        available = ", ".join(_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    return _DATASETS[dataset_name]()    

def download_and_split(dataset_name, user_adaptation=False, extended=False, named=False, baseline=False,):
    """
    Download a dataset and generate its requested task split.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Case-insensitive.

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
    
    dataset = download(dataset_name)

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

    for name in config.dataset_info:
        dataset = download(name)
        descriptions[name] = dataset.describe(verbose=verbose)

    return descriptions