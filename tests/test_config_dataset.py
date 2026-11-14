import logging
import textwrap

import pandas as pd
import pytest
from datasets import Dataset

import perseval
import perseval.data as data_module
from perseval import config
from perseval.data import ConfigDataset, UNKNOWN_TRAIT, age_to_generation

ROWS = [
    # annotator, context, speaker, response, score, sex, age, ethnicity
    ("a1", "c1", "s1", "first", 5, "Male", "30", "White"),
    ("a1", "c1", "s2", "second", 2, "Male", "30", "White"),
    ("a1", "c1", "s1", "first", 1, "Male", "30", "White"),  # same text again
    ("a2", "c1", "s1", "first", 4, "male", "DATA_EXPIRED", None),
    ("a2", "c1", "s2", "second", 3, "female", "61", "Asian"),
    ("a3", "c2", "s1", "third", "", "non-binary", "20", "Black"),  # no label
]


def write_csv(path, rows=ROWS):
    lines = ["annotator,context,speaker,response,score,sex,age,ethnicity"]
    lines += [",".join("" if v is None else str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def spec(path, **changes):
    entry = {
        "source": {"csv": str(path)},
        "annotator_column": "annotator",
        "text_id_column": ["context", "speaker"],
        "text_columns": {"response": "response"},
        "labels": {"positive": {"column": "score", "positive_if_at_least": 4}},
        "traits": {
            "Gender": {"column": "sex", "lowercase": True, "values": {"male": "male", "female": "female"}},
            "Generation": {"column": "age", "age_to_generation": True},
            "Ethnicity": {"column": "ethnicity"},
        },
        "missing_values": ["DATA_EXPIRED"],
        "drop_duplicate_annotations": True,
    }
    entry.update(changes)
    return entry


@pytest.fixture
def toy(tmp_path, monkeypatch):
    path = tmp_path / "toy.csv"
    write_csv(path)

    def load(**changes):
        monkeypatch.setitem(config.dataset_specs, "TOY", spec(path, **changes))
        return ConfigDataset("TOY")
    return load


def rows_of(dataset):
    return {(row[dataset.key_user], row[dataset.key_text]): row for row in dataset.dataset}


def test_columns_text_ids_and_labels(toy):
    dataset = toy()
    rows = rows_of(dataset)
    assert set(rows) == {("a1", "c1_s1"), ("a1", "c1_s2"), ("a2", "c1_s1"), ("a2", "c1_s2")}
    assert dataset.read_labels(rows[("a1", "c1_s1")]) == {"positive": 1}
    assert dataset.read_labels(rows[("a2", "c1_s2")]) == {"positive": 0}
    assert dataset.read_text(rows[("a1", "c1_s2")]) == {"response": "second"}
    assert dataset.label == "positive"


def test_traits(toy):
    dataset = toy()
    rows = rows_of(dataset)
    assert dataset.read_traits(rows[("a1", "c1_s1")]) == {"Gender": "male", "Generation": "GenY", "Ethnicity": "White"}
    assert dataset.read_traits(rows[("a2", "c1_s1")]) == {"Gender": "male", "Generation": UNKNOWN_TRAIT,
                                                         "Ethnicity": UNKNOWN_TRAIT}
    assert dataset.read_traits(rows[("a2", "c1_s2")])["Generation"] == "Boomer"


def test_missing_traits_can_be_skipped(toy):
    dataset = toy(missing_traits="skip")
    assert dataset.read_traits(rows_of(dataset)[("a2", "c1_s1")]) == {"Gender": "male"}


def test_value_not_in_the_table_is_unknown(toy):
    dataset = toy(traits={"Gender": {"column": "sex", "values": {"male": "M"}}})
    rows = rows_of(dataset)
    assert dataset.read_traits(rows[("a2", "c1_s1")]) == {"Gender": "M"}
    assert dataset.read_traits(rows[("a1", "c1_s1")]) == {"Gender": UNKNOWN_TRAIT}


def test_dataset_without_traits_cannot_be_named(toy):
    dataset = toy(traits={})
    with pytest.raises(ValueError, match="named=True is not available"):
        dataset.read_traits(rows_of(dataset)[("a1", "c1_s1")])


def test_unknown_column_lists_the_available_ones(toy):
    with pytest.raises(ValueError, match="no column 'rater' .annotator_column.*Available columns: annotator"):
        toy(annotator_column="rater")


def test_label_that_is_not_a_number_explains_the_fix(toy):
    with pytest.raises(ValueError, match="is not a number"):
        toy(labels={"gender": {"column": "sex"}})


def test_label_table_rejects_unknown_values(toy):
    with pytest.raises(ValueError, match="is not in the 'values'"):
        toy(labels={"gender": {"column": "sex", "values": {"male": 0}}})


def test_label_that_is_not_a_whole_number_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "toy.csv"
    write_csv(path, [("a1", "c1", "s1", "first", 0.7, "male", "30", "White")])
    monkeypatch.setitem(config.dataset_specs, "TOY", spec(path, labels={"score": {"column": "score"}}))
    with pytest.raises(ValueError, match="not a whole number"):
        ConfigDataset("TOY")


def test_default_label_is_the_first_one(toy):
    dataset = toy(labels={"high": {"column": "score", "positive_if_at_least": 4},
                          "low": {"column": "score", "positive_if_at_least": 2}})
    assert dataset.label == "high"
    assert dataset.read_labels(rows_of(dataset)[("a2", "c1_s2")]) == {"high": 0, "low": 1}


def test_dataset_label_follows_the_specs():
    assert set(config.dataset_label) == set(config.dataset_specs)
    assert config.dataset_label["BREXIT"] == "hs"
    assert config.dataset_label["TAS"] == "hate_speech"


@pytest.mark.parametrize("name", sorted(config.dataset_specs))
def test_library_entries_have_the_required_keys(name):
    assert all(key in config.dataset_specs[name] for key in ConfigDataset.REQUIRED)


def test_numbered_rows_and_label_from_several_columns(tmp_path, monkeypatch):
    path = tmp_path / "conversations.csv"
    lines = ["annotator,conv,prev,turn,row_no,mild,strong",
             "a1,k1,,hi,10,0,0",
             "a2,k1,,hi,11,1,0",  # same exchange as row 10
             "a1,k1,hi,you idiot,12,0,1",
             "a2,k1,hi,you idiot,13,0,0"]
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.setitem(config.dataset_specs, "TOY", {
        "source": {"csv": str(path)},
        "annotator_column": "annotator",
        "text_id_column": "row_no",
        "same_text_columns": ["conv", "prev", "turn"],
        "text_columns": {"previous": "prev", "turn": "turn"},
        "labels": {"abuse": {"positive_if_any": ["mild", "strong"]}},
    })
    dataset = ConfigDataset("TOY")
    rows = rows_of(dataset)
    pairs = [("a1", 10), ("a2", 10), ("a1", 12), ("a2", 12)]
    assert set(rows) == set(pairs)
    assert [dataset.read_labels(rows[pair])["abuse"] for pair in pairs] == [0, 1, 1, 0]
    assert dataset.read_text(rows[("a1", 10)]) == {"previous": "", "turn": "hi"}


def test_unknown_label(tmp_path, monkeypatch):
    path = tmp_path / "toy.csv"
    write_csv(path)
    monkeypatch.setitem(config.dataset_specs, "TOY", spec(path))
    with pytest.raises(ValueError, match="no label 'sarcasm'.*Available labels: positive"):
        ConfigDataset("TOY", "sarcasm")


def test_missing_required_key(tmp_path, monkeypatch):
    entry = spec(tmp_path / "toy.csv")
    del entry["labels"]
    monkeypatch.setitem(config.dataset_specs, "TOY", entry)
    with pytest.raises(ValueError, match="has no labels"):
        ConfigDataset("TOY")


def test_warns_about_repeated_annotations(toy, caplog):
    with caplog.at_level(logging.WARNING):
        toy(drop_duplicate_annotations=False)
    assert "repeat an (annotator, text) pair" in caplog.text


def test_warns_when_text_ids_number_the_rows(tmp_path, monkeypatch, caplog):
    path = tmp_path / "toy.csv"
    write_csv(path, [("a1", "c1", "s1", f"text {i}", 5, "male", "30", "White") for i in range(3)])
    monkeypatch.setitem(config.dataset_specs, "TOY", spec(path, text_id_column="response"))
    with caplog.at_level(logging.WARNING):
        ConfigDataset("TOY")
    assert "no text has more than one annotation" in caplog.text


def test_dataset_added_at_runtime_is_available(toy):
    toy()
    assert "toy" in data_module.available_datasets()
    dataset = data_module.download("TOY", label="positive")
    assert isinstance(dataset, ConfigDataset) and dataset.name == "TOY"


def test_prepare_runs_before_the_columns_are_read(toy):
    def prepare(df):
        df["text"] = df["context"] + "/" + df["speaker"]
        return df
    dataset = toy(prepare=prepare, text_id_column="text")
    assert {row[dataset.key_text] for row in dataset.dataset} == {"c1/s1", "c1/s2"}


def test_relative_paths_are_read_from_the_data_dir(tmp_path, monkeypatch):
    write_csv(tmp_path / "toy.csv")
    monkeypatch.setenv("PERSEVAL_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(config.dataset_specs, "TOY", spec("toy.csv"))
    assert len(ConfigDataset("TOY").dataset) == 4


def test_missing_local_file_names_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSEVAL_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(config.dataset_specs, "TOY", spec("nope.csv"))
    with pytest.raises(FileNotFoundError, match="PERSEVAL_DATA_DIR"):
        ConfigDataset("TOY")


def test_dataset_saved_to_disk(tmp_path, monkeypatch):
    write_csv(tmp_path / "toy.csv")
    Dataset.from_pandas(pd.read_csv(tmp_path / "toy.csv")).save_to_disk(str(tmp_path / "toy"))
    entry = spec(tmp_path, source={"disk": str(tmp_path / "toy")})
    monkeypatch.setitem(config.dataset_specs, "TOY", entry)
    assert len(ConfigDataset("TOY").dataset) == 4


def test_generation_depends_on_the_year_of_the_annotation():
    assert age_to_generation(57) == "GenX"
    assert age_to_generation(57, 2020) == "Boomer"


def test_unreadable_age_is_unknown(toy):
    dataset = toy(missing_values=[])
    assert dataset.read_traits(rows_of(dataset)[("a2", "c1_s1")])["Generation"] == UNKNOWN_TRAIT


def test_warns_about_trait_values_that_differ_in_case(toy, caplog):
    with caplog.at_level(logging.WARNING):
        toy(traits={"Gender": {"column": "sex"}})
    assert "differ only in case" in caplog.text


def test_old_class_names_still_work(tmp_path, monkeypatch):
    path = tmp_path / "toy.csv"
    write_csv(path)
    monkeypatch.setitem(config.dataset_specs, "DICES", spec(path))
    dataset = data_module.DICES("positive")
    assert isinstance(dataset, ConfigDataset) and dataset.name == "DICES"


TOY_YAML = textwrap.dedent("""\
    TOY:
      source:
        csv: toy.csv
      annotator_column: annotator
      text_id_column: [context, speaker]
      text_columns:
        response: response
      labels:
        positive:
          column: score
          values: {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
      traits:
        Generation:
          column: age
          age_to_generation: true
      missing_values: [DATA_EXPIRED]
      drop_duplicate_annotations: true
    """)


@pytest.fixture
def isolated_config(monkeypatch):
    monkeypatch.setattr(config, "dataset_specs", dict(config.dataset_specs))
    monkeypatch.setattr(config, "dataset_label", dict(config.dataset_label))


def test_dataset_from_a_yaml_file(tmp_path, monkeypatch, isolated_config):
    write_csv(tmp_path / "toy.csv")
    (tmp_path / "toy.yaml").write_text(TOY_YAML)
    monkeypatch.chdir(tmp_path)
    assert perseval.add_dataset("toy.yaml") == ["TOY"]
    monkeypatch.chdir(tmp_path.parent)
    assert "toy" in perseval.available_datasets()
    assert config.dataset_label["TOY"] == "positive"
    dataset = perseval.download("toy")
    rows = rows_of(dataset)
    assert dataset.read_labels(rows[("a1", "c1_s1")]) == {"positive": 1}
    assert dataset.read_traits(rows[("a2", "c1_s2")]) == {"Generation": "Boomer"}


def test_yaml_entry_must_be_under_its_name(tmp_path, isolated_config):
    path = tmp_path / "toy.yaml"
    path.write_text("source:\n  csv: toy.csv\nannotator_column: annotator\n")
    with pytest.raises(ValueError, match="under its name"):
        perseval.add_dataset(path)


def test_yaml_entry_is_checked_before_it_is_added(tmp_path, isolated_config):
    path = tmp_path / "toy.yaml"
    path.write_text("TOY:\n  source:\n    csv: toy.csv\n")
    with pytest.raises(ValueError, match="TOY has no annotator_column"):
        perseval.add_dataset(path)
    path.write_text(TOY_YAML.replace("source:\n    csv: toy.csv", "source: toy.csv"))
    with pytest.raises(ValueError, match="must name its type"):
        perseval.add_dataset(path)
    assert "TOY" not in config.dataset_specs
