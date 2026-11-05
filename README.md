# PersEval

## A Library for Perspectivist Classification Evaluation

**PersEval** is a library for the evaluation of perspectivist classification models, developed to streamline the current fragmented perspectivist evaluation practices with a new unified framework. 

The framework is presented in the paper [PersEval: A Framework for Perspectivist Classification Evaluation](https://aclanthology.org/2025.emnlp-main.1137), accepted at the main track of **EMNLP 2025**.

## 📝 Abstract
Data perspectivism goes beyond majority vote label aggregation by recognizing various perspectives as legitimate ground truths.
However, current evaluation practices remain fragmented, making it difficult to compare perspectivist approaches and analyze their impact on different users and demographic subgroups. To address this gap, we introduce PersEval, the first unified framework for evaluating perspectivist models in NLP. A key innovation is its evaluation at the individual annotator level and its treatment of annotators and users as distinct entities, consistently with real-world scenarios.
We demonstrate PersEval's capabilities through experiments with both Encoder-based and Decoder-based approaches, as well as an analysis of the effect of sociodemographic prompting. 
By considering global, text-, trait- and user-level evaluation metrics, we show that PersEval is a powerful tool for examining how models are influenced by user-specific information and identifying the biases this information may introduce. 


## ⚙️ Framework

![](<./framework_diagram.png>)

### Installation

Install PersEval from the repository with:

```bash
python -m pip install .
```

### Named or unnamed classification
Users can be represented in two ways:

- **Unnamed classification** — by their identifier only.

- **Named classification** — by a set of explicit metadata (e.g., traits).

To enable named classification when generating the dataset splits, set `named=True`:
```python
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=True)
```
Set `named=False` for unnamed classification.

### Adaptation set 
Mirroring real-world scenarios, we assume that the annotators who provided the bulk of the annotations to train the system, and the users for which the system is tested, are disjoint. 

When explicit knowledge about the users is available (e.g. metadata), a model can attempt to learn biases toward such characteristics, without knowing any preference of the unkown users.
 
```python
perspectivist_dataset.get_splits(user_adaptation=False, extended=False, named=True)
```

However, we also assume two adaptation scenarios where few labels from test users are available: 

**Adaptation at training time**: we assume minimal annotation from users has been obtained before training the system. A few annotations from test users are thus included in the training split.
```python
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=False)
perspectivist_dataset.get_splits(user_adaptation="train", extended=False, named=True)
```

**Adaptation at inference time**: we assume an already trained system has to be adapted to new users. A few test users instances can thus be used to adapt an existing model.
```python
perspectivist_dataset.get_splits(user_adaptation="test", extended=False, named=False)
perspectivist_dataset.get_splits(user_adaptation="test", extended=False, named=True)
```

### Training set 
In the default setting, textual examples in the test split are disjoint from those in the training split. 


We also provide a variant for which texts that are also found in test instances, but annotated by different users, can be included in the trainig split.
```python
perspectivist_dataset.get_splits(user_adaptation="train", extended=True, named=False)
```

### Test set
The difference between the task variants manifests in different training splits (or in some cases additional sets when using an adaptation set at inference time), while **the test set depends uniquely on the chosen dataset and it remains the same across all task variants.**

### How the splits are computed
Test users must be disjoint from training users and, without `--extended`, test texts must be disjoint from training texts. On a dense dataset (many annotators per text) almost every text is annotated by some test user, so a split by user alone would leave nothing to train on.

Datasets are therefore split automatically (`perseval/splitting.py`):
1. 20% of the annotators become test users, sampled among those with enough annotations to be evaluated (optionally stratified by an annotator column, e.g. BREXIT's groups).
2. A fraction of the texts is reserved for training only. The smallest reserve that brings the test split down to 20% of (training + test) is chosen, so sparse datasets get little or no reserve and dense ones get a larger one.
3. 5% of the remaining texts of the test users form the adaptation set. Every test user has at least one adaptation text and at least ten test texts. A test user left without an adaptation text gets the text of theirs annotated by the fewest test users, so that texts annotated by many users (such as the reference texts of MHS) do not end up in everyone's adaptation set.

These targets live in `config.auto_split`.


## 🗃️ Datasets

The library comes with eight datasets. 


| Dataset  | Reference | Task| #Annotators | Metadata | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Brexit  | [Akhtar et al. 2021](https://arxiv.org/abs/2106.15896)  | Abusive language | 6 | Target an control group | 
| EPIC  | [Frenda et al. 2023](https://aclanthology.org/2023.acl-long.774/)  | Irony | 74 | Gender, Nationality, Age/Generation | 
| MHS | [Sachdeva et al. 2022](https://aclanthology.org/2022.nlperspectives-1.11/)  | Hate Speech | 7,912 | Gender, Age/Generation, Education, Income, Ideology | 
| MD-Agreement |[Leonaredlli et al. 2022](https://aclanthology.org/2023.semeval-1.314/)  | Offensiveness | 819 | --- | 
|Dices | [Aroyo et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html)  | AI Safety | 123 | Gender, Age/Generation, Education, Ethinicity | 
| TAS | [Kern et al. 2023](https://aclanthology.org/2023.findings-emnlp.992/) | Hate speech, Offensive language | 1,807 | Annotation condition, Age/Generation, Education, Party | 
| ConvAbuse | [Cercas Curry et al. 2021](https://aclanthology.org/2021.emnlp-main.587/) | Abuse towards conversational agents | 8 | --- | 
| CSC | [Jang and Frassinelli 2024](https://aclanthology.org/2024.naacl-long.238/) | Sarcasm | 838 | Gender, Age/Generation, Ethnicity | 

BREXIT, EPIC, MHS and TAS are directly loaded from Hugging Face, ConvAbuse and CSC from a fixed commit of their GitHub repositories, while DICES and MD-Agreement are loaded from the directory configured through the `PERSEVAL_DATA_DIR` environment variable.

In ConvAbuse an utterance is abusive when rated from -1 to -3 (ambiguous and non-abusive are not). In CSC a response is sarcastic when rated 4 ("somewhat") or more on the 1-6 scale, as in the original paper.

Other than the dataset name, it is important to also specify the corresponding label.

```python
from perseval.data import DICES

perspectivist_dataset = DICES("Q2_harmful_content_overall")
```
Options for labels are listed below:

```
# EPIC   -> ["irony"]
# BREXIT -> ["hs", "offensiveness", "aggressiveness", "stereotype"]
# DICES  -> ["Q2_harmful_content_overall"]
# MHS    -> ["hateful"]
# MD     -> ["offensiveness"]
# TAS    -> ["hate_speech", "offensive_language"]
# CONVABUSE -> ["abuse"]
# CSC    -> ["sarcasm"]
```

## 📈 Metrics

We require models to output a **label** for each **<user, text> tuple**. Starting from these prediction, we compute standard classification metrics. Specifically precision, recall and F1 scores for each class, as well as their macro- and micro-average. This same metrics are computed also at more fine-grained levels: 
- annotator-level: computed individually for each annotator and then averaged; 
- text-level: computed individually for each text in the test set and then averaged; 
- trait-level: computed for each trait and then averaged for each dimension.

## 🤖 Baseline Models 
The optional baseline implementations are available in the `perseval.baselines` submodule.

### Encoder-based 
We fine-tuned RoBERTa, customized implementing Focal Loss.

We added annotators' identifiers and their traits to the text embedding as a special token. The model input thus concatenates the annotator id, a special token for each of the annotator's traits, and the input text to classify. 
The model is then trained with a classification head to predict the binary label.

### Decoder-based 
We chose open-source models of medium size: Mixtral-8 7B and Llama-3.1 8B, both instruction tuned. We considered three possible settings: 

**Base zero**: we prompt the models to classify the test set examples, with no additional information.

**Perspective**: we ask the models to impersonate each user’s trait. We use this variant to test models without adaptation with a named user representation. We prompt the model for each available user trait.

**In-Prompt Augmentation** We reproduced
 [Salemi et al. (2024)](https://aclanthology.org/2024.acl-long.399/)’s approach,  prompting the model with user-specific input selected via retrieval augmentation. We used this approach both giving information about the user’s trait value (Named with Adaptation-T) and without providing demographic information. 

## 🧰 Our setup
The required dependencies were verified on the following system:

- Operating System: Ubuntu 22.04.3 LTS (Jammy)
- Kernel: Linux 5.15.0-113-generic x86_64
- CPU: AMD EPYC-Rome Processor — 24 cores, 1 thread per core
- RAM: 503 GiB
- GPU: 4× NVIDIA A40 (46 GiB each)
- Python: 3.10.12

Performance and compatibility may vary on different hardware or OS versions.

## 📊 Qualitative analysis
Representing users through their sociodemographic traits could lead to the risk of stereotype propagation. So we evaluated the effect sociodemographic prompting through a qualitative analysis based on the following Questions: 
- Q1: What is the contribution of each trait when ensembling the model’s outputs?
- Q2: Which demographic trait most significantly impacts the model’s label predictions in the presence of varying annotator characteristics?
- Q3: How similar is the distribution of models’ predictions to that of the annotators’ chosen labels

## Other information

Note that the current version of the library does not support multi-GPU settings.

The In-Prompt Augmentation strategy is part of a comprehensive evaluation framework for personalization with LLMs (LaMP). The full framework is available at [this repository](https://github.com/LaMP-Benchmark/LaMP/tree/main).

## 📖 Reference
If you use the library, please cite:

```
@inproceedings{lo-etal-2025-perseval,
    title = "{PERSEVAL}: A Framework for Perspectivist Classification Evaluation",
    author = "Lo, Soda Marem  and
      Casola, Silvia  and
      Sezerer, Erhan  and
      Basile, Valerio  and
      Sansonetti, Franco  and
      Uva, Antonio  and
      Bernardi, Davide",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1137/",
    pages = "22345--22370",
    ISBN = "979-8-89176-332-6",
    abstract = "Data perspectivism goes beyond majority vote label aggregation by recognizing various perspectives as legitimate ground truths.However, current evaluation practices remain fragmented, making it difficult to compare perspectivist approaches and analyze their impact on different users and demographic subgroups. To address this gap, we introduce PersEval, the first unified framework for evaluating perspectivist models in NLP. A key innovation is its evaluation at the individual annotator level and its treatment of annotators and users as distinct entities, consistently with real-world scenarios. We demonstrate PersEval{'}s capabilities through experiments with both Encoder-based and Decoder-based approaches, as well as an analysis of the effect of sociodemographic prompting. By considering global, text-, trait- and user-level evaluation metrics, we show that PersEval is a powerful tool for examining how models are influenced by user-specific information and identifying the biases this information may introduce."
}
```
