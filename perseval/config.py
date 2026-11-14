import os
from pathlib import Path

output_root = Path(
    os.environ.get("PERSEVAL_OUTPUT_DIR", "./outputs")
).expanduser()

seed = 42

auto_split = {
    "test_user_fraction": 0.2,
    "adaptation_fraction": 0.05,
    "target_test_share": 0.2,  # test / (test + training with extended=False)
    "min_test_annotations": 10,
    "min_adaptation_annotations": 1,
}

# keys explained in "Adding a dataset" in the README
dataset_specs = {
    "EPIC": {
        "source": {"huggingface": "Multilingual-Perspectivist-NLU/EPIC"},
        "annotator_column": "user",
        "text_id_column": "id_original",
        "text_columns": {"post": "parent_text", "reply": "text"},
        "labels": {"irony": {"column": "label", "values": {"iro": 1, "not": 0}}},
        "traits": {
            "Gender": {"column": "Sex"},
            "Nationality": {"column": "Nationality"},
            "Generation": {"column": "Age", "age_to_generation": True},
        },
        "missing_values": ["DATA_EXPIRED"],
    },
    "BREXIT": {
        "source": {"huggingface": "silvia-casola/BREXIT", "split": ["train", "validation", "test"]},
        "annotator_column": "annotator_id",
        "text_id_column": "instance_id",
        "text_columns": {"tweet": "tweet"},
        "labels": {
            "hs": {"column": "hs"},
            "offensiveness": {"column": "offensiveness"},
            "aggressiveness": {"column": "aggressiveness"},
            "stereotype": {"column": "stereotype"},
        },
        "traits": {"Group": {"column": "annotator_group"}},
        "group_column": "annotator_group",
    },
    "DICES": {
        "source": {"disk": "diverse_safety_adversarial_dialog_350_enhanced"},
        "annotator_column": "rater_id",
        "text_id_column": "text_id",
        "text_columns": {"context": "context", "reply": "response"},
        "labels": {"Q2_harmful_content_overall": {"column": "Q2_harmful_content_overall",
                                                  "values": {"Yes": 2, "Unsure": 1, "No": 0}}},
        "traits": {
            "Gender": {"column": "rater_gender"},
            "Generation": {"column": "rater_age"},
            "Race": {"column": "rater_race"},
            "Education": {"column": "rater_education"},
        },
    },
    "MHS": {
        "source": {"huggingface": "ucberkeley-dlab/measuring-hate-speech"},
        "annotator_column": "annotator_id",
        "text_id_column": "comment_id",
        "text_columns": {"post": "text"},
        "labels": {"hateful": {"column": "hatespeech", "positive_if_at_least": 1}},
        "traits": {
            "Education": {"column": "annotator_educ", "values": {
                "college_grad_aa": "educ-high", "college_grad_ba": "educ-high", "masters": "educ-high", "phd": "educ-high",
                "high_school_grad": "educ-low", "professional_degree": "educ-low", "some_college": "educ-low",
                "some_high_school": "educ-low"}},
            "Gender": {"column": "annotator_gender"},
            "Ideology": {"column": "annotator_ideology", "values": {
                "conservative": "conservative", "extremely_conservative": "conservative", "slightly_conservative": "conservative",
                "liberal": "liberal", "extremely_liberal": "liberal", "slightly_liberal": "liberal",
                "neutral": "neutral", "no_opinion": "neutral"}},
            "Income": {"column": "annotator_income", "values": {
                "<10k": "income-low", "10k-50k": "income-low",
                "50k-100k": "income-high", "100k-200k": "income-high", ">200k": "income-high"}},
            # annotated in 2020
            "Age": {"column": "annotator_age", "age_to_generation": 2020},
        },
        "missing_traits": "skip",
    },
    "MD": {
        "source": {"csv": "MD-Agreement_dataset/MD_agreement.csv"},
        "annotator_column": "annotators",
        "text_id_column": "text_id",
        "text_columns": {"text": "text"},
        "labels": {"offensiveness": {"column": "annotations"}},
    },
    "CONVABUSE": {
        "source": {"csv": "https://raw.githubusercontent.com/amandacurry/convabuse/c0a9469f48956e868276c605d499010ea3c7c0d0/1_full/ConvAbuseEMNLPfull.csv"},
        "annotator_column": "annotator_id",
        "text_id_column": "example_no",
        "same_text_columns": ["conv_id", "prev_agent", "prev_user", "agent", "user"],
        "text_columns": {"prev_agent": "prev_agent", "prev_user": "prev_user", "agent": "agent", "user": "user"},
        "labels": {"abuse": {"positive_if_any": ["is_abuse.-1", "is_abuse.-2", "is_abuse.-3"]}},
        "drop_duplicate_annotations": True,
    },
    "TAS": {
        "source": {"huggingface": "soda-lmu/tweet-annotation-sensitivity-2"},
        "annotator_column": "annotator_id",
        "text_id_column": "tweet_id",
        "text_columns": {"tweet": "tweet_hashed"},
        "labels": {
            "hate_speech": {"column": "hate_speech"},
            "offensive_language": {"column": "offensive_language"},
        },
        "traits": {
            "Condition": {"column": "condition"},
            "Generation": {"column": "age", "age_to_generation": True},
            "Education": {"column": "education", "integer": True,
                          "values": {1: "educ-low", 2: "educ-low", 3: "educ-low", 4: "educ-high", 5: "educ-high", 6: "educ-high"}},
            "Party": {"column": "party_affiliation", "integer": True,
                      "values": {1: "Republican", 2: "Democrat", 3: "Independent"}},
        },
        "missing_traits": "skip",
    },
    "CSC": {
        "source": {"csv": "https://raw.githubusercontent.com/CoPsyN/CSC/f11c92b8d54686f64cf1d92766cef07201efb986/data_full_w_demo.tsv",
                   "separator": "\t"},
        "annotator_column": "global_evaluator",
        "text_id_column": ["global_context_name", "global_speaker"],
        "text_columns": {"context": "context_text", "response": "response_text"},
        "labels": {"sarcasm": {"column": "sarcasm_score_by_evaluator", "positive_if_at_least": 4}},
        "traits": {
            "Gender": {"column": "evaluator_sex", "lowercase": True,
                       "values": {"female": "female", "male": "male", "non-binary": "non-binary"}},
            "Generation": {"column": "evaluator_age", "age_to_generation": True},
            "Ethnicity": {"column": "evaluator_ethnicity"},
        },
        "missing_values": ["DATA_EXPIRED"],
        "drop_duplicate_annotations": True,
    },
}

dataset_label ={name: next(iter(spec["labels"])) for name, spec in dataset_specs.items()}

padding="max_length"
truncation=True
max_length=512

prediction_dir = "predictions"

prediction_dir_llama = "predictions_llama"
prediction_dir_mixtral = "predictions_mixtral"

label_map = {
    "irony": {"iro":1, "not":0},
    "irony_pred": {"irony":1, "not irony":0},
    "hs": {"hate speech":1, "not hate speech":0},
    "hs_pred": {"hate speech":1, "not hate speech":0},
    "Q2_harmful_content_overall": {"Yes":2, "Unsure":1, "No":0},
    "Q2_harmful_content_overall_pred": {"yes":2, "unsure":1, "no":0},
    "hateful_pred": {"hateful":1, "not hateful":0},
    "offensiveness_pred": {"offensive":1, "not offensive":0}
}

model_config = {
    "roberta-base": {
        "eval_strategy": "epoch",
        "greater_is_better":False,
        "learning_rate": 5e-6,
        "load_best_model_at_end": True,
        "logging_dir": str(output_root / "logs"),
        "logging_strategy": "epoch",
        "metric_for_best_model":"eval_loss",
        "num_train_epochs": 5,
        "output_dir": str(output_root / "checkpoints"),
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 32, 
        "per_device_train_batch_size": 16,
        "report_to": "none",
        "save_strategy": "epoch",
        "save_total_limit": 1
    }
}

padding="max_length"
truncation=True
max_length=512

prediction_dir = "predictions"

eval_percentage = 0.2
data_lamp_dir="data_LaMP"

prompts = {
    "EPIC": {
        "prelude": "You are provided as input a pair of sentences extracted from social media conversations.", 
        "task": "Your task is to determine if the Reply is Ironic.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {irony} or {not irony}), followed with your thoughts",
        "pred_opt": ['irony', 'not irony'],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"United Kingdom":"British",
               	    "United States":"American",
                	"Australia":"Australian",  
                	"Ireland":"Irish",
                	"India":"Indian",
                	"Male":"a self-identified male",
                	"Female":"a self-identified female",
                	"Boomer":"an older person of the baby boomer generation",
                	"GenX":"an older person of generation X",
                	"GenY":"a millennial",
                	"GenZ":"a young person of generation Z",},
    },

    "BREXIT": {
        "prelude": "You are provided as input a tweet extracted from twitter.", 
        "task": "Your task is to determine if the tweet contains hateful speech. You do not need to produce or promote hate speech, You just need to make a decision whether the text you see is hate speech or not.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {hate speech} or {not hate speech}) depending on your decision whether it is hate speech or not, followed with your thoughts",
        "pred_opt": ["hate speech", "not hate speech"],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"target":"a Muslim immigrant in the UK",
               	"control":"a researcher",}
    },

    "DICES": {
        "prelude": "You are provided as input a dialog extracted from a conversation between a user and an AI-chatbot.", 
        "task": "Your task is to determine if the dialog is harmful.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {yes}, {no} or {unsure}), followed with your thoughts",
        "pred_opt": ['Yes', 'No', 'Unsure'],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"Asian/Asian subcontinent":"Asian",
               	    "White":"White",
                	"Black/African American":"Black",  
                	"LatinX, Latino, Hispanic or Spanish Origin":"Latinx",
                	"Multiracial":"Multiracial",
                	"Man":"a self-identified male",
                	"Woman":"a self-identified female",
                	"gen x+":"an older person of generation X",
                	"millenial":"a millennial",
                	"gen z":"a young person of generation Z",
                	"College degree or higher":"a person with a college degree or higher",
                	"High school or below":"a person with a high school diploma or lower"}
    },

    "MHS": {
        "prelude": "You are provided as input a text extracted from social media platforms.", 
        "task": "Your task is to determine if the text is hateful.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {hateful} or {not hateful}) depending on your decision whether it is hateful or not, followed with your thoughts",
        "pred_opt": ["hateful", "not hateful"],
        "context_pre": "Input:",
        "context_post": "Output:",
        "traits": {"educ-high":"a person with a college degree or higher education",
                    "educ-low":"a person with a high school diploma or lower education",
                    "male":"a male",
                    "female":"a female",
                    "non-binary":"a non-binary person",
                    "Boomer":"a person of the baby boomer generation",
                    "GenX":"a person of generation X",
                    "GenY":"a person of generation Y",
                    "GenZ":"a person of generation Z",
                    "conservative":"a conservative",
                    "no_opinion":"a person with no political opinion",
                    "liberal":"a liberal",
                    "income-high":"a person with a more than 50k annual income",
                    "income-low":"a person with a less than 50k annual income"}
    },

    "MD": {
        "prelude": "You are provided as input a tweet extracted from twitter.", 
        "task": "Your task is to determine if the tweet is offensive.",
        "instr_pre": "Please provide as an output, a single label among the following options:",
        "instr_post": " within curly brackets (i.e. {offensive} or {not offensive}) depending on your decision whether it is offensive or not, followed with your thoughts",
        "pred_opt": ["offensive", "not offensive"],
        "context_pre": "Input:",
        "context_post": "Output:"
    }
}