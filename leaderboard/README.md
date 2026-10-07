# PersEval Leaderboard

This leaderboard reports the benchmark results ranked by global macro-F1 only within the same dataset and task variant.

The annotator and text columns report the corresponding macro-averaged F1 scores. The complete machine-readable data is available in [scores.csv](./scores.csv).

Only configurations with an existing aggregate report are listed. An absent configuration does not mean that PersEval does not support that configuration, but simply that there is no corresponding result.

## BREXIT — hs

| Rank | Family | Model | Variant | Method | Representation | Global F1 | User F1 | Text F1 | Source |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: no | Text-only baseline | — | **0.567** | 0.519 | 0.403 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Baseline · no adaptation · Extended: no | Base-zero | — | 0.502 | 0.476 | 0.340 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Baseline · no adaptation · Extended: no | Base-zero | — | 0.255 | 0.235 | 0.128 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: no | Fine-tuned encoder | Traits | **0.558** | 0.524 | 0.416 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.527 | 0.502 | 0.371 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.344 | 0.323 | 0.193 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: no | Fine-tuned encoder | Traits | **0.544** | 0.512 | 0.405 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: no | Fine-tuned encoder | User ID | **0.512** | 0.484 | 0.378 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | **0.406** | 0.382 | 0.263 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | 0.364 | 0.362 | 0.238 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | **0.448** | 0.410 | 0.313 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | 0.330 | 0.319 | 0.231 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: yes | Text-only baseline | — | **0.592** | 0.543 | 0.427 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.587** | 0.557 | 0.455 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.540** | 0.509 | 0.424 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: yes | Fine-tuned encoder | User ID | **0.554** | 0.524 | 0.436 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |

## DICES — Q2_harmful_content_overall

| Rank | Family | Model | Variant | Method | Representation | Global F1 | User F1 | Text F1 | Source |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: no | Text-only baseline | — | **0.340** | 0.311 | 0.245 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Baseline · no adaptation · Extended: no | Base-zero | — | 0.290 | 0.282 | 0.310 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Baseline · no adaptation · Extended: no | Base-zero | — | 0.232 | 0.228 | 0.402 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: no | Fine-tuned encoder | Traits | **0.400** | 0.391 | 0.361 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.298 | 0.290 | 0.289 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.256 | 0.323 | 0.412 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: no | Fine-tuned encoder | Traits | **0.420** | 0.407 | 0.373 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: no | Fine-tuned encoder | User ID | **0.434** | 0.424 | 0.389 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Decoder | Llama-3.1-8B-Instruct | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | **0.365** | 0.354 | 0.428 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | 0.303 | 0.350 | 0.448 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Decoder | Llama-3.1-8B-Instruct | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | **0.355** | 0.340 | 0.425 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | 0.306 | 0.356 | 0.443 | [Table 6](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: yes | Text-only baseline | — | **0.440** | 0.448 | 0.335 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.453** | 0.439 | 0.378 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.457** | 0.445 | 0.389 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: yes | Fine-tuned encoder | User ID | **0.456** | 0.446 | 0.388 | [Table 4](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |

## EPIC — irony

| Rank | Family | Model | Variant | Method | Representation | Global F1 | User F1 | Text F1 | Source |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: no | Text-only baseline | — | **0.555** | 0.538 | 0.376 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Baseline · no adaptation · Extended: no | Base-zero | — | 0.529 | 0.511 | 0.363 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Baseline · no adaptation · Extended: no | Base-zero | — | 0.487 | 0.477 | 0.305 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: no | Fine-tuned encoder | Traits | **0.542** | 0.527 | 0.364 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.507 | 0.494 | 0.328 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Llama-3.1-8B-Instruct | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.484 | 0.467 | 0.322 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: no | Fine-tuned encoder | Traits | **0.550** | 0.534 | 0.371 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: no | Fine-tuned encoder | User ID | **0.534** | 0.518 | 0.352 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | **0.554** | 0.521 | 0.380 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | 0.547 | 0.528 | 0.387 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | **0.553** | 0.528 | 0.384 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | 0.546 | 0.530 | 0.386 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: yes | Text-only baseline | — | **0.579** | 0.559 | 0.405 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.575** | 0.560 | 0.392 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.578** | 0.564 | 0.594 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: yes | Fine-tuned encoder | User ID | **0.586** | 0.571 | 0.405 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |

## MD — offensiveness

| Rank | Family | Model | Variant | Method | Representation | Global F1 | User F1 | Text F1 | Source |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: no | Text-only baseline | — | **0.665** | 0.591 | 0.500 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Baseline · no adaptation · Extended: no | Base-zero | — | 0.556 | 0.515 | 0.381 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Baseline · no adaptation · Extended: no | Base-zero | — | 0.538 | 0.495 | 0.678 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: no | Fine-tuned encoder | User ID | **0.665** | 0.597 | 0.499 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Decoder | Llama-3.1-8B-Instruct | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | **0.613** | 0.535 | 0.451 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | 0.531 | 0.398 | 0.643 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: yes | Text-only baseline | — | **0.667** | 0.603 | 0.495 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: yes | Fine-tuned encoder | User ID | **0.681** | 0.620 | 0.518 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |

## MHS — hateful

| Rank | Family | Model | Variant | Method | Representation | Global F1 | User F1 | Text F1 | Source |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: no | Text-only baseline | — | **0.688** | 0.642 | 0.515 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Baseline · no adaptation · Extended: no | Base-zero | — | 0.648 | 0.599 | 0.483 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Llama-3.1-8B-Instruct | Baseline · no adaptation · Extended: no | Base-zero | — | 0.593 | 0.543 | 0.425 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: no | Fine-tuned encoder | Traits | **0.691** | 0.640 | 0.518 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.644 | 0.594 | 0.480 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 3 | Decoder | Llama-3.1-8B-Instruct | Named · no adaptation · Extended: no | Perspective prompting | Traits | 0.515 | 0.454 | 0.354 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: no | Fine-tuned encoder | Traits | **0.689** | 0.641 | 0.516 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: no | Fine-tuned encoder | User ID | **0.692** | 0.643 | 0.521 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Decoder | Llama-3.1-8B-Instruct | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | **0.637** | 0.573 | 0.467 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Named · inference-time adaptation · Extended: no | IPA (LaMP) | Traits + user history | 0.634 | 0.569 | 0.459 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Decoder | Mixtral-8x7B-Instruct-v0.1 | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | **0.632** | 0.571 | 0.457 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 2 | Decoder | Llama-3.1-8B-Instruct | Unnamed · inference-time adaptation · Extended: no | IPA (LaMP) | User history | 0.626 | 0.570 | 0.456 | [Table 5](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=8) |
| 1 | Encoder | FacebookAI/roberta-base | Baseline · no adaptation · Extended: yes | Text-only baseline | — | **0.696** | 0.647 | 0.526 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · no adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.700** | 0.651 | 0.530 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Named · training-time adaptation · Extended: yes | Fine-tuned encoder | Traits | **0.700** | 0.650 | 0.532 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |
| 1 | Encoder | FacebookAI/roberta-base | Unnamed · training-time adaptation · Extended: yes | Fine-tuned encoder | User ID | **0.697** | 0.648 | 0.527 | [Table 3](https://aclanthology.org/2025.emnlp-main.1137.pdf#page=7) |

## Updating the leaderboard

To add a verified result:

1. Add one row to [scores.csv](./scores.csv), including its model family and source.
2. Add the result to the matching dataset table above.
3. Rank entries by global F1 only within the same task, adaptation, and extended-training variant.
