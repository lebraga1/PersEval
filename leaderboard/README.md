# PersEval Leaderboard

This leaderboard reports the benchmark results ranked by **global macro-F1** only within the
same dataset and task variant.

The annotator and text columns report the corresponding macro-averaged F1
scores. The complete machine-readable data is available in [scores.csv](./scores.csv).

Only configurations with an existing aggregate report are listed. An absent
configuration does not mean that PersEval does not support that configuration, but simply 
that there is no corresponding result. 

## BREXIT — hs

### Unnamed · Adaptation: training time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Llama-3.1-8B-Instruct | Base zero | — | **0.459** | 0.451 | 0.753 | [report](../results/result_llama_BREXIT_False.txt) |
| 2 | Mixtral-8x7B-Instruct-v0.1 | Base zero | — | 0.383 | 0.376 | 0.768 | [report](../results/result_mixtral_BREXIT_False.txt) |

### Named · Adaptation: training time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Llama-3.1-8B-Instruct | Perspective prompting | Group | **0.475** | 0.466 | 0.797 | [report](../results/result_llama_BREXIT_Group.txt) |
| 2 | Mixtral-8x7B-Instruct-v0.1 | Perspective prompting | Group | 0.405 | 0.398 | 0.729 | [report](../results/result_mixtral_BREXIT_Group.txt) |

### Unnamed · Adaptation: inference time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Mixtral-8x7B-Instruct-v0.1 | LaMP | — | **0.651** | 0.632 | 0.736 | [report](../results/result_lamp_mixtral_Brexit_False.txt) |
| 2 | Llama-3.1-8B-Instruct | LaMP | — | 0.407 | 0.402 | 0.383 | [report](../results/result_lamp_llama_Brexit_False.txt) |

### Named · Adaptation: inference time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Mixtral-8x7B-Instruct-v0.1 | LaMP | Group | **0.627** | 0.615 | 0.722 | [report](../results/result_lamp_mixtral_Brexit_Group.txt) |
| 2 | Llama-3.1-8B-Instruct | LaMP | Group | 0.481 | 0.481 | 0.454 | [report](../results/result_lamp_llama_Brexit_Group.txt) |

## DICES — Q2_harmful_content_overall

### Unnamed · Adaptation: training time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Llama-3.1-8B-Instruct | Base zero | — | **0.290** | 0.282 | 0.310 | [report](../results/result_llama_DICES_False.txt) |
| 2 | Mixtral-8x7B-Instruct-v0.1 | Base zero | — | 0.232 | 0.228 | 0.402 | [report](../results/result_mixtral_DICES_False.txt) |

### Unnamed · Adaptation: inference time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Llama-3.1-8B-Instruct | LaMP | — | **0.355** | 0.340 | 0.425 | [report](../results/result_lamp_llama_Dices_False.txt) |
| 2 | Mixtral-8x7B-Instruct-v0.1 | LaMP | — | 0.306 | 0.356 | 0.443 | [report](../results/result_lamp_mixtral_Dices_False.txt) |

## EPIC — irony

### Unnamed · Adaptation: training time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Mixtral-8x7B-Instruct-v0.1 | Base zero | — | **0.405** | 0.479 | 0.620 | [report](../results/result_mixtral_EPIC_False.txt) |
| 2 | Llama-3.1-8B-Instruct | Base zero | — | 0.399 | 0.534 | 0.592 | [report](../results/result_llama_EPIC_False.txt) |

### Unnamed · Adaptation: inference time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Mixtral-8x7B-Instruct-v0.1 | LaMP | — | **0.562** | 0.546 | 0.534 | [report](../results/result_lamp_mixtral_Epic_False.txt) |
| 2 | Llama-3.1-8B-Instruct | LaMP | — | 0.433 | 0.425 | 0.427 | [report](../results/result_lamp_llama_Epic_False.txt) |

## MD — offensiveness

### Unnamed · Adaptation: training time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Llama-3.1-8B-Instruct | Base zero | — | **0.398** | 0.397 | 0.528 | [report](../results/result_llama_MD_False.txt) |

### Unnamed · Adaptation: inference time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Llama-3.1-8B-Instruct | LaMP | — | **0.604** | 0.545 | 0.581 | [report](../results/result_lamp_llama_MD_False.txt) |
| 2 | Mixtral-8x7B-Instruct-v0.1 | LaMP | — | 0.447 | 0.582 | 0.708 | [report](../results/result_lamp_mixtral_MD_False.txt) |

## MHS — hateful

### Unnamed · Adaptation: training time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Mixtral-8x7B-Instruct-v0.1 | Base zero | — | **0.437** | 0.514 | 0.624 | [report](../results/result_mixtral_MHS_False.txt) |
| 2 | Llama-3.1-8B-Instruct | Base zero | — | 0.419 | 0.489 | 0.603 | [report](../results/result_llama_MHS_False.txt) |

### Unnamed · Adaptation: inference time · Extended training: no

| Rank | Model | Method | Representation | Global macro-F1 | Annotator macro-F1 | Text macro-F1 | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Mixtral-8x7B-Instruct-v0.1 | LaMP | — | **0.713** | 0.672 | 0.726 | [report](../results/result_lamp_mixtral_MHS_False.txt) |
| 2 | Llama-3.1-8B-Instruct | LaMP | — | 0.666 | 0.624 | 0.659 | [report](../results/result_lamp_llama_MHS_False.txt) |

## Updating the leaderboard

To add a verified result:

1. Add one row to [scores.csv](./scores.csv).
2. Add the result to the matching task-variant table above.
3. Rank entries by global macro-F1 within that table.

