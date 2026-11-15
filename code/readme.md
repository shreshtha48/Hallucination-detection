# code/: probes, control and steering

Each notebook is standalone and runs on a free Kaggle **T4 x2** with Internet on and a secret `HF_TOKEN`.
Every notebook checkpoints to [`shreshtha2002/hallucination-probes-multiturn`](https://huggingface.co/shreshtha2002/hallucination-probes-multiturn)
and resumes where it stopped if Kaggle restarts. The notebooks reuse the paper repo's code (losses, span search, metrics, label map)
and ARENA 1.3.1's probes (`LRProbe`, `MMProbe`, PCA).

| Notebook | What it does | Results folder on the Hub |
|---|---|---|
| [`probing/hindi-data/C_probe_hindi_llama.ipynb`](probing/hindi-data/C_probe_hindi_llama.ipynb), [`…_aya.ipynb`](probing/hindi-data/C_probe_hindi_aya.ipynb) | Trains and evaluates every detector on the annotated Hindi answers | `hindi-llama/`, `hindi-aya/` |
| [`probing/multiturn/C_probe_multiturn_llama.ipynb`](probing/multiturn/C_probe_multiturn_llama.ipynb), [`…_aya.ipynb`](probing/multiturn/C_probe_multiturn_aya.ipynb) | The same on the multi-turn conversations, reported overall and per turn | `multiturn-llama/`, `multiturn-aya/` |
| [`steer and control/D_control_llama3b.ipynb`](steer%20and%20control/D_control_llama3b.ipynb) | English control, end to end: Llama 3.2 3B answers the English versions of the Hindi prompts, the same judge labels them, the same probes are trained, and the results are compared with Hindi | `control-en-llama3.2-3b-instruct-q8-0/` |
| [`steer and control/E_steer_hindi.ipynb`](steer%20and%20control/E_steer_hindi.ipynb) | Finds the hallucination direction in both 8B models, locates it (MLP neurons, attention heads), removes or steers it with SafeTune, and judges new Hindi answers | `steer-hindi/` |

## Detectors (probe notebooks)

All detectors read the frozen model in fp16, from one forward pass per answer, at every layer.

| Detector | What it is | Source |
|---|---|---|
| `lin` | Linear probe on the residual stream | Obeso et al. (2025) |
| `mlp` | Non-linear probe: 64 hidden units, GELU | Liang & Wang (2025), [arXiv 2512.20949](https://arxiv.org/abs/2512.20949) |
| `attn` | Linear probe on each layer's attention output | ours |
| `head` | A linear probe on every attention head's output | ours |
| `LR` | Logistic regression on entity tokens | ARENA 1.3.1 `LRProbe` |
| `MM` | Difference of means on entity tokens | ARENA 1.3.1 `MMProbe`; Marks & Tegmark (2023) |
| `entropy` | Next-token entropy / log(vocabulary size), no training | baseline |
| `paper` | The paper's released `llama3_1_8b_linear` probe, layer 30, used as is (Llama only) | Obeso et al. (2025) |

## Settings

| Setting | Hindi | Multi-turn | English control |
|---|---|---|---|
| Training data | our answers only | ours + 150 paper answers per source | our answers only |
| Max length (cut from the left, latest answer kept) | 2,560 | 2,048 | 2,560 |
| Evaluation | 2-fold cross-validation by prompt | 2-fold by conversation | 2-fold by prompt |
| Optimizer | AdamW, lr 5e-4, weight decay 0, batch 8 | same | same |
| Epochs | 10 | 2 | 10 |
| Learning-rate schedule | linear decay to 0, gradients clipped at 1.0 | same | same |
| Token weights | entity tokens 10, other answer tokens 1, prompt ignored | same | same |
| Logistic-regression C | chosen per layer from {1e-4 … 1} by grouped 3-fold CV | same | same |
| Seed | 42 | 42 | 42 |

**Loss** (`lin`, `mlp`, `attn`, `head`): the paper repo's token-level weighted BCE and span-max BCE,
`L = (1 − ω) · L_BCE + ω · L_max`, with ω rising linearly from 0 to 1 over training.

**Metrics:** the repo's `compute_clf_metrics` at three levels: all answer tokens, entity tokens, and span-max (each entity
scored by its highest token). We report AUC, recall at 10% FPR, F1, precision, recall, accuracy and the accuracy-optimal threshold.

## Steering notebook (E)

1. **Direction:** the difference of means between hallucinated and supported entity tokens. It is cross-fitted over 2
   folds, and the layer is chosen from the middle 60% of layers by held-out span-max AUC.
2. **Locating it:**
   - SafeTune `identify_safety_neurons` (weight mode) ranks the top 16 MLP neurons per layer.
   - Per head, it records how strongly the output weights align with the direction, how much the head actually writes along
     it, and where the head attends.
3. **Teacher-forced sweep:** SafeTune `RefusalDirectionModel` either ablates the direction at every layer or steers with
   α ∈ {±1, ±2, ±4, ±8} × the mean difference. For each condition it measures the change in log-probability of
   hallucinated vs supported entities, the KL from the unchanged model, and a held-out logistic-regression detector.
   The generation strength is the most negative α with KL < 0.1.
4. **New answers:** 20 new Hindi prompts are answered unchanged, with the direction ablated, and steered away. The same
   web-search judge labels them, and changes come with paired bootstrap 95% intervals.

## Outputs per run (Hub, `RUN/`)

- **Probe notebooks:**
  - `fold0.pt`, `fold1.pt`: probe weights, LR/MM directions per layer, and per-token scores of every held-out answer
  - `results_<run>.csv`: every metric for every detector and layer
  - `head_auc_<run>.npy`
  - `figs/`
- **Steering:** `direction.pt`, `neurons.json`, `heads.pt`, `sweep.jsonl`, `gen_*.jsonl`, `annotations.jsonl`, CSVs and `figs/`.

