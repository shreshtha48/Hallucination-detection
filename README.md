# Hallucination-detection

Extension of the real-time hallucination detection paper for multiturn settings and hindi language models.

This project extends **Obeso et al. (2025), [Real-Time Detection of Hallucinated Entities in Long-Form Generation](https://arxiv.org/abs/2509.03531)**
([code](https://github.com/obalcells/hallucination_probes)). That paper trains probes on a frozen LLM's hidden states to flag
hallucinated *entities* (names, dates, numbers, places, citations) token by token while the model writes a long answer.
The paper works in English and on single-turn answers. We take the same idea to:

- **Hindi:** LongFact prompts translated with [Sarvam-Translate](https://huggingface.co/sarvamai/sarvam-translate), answered natively in Hindi.
- **Multi-turn conversations:** three-turn LongFact conversations in which every follow-up asks for more specific facts about the previous answer.
- **Steering:** we find a "hallucination direction" in the model and test whether removing it reduces hallucination.

Target models: [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[Aya Expanse 8B](https://huggingface.co/CohereLabs/aya-expanse-8b). The English control uses
[Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on the same prompts.
Everything runs in free Kaggle notebooks (T4 x2).

**[Interactive examples →](https://shreshtha48.github.io/Hallucination-detection/)**: answers in Hindi and in multi-turn conversations, each word
shaded by the probe's score and underlined with the judge's label.

![Pipeline](results/pipeline-figures/fig3_pipeline_methods.png)

## What we did

1. **Generate:** the target model answers every prompt with the paper's settings (temperature 0.1, up to 2,048 new tokens),
   through Ollama `q8_0` builds.
2. **Annotate:** an LLM judge with web search labels every entity in the answer as *Supported*, *Not Supported* or
   *Insufficient Information*. It uses the paper repo's prompt and parsing code. The judge is `openai/gpt-oss-120b` through
   Hugging Face Inference Providers (Groq), searching with Exa MCP and DuckDuckGo as fallback. For Hindi it gets one extra
   note (copy spans exactly in Devanagari). For multi-turn it sees the earlier turns but labels only the latest answer.
   As in the paper's repo, *Insufficient Information* counts as hallucinated.
3. **Train detectors** on the frozen model's activations at every layer:
   - the paper's linear probe (token BCE annealed into span-max BCE)
   - a non-linear (MLP) probe
   - linear probes on the attention output and on every attention head
   - logistic regression and difference of means (ARENA 1.3.1)
   - next-token entropy, with no training
   - for Llama, the paper's released probe as a reference
4. **Evaluate** with 2-fold cross-validation (by prompt or by conversation), so every answer is scored by a detector that never
   saw it. Metrics come from the paper repo's `compute_clf_metrics`; the headline is **span-max AUC**.
5. **Steer:** with [SafeTune](https://github.com/Lexsi-Labs/SafeTune), remove the hallucination direction (difference of
   means between hallucinated and supported entities) or steer away from it, then judge new Hindi answers.

## Results

### Hindi hallucinates more than English

Judge labels on the same 60 LongFact prompts. NS = Not Supported, II = Insufficient Information.

| Run | Entities / answer | NS % | NS + II % | Answers with ≥ 1 NS % |
|---|---|---|---|---|
| English · Llama 3.2 3B | 21.4 | 28.5 | 32.3 | 76.7 |
| Hindi · Llama 3.1 8B | 6.4 | 37.3 | 40.2 | 48.3 |
| Hindi · Aya Expanse 8B | 10.0 | 33.5 | 43.6 | 71.7 |

Hindi answers name far fewer entities, and a larger share of them are hallucinated. The English control is a smaller model (3B).

### Hallucination grows over a conversation

![Hallucinated entities by turn](results/results-figures/R5_multiturn_hallucination_by_turn.png)

For Llama the share of hallucinated entities more than doubles after the first follow-up (about 27% → 64%). For Aya it rises
more slowly (about 25% → 35%).

### Probes find hallucinated entities in Hindi and across turns

Best span-max AUC per detector, held-out answers. The layer is chosen on the same scores, so these numbers are optimistic.

| Detector | Hindi · Llama | Hindi · Aya | Multi-turn · Llama | Multi-turn · Aya | English · Llama 3B |
|---|---|---|---|---|---|
| Linear probe | 0.638 (L13) | 0.702 (L20) | 0.818 (L9) | 0.781 (L16) | 0.652 (L25) |
| MLP probe | 0.631 (L12) | 0.725 (L15) | 0.829 (L12) | **0.812** (L18) | 0.699 (L26) |
| Attention-output probe | 0.652 (L7) | 0.699 (L18) | 0.753 (L31) | 0.779 (L7) | 0.645 (L26) |
| Logistic regression | 0.685 (L7) | **0.732** (L14) | 0.836 (L13) | 0.783 (L18) | **0.781** (L11) |
| Difference of means | 0.691 (L14) | 0.711 (L15) | 0.768 (L11) | 0.677 (L15) | 0.698 (L17) |
| Next-token entropy | 0.639 | 0.681 | 0.612 | 0.685 | 0.731 |
| Paper's released probe (L30, English-trained) | **0.762** | – | **0.873** | – | – |

- **In Hindi:** simple detectors (logistic regression, difference of means) do as well as or better than the trained linear
  probe. The paper's English-trained probe still reaches 0.76 on Llama's Hindi answers.
- **Across turns:** probes hold up turn by turn (paper probe on Llama: 0.86 / 0.85 / 0.88). Llama's entropy baseline falls
  from 0.79 at turn 1 to 0.49 / 0.59 at turns 2 and 3.

![AUC by turn](results/results-figures/R4_multiturn_auc_by_turn.png)

### Removing the hallucination direction lowers Aya's hallucination in Hindi

![Steering](results/results-figures/R8_steering_behaviour.png)

New Hindi prompts, judged by the same web-search judge. Changes are paired against the unchanged model, with 95% bootstrap intervals over prompts.

| Model | Direction layer | Unchanged | Direction removed | Steered away (α = −2) |
|---|---|---|---|---|
| Aya Expanse 8B | L15 | 61.9% | **47.8%** (−14.2 pts, CI −22.7 to −3.9) | 51.2% (−10.7, CI −26.0 to +2.6) |
| Llama 3.1 8B | L14 | 63.8% | 65.9% (+2.1, CI −18.0 to +26.4) | 59.9% (−3.9, CI −22.4 to +15.3) |

- **Effect on the detector:** removing the direction drops a held-out detector at that layer to chance (0.65 → 0.40 for
  Llama, 0.71 → 0.47 for Aya), with a KL of about 0.02 nats per token.
- **Side effects:** no answer refused, and answers stayed in Hindi. For Llama, removal made answers about 40% shorter.
- **Sample size:** 18–19 prompts per model were judged under all three conditions.

More figures: [`results/`](results/) (per-run layer curves, per-head heatmaps, attention, correlation matrices).

## Interactive examples

Each page shows one answer: the background is the linear probe's score at its best layer (red = above the probe's
threshold), and the underline is the web-search judge's label (red = hallucinated, green = supported). Hover a word for its score.

- **Hindi:** [Llama 3.1 8B](https://shreshtha48.github.io/Hallucination-detection/#hindi-llama) · [Aya Expanse 8B](https://shreshtha48.github.io/Hallucination-detection/#hindi-aya)
- **Multi-turn (one conversation, three turns):** [Llama 3.1 8B](https://shreshtha48.github.io/Hallucination-detection/#multiturn-llama) · [Aya Expanse 8B](https://shreshtha48.github.io/Hallucination-detection/#multiturn-aya)
- **English control:** [Llama 3.2 3B](https://shreshtha48.github.io/Hallucination-detection/#control)
- **Attention patterns** of the head that pushes hardest toward "hallucinated": [all models](https://shreshtha48.github.io/Hallucination-detection/#attention)

## Repository layout

| Folder | What's in it |
|---|---|
| [`data/`](data/) | Notebooks that build the datasets: translation, generation, multi-turn conversations, web-search annotation |
| [`code/`](code/) | Probe training and evaluation (Hindi, multi-turn), the English control, steering |
| [`results/`](results/) | Metrics CSVs and figures for every run, plus the scripts that drew the summary figures |
| [`docs/`](docs/) | The interactive examples (served with GitHub Pages) |
| [`demo/`](demo/) | A live demo: the model answers and the probe shades every new token (Gradio, Kaggle or a Space) |

## Data, probes and results on Hugging Face

| What | Link |
|---|---|
| Hindi prompts (Sarvam-Translate) | [longfact-hindi](https://huggingface.co/datasets/shreshthamodi02/longfact-hindi) · [longfact-hindi-paper](https://huggingface.co/datasets/shreshthamodi02/longfact-hindi-paper) (paper test split, Hindi + English) |
| Hindi answers | [Llama](https://huggingface.co/datasets/shreshthamodi02/longfact-hi-llama3.1-8b-instruct-q8-0) · [Aya](https://huggingface.co/datasets/shreshthamodi02/longfact-hi-aya-expanse-8b-q8-0) |
| Hindi answers, annotated | [Llama](https://huggingface.co/datasets/shreshtha2002/longfact-hi-llama3.1-8b-instruct-q8-0-annotated) · [Aya](https://huggingface.co/datasets/shreshtha2002/longfact-hi-aya-expanse-8b-q8-0-annotated) |
| Multi-turn conversations | [Llama](https://huggingface.co/datasets/shreshthamodi02/longfact-multiturn-llama3.1-8b-instruct-q8-0) · [Aya](https://huggingface.co/datasets/shreshthamodi02/longfact-multiturn-aya-expanse-8b-q8-0) |
| Multi-turn conversations, annotated | [Llama](https://huggingface.co/datasets/shreshtha2002/longfact-multiturn-llama3.1-8b-instruct-q8-0-annotated) · [Aya](https://huggingface.co/datasets/shreshtha2002/longfact-multiturn-aya-expanse-8b-q8-0-annotated) |
| English control (Llama 3.2 3B) | [answers](https://huggingface.co/datasets/shreshtha2002/longfact-en-llama3.2-3b-instruct-q8-0) · [annotated](https://huggingface.co/datasets/shreshtha2002/longfact-en-llama3.2-3b-instruct-q8-0-annotated) |
| Trained probes, metrics, figures, steering outputs | [shreshtha2002/hallucination-probes-multiturn](https://huggingface.co/shreshtha2002/hallucination-probes-multiturn) (`hindi-*`, `multiturn-*`, `control-en-*`, `steer-hindi`) |
| Paper data we build on | [longfact-annotations](https://huggingface.co/datasets/obalcells/longfact-annotations) · [longfact-augmented-annotations](https://huggingface.co/datasets/obalcells/longfact-augmented-annotations) · [longfact-augmented-prompts](https://huggingface.co/datasets/obalcells/longfact-augmented-prompts) · [released probes](https://huggingface.co/obalcells/hallucination-probes) |

## Reproduce

Run the notebooks in order on Kaggle (GPU **T4 x2**, Internet on, an `HF_TOKEN` secret with access to the gated models and
"Make calls to Inference Providers" for the judge). See [`data/`](data/readme.md) and [`code/`](code/readme.md) for the
order and settings. Every notebook checkpoints to the Hub and resumes where it stopped.

## Limitations

- **Small samples:** about 60 annotated answers per run (19–22 per turn), so differences of a few AUC points are within noise.
- **Optimistic AUCs:** the best layer is picked on the same held-out scores it is reported on.
- **The English control is Llama 3.2 3B** while the Hindi runs are 8B, so the English vs Hindi gap mixes language with model size.
- **The judge** is gpt-oss-120b with free web search, not the paper's Claude judge.
- **No random-direction control yet** for the steering result.

## Citation

If you use this work, please also cite the paper it builds on:

```bibtex
@article{obeso2025realtime,
  title   = {Real-Time Detection of Hallucinated Entities in Long-Form Generation},
  author  = {Obeso, Oscar and Arditi, Andy and Ferrando, Javier and Freeman, Joshua and Holmes, Cameron and Nanda, Neel},
  journal = {arXiv preprint arXiv:2509.03531},
  year    = {2025}
}
```
