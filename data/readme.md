# data/: building the datasets

These notebooks make every dataset the probes are trained on: translate the prompts, generate answers, and label every
entity in them with a web-search judge. Each notebook saves to the Hugging Face Hub as it goes and resumes where it stopped.

## Run order

| # | Notebook | What it does | Output on the Hub |
|---|---|---|---|
| 1 | [`longfact_hindi.ipynb`](longfact_hindi.ipynb) | Translates every LongFact (2,280 prompts) and LongFact++ prompt into Hindi with [Sarvam-Translate](https://huggingface.co/sarvamai/sarvam-translate), keeping the original files' format and line order. Colab/Kaggle T4. | [`shreshthamodi02/longfact-hindi`](https://huggingface.co/datasets/shreshthamodi02/longfact-hindi) |
| 2 | [`longfact_hindi_paper.ipynb`](longfact_hindi_paper.ipynb) | Matches each translation to the paper's own prompts and splits (`obalcells/longfact-annotations`, `obalcells/longfact-augmented-annotations`) and repackages them in the paper's format. Every row keeps the paper's exact English prompt (`prompt_en`). CPU only. | [`shreshthamodi02/longfact-hindi-paper`](https://huggingface.co/datasets/shreshthamodi02/longfact-hindi-paper) |
| 3 | [`A_longform_hindi_generation.ipynb`](A_longform_hindi_generation.ipynb) | Llama 3.1 8B and Aya Expanse 8B answer the Hindi test prompts with the paper's settings (single turn, temperature 0.1, up to 2,048 new tokens), through Ollama `q8_0` builds, one model per T4. | [`longfact-hi-llama3.1-8b-instruct-q8-0`](https://huggingface.co/datasets/shreshthamodi02/longfact-hi-llama3.1-8b-instruct-q8-0), [`longfact-hi-aya-expanse-8b-q8-0`](https://huggingface.co/datasets/shreshthamodi02/longfact-hi-aya-expanse-8b-q8-0) |
| 4 | [`longfact-multiturn.ipynb`](longfact-multiturn.ipynb) | Three-turn conversations. Turn 1 is a paper prompt. Each follow-up is written by Aya Expanse 8B (`aya-expanse:8b`) and asks for more specific facts about the target model's own previous answer. Settings as in the paper, with an 8,192-token context. Ends with a CuratorKIT pass (length checks, duplicate removal). | [`longfact-multiturn-llama3.1-8b-instruct-q8-0`](https://huggingface.co/datasets/shreshthamodi02/longfact-multiturn-llama3.1-8b-instruct-q8-0), [`longfact-multiturn-aya-expanse-8b-q8-0`](https://huggingface.co/datasets/shreshthamodi02/longfact-multiturn-aya-expanse-8b-q8-0) |
| 5 | [`annotation/B_annotate_hindi_llama.ipynb`](annotation/B_annotate_hindi_llama.ipynb), [`annotation/B_annotate_hindi_aya.ipynb`](annotation/B_annotate_hindi_aya.ipynb) | Labels every entity in the Hindi answers (one notebook per model, so they can run in parallel). | [`longfact-hi-…-annotated`](https://huggingface.co/datasets/shreshtha2002/longfact-hi-llama3.1-8b-instruct-q8-0-annotated) |
| 6 | [`annotation/B_annotate_websearch.ipynb`](annotation/B_annotate_websearch.ipynb) | Labels every entity in the multi-turn answers. The judge sees the earlier turns but labels only the latest answer. | [`longfact-multiturn-…-annotated`](https://huggingface.co/datasets/shreshtha2002/longfact-multiturn-llama3.1-8b-instruct-q8-0-annotated) |

The English control (Llama 3.2 3B on the same prompts as Hindi) builds its own data inside
[`code/steer and control/D_control_llama3b.ipynb`](../code/steer%20and%20control/D_control_llama3b.ipynb).

## The annotation judge

| Setting | Value |
|---|---|
| Judge model | `openai/gpt-oss-120b` through Hugging Face Inference Providers (Groq), temperature 0, up to 8,192 tokens |
| Web search | Exa's hosted MCP server (`web_search_exa`), with DuckDuckGo as fallback; at most 15 searches per answer |
| Prompt | The paper repo's `entity_annotation.prompt`, unchanged for English. Hindi adds one note: copy spans exactly in Devanagari, search in English or Hindi, write notes in English. |
| Parsing | The repo's `parse_and_validate_json` and `assign_span_positions`. For Hindi, ROUGE uses a Unicode tokenizer, because the default one drops Devanagari. |
| Labels | `Supported` (verified as correct), `Not Supported` (appears fabricated or incorrect), `Insufficient Information` (cannot verify). As in the paper's repo, `Insufficient Information` counts as hallucinated. |
| Cost | about $0.05 per answer |

## Format

Every dataset follows the paper's format: one row per answer, with `conversation` (the messages up to and including the
answer), `subset`, `split` and `canary`, plus:

- **Hindi rows:** `lang`, `prompt_id` and `ollama_tag`.
- **Multi-turn rows:** `conversation_id`, `turn` and `followup_writer`.
- **Annotated rows:** `annotations`, a list of `{span, label, verification_note, index}`, where `index` is the span's character
  position in the answer (or `null` if it could not be found), and `annotator`.

Annotated datasets keep their progress in `progress/progress.jsonl`. That is the file the probe notebooks read.

## Kaggle / Colab setup

GPU **T4 x2** (translation needs one T4; the annotation notebooks run on CPU), Internet **on**, and a secret `HF_TOKEN`:
a write token for the account that receives the outputs. The annotation notebooks need "Make calls to Inference Providers".

