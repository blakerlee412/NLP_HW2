# Does Option Label Format Affect Korean MCQA?

Reproducibility materials for NLP HW2. This repository reorganizes the code used to compare Latin labels (A/B/C/D) and Hangul consonants (ㄱ/ㄴ/ㄷ/ㄹ) on Qwen2.5-3B-Instruct with 160 questions from four KMMLU test subjects.

**Research status:** 1,280 primary responses were collected in Google Colab. The optional 400-response exhaustive-permutation check was **not** run. This repository's clean notebook has **not** been re-executed end-to-end on a new GPU; the original user notebook was executed in Colab. Please do not claim fresh-machine reproduction until tested.

## Files

- `NLP_HW2_clean.ipynb`: cleaned Colab notebook for sampling, reviewing selected IDs, prompt generation, inference, statistics, and charts. Outputs and unrelated diagnostic cells have been removed.
- `analyze_results.py`: **CPU-only** script to recompute the main table, bootstrap CIs, paired sign-flip tests, Holm adjustment, subject-level analysis and position-level analysis from the main saved result CSV.
- `requirements.txt`: direct versions for Transformers, Datasets and Accelerate. Other packages may be preinstalled in Colab.
- `.gitignore`: prevents accidental inclusion of some dataset-text files. **Review staged files manually before publishing.**

The actual `main_1280_results.csv` was saved to the user's Google Drive and was **not included in the uploaded notebook**. Copy it from `MyDrive/NLP_HW2/results/main_1280_results.csv` to a local `results/` folder to reproduce published results. The `sampling_metadata.json`, `review_log.json` and `inference_config.json` may be added from Google Drive for documented provenance; review their contents before uploading.

## Environment

Original experiment: Google Colab, NVIDIA Tesla T4, FP16 model, greedy generation (`do_sample=False`), `max_new_tokens=16`, Korean zero-shot instruction `정답 선택지의 기호 하나만 출력하세요.`, Python 3.13 in the exported runtime, and `torch 2.14.1+cu130` as observed in the user's environment. Exact Colab runtime library versions beyond explicitly pinned packages were not archived in this repository. For more complete provenance, export `pip freeze > environment-original.txt` **from the original environment only**; a newly connected session does not necessarily have the same packages.

The notebook's original run recorded:

| Resource | Revision |
| --- | --- |
| `HAERAE-HUB/KMMLU` | `d61b3f19e552c576bf5960dd24289763edc36a88` |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |

Dataset: [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU). Model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

## Reproducing the main statistical results (no GPU needed)

1. Download the saved results CSV from Google Drive and put it at `results/main_1280_results.csv` in this repo.
2. Install dependencies (e.g. `python -m pip install -r requirements.txt` in a suitable Python environment).
3. Run:

```bash
python analyze_results.py --results results/main_1280_results.csv --output derived_tables
```

This command reconstructs the main summary, the 160-question paired bootstrap confidence intervals (2,000 replicates), paired sign-flip tests (10,000 randomizations), Holm-adjusted p-values, and two exploratory tables. The original script uses one `numpy.random.default_rng(42)` for both paired tests, in H1 then H2 order, so these calculations reproduce the same pseudo-random sequence given the same NumPy behavior and inputs.

Expected summary from the original execution:

| Metric | Latin | Hangul |
| --- | ---: | ---: |
| Accuracy | 40.625% | 38.90625% |
| Four-order consistency | 21.875% | 19.375% |
| Invalid responses | 4/640 | 5/640 |

Main comparisons (Hangul minus Latin): accuracy −1.71875 percentage points; consistency −2.5 percentage points. The original Holm-adjusted p-values were both ≈ 0.865113. Neither contrast was statistically significant at α=0.05.

## Reproducing the inference outputs (GPU required)

1. Open `NLP_HW2_clean.ipynb` in Google Colab and select a T4 GPU, if available.
2. Mount Google Drive. The notebook writes to `MyDrive/NLP_HW2/`.
3. Execute cells from top to bottom. The dataset filtering step and the **19 explicitly documented replacement IDs** reconstruct the final 160 questions, provided the pinned revision remains available.
4. Inference stores raw model responses to `results/main_1280_results.csv`, saving every 8 new outputs. The notebook uses an input fingerprint and resumes already completed experiment IDs. Avoid changing prompts, parser, model revision, or generation parameters mid-run.
5. Inspect saved files and compare counts and accuracy against the main results.

**Important:** This work used an AI-assisted review to flag 19 items. Those manual-review choices are recorded in the notebook as IDs; the automatic text filter alone does not determine those replacement choices. Re-running does **not** independently validate source answer keys. No guarantee of bitwise identical GPU outputs is made.

## Dataset permissions and privacy

The KMMLU dataset card lists **CC BY-ND 4.0**. In this example package, CSV/JSONL files containing question stems, options or complete prompts are **excluded** from distribution; users should obtain data from the official source and review the license before redistributing any dataset-derived text. When publishing, also remove Colab outputs that might contain private Google Drive folder names, mounted paths, or raw questions.

The repository contains no model weights. All experimental outputs and the accompanying paper should disclose the use of ChatGPT for source-code drafting, debugging, dataset review and manuscript assistance.
