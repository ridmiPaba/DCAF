# DCAF — Dynamic Content-Augmented Factorization Machine with Inverse Propensity Weighting

A cold-start recommender system that addresses **zero-interaction items**, **inherited popularity bias**, and **dynamic profile instability** in a single end-to-end trained model — evaluated on the MovieLens 1M dataset.

---

## What is DCAF?

Standard collaborative filtering cannot recommend new items because it has no interaction data to learn from. DCAF solves this through three integrated mechanisms:

| Mechanism | Problem it solves |
|---|---|
| **SBERT Content Encoding** | Gives new items a meaningful representation on day one using title and genre text |
| **Inverse Propensity Weighting (IPW)** | Corrects popularity bias by upweighting rare items during BPR training |
| **Exponential Moving Average (EMA) Fusion** | Smoothly transitions item profiles from content-driven to interaction-driven as interactions accumulate |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Loads MovieLens 1M, encodes content with SBERT, trains the model, evaluates, and saves all artifacts to `saved_model/`.

```bash
python train.py
```

> **Note:** This must be run before launching the dashboard. All files in `saved_model/` are required by `app.py`.

### 3. (Optional) Run the ablation study

Trains five model variants and saves results to `saved_model/ablation_results.pkl`.

```bash
python ablation.py
```

Variants compared: Full DCAF, No IPW, No EMA, No SBERT, MF-BPR baseline.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## Project Structure

```
├── model_logic.py       # DCAFSystem class and custom Keras layers
├── train.py             # Full training pipeline — run this first
├── app.py               # Streamlit research dashboard
├── ablation.py          # Ablation study across five model variants
├── data/                # MovieLens 1M data files (see data/README)
├── saved_model/         # Created by train.py — required by app.py
│   ├── dcaf_keras_model.keras
│   ├── dcaf_state.pkl
│   ├── splits.pkl
│   ├── evaluation_results.pkl
│   ├── content_embeddings.npy
│   └── ablation_results.pkl  (created by ablation.py)
└── requirements.txt
```

---

## System Architecture

DCAF is a six-input Factorization Machine with three parallel input paths:

**User Path**
- `user_idx` → `Embedding(6040, 64)` → user embedding
- `gender`, `age`, `occupation` → concatenated → `Dense(64, relu)` → demographic projection
- Both vectors are added → combined user representation (64-dim)

**Content Path**
- `movie_idx` → `ContentAnchorLayer` → frozen SBERT vector (384-dim)
- → `Dense(64, linear)` → content projection (64-dim)

**Interaction Path**
- `movie_idx` → `Embedding(3706, 64)` → interaction embedding (64-dim)
- `count` → passed to EMAFusionLayer as interaction count t

**EMA Fusion**

The `EMAFusionLayer` blends the content projection and interaction embedding:

$$e_i(t) = \alpha(t) \times \text{content\_projection} + (1 - \alpha(t)) \times \text{interaction\_embedding}$$

$$\alpha(t) = \frac{k}{k + t}$$

where `t` is the item's historical interaction count and `k = 5.0` (default). At `t=0`, α=1.0 — the representation is 100% content-driven. At `t=50`, α=0.091 — 90.9% interaction-driven.

**Scoring**

The fused movie vector and combined user vector are combined via dot product, plus per-user and per-item bias terms and a global learnable bias, producing a scalar relevance score.

---

## Training Details

- **Loss function:** Custom BPR (Bayesian Personalised Ranking) via `tf.GradientTape` — not `model.compile()`
- **Negative sampling:** Uniform random (not popularity-weighted)
- **IPW:** Per-item weights computed from training frequency, log-normalised, scaled to [1.0, 5.0], capped at 99th percentile
- **Pair weighting:** `pair_weight = (w_pos + w_neg) / 2` — two-sided weighting
- **Epochs:** 30 | **Batch size:** 1024 | **Embedding dim:** 64 | **Learning rate:** 0.001
- **Reproducibility:** Seeds set to 42 for Python, NumPy, and TensorFlow

---

## Evaluation

Two protocols are used:

| Protocol | Description | Purpose |
|---|---|---|
| **Sampled Ranking** (primary) | 1 positive vs 100 random negatives | Within-project comparisons |
| **Full Ranking** (secondary) | 1 positive vs all 3,706 catalogue items | Literature comparisons |

Items are segmented by training interaction count:

| Segment | Interaction count | Description |
|---|---|---|
| Cold | 0 – 5 | New items with no or minimal history |
| Maturing | 6 – 50 | Items in transition |
| Warm | 51+ | Established items with sufficient history |

Metrics reported: **NDCG@10**, **Hit Rate@10**, **Novelty@10**, **Item Coverage**

---

## Key Results

| Model | Cold NDCG@10 | Cold/Warm Hit Rate Ratio |
|---|---|---|
| MF-BPR Baseline | 0.0014 | 0.74% |
| **Full DCAF** | **0.2377** | **83.5%** |

Ablation study — contribution of each component:

| Component removed | Cold NDCG@10 | Drop vs Full DCAF |
|---|---|---|
| Full DCAF | 0.2177 | — |
| No SBERT (random vectors) | 0.0213 | 10.2× worse |
| No EMA (α fixed at 1.0) | 0.0150 | 14.5× worse |
| No IPW (uniform weights) | 0.0384 | 5.7× worse |
| MF-BPR (no content or EMA) | 0.0014 | 155× worse |

---

## Dashboard Pages

The Streamlit dashboard (`app.py`) has six pages:

| Page | What it shows |
|---|---|
| **Overview** | Research problem, architecture summary, research questions |
| **Evaluation & Metrics** | NDCG, Hit Rate, Novelty, Item Coverage — sampled and full ranking |
| **Live Recommendations** | Select a user, generate top-K recommendations, genre match analysis |
| **EMA Profile Explorer** | α(t) decay curve and content/interaction weight breakdown for any movie |
| **Content Similarity** | Most similar movies by SBERT content, coloured by lifecycle segment |
| **Ablation Study** | Side-by-side comparison of all five model variants |

---

## Reproducing Results

All results are fully reproducible with fixed random seeds (42). To reproduce:

```bash
python train.py       # trains model and saves evaluation results
python ablation.py    # runs ablation study
streamlit run app.py  # view results in dashboard
```

Training runs on CPU only. GPU is explicitly disabled to ensure reproducibility on Apple M2 hardware. Expected training time: approximately 45–60 minutes on Apple M2.

---

## Limitations

- The **Maturing segment** (6–50 interactions) collapses to NDCG@10 = 0.0091 due to a count regime mismatch between training (running count) and evaluation (total countmap). This is a known limitation documented in the thesis.
- Evaluated on **MovieLens 1M only** — generalisation to other datasets has not been established.
- Content anchor uses **title and genre only** — no plot synopsis, director, or cast.


---

## Dependencies

See `requirements.txt`. Key libraries:

| Library | Purpose |
|---|---|
| TensorFlow 2.x | Model construction and custom BPR training loop |
| sentence-transformers | SBERT content encoding (all-MiniLM-L6-v2) |
| pandas / numpy | Data loading and preprocessing |
| scikit-learn | Label encoding |
| streamlit | Research dashboard |
| matplotlib | Evaluation visualisation |

---

## Citation

If you use this work, please cite:

> Hemachandra, G. P. R. (2026). *DCAF: A Recommender System with Dynamic Content-Augmented Factorization Machine and Inverse Propensity Weighting*. BEng Final Year Project, Informatics Institute of Technology / University of Westminster.
