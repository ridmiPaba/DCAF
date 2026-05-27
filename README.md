# DCAF — Dynamic Content-Augmented Factorization

Brief overview, setup and system architecture for the DCAF research project.

**Project**
- DCAF augments a factorization-machine BPR recommender with semantic content (SBERT),
  exponential moving-average (EMA) fusion, and inverse propensity weighting (IPW).

**Quick start**
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training (creates `saved_model/` artifacts used by the dashboard):

```bash
python train.py
```

3. (Optional) Run ablation study:

```bash
python ablation.py
```

4. Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

**Dependencies**
- See [requirements.txt](requirements.txt#L1-L8).

**Repository files**
- `train.py`: pipeline to load MovieLens, encode content, train model, evaluate, and save artifacts.
- `app.py`: Streamlit dashboard that loads saved artifacts and exposes visualizations and live recommendations.
- `model_logic.py`: core `DCAFSystem` class and custom Keras layers (`ContentAnchorLayer`, `EMAFusionLayer`, `GlobalBiasLayer`).
- `ablation.py`: trains multiple variants (No IPW / No EMA / No SBERT / MF baseline) and saves results.
- `data/`: expected MovieLens 1M files (see `data/README`).
- `saved_model/`: contains trained model and artifacts (created by `train.py`).

**System Architecture (high level)**

Components:
- Content Encoder (SBERT): encodes `Title + Genres` → 384-dim semantic vectors saved as `content_embeddings.npy`.
- ContentAnchorLayer: a Keras layer that stores the fixed content matrix and returns content anchors by `movie_idx`.
- Interaction Embeddings: learnable `movie_interaction_embedding` that captures collaborative signals.
- EMAFusionLayer: fuses content and interaction vectors using an exponential moving average:

  $\alpha(t) = \dfrac{k}{k + t}$

  fused_vector = $\alpha(t) \cdot$ content_proj + $(1-\alpha(t))\cdot$ interaction_vec

  where $t$ = historical interaction count for the item, and $k$ = `k_decay` hyperparameter (default 5.0).
- Factorisation backbone: user and movie embeddings, dot-product score, per-user and per-item bias, and a global bias.
- Inverse Propensity Weighting (IPW): training pair weights computed from movie popularity, log-scaled and capped (p99), used to upweight rare items during BPR training.

Data flow (training):
1. `DCAFSystem.load_data()` — load MovieLens files, create `user_idx`/`movie_idx` encodings.
2. `DCAFSystem.encode_content()` — encode title+genres with SBERT → `content_embeddings.npy`.
3. `DCAFSystem.prepare_training_data()` — chronological per-user train/test split, compute `countmap` and IPW weights.
4. `DCAFSystem.build_model()` — assemble model with `ContentAnchorLayer`, `EMAFusionLayer` and embedding layers.
5. `DCAFSystem.train()` — custom BPR training loop with two-sided IPW, uniform negative sampling, and demographic features.
6. Evaluation: sampled ranking (1 vs 100, primary) and full ranking (1 vs all) are computed and saved.

Runtime (inference / dashboard):
- `app.py` loads the saved Keras model and system state (from `saved_model/`), restores custom layers, and exposes:
  - Live recommendations (`DCAFSystem.get_recommendations()`)
  - EMA evolution visualiser (`get_ema_evolution_data()`)
  - Content similarity queries (`get_similar_items_by_content()`)
  - Ablation results (if `ablation_results.pkl` is present)

**Evaluation protocols**
- Sampled Ranking — primary: 1 positive vs 100 negatives (used for within-project comparisons).
- Full Ranking — 1 vs all catalogue items (comparable to literature benchmarks; sampled users for speed).

**Ablation study**
- Implemented in `ablation.py` — compares Full DCAF against No IPW, No EMA, No SBERT, and an MF-BPR baseline.
- Results saved to `saved_model/ablation_results.pkl` and surfaced in the dashboard.

**Notes & reproducibility**
- Training saves artifacts to `saved_model/` (see `train.py` paths). `app.py` expects those files to exist before launching.
- Training uses a custom TensorFlow training loop (no `model.compile()`), so model saving/loading preserves custom layers via `@register_keras_serializable()`.

**Where to look next**
- Model implementation and layers: [model_logic.py](model_logic.py#L1-L400)
- Training pipeline and saved artifact layout: [train.py](train.py#L1-L400)
- Dashboard & visualisation: [app.py](app.py#L1-L200)

If you want, I can add a visual Mermaid diagram of the architecture to this README. Would you like that?
# DCAF
