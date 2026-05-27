"""
train.py  —  Run this ONCE before launching the frontend.

    python train.py

Saves everything to saved_model/ so app.py can load it.
"""

import os
import pickle
import numpy as np
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    print("GPU disabled — running on CPU.")
except Exception:
    pass

from model_logic import DCAFSystem

# ── Where to save ─────────────────────────────────────────────────────
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Paths  (these MUST match the paths in app.py exactly) ─────────────
PATH_KERAS_MODEL    = os.path.join(SAVE_DIR, "dcaf_keras_model.keras")
PATH_STATE          = os.path.join(SAVE_DIR, "dcaf_state.pkl")
PATH_SPLITS         = os.path.join(SAVE_DIR, "splits.pkl")
PATH_EVAL           = os.path.join(SAVE_DIR, "evaluation_results.pkl")
PATH_EMBEDDINGS     = os.path.join(SAVE_DIR, "content_embeddings.npy")

# ── Hyperparameters ────────────────────────────────────────────────────
EMBEDDING_DIM = 64
EMA_K_DECAY   = 5.0
LEARNING_RATE = 0.001
EPOCHS        = 30
BATCH_SIZE    = 1024
SPLIT_RATIO   = 0.8
EVAL_K        = 10


def main():
    print("\n" + "="*60)
    print("  DCAF  —  Training Script")
    print("="*60)

    dcaf = DCAFSystem(data_path="data")

    # 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/5]  Loading MovieLens 1M dataset …")
    dcaf.load_data()
    print(f"       Users: {dcaf.num_users}  |  Movies: {dcaf.num_movies}")

    # 2. SBERT content encoding ─────────────────────────────────────────
    print("\n[2/5]  Encoding movie content with SBERT …")
    content_embeddings = dcaf.encode_content()
    print(f"       Embeddings shape: {content_embeddings.shape}")

    # 3. Train / test split + IPW ───────────────────────────────────────
    print("\n[3/5]  Preparing train/test splits with IPW …")
    traindf, testdf = dcaf.prepare_training_data(split_ratio=SPLIT_RATIO)
    print(f"       Train rows: {len(traindf)}  |  Test rows: {len(testdf)}")

    # 4. Build + train model ────────────────────────────────────────────
    print("\n[4/5]  Building and training DCAF model …")
    dcaf.build_model(
        content_embeddings,
        embedding_dim=EMBEDDING_DIM,
        k_decay=EMA_K_DECAY,
        learning_rate=LEARNING_RATE,
    )
    dcaf.train(dcaf.model, traindf, testdf, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 5. Evaluate ───────────────────────────────────────────────────────
    print("\n[5/5]  Running evaluation …")

    # 5a. Sampled ranking (1 vs 100) — primary protocol
    print("\n  [5a] Sampled ranking (1 vs 100 negatives) …")
    overall_sampled  = dcaf.evaluate_metrics_sampled(testdf, k=EVAL_K)
    segments_sampled = dcaf.segment_performance(testdf, k=EVAL_K)

    # 5b. Full ranking (1 vs all 3,900) — comparable to CLCRec / CGRC
    print("\n  [5b] Full ranking (1 vs all catalogue items) …")
    overall_full = dcaf.evaluate_metrics(testdf, traindf, k=EVAL_K, sample_size=500)

    print("       Full ranking segmentation …")
    all_movies_list = list(range(dcaf.num_movies))
    cold_set     = set(m for m in all_movies_list if dcaf.countmap.get(m, 0) <= 5)
    maturing_set = set(m for m in all_movies_list if 6 <= dcaf.countmap.get(m, 0) <= 50)
    warm_set     = set(m for m in all_movies_list if dcaf.countmap.get(m, 0) > 50)

    segments_full = {}
    for seg_name, seg_set in [("Cold", cold_set), ("Maturing", maturing_set), ("Warm", warm_set)]:
        seg_testdf = testdf[testdf["movie_idx"].isin(seg_set)]
        if len(seg_testdf) > 0:
            seg_metrics = dcaf.evaluate_metrics(seg_testdf, traindf, k=EVAL_K, sample_size=300)
            segments_full[seg_name] = seg_metrics
            print(f"       {seg_name}: NDCG@{EVAL_K}={seg_metrics['ndcg']:.4f}, "
                  f"Hit Rate={seg_metrics['hit_rate']:.4f}")

    evaluation_results = {
        "k":             EVAL_K,
        "overall":       overall_sampled,    # sampled — primary
        "segments":      segments_sampled,   # sampled segmentation
        "overall_full":  overall_full,       # full ranking overall
        "segments_full": segments_full,      # full ranking segmentation
    }

    # ── Save everything ────────────────────────────────────────────────
    print("\n  Saving artifacts …")

    # 5a. Keras model
    print(f"  → {PATH_KERAS_MODEL}")
    dcaf.model.save(PATH_KERAS_MODEL)

    # 5b. System state (all non-model attributes)
    state = {
        "movies":           dcaf.movies,
        "ratings":          dcaf.ratings,
        "users":            dcaf.users,
        "countmap":         dcaf.countmap,
        "num_users":        dcaf.num_users,
        "num_movies":       dcaf.num_movies,
        "user_encoder":     dcaf.user_encoder,
        "movie_encoder":    dcaf.movie_encoder,
        "movie_popularity": dcaf.movie_popularity,
        "ipw_weights":      dcaf.ipw_weights,
        # Bug 3 fix: persist train_seen so app.py's evaluation uses the
        # correct full-exclusion negative pool (train + test seen items).
        "train_seen":       dcaf.train_seen,
    }
    print(f"  → {PATH_STATE}")
    with open(PATH_STATE, "wb") as f:
        pickle.dump(state, f)

    # 5c. Train / test splits
    print(f"  → {PATH_SPLITS}")
    with open(PATH_SPLITS, "wb") as f:
        pickle.dump({"traindf": traindf, "testdf": testdf}, f)

    # 5d. Evaluation results
    print(f"  → {PATH_EVAL}")
    with open(PATH_EVAL, "wb") as f:
        pickle.dump(evaluation_results, f)

    # 5e. Content embeddings
    print(f"  → {PATH_EMBEDDINGS}")
    np.save(PATH_EMBEDDINGS, content_embeddings)

    # ── Verify files exist ─────────────────────────────────────────────
    print("\n  Verifying saved files …")
    all_ok = True
    for path in [PATH_KERAS_MODEL, PATH_STATE, PATH_SPLITS, PATH_EVAL, PATH_EMBEDDINGS]:
        exists = os.path.exists(path)
        size   = ""
        if exists and os.path.isfile(path):
            size = f"  ({os.path.getsize(path)/1024:.0f} KB)"
        elif exists and os.path.isdir(path):
            size = "  (directory)"
        status = "  ✅" if exists else "  ❌  MISSING!"
        print(f"  {status}  {path}{size}")
        if not exists:
            all_ok = False

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    if all_ok:
        print("  ✅  All files saved successfully.")
        print(f"\n  ── Sampled Ranking (1 vs 100) ──────────────")
        print(f"  NDCG@{EVAL_K}:        {overall_sampled['ndcg']:.4f}")
        print(f"  Hit Rate@{EVAL_K}:    {overall_sampled['hit_rate']:.4f}")
        print(f"  Novelty@{EVAL_K}:     {overall_sampled['novelty']:.4f}")
        print(f"  Item Coverage:  {overall_sampled['item_coverage']:.2%}")
        print(f"\n  ── Full Ranking (1 vs 3,900) ───────────────")
        print(f"  NDCG@{EVAL_K}:        {overall_full['ndcg']:.4f}")
        print(f"  Hit Rate@{EVAL_K}:    {overall_full['hit_rate']:.4f}")
        print(f"  Novelty@{EVAL_K}:     {overall_full['novelty']:.4f}")
        print(f"  Item Coverage:  {overall_full['item_coverage']:.2%}")
        print("\n  Now launch the frontend:")
        print("      streamlit run app.py")
    else:
        print("  ❌  Some files are missing — check the errors above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()