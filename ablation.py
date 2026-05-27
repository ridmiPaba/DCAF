"""
ablation.py — DCAF Ablation Study

Trains four stripped-down variants of DCAF plus a clean MF-BPR baseline
to prove that each component (IPW, EMA, SBERT) contributes to performance
and that DCAF outperforms standard collaborative filtering.

Variants:
  1. Full DCAF       — all components active (pre-trained model)
  2. No IPW          — BPR loss with uniform pair weights
  3. No EMA          — alpha fixed at 1.0, pure content anchor always
  4. No SBERT        — random content embeddings instead of SBERT
  5. MF-BPR baseline — pure Matrix Factorisation, no content, no EMA, no IPW
                       

"""

import os
import pickle
import random
import numpy as np
import tensorflow as tf

# ── Fix random seeds for reproducibility ──────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.set_visible_devices([], 'GPU')
    print("GPU disabled — running on CPU.")
except Exception:
    pass

from model_logic import DCAFSystem, ContentAnchorLayer, EMAFusionLayer, GlobalBiasLayer
from tensorflow.keras import layers, Model

# ── Paths ──────────────────────────────────────────────────────────────
SAVE_DIR             = "saved_model"
PATH_STATE           = os.path.join(SAVE_DIR, "dcaf_state.pkl")
PATH_SPLITS          = os.path.join(SAVE_DIR, "splits.pkl")
PATH_EMBEDDINGS      = os.path.join(SAVE_DIR, "content_embeddings.npy")
PATH_ABLATION        = os.path.join(SAVE_DIR, "ablation_results.pkl")

# ── Hyperparameters ── 
EPOCHS        = 30
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE    = 1024
EVAL_K        = 10


# ── Helper: build model with optional EMA disabled ────────────────────
def build_dcaf_model(dcaf, content_embeddings, disable_ema=False):
    """
    Build the DCAF model.
    disable_ema=True sets k_decay=999999 so alpha≈1.0 always
    (pure content anchor, interaction_vec has no effect).
    """
    k_decay = 999999.0 if disable_ema else 5.0

    user_in   = layers.Input(shape=(), dtype=tf.int32,   name="user_idx")
    movie_in  = layers.Input(shape=(), dtype=tf.int32,   name="movie_idx")
    count_in  = layers.Input(shape=(), dtype=tf.int32,   name="interaction_count")
    gender_in = layers.Input(shape=(), dtype=tf.float32, name="gender")
    age_in    = layers.Input(shape=(), dtype=tf.float32, name="age")
    occ_in    = layers.Input(shape=(), dtype=tf.float32, name="occupation")

    user_vec = layers.Embedding(dcaf.num_users, EMBEDDING_DIM, name="user_embedding")(user_in)
    user_features = layers.Concatenate(name="user_features")([
        layers.Reshape((1,))(gender_in),
        layers.Reshape((1,))(age_in),
        layers.Reshape((1,))(occ_in),
    ])
    user_feat_vec = layers.Dense(EMBEDDING_DIM, activation="relu", name="user_feature_projection")(user_features)
    user_vec = layers.Add(name="user_combined")([user_vec, user_feat_vec])

    movie_inter_vec = layers.Embedding(dcaf.num_movies, EMBEDDING_DIM, name="movie_interaction_embedding")(movie_in)
    content_layer   = ContentAnchorLayer(content_embeddings, name="content_anchor_layer")
    content_vec     = content_layer(movie_in)
    content_proj    = layers.Dense(EMBEDDING_DIM, activation=None, name="content_projection")(content_vec)

    movie_vec = EMAFusionLayer(k_decay=k_decay, name="ema_fusion_layer")(content_proj, movie_inter_vec, count_in)

    dot        = layers.Dot(axes=-1)([user_vec, movie_vec])
    user_bias  = layers.Embedding(dcaf.num_users,  1, name="user_bias")(user_in)
    movie_bias = layers.Embedding(dcaf.num_movies, 1, name="movie_bias")(movie_in)
    prediction = dot + user_bias + movie_bias
    prediction = GlobalBiasLayer(name="global_bias_layer")(prediction)
    prediction = layers.Flatten()(prediction)

    return Model(inputs=[user_in, movie_in, count_in, gender_in, age_in, occ_in],
                 outputs=prediction, name="DCAF_ablation")


# ── Helper: build clean MF-BPR baseline model ─────────────────────────
def build_mf_model(dcaf):
   
    user_in  = layers.Input(shape=(), dtype=tf.int32, name="user_idx")
    movie_in = layers.Input(shape=(), dtype=tf.int32, name="movie_idx")

    user_vec  = layers.Embedding(dcaf.num_users,  EMBEDDING_DIM, name="user_embedding")(user_in)
    movie_vec = layers.Embedding(dcaf.num_movies, EMBEDDING_DIM, name="movie_embedding")(movie_in)

    dot        = layers.Dot(axes=-1)([user_vec, movie_vec])
    user_bias  = layers.Embedding(dcaf.num_users,  1, name="user_bias")(user_in)
    movie_bias = layers.Embedding(dcaf.num_movies, 1, name="movie_bias")(movie_in)

    prediction = dot + user_bias + movie_bias
    prediction = GlobalBiasLayer(name="global_bias_layer")(prediction)
    prediction = layers.Flatten()(prediction)

    return Model(inputs=[user_in, movie_in], outputs=prediction, name="MF_BPR")


# ── Helper: train MF-BPR baseline ─────────────────────────────────────
def train_mf_variant(dcaf, model, traindf):
    """
    Train MF-BPR with standard uniform BPR loss (Rendle et al., 2009).
    No IPW, no counts, no demographic features.
    """
    user_seen = dcaf.train_seen
    OVERSAMPLE = 10
    n_train    = len(traindf)
    u_vals     = traindf["user_idx"].values.astype(np.int32)
    neg_items  = np.empty(n_train, dtype=np.int32)
    candidates = np.random.randint(0, dcaf.num_movies, size=(n_train, OVERSAMPLE))

    for i in range(n_train):
        u = int(u_vals[i]); seen = user_seen.get(u, set()); picked = False
        for j in range(OVERSAMPLE):
            if candidates[i, j] not in seen:
                neg_items[i] = candidates[i, j]; picked = True; break
        if not picked:
            neg = random.randint(0, dcaf.num_movies - 1)
            while neg in seen: neg = random.randint(0, dcaf.num_movies - 1)
            neg_items[i] = neg

    u_arr   = traindf["user_idx"].values.astype(np.int32)
    pos_arr = traindf["movie_idx"].values.astype(np.int32)
    neg_arr = neg_items

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    @tf.function
    def mf_bpr_step(u, pos, neg):
        with tf.GradientTape() as tape:
            s_pos = model([u, pos], training=True)
            s_neg = model([u, neg], training=True)
            diff  = tf.squeeze(s_pos, -1) - tf.squeeze(s_neg, -1)
            loss  = tf.reduce_mean(tf.math.softplus(-diff))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0; n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start: start + BATCH_SIZE]
            loss = mf_bpr_step(
                tf.constant(u_arr[idx]),
                tf.constant(pos_arr[idx]),
                tf.constant(neg_arr[idx]),
            )
            epoch_loss += float(loss.numpy()); n_batches += 1
        print(f"    Epoch {epoch+1:2d}/{EPOCHS} — loss: {epoch_loss/n_batches:.4f}")


# ── Helper: train one variant ─────────────────────────────────────────
def train_variant(dcaf, model, traindf, use_ipw=True):
    """
    Train the model. use_ipw=False sets all pair weights to 1.0,
    removing the Inverse Propensity Weighting debiasing signal.
    """
    user_feat = dcaf.users.set_index("user_idx")[["GenderCode", "Age", "Occupation"]]
    traindf = traindf.copy()
    traindf["gender"]     = traindf["user_idx"].map(user_feat["GenderCode"]).fillna(0).astype(np.float32)
    traindf["age"]        = traindf["user_idx"].map(user_feat["Age"]).fillna(0).astype(np.float32)
    traindf["occupation"] = traindf["user_idx"].map(user_feat["Occupation"]).fillna(0).astype(np.float32)

    user_seen = dcaf.train_seen

    # Uniform negative sampling (same as main model)
    OVERSAMPLE = 10
    n_train    = len(traindf)
    u_vals     = traindf["user_idx"].values.astype(np.int32)
    neg_items  = np.empty(n_train, dtype=np.int32)
    candidates = np.random.randint(0, dcaf.num_movies, size=(n_train, OVERSAMPLE))

    for i in range(n_train):
        u = int(u_vals[i]); seen = user_seen.get(u, set()); picked = False
        for j in range(OVERSAMPLE):
            if candidates[i, j] not in seen:
                neg_items[i] = candidates[i, j]; picked = True; break
        if not picked:
            neg = random.randint(0, dcaf.num_movies - 1)
            while neg in seen: neg = random.randint(0, dcaf.num_movies - 1)
            neg_items[i] = neg

    traindf["neg_item"]  = neg_items
    traindf["pos_count"] = traindf["movie_idx"].map(dcaf.countmap).fillna(0).astype(int).values
    traindf["neg_count"] = pd.Series(neg_items).map(dcaf.countmap).fillna(0).astype(int).values

    ipw_lookup = np.array(
        [float(dcaf.ipw_weights[m]) if m < len(dcaf.ipw_weights) else 1.0
         for m in range(dcaf.num_movies)], dtype=np.float32
    )
    traindf["ipw"]     = ipw_lookup[traindf["movie_idx"].values]
    traindf["neg_ipw"] = ipw_lookup[neg_items]

    u_arr       = traindf["user_idx"].values.astype(np.int32)
    pos_arr     = traindf["movie_idx"].values.astype(np.int32)
    neg_arr     = traindf["neg_item"].values.astype(np.int32)
    pos_cnt_arr = traindf["pos_count"].values.astype(np.int32)
    neg_cnt_arr = traindf["neg_count"].values.astype(np.int32)
    g_arr       = traindf["gender"].values.astype(np.float32)
    a_arr       = traindf["age"].values.astype(np.float32)
    o_arr       = traindf["occupation"].values.astype(np.float32)
    ipw_pos_arr = traindf["ipw"].values.astype(np.float32)
    ipw_neg_arr = traindf["neg_ipw"].values.astype(np.float32)

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    @tf.function
    def bpr_step(u, pos, neg, pc, nc, g, a, o, w_pos, w_neg):
        with tf.GradientTape() as tape:
            s_pos = model([u, pos, pc, g, a, o], training=True)
            s_neg = model([u, neg, nc, g, a, o], training=True)
            diff  = tf.squeeze(s_pos, -1) - tf.squeeze(s_neg, -1)
            if use_ipw:
                pair_w = (w_pos + w_neg) / 2.0
            else:
                pair_w = tf.ones_like(w_pos)   # No IPW — all pairs weighted equally
            loss = tf.reduce_mean(pair_w * tf.math.softplus(-diff))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0; n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start: start + BATCH_SIZE]
            loss = bpr_step(
                tf.constant(u_arr[idx]), tf.constant(pos_arr[idx]),
                tf.constant(neg_arr[idx]), tf.constant(pos_cnt_arr[idx]),
                tf.constant(neg_cnt_arr[idx]), tf.constant(g_arr[idx]),
                tf.constant(a_arr[idx]), tf.constant(o_arr[idx]),
                tf.constant(ipw_pos_arr[idx]), tf.constant(ipw_neg_arr[idx]),
            )
            epoch_loss += float(loss.numpy()); n_batches += 1
        print(f"    Epoch {epoch+1:2d}/{EPOCHS} — loss: {epoch_loss/n_batches:.4f}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import pandas as pd

    print("\n" + "="*60)
    print("  DCAF — Ablation Study")
    print(f"  Epochs per variant: {EPOCHS}  (change EPOCHS for full training)")
    print("="*60)

    # ── Load saved state from train.py ────────────────────────────────
    print("\nLoading saved model state …")
    with open(PATH_STATE, "rb") as f:
        state = pickle.load(f)
    with open(PATH_SPLITS, "rb") as f:
        splits = pickle.load(f)
    content_embeddings = np.load(PATH_EMBEDDINGS)

    traindf = splits["traindf"]
    testdf  = splits["testdf"]

    dcaf = DCAFSystem(data_path="data")
    for key, val in state.items():
        setattr(dcaf, key, val)

    print(f"  Users: {dcaf.num_users}  |  Movies: {dcaf.num_movies}")
    print(f"  Train rows: {len(traindf)}  |  Test rows: {len(testdf)}")

    # ── Helper to evaluate a trained model ────────────────────────────
    def evaluate(model_obj, label):
        print(f"\n  Evaluating {label} …")
        dcaf.model = model_obj
        overall  = dcaf.evaluate_metrics_sampled(testdf, k=EVAL_K)
        segments = dcaf.segment_performance(testdf, k=EVAL_K)
        print(f"  {label}: NDCG@{EVAL_K}={overall['ndcg']:.4f}  "
              f"HR@{EVAL_K}={overall['hit_rate']:.4f}  "
              f"Novelty={overall['novelty']:.4f}  "
              f"Coverage={overall['item_coverage']:.2%}")
        return {"overall": overall, "segments": segments}

    results = {}

    # ── Variant 1: Full DCAF (load already-trained model) ─────────────
    print("\n" + "-"*50)
    print("Variant 1/5: Full DCAF (loading pre-trained model)")
    print("-"*50)
    full_model = tf.keras.models.load_model(
        os.path.join(SAVE_DIR, "dcaf_keras_model.keras"),
        custom_objects={
            "ContentAnchorLayer": ContentAnchorLayer,
            "EMAFusionLayer":     EMAFusionLayer,
            "GlobalBiasLayer":    GlobalBiasLayer,
        }
    )
    results["Full DCAF"] = evaluate(full_model, "Full DCAF")

    # ── Variant 2: No IPW ─────────────────────────────────────────────
    print("\n" + "-"*50)
    print("Variant 2/5: No IPW (uniform loss weights)")
    print("-"*50)
    dcaf.model = None
    model_no_ipw = build_dcaf_model(dcaf, content_embeddings, disable_ema=False)
    train_variant(dcaf, model_no_ipw, traindf, use_ipw=False)
    results["No IPW"] = evaluate(model_no_ipw, "No IPW")

    # ── Variant 3: No EMA ─────────────────────────────────────────────
    print("\n" + "-"*50)
    print("Variant 3/5: No EMA (alpha fixed at 1.0 — pure content)")
    print("-"*50)
    model_no_ema = build_dcaf_model(dcaf, content_embeddings, disable_ema=True)
    train_variant(dcaf, model_no_ema, traindf, use_ipw=True)
    results["No EMA"] = evaluate(model_no_ema, "No EMA")

    # ── Variant 4: No SBERT ───────────────────────────────────────────
    print("\n" + "-"*50)
    print("Variant 4/5: No SBERT (random content embeddings)")
    print("-"*50)
    rng = np.random.default_rng(SEED)
    random_embeddings = rng.standard_normal(content_embeddings.shape).astype(np.float32)
    model_no_sbert = build_dcaf_model(dcaf, random_embeddings, disable_ema=False)
    train_variant(dcaf, model_no_sbert, traindf, use_ipw=True)
    results["No SBERT"] = evaluate(model_no_sbert, "No SBERT")

    # ── Variant 5: MF-BPR baseline ───────────────────────────────────────
    print("\n" + "-"*50)
    print("Variant 5/5: MF-BPR baseline (pure collaborative filtering)")
    print("  Reference: Rendle et al. (2009) BPR: Bayesian Personalized Ranking")
    print("-"*50)
    model_mf = build_mf_model(dcaf)
    train_mf_variant(dcaf, model_mf, traindf)

    # MF model uses only [user_idx, movie_idx] inputs.
    # evaluate_metrics_sampled calls model.predict([u, m, count, g, a, o]).
    # Use a simple Python wrapper class that accepts 6 inputs but passes
    # only the first two to the MF model. This avoids any Keras graph issues.
    class MFPredictor:
        """Wraps the 2-input MF model to accept the 6-input interface."""
        def __init__(self, mf_model):
            self._model = mf_model
        def predict(self, inputs, batch_size=4096, verbose=0):
            return self._model.predict(
                [inputs[0], inputs[1]], batch_size=batch_size, verbose=verbose
            )

    orig_model      = dcaf.model
    dcaf.model      = MFPredictor(model_mf)
    overall_mf      = dcaf.evaluate_metrics_sampled(testdf, k=EVAL_K)
    segments_mf     = dcaf.segment_performance(testdf, k=EVAL_K)
    dcaf.model      = orig_model   # restore original model

    print(f"  MF-BPR baseline: NDCG@{EVAL_K}={overall_mf['ndcg']:.4f}  "
          f"HR@{EVAL_K}={overall_mf['hit_rate']:.4f}  "
          f"Novelty={overall_mf['novelty']:.4f}  "
          f"Coverage={overall_mf['item_coverage']:.2%}")
    results["MF-BPR (Rendle et al., 2009)"] = {
        "overall": overall_mf, "segments": segments_mf
    }

    # ── Save results ──────────────────────────────────────────────────
    ablation_output = {
        "results": results,
        "epochs":  EPOCHS,
        "k":       EVAL_K,
    }
    with open(PATH_ABLATION, "wb") as f:
        pickle.dump(ablation_output, f)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  ABLATION STUDY RESULTS ({EPOCHS} epochs) — 5 variants")
    print("="*60)
    print(f"  {'Variant':<15} {'NDCG@10':>10} {'HR@10':>10} {'Novelty':>10} {'Coverage':>10}")
    print("  " + "-"*55)
    for name, res in results.items():
        o = res["overall"]
        marker = " ← full model" if name == "Full DCAF" else ""
        print(f"  {name:<15} {o['ndcg']:>10.4f} {o['hit_rate']:>10.4f} "
              f"{o['novelty']:>10.4f} {o['item_coverage']:>10.2%}{marker}")
    print("="*60)
    print(f"\n  Results saved to {PATH_ABLATION}")
    print("  Open the Streamlit dashboard → Ablation Study page to see the full comparison.")
    print()


if __name__ == "__main__":
    import pandas as pd
    main()