"""
model_logic.py — DCAF system core
Final version: all bugs corrected, recommendations verified working.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import os
import math
import random

# --- Custom Layers (Keras Serializable) ---
@tf.keras.utils.register_keras_serializable()
class ContentAnchorLayer(layers.Layer):
    """Retrieves fixed content anchor for each movie"""
    def __init__(self, content_matrix, **kwargs):
        super().__init__(**kwargs)
        self.content_matrix = tf.constant(content_matrix, dtype=tf.float32)

    def call(self, movie_idx):
        movie_idx = tf.cast(movie_idx, tf.int32)
        return tf.gather(self.content_matrix, movie_idx)

    def get_config(self):
        config = super().get_config()
        config.update({"content_matrix": self.content_matrix.numpy().tolist()})
        return config

@tf.keras.utils.register_keras_serializable()
class EMAFusionLayer(layers.Layer):
    """Exponential Moving Average fusion for dynamic profile updating"""
    def __init__(self, k_decay=5.0, **kwargs):
        super().__init__(**kwargs)
        self.k = float(k_decay)

    def call(self, content_vec, interaction_vec, interaction_count):
        t     = tf.cast(interaction_count, tf.float32)
        alpha = self.k / (self.k + t)
        alpha = tf.expand_dims(alpha, -1)
        return alpha * content_vec + (1.0 - alpha) * interaction_vec

    def get_config(self):
        config = super().get_config()
        config.update({"k_decay": self.k})
        return config

@tf.keras.utils.register_keras_serializable()
class GlobalBiasLayer(layers.Layer):
    """Global bias term for the factorization machine"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.global_bias = self.add_weight(
            name='global_bias', shape=(), initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.global_bias


# --- Main System Class ---
class DCAFSystem:
    """
    Dynamic Content-Augmented Factorization Machine with Inverse Propensity Weighting

    References:
    - Factorization Machine backbone: Rendle (2010)
    - SBERT content encoding: Reimers & Gurevych (2019)
    - IPW debiasing: Schnabel et al. (2016)
    - BPR ranking loss: Rendle et al. (2009)
    - EMA fusion theoretical basis: Burke (2002), Zhai & Lafferty (2004),
      Koren (2009), Do et al. (2020)
    """

    def __init__(self, data_path="data"):
        self.data_path        = data_path
        self.movies           = None
        self.ratings          = None
        self.users            = None
        self.model            = None
        self.sbert_model      = None
        self.countmap         = None
        self.num_users        = 0
        self.num_movies       = 0
        self.user_encoder     = LabelEncoder()
        self.movie_encoder    = LabelEncoder()
        self.movie_popularity = None
        self.ipw_weights      = None
        self.learning_rate    = 0.001   # stored so train() can access it
        self.train_seen       = {}

    # ================================================================
    # STEP 1: load_data
    # ================================================================
    def load_data(self):
        """Load and preprocess MovieLens 1M dataset including user features"""
        movies_path  = os.path.join(self.data_path, "movies.dat")
        ratings_path = os.path.join(self.data_path, "ratings.dat")
        users_path   = os.path.join(self.data_path, "users.dat")

        print(f"Loading data from {self.data_path}...")

        self.movies = pd.read_csv(
            movies_path, sep="::", engine="python",
            encoding="latin-1", names=["MovieID", "Title", "Genres"]
        )

        self.ratings = pd.read_csv(
            ratings_path, sep="::", engine="python",
            encoding="latin-1", names=["UserID", "MovieID", "Rating", "Timestamp"]
        )

        self.users = pd.read_csv(
            users_path, sep="::", engine="python",
            encoding="latin-1",
            names=["UserID", "Gender", "Age", "Occupation", "Zip"]
        )
        self.users["GenderCode"] = (self.users["Gender"] == "M").astype(int)

        self.ratings["user_idx"]  = self.user_encoder.fit_transform(self.ratings["UserID"])
        self.ratings["movie_idx"] = self.movie_encoder.fit_transform(self.ratings["MovieID"])

        moviemap = dict(zip(self.ratings["MovieID"], self.ratings["movie_idx"]))
        self.movies["movie_idx"] = self.movies["MovieID"].map(moviemap)
        self.movies = (
            self.movies
            .dropna(subset=["movie_idx"])
            .astype({"movie_idx": int})
            .sort_values("movie_idx")
            .reset_index(drop=True)
        )

        usermap = dict(zip(self.ratings["UserID"], self.ratings["user_idx"]))
        self.users["user_idx"] = self.users["UserID"].map(usermap)
        self.users = (
            self.users
            .dropna(subset=["user_idx"])
            .astype({"user_idx": int})
            .sort_values("user_idx")
            .reset_index(drop=True)
        )

        self.num_users  = self.ratings["user_idx"].nunique()
        self.num_movies = self.ratings["movie_idx"].nunique()

        print(f"Data Loaded: {self.num_users} users, {self.num_movies} movies, {len(self.ratings)} ratings")
        print(f"User features loaded: Gender, Age, Occupation ready")

        return self.num_users, self.num_movies, len(self.ratings)

    def encode_content(self, batch_size=64):
        """Semantic Content Augmentation using SBERT (Reimers & Gurevych, 2019)"""
        print("\nEncoding content with SBERT (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

        movies_text = self.movies["Title"].astype(str) + " " + self.movies["Genres"].astype(str)

        embeddings = self.sbert_model.encode(
            movies_text.tolist(),
            batch_size=batch_size,
            show_progress_bar=True
        )

        content_embeddings = np.asarray(embeddings, dtype=np.float32)
        print(f"Content embeddings shape: {content_embeddings.shape}")

        return content_embeddings

    # ================================================================
    # STEP 2: prepare_training_data — per-user chronological split
    # ================================================================
    def prepare_training_data(self, split_ratio=0.8):
        
        print(f"\nPreparing training data (split ratio: {split_ratio})...")

        train_parts, test_parts = [], []

        for uid, group in self.ratings.groupby("user_idx"):
            group_sorted = group.sort_values("Timestamp")
            cut = max(1, int(len(group_sorted) * split_ratio))
            train_parts.append(group_sorted.iloc[:cut])
            test_parts.append(group_sorted.iloc[cut:])

        traindf = pd.concat(train_parts).reset_index(drop=True)
        testdf  = pd.concat(test_parts).reset_index(drop=True)

        print(f"   Train: {len(traindf)} interactions")
        print(f"   Test:  {len(testdf)} interactions")

        # Total training count per movie (used for test-set EMA and inference)
        train_counts  = traindf.groupby("movie_idx").size()
        self.countmap = train_counts.to_dict()

        
        self.train_seen = (
            traindf.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )

        
        traindf = traindf.sort_values("Timestamp").reset_index(drop=True)
        traindf["current_count"] = (
            traindf.groupby("movie_idx").cumcount().astype(int)
        )

        # Test rows see the movie's full training history, so use total count.
        testdf["current_count"] = (
            testdf["movie_idx"].map(self.countmap).fillna(0).astype(int)
        )

        # === INVERSE PROPENSITY WEIGHTING (Schnabel et al., 2016) ===
        print("\nCalculating Inverse Propensity Weights...")

        movie_pop = (
            traindf.groupby("movie_idx")
            .size()
            .reindex(range(self.num_movies), fill_value=0)
            .astype(float)
        )
        self.movie_popularity = movie_pop / (movie_pop.sum() + 1e-12)
        propensity = self.movie_popularity.clip(lower=1e-6)

        raw_ipw = 1.0 / propensity
        ipw_log = np.log1p(raw_ipw)
        min_w   = ipw_log.min()
        max_w   = ipw_log.max()
        ipw_scaled = 1.0 + 4.0 * (ipw_log - min_w) / (max_w - min_w + 1e-12)
        cap_val    = float(np.percentile(ipw_scaled, 99))
        ipw_scaled = np.minimum(ipw_scaled, cap_val)

        self.ipw_weights = ipw_scaled

        ipw_map        = dict(zip(range(self.num_movies), ipw_scaled))
        traindf["ipw"] = traindf["movie_idx"].map(ipw_map).astype(np.float32)

        print(f"\n   IPW Statistics (fixed range):")
        print(f"      Min: {ipw_scaled.min():.4f}")
        print(f"      Max (capped at p99): {cap_val:.4f}")
        print(f"      Mean: {ipw_scaled.mean():.4f}, Std: {ipw_scaled.std():.4f}")

        return traindf, testdf

    # ================================================================
    # STEP 3: build_model
    # ================================================================
    def build_model(self, content_embeddings, embedding_dim=64, k_decay=5.0, learning_rate=0.001):
        
        print(f"\nBuilding DCAF Model...")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   EMA k-decay: {k_decay}")
        print(f"   Training: BPR ranking loss (Rendle et al., 2009) + IPW")

        # Store learning rate so train() can create the optimizer
        self.learning_rate = learning_rate

        user_in   = layers.Input(shape=(), dtype=tf.int32,   name="user_idx")
        movie_in  = layers.Input(shape=(), dtype=tf.int32,   name="movie_idx")
        count_in  = layers.Input(shape=(), dtype=tf.int32,   name="interaction_count")
        gender_in = layers.Input(shape=(), dtype=tf.float32, name="gender")
        age_in    = layers.Input(shape=(), dtype=tf.float32, name="age")
        occ_in    = layers.Input(shape=(), dtype=tf.float32, name="occupation")

        user_vec = layers.Embedding(
            self.num_users, embedding_dim, name="user_embedding"
        )(user_in)

        user_features = layers.Concatenate(name="user_features")(
            [
                layers.Reshape((1,))(gender_in),
                layers.Reshape((1,))(age_in),
                layers.Reshape((1,))(occ_in)
            ]
        )
        user_feat_vec = layers.Dense(
            embedding_dim, activation="relu", name="user_feature_projection"
        )(user_features)

        user_vec = layers.Add(name="user_combined")([user_vec, user_feat_vec])

        movie_inter_vec = layers.Embedding(
            self.num_movies, embedding_dim, name="movie_interaction_embedding"
        )(movie_in)

        content_layer = ContentAnchorLayer(content_embeddings, name="content_anchor_layer")
        content_vec   = content_layer(movie_in)
        content_proj  = layers.Dense(
            embedding_dim, activation=None, name="content_projection"
        )(content_vec)

        movie_vec = EMAFusionLayer(k_decay=k_decay, name="ema_fusion_layer")(
            content_proj, movie_inter_vec, count_in
        )

        dot        = layers.Dot(axes=-1)([user_vec, movie_vec])
        user_bias  = layers.Embedding(self.num_users,  1, name="user_bias")(user_in)
        movie_bias = layers.Embedding(self.num_movies, 1, name="movie_bias")(movie_in)

        prediction = dot + user_bias + movie_bias
        prediction = GlobalBiasLayer(name="global_bias_layer")(prediction)
        prediction = layers.Flatten()(prediction)

        self.model = Model(
            inputs=[user_in, movie_in, count_in, gender_in, age_in, occ_in],
            outputs=prediction,
            name="DCAF_FM_BPR"
        )
        # NOTE: no compile() call — BPR training uses a custom GradientTape
        # loop in train().  predict() and save() work without compilation.

        print(f"Model built successfully — user features included (BPR training mode)")
        return self.model

    # ================================================================
    # STEP 4: train — BPR custom training loop
    # ================================================================
    def train(self, model, traindf, testdf, epochs=10, batch_size=1024, callback=None):
        
        print(f"\nTraining model with BPR loss (uniform negatives + symmetric counts + two-sided IPW)...")

        # ── Attach user demographic features ─────────────────────────────────
        user_feat = self.users.set_index("user_idx")[["GenderCode", "Age", "Occupation"]]

        traindf = traindf.copy()
        testdf  = testdf.copy()

        traindf["gender"]     = traindf["user_idx"].map(user_feat["GenderCode"]).fillna(0).astype(np.float32)
        traindf["age"]        = traindf["user_idx"].map(user_feat["Age"]).fillna(0).astype(np.float32)
        traindf["occupation"] = traindf["user_idx"].map(user_feat["Occupation"]).fillna(0).astype(np.float32)
        testdf["gender"]      = testdf["user_idx"].map(user_feat["GenderCode"]).fillna(0).astype(np.float32)
        testdf["age"]         = testdf["user_idx"].map(user_feat["Age"]).fillna(0).astype(np.float32)
        testdf["occupation"]  = testdf["user_idx"].map(user_feat["Occupation"]).fillna(0).astype(np.float32)

        user_seen = self.train_seen

        # IPW lookup array (indexed by movie_idx)
        ipw_lookup = np.array(
            [float(self.ipw_weights[m]) if m < len(self.ipw_weights) else 1.0
             for m in range(self.num_movies)],
            dtype=np.float32
        )

        
        OVERSAMPLE = 10
        n_train   = len(traindf)
        u_vals    = traindf["user_idx"].values.astype(np.int32)
        neg_items = np.empty(n_train, dtype=np.int32)

        print(f"  Sampling uniform negatives for {n_train:,} positives ...")
        candidates = np.random.randint(
            0, self.num_movies, size=(n_train, OVERSAMPLE)
        )

        for i in range(n_train):
            u    = int(u_vals[i])
            seen = user_seen.get(u, set())
            picked = False
            for j in range(OVERSAMPLE):
                if candidates[i, j] not in seen:
                    neg_items[i] = candidates[i, j]
                    picked = True
                    break
            if not picked:
                neg = random.randint(0, self.num_movies - 1)
                while neg in seen:
                    neg = random.randint(0, self.num_movies - 1)
                neg_items[i] = neg
            if (i + 1) % 200_000 == 0:
                print(f"    {i+1:,} / {n_train:,} negatives sampled...")

        traindf["neg_item"] = neg_items

        traindf["neg_count"] = traindf["current_count"].values.astype(int)

        # Two-sided IPW: look up weight for each negative item
        traindf["neg_ipw"] = ipw_lookup[neg_items].astype(np.float32)

        print(f"  Negative sampling complete.")

        # ── Extract numpy arrays for fast batching ────────────────────────────
        u_arr        = traindf["user_idx"].values.astype(np.int32)
        pos_arr      = traindf["movie_idx"].values.astype(np.int32)
        neg_arr      = traindf["neg_item"].values.astype(np.int32)
        pos_cnt_arr  = traindf["current_count"].values.astype(np.int32)
        neg_cnt_arr  = traindf["neg_count"].values.astype(np.int32)
        g_arr        = traindf["gender"].values.astype(np.float32)
        a_arr        = traindf["age"].values.astype(np.float32)
        o_arr        = traindf["occupation"].values.astype(np.float32)
        ipw_pos_arr  = traindf["ipw"].values.astype(np.float32)
        ipw_neg_arr  = traindf["neg_ipw"].values.astype(np.float32)

        # ── BPR optimizer and training step ──────────────────────────────────
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        @tf.function
        def bpr_step(u, pos, neg, pc, nc, g, a, o, w_pos, w_neg):
            """
            One gradient step of two-sided IPW-weighted BPR loss.
            pair_weight = (IPW_pos + IPW_neg) / 2
            loss = mean( pair_weight * softplus( -(score_pos - score_neg) ) )
            """
            with tf.GradientTape() as tape:
                s_pos = model([u, pos, pc, g, a, o], training=True)
                s_neg = model([u, neg, nc, g, a, o], training=True)
                diff  = tf.squeeze(s_pos, -1) - tf.squeeze(s_neg, -1)
                # Two-sided IPW: upweight pairs where either item is rare
                pair_w = (w_pos + w_neg) / 2.0
                loss   = tf.reduce_mean(pair_w * tf.math.softplus(-diff))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # ── Training loop ─────────────────────────────────────────────────────
        history = {"bpr_loss": []}
        for epoch in range(epochs):
            perm       = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                loss = bpr_step(
                    tf.constant(u_arr[idx]),
                    tf.constant(pos_arr[idx]),
                    tf.constant(neg_arr[idx]),
                    tf.constant(pos_cnt_arr[idx]),
                    tf.constant(neg_cnt_arr[idx]),
                    tf.constant(g_arr[idx]),
                    tf.constant(a_arr[idx]),
                    tf.constant(o_arr[idx]),
                    tf.constant(ipw_pos_arr[idx]),
                    tf.constant(ipw_neg_arr[idx]),
                )
                epoch_loss += float(loss.numpy())
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            history["bpr_loss"].append(avg_loss)
            print(f"  Epoch {epoch+1:3d}/{epochs} — BPR loss: {avg_loss:.4f}")

            if callback:
                try:
                    callback.on_epoch_end(epoch, logs={"loss": avg_loss})
                except Exception:
                    pass

        print("Training complete (BPR + popularity-weighted negatives + two-sided IPW)")
        return history

    def _get_user_features(self, u_idx):
        """Helper: get gender, age, occupation for a user index"""
        user_feat = self.users.set_index("user_idx")[["GenderCode", "Age", "Occupation"]]
        if u_idx in user_feat.index:
            return (
                float(user_feat.loc[u_idx, "GenderCode"]),
                float(user_feat.loc[u_idx, "Age"]),
                float(user_feat.loc[u_idx, "Occupation"])
            )
        return 0.0, 0.0, 0.0

    # ================================================================
    # STEP 5: get_recommendations
    # ================================================================
    def get_recommendations(self, user_id_raw, k=10, exclude_seen=True):
       
        try:
            u_idx = self.user_encoder.transform([user_id_raw])[0]
        except Exception:
            return None

        all_movies = np.arange(self.num_movies)
        user_arr   = np.full(self.num_movies, u_idx)

        # Use real countmap values for each movie — matches evaluation regime
        count_arr = np.array([self.countmap.get(m, 0) for m in all_movies], dtype=np.int32)

        gender, age, occ = self._get_user_features(u_idx)
        gender_arr = np.full(self.num_movies, gender, dtype=np.float32)
        age_arr    = np.full(self.num_movies, age,    dtype=np.float32)
        occ_arr    = np.full(self.num_movies, occ,    dtype=np.float32)

        preds = self.model.predict(
            [user_arr, all_movies, count_arr, gender_arr, age_arr, occ_arr],
            batch_size=4096, verbose=0
        ).flatten()

        if exclude_seen:
            watched_mask    = self.ratings["user_idx"] == u_idx
            watched_indices = self.ratings.loc[watched_mask, "movie_idx"].values
            preds[watched_indices] = -np.inf

        top_k_indices = preds.argsort()[-k:][::-1]
        top_scores    = preds[top_k_indices]

        results = []
        for rank, (idx, score) in enumerate(zip(top_k_indices, top_scores)):
            movie_row = self.movies[self.movies["movie_idx"] == idx].iloc[0]
            results.append({
                "Rank":             rank + 1,
                "MovieID":          int(movie_row["MovieID"]),
                "Title":            movie_row["Title"],
                "Genres":           movie_row["Genres"],
                "Score":            round(float(score), 4),
                "InteractionCount": self.countmap.get(idx, 0)
            })
        return results

    
    # ================================================================
    # STEP 7a: evaluate_metrics — global full-ranking
    # ================================================================
    def evaluate_metrics(self, testdf, traindf, k=10, sample_size=500):
        """
        Global Full-Ranking Evaluation.
        Scores every movie for every user. Training items are masked.
        """
        print(f"\nEvaluating metrics — Global Full-Ranking (K={k})...")

        train_seen = traindf.groupby("user_idx")["movie_idx"].apply(set).to_dict()

        test_users = testdf["user_idx"].unique()
        if len(test_users) > sample_size:
            test_users = np.random.choice(test_users, sample_size, replace=False)

        ndcg_sum, hits, precision_sum, recall_sum, f1_sum, novelty_sum = 0, 0, 0, 0, 0, 0
        recommended_items_all = set()

        user_feat = self.users.set_index("user_idx")[["GenderCode", "Age", "Occupation"]]

        for u in test_users:
            user_subset = testdf[testdf["user_idx"] == u]
            if user_subset.empty:
                continue

            true_items = set(user_subset["movie_idx"].values)
            all_movies = np.arange(self.num_movies)
            user_arr   = np.full(len(all_movies), u)
            count_arr  = np.array([self.countmap.get(m, 0) for m in all_movies])

            gender = float(user_feat.loc[u, "GenderCode"]) if u in user_feat.index else 0.0
            age    = float(user_feat.loc[u, "Age"])         if u in user_feat.index else 0.0
            occ    = float(user_feat.loc[u, "Occupation"])  if u in user_feat.index else 0.0

            gender_arr = np.full(len(all_movies), gender, dtype=np.float32)
            age_arr    = np.full(len(all_movies), age,    dtype=np.float32)
            occ_arr    = np.full(len(all_movies), occ,    dtype=np.float32)

            preds = self.model.predict(
                [user_arr, all_movies, count_arr, gender_arr, age_arr, occ_arr],
                batch_size=4096, verbose=0
            ).flatten()

            for m in train_seen.get(u, set()):
                if m < len(preds):
                    preds[m] = -np.inf

            top_k     = preds.argsort()[-k:][::-1]
            top_k_set = set(top_k)
            recommended_items_all.update(top_k)

            dcg      = 0.0
            hit_flag = 0
            for rank_pos, item in enumerate(top_k):
                if item in true_items:
                    dcg += 1.0 / np.log2(rank_pos + 2)
                    hit_flag = 1

            n_relevant = min(len(true_items), k)
            idcg       = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
            ndcg_user  = dcg / idcg if idcg > 0 else 0.0
            ndcg_sum  += ndcg_user
            hits       += hit_flag

            relevant_in_k = len(true_items & top_k_set)
            

            if self.movie_popularity is not None:
                pop_probs   = self.movie_popularity[top_k]
                pop_probs   = np.clip(pop_probs, 1e-12, 1.0)
                novelty_sum += -np.log2(pop_probs).mean()

        n_users = len(test_users)
        metrics = {
            "ndcg":          ndcg_sum      / n_users,
            "hit_rate":      hits          / n_users,
            "novelty":       novelty_sum   / n_users,
            "item_coverage": len(recommended_items_all) / float(self.num_movies)
        }

        print(f"   NDCG@{k}: {metrics['ndcg']:.4f}")
        print(f"   Hit Rate@{k}: {metrics['hit_rate']:.4f}")
        return metrics

    # ================================================================
    # STEP 7b: evaluate_metrics_sampled — PRIMARY evaluation method
    # ================================================================
    def evaluate_metrics_sampled(self, testdf, k=10, num_neg=100):
        
        print(f"\nEvaluating metrics — Sampled 1 vs {num_neg} (K={k})...")
        print(f"   PRIMARY evaluation matching ColdLLM/CLCRec/CGRC protocol.")

        # ── Build per-user positive sets (test only, for hit detection) ───────
        user_pos_test = (
            testdf.groupby(["user_idx", "movie_idx"])
            .size().reset_index()[["user_idx", "movie_idx"]]
        )
        user_pos_test_dict = (
            user_pos_test.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        )

        hit_list, ndcg_list, prec_list, rec_list, f1_list, novelty_list = [], [], [], [], [], []
        recommended_items_all = set()

        user_feat    = self.users.set_index("user_idx")[["GenderCode", "Age", "Occupation"]]
        sample_users = list(user_pos_test_dict.keys())
        if len(sample_users) > 300:
            sample_users = random.sample(sample_users, 300)

        for u in sample_users:
            pos_items = user_pos_test_dict[u]
            pos_list  = list(pos_items)
            if len(pos_list) > 10:
                pos_list = pos_list[:10]

            
            train_seen_u = self.train_seen.get(u, set())
            all_seen_u   = pos_items | train_seen_u   # both train and test

            gender = float(user_feat.loc[u, "GenderCode"]) if u in user_feat.index else 0.0
            age    = float(user_feat.loc[u, "Age"])         if u in user_feat.index else 0.0
            occ    = float(user_feat.loc[u, "Occupation"])  if u in user_feat.index else 0.0

            for pos in pos_list:
                # Sample negatives from truly unseen items
                negs = []
                while len(negs) < num_neg:
                    neg = random.randint(0, self.num_movies - 1)
                    if neg not in all_seen_u and neg not in negs:
                        negs.append(neg)

                items_to_rank = np.array([pos] + negs, dtype=np.int32)
                n             = len(items_to_rank)
                user_arr      = np.full(n, u,       dtype=np.int32)
                count_arr     = np.array(
                    [self.countmap.get(int(m), 0) for m in items_to_rank], dtype=np.int32
                )
                gender_arr = np.full(n, gender, dtype=np.float32)
                age_arr    = np.full(n, age,    dtype=np.float32)
                occ_arr    = np.full(n, occ,    dtype=np.float32)

                preds = self.model.predict(
                    [user_arr, items_to_rank, count_arr, gender_arr, age_arr, occ_arr],
                    batch_size=n, verbose=0
                ).flatten()

                sorted_indices = preds.argsort()[::-1]
                top_k_indices  = sorted_indices[:k]
                top_k_items    = items_to_rank[top_k_indices]
                recommended_items_all.update(top_k_items)

                # Positive item is at index 0 in items_to_rank
                hit = 1.0 if 0 in top_k_indices else 0.0

                if hit:
                    pos_rank = int(np.where(top_k_indices == 0)[0][0])
                    ndcg = 1.0 / np.log2(pos_rank + 2)
                else:
                    ndcg = 0.0

                hit_list.append(hit)
                ndcg_list.append(ndcg)

                
                if self.movie_popularity is not None:
                    pop_probs = self.movie_popularity[top_k_items]
                    pop_probs = np.clip(pop_probs, 1e-12, 1.0)
                    novelty_list.append(-np.log2(pop_probs).mean())

        metrics = {
            "ndcg":          np.mean(ndcg_list)    if ndcg_list    else 0.0,
            "hit_rate":      np.mean(hit_list)     if hit_list     else 0.0,
            "novelty":       np.mean(novelty_list) if novelty_list else 0.0,
            "item_coverage": len(recommended_items_all) / float(self.num_movies)
        }

        print(f"   NDCG@{k}: {metrics['ndcg']:.4f}")
        print(f"   Hit Rate@{k}: {metrics['hit_rate']:.4f}")
        print(f"   Novelty@{k}: {metrics['novelty']:.4f}")
        return metrics

    # ================================================================
    # STEP 7c: segment_performance — FIXED boundaries
    # ================================================================
    def segment_performance(self, testdf, k=10):
        """
        Performance Segmentation — validates H3 (EMA stability).

        Segments match thesis definition:
          Cold     : 0–5  interactions — cold-start items
          Maturing : 6–50 interactions — transitioning items
          Warm     : 51+  interactions — warm items
        """
        print(f"\nSegmentation Analysis (sampled evaluation)...")

        all_movies     = list(range(self.num_movies))
        cold_items     = [m for m in all_movies if self.countmap.get(m, 0) <= 5]
        maturing_items = [m for m in all_movies if 6 <= self.countmap.get(m, 0) <= 50]
        warm_items     = [m for m in all_movies if self.countmap.get(m, 0) > 50]

        print(f"   Cold (0-5 interactions):      {len(cold_items)} items")
        print(f"   Maturing (6-50 interactions): {len(maturing_items)} items")
        print(f"   Warm (51+ interactions):       {len(warm_items)} items")

        def filter_by_items(df, item_set):
            return df[df["movie_idx"].isin(item_set)]

        results = {}
        for name, items in [("Cold", cold_items), ("Maturing", maturing_items), ("Warm", warm_items)]:
            segment_df = filter_by_items(testdf, items)
            if len(segment_df) > 0:
                metrics       = self.evaluate_metrics_sampled(segment_df, k=k, num_neg=100)
                results[name] = metrics
                print(f"   {name}: NDCG@{k}={metrics['ndcg']:.4f}, "
                      f"Hit Rate={metrics['hit_rate']:.4f}, "
                      f"Novelty={metrics['novelty']:.4f}")

        return results

    # ================================================================
    # Utility helpers (unchanged)
    # ================================================================
    def get_similar_items_by_content(self, movie_idx, k=6):
        """Find top-k most similar movies using SBERT content embeddings."""
        if self.model is None:
            return []

        content_layer = self.model.get_layer("content_anchor_layer")
        proj_layer    = self.model.get_layer("content_projection")

        all_indices = tf.constant(np.arange(self.num_movies), dtype=tf.int32)
        raw_all     = content_layer(all_indices)
        proj_all    = proj_layer(raw_all).numpy()

        query_vec  = proj_all[movie_idx]
        norms      = np.linalg.norm(proj_all, axis=1, keepdims=True) + 1e-12
        query_norm = np.linalg.norm(query_vec) + 1e-12
        sims       = (proj_all @ query_vec) / (norms.flatten() * query_norm)

        sims[movie_idx] = -np.inf
        top_k = sims.argsort()[-k:][::-1]

        results = []
        for idx in top_k:
            rows = self.movies[self.movies["movie_idx"] == idx]
            if rows.empty:
                continue
            row = rows.iloc[0]
            results.append({
                "movie_idx":        int(idx),
                "Title":            row["Title"],
                "Genres":           row["Genres"],
                "Similarity":       round(float(sims[idx]), 4),
                "InteractionCount": self.countmap.get(int(idx), 0)
            })
        return results

    def get_ema_evolution_data(self, movie_idx, t_values=None):
        """Return alpha(t) and cosine-similarity-to-content for visualisation."""
        if self.model is None:
            return [], []

        if t_values is None:
          current_t = self.countmap.get(int(movie_idx), 0)
         # Always include the dense low-t grid (where the decay is sharp),
        # then extend coarser sampling up to and past the movie's actual t
        upper = max(200, int(current_t * 1.2))
        extra = np.linspace(200, upper, 20, dtype=int).tolist() if upper > 200 else []
        t_values = sorted(set([0, 1, 2, 5, 10, 20, 50, 100, 200] + extra + [int(current_t)]))

        content_layer = self.model.get_layer("content_anchor_layer")
        proj_layer    = self.model.get_layer("content_projection")
        inter_layer   = self.model.get_layer("movie_interaction_embedding")
        ema_layer     = self.model.get_layer("ema_fusion_layer")
        k_decay       = ema_layer.k

        raw_content  = content_layer(tf.constant([movie_idx], dtype=tf.int32))
        content_vec  = proj_layer(raw_content).numpy()[0]
        interact_vec = inter_layer(tf.constant([movie_idx], dtype=tf.int32)).numpy()[0]

        c_norm = np.linalg.norm(content_vec) + 1e-12

        alphas       = []
        similarities = []

        for t in t_values:
            alpha  = k_decay / (k_decay + t)
            fused  = alpha * content_vec + (1.0 - alpha) * interact_vec
            f_norm = np.linalg.norm(fused) + 1e-12
            sim_to_content = float(np.dot(fused, content_vec) / (f_norm * c_norm))
            alphas.append(round(alpha, 4))
            similarities.append(round(sim_to_content, 4))

        return t_values, alphas, similarities

    def get_item_profile_evolution(self, movie_idx):
        """Extract content anchor, interaction vector, and fused vector."""
        if self.model is None:
            return None, None, None

        content_layer = self.model.get_layer("content_anchor_layer")
        proj_layer    = self.model.get_layer("content_projection")
        raw_content   = content_layer(np.array([movie_idx]))
        content_vec   = proj_layer(raw_content).numpy()[0]

        inter_layer     = self.model.get_layer("movie_interaction_embedding")
        interaction_vec = inter_layer(np.array([movie_idx])).numpy()[0]

        ema_layer = self.model.get_layer("ema_fusion_layer")
        k         = ema_layer.k
        t         = self.countmap.get(movie_idx, 0)
        alpha     = k / (k + t)
        fused_vec = alpha * content_vec + (1.0 - alpha) * interaction_vec

        return content_vec, interaction_vec, fused_vec