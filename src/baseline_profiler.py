"""
Honeywell Hackathon Anomaly Detection System - Phase 2: Baseline Profiler Engine
----------------------------------------------------------------------------------
Establishes unsupervised entity behavioral baselines using Isolation Forest
and EWMA (Exponentially Weighted Moving Average) concept drift adaptation.

Input  : data/features.csv
Output : data/baseline_scores.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Core Feature List for ML Model
# -----------------------------------------------------------------------------

NUMERICAL_FEATURE_COLS = [
    "login_hour_deviation",
    "geo_velocity_kmh",
    "time_since_last_activity_sec",
    "resource_novelty_score",
    "session_duration_dev",
    "failed_auth_rate_5m",
    "device_changed",
    "auth_method_deviation",
    "resource_transition_freq",
    "distinct_resources_1h",
    "rolling_activity_count_15m",
    "resource_ngram_score"
]

# -----------------------------------------------------------------------------
# Baseline Profiler & Concept Drift Engine
# -----------------------------------------------------------------------------

class BaselineProfiler:
    def __init__(self, contamination=0.025, ewma_alpha=0.3, random_state=42):
        self.contamination = contamination
        self.ewma_alpha = ewma_alpha
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit_predict(self, features_df):
        """
        Trains IsolationForest model and computes concept-drift adapted scores.
        """
        print("[*] Extracting feature vectors for baseline model training...")
        X_raw = features_df[NUMERICAL_FEATURE_COLS].copy()

        # Handle any infinite or missing values safely
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_raw)

        print("[*] Fitting unsupervised Isolation Forest baseline model...")
        self.model.fit(X_scaled)

        # Raw decision function: positive for inliers, negative for outliers
        raw_decision_scores = self.model.decision_function(X_scaled)

        # Invert scores so higher = more anomalous
        raw_anomaly_scores = -raw_decision_scores

        # Min-Max normalize raw scores strictly to [0.0, 1.0]
        min_score = np.min(raw_anomaly_scores)
        max_score = np.max(raw_anomaly_scores)
        if max_score > min_score:
            norm_scores = (raw_anomaly_scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.zeros_like(raw_anomaly_scores)

        features_df['raw_anomaly_score'] = norm_scores

        # ---------------------------------------------------------------------
        # Concept Drift Adaptation via EWMA
        # ---------------------------------------------------------------------
        print("[*] Applying per-entity EWMA concept drift adaptation...")
        
        entity_ewma_state = {}  # entity_id -> running EWMA anomaly score
        adapted_scores = []
        ewma_historical = []

        for idx, row in features_df.iterrows():
            entity_id = row['entity_id']
            s_raw = row['raw_anomaly_score']

            if entity_id not in entity_ewma_state:
                # First event for entity: initialize EWMA state with current score
                s_ewma = s_raw
            else:
                # EWMA Update Formula: S_t = alpha * S_curr + (1 - alpha) * S_prev
                s_prev = entity_ewma_state[entity_id]
                s_ewma = self.ewma_alpha * s_raw + (1.0 - self.ewma_alpha) * s_prev

            entity_ewma_state[entity_id] = s_ewma

            # Final baseline anomaly score blends raw point score with entity historical EWMA baseline
            # Sharp spikes retain high anomaly signal while sustained low-grade drift evolves the baseline
            baseline_score = 0.7 * s_raw + 0.3 * s_ewma

            adapted_scores.append(round(float(baseline_score), 6))
            ewma_historical.append(round(float(s_ewma), 6))

        features_df['ewma_entity_score'] = ewma_historical
        features_df['baseline_anomaly_score'] = adapted_scores
        features_df['is_baseline_anomaly'] = (features_df['baseline_anomaly_score'] >= 0.65).astype(int)

        # Prepare final output dataset
        output_cols = [
            "event_id",
            "entity_id",
            "raw_anomaly_score",
            "ewma_entity_score",
            "baseline_anomaly_score",
            "is_baseline_anomaly"
        ]
        return features_df[output_cols]

# -----------------------------------------------------------------------------
# Main Execution CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 2 Baseline Profiler")
    parser.add_argument("--features", type=str, default="data/features.csv", help="Path to features CSV (default: data/features.csv)")
    parser.add_argument("--output", type=str, default="data/baseline_scores.csv", help="Output path for baseline scores (default: data/baseline_scores.csv)")
    parser.add_argument("--contamination", type=float, default=0.025, help="Contamination rate for Isolation Forest (default: 0.025)")

    args = parser.parse_args()

    print(f"[*] Loading feature matrix from {args.features}...")
    features_df = pd.read_csv(args.features)

    profiler = BaselineProfiler(contamination=args.contamination, ewma_alpha=0.3, random_state=42)
    scores_df = profiler.fit_predict(features_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scores_df.to_csv(args.output, index=False)
    print(f"[+] Successfully generated baseline anomaly scores for {len(scores_df)} events -> {args.output}")

    # Summary statistics
    num_anomalies = (scores_df['is_baseline_anomaly'] == 1).sum()
    print("\n" + "=" * 50)
    print(" BASELINE PROFILER SUMMARY")
    print("=" * 50)
    print(f" Total Events Scored  : {len(scores_df)}")
    print(f" Flagged Baseline Anom: {num_anomalies} ({num_anomalies / len(scores_df) * 100:.2f}%)")
    print(f" Score Range          : [{scores_df['baseline_anomaly_score'].min():.4f}, {scores_df['baseline_anomaly_score'].max():.4f}]")
    print(f" Score Mean / Std     : {scores_df['baseline_anomaly_score'].mean():.4f} +/- {scores_df['baseline_anomaly_score'].std():.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
