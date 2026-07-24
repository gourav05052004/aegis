"""
Honeywell Hackathon Anomaly Detection System - Phase 3: Single-Pass Inference Engine
--------------------------------------------------------------------------------------
Executes single-pass batch inference on raw SOC event logs:
  1. Feature extraction
  2. Baseline profiling & concept drift score
  3. Supervised ML model attack classification
  4. Hybrid risk score fusion

Input  : data/events.csv, data/profiles.json, models/lightgbm_detector.pkl
Output : data/predictions.csv
"""

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineeringEngine
from baseline_profiler import BaselineProfiler

def run_prediction_pipeline(events_path, profiles_path, model_path, output_path):
    print(f"[*] Loading model package from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please run src/train_detector.py first.")

    model_package = joblib.load(model_path)
    clf = model_package["model"]
    label_encoder = model_package["label_encoder"]
    prev_res_encoder = model_package["prev_res_encoder"]
    model_features = model_package["features"]
    normal_idx = model_package["normal_idx"]
    w1 = model_package["best_w1"]
    w2 = model_package["best_w2"]

    print(f"[*] Loading profiles from {profiles_path}...")
    with open(profiles_path, "r") as f:
        profiles_list = json.load(f)
    profiles_dict = {p["entity_id"]: p for p in profiles_list}

    print(f"[*] Loading raw events from {events_path}...")
    events_df = pd.read_csv(events_path)

    # 1. Feature Extraction
    print(f"[*] Step 1/3: Extracting point-in-time and sequential features ({len(events_df)} events)...")
    feature_engine = FeatureEngineeringEngine(profiles_dict)
    features_df = feature_engine.extract_features(events_df)

    # 2. Baseline Profiler
    print("[*] Step 2/3: Computing baseline anomaly scores & EWMA concept drift...")
    profiler = BaselineProfiler(contamination=0.025, ewma_alpha=0.3, random_state=42)
    scores_df = profiler.fit_predict(features_df)

    features_df["baseline_anomaly_score"] = scores_df["baseline_anomaly_score"]

    # Encode prev_resource_accessed if present
    if "prev_resource_encoded" in model_features:
        if prev_res_encoder is not None:
            # Handle unseen resource categories safely
            known_classes = set(prev_res_encoder.classes_)
            features_df['prev_resource_clean'] = features_df['prev_resource_accessed'].apply(
                lambda x: x if str(x) in known_classes else "NONE"
            )
            features_df['prev_resource_encoded'] = prev_res_encoder.transform(features_df['prev_resource_clean'].astype(str))
        else:
            features_df['prev_resource_encoded'] = 0

    X_infer = features_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3. Supervised Model Prediction & Risk Score Fusion
    print("[*] Step 3/3: Running classifier inference & hybrid risk score fusion...")
    probs = clf.predict_proba(X_infer)
    pred_indices = np.argmax(probs, axis=1)
    predicted_labels = label_encoder.inverse_transform(pred_indices)

    p_attack = 1.0 - probs[:, normal_idx]
    baseline_scores = features_df["baseline_anomaly_score"].values

    hybrid_risk_scores = w1 * baseline_scores + w2 * p_attack
    confidences = np.max(probs, axis=1)

    # Build Output Predictions DataFrame
    predictions_df = pd.DataFrame({
        "event_id": features_df["event_id"],
        "entity_id": features_df["entity_id"],
        "timestamp": features_df["timestamp"],
        "baseline_score": np.round(baseline_scores, 4),
        "model_probability": np.round(p_attack, 4),
        "hybrid_risk_score": np.round(hybrid_risk_scores, 4),
        "predicted_attack_type": predicted_labels,
        "confidence": np.round(confidences, 4)
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"[+] Successfully generated predictions -> {output_path}")

    # Summary
    anom_count = (predictions_df["predicted_attack_type"] != "Normal").sum()
    print("\n" + "=" * 55)
    print(" BATCH INFERENCE SUMMARY")
    print("=" * 55)
    print(f" Total Events Processed  : {len(predictions_df)}")
    print(f" Predicted Attack Events : {anom_count} ({anom_count / len(predictions_df) * 100:.2f}%)")
    print(f" Mean Hybrid Risk Score  : {predictions_df['hybrid_risk_score'].mean():.4f}")
    print(" Predicted Class Distribution:")
    pred_counts = predictions_df["predicted_attack_type"].value_counts()
    for cname, cnt in pred_counts.items():
        print(f"   - {cname:<22}: {cnt:>6} ({cnt/len(predictions_df)*100:.2f}%)")
    print("=" * 55)

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 3 Predict Engine")
    parser.add_argument("--events", type=str, default="data/events.csv", help="Path to events CSV")
    parser.add_argument("--profiles", type=str, default="data/profiles.json", help="Path to profiles JSON")
    parser.add_argument("--model", type=str, default="models/lightgbm_detector.pkl", help="Path to trained model package")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Output path for predictions")

    args = parser.parse_args()

    run_prediction_pipeline(args.events, args.profiles, args.model, args.output)

if __name__ == "__main__":
    main()
