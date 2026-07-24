"""
Honeywell Hackathon Anomaly Detection System - Phase 3: Training & Fusion Pipeline
-----------------------------------------------------------------------------------
Trains multi-class LightGBM classifier, optimizes baseline & ML hybrid risk fusion
weights, and evaluates held-out performance.

Outputs:
  - models/lightgbm_detector.pkl : Trained model artifact
  - data/eval_results.json       : Full benchmark evaluation results
"""

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, classification_report
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# Configuration & Feature Columns
# -----------------------------------------------------------------------------

FEATURE_COLS = [
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
    "resource_ngram_score",
    "baseline_anomaly_score"
]

# -----------------------------------------------------------------------------
# Metric & Evaluation Helpers
# -----------------------------------------------------------------------------

def compute_pr_auc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))

def compute_top_percent_precision(y_true, scores, top_percent=0.01):
    n_top = max(1, int(len(scores) * top_percent))
    top_indices = np.argsort(scores)[::-1][:n_top]
    top_y_true = np.array(y_true)[top_indices]
    return float(np.mean(top_y_true == 1))

# -----------------------------------------------------------------------------
# Main Training & Tuning Engine
# -----------------------------------------------------------------------------

def run_training_pipeline(features_path, baseline_path, labels_path, model_out_path, eval_out_path):
    print(f"[*] Loading dataset files...")
    df_feat = pd.read_csv(features_path)
    df_base = pd.read_csv(baseline_path)
    df_lbl = pd.read_csv(labels_path)

    # Join on event_id
    merged_df = df_feat.merge(df_base[["event_id", "baseline_anomaly_score"]], on="event_id")
    merged_df = merged_df.merge(df_lbl[["event_id", "is_attack", "attack_type"]], on="event_id")

    # Encode target labels
    label_encoder = LabelEncoder()
    merged_df['target_encoded'] = label_encoder.fit_transform(merged_df['attack_type'])
    class_names = list(label_encoder.classes_)
    normal_idx = label_encoder.transform(["Normal"])[0]

    # Preprocess categorical prev_resource_accessed if present
    if "prev_resource_accessed" in merged_df.columns:
        prev_res_encoder = LabelEncoder()
        merged_df['prev_resource_encoded'] = prev_res_encoder.fit_transform(merged_df['prev_resource_accessed'].astype(str))
        model_features = FEATURE_COLS + ["prev_resource_encoded"]
    else:
        model_features = FEATURE_COLS
        prev_res_encoder = None

    # Handle missing/infinite values
    X = merged_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = merged_df['target_encoded'].values
    y_binary = merged_df['is_attack'].values

    # Stratified Split: 70% Train, 15% Val, 15% Test
    print("[*] Performing 70/15/15 stratified train/validation/test split...")
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, merged_df.index, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    y_val_binary = y_binary[idx_val]
    y_test_binary = y_binary[idx_test]

    # Model Training
    if HAS_LGBM:
        print("[*] Training Multi-Class LightGBM Classifier with balanced class weights...")
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(class_names),
            class_weight="balanced",
            n_estimators=120,
            learning_rate=0.08,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        print("[*] LightGBM not found. Falling back to RandomForestClassifier...")
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    clf.fit(X_train, y_train)

    # Compute Model Attack Probabilities P(Attack) = 1 - P(Normal)
    val_probs = clf.predict_proba(X_val)
    val_p_attack = 1.0 - val_probs[:, normal_idx]

    val_baseline_scores = X_val["baseline_anomaly_score"].values

    # -------------------------------------------------------------------------
    # Weight Fusion Grid Sweep on Validation Set
    # -------------------------------------------------------------------------
    weight_candidates = [
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4)
    ]

    print("\n" + "=" * 65)
    print(" VALIDATION SET HYBRID WEIGHT FUSION GRID SWEEP")
    print("=" * 65)
    print(f" {'w1 (Baseline)':<15} | {'w2 (Model P)':<15} | {'Top-1% Alert Precision':<25}")
    print("-" * 65)

    grid_sweep_results = []
    best_w1, best_w2 = 0.5, 0.5
    best_top1_prec = -1.0

    for w1, w2 in weight_candidates:
        hybrid_val_scores = w1 * val_baseline_scores + w2 * val_p_attack
        top1_prec = compute_top_percent_precision(y_val_binary, hybrid_val_scores, top_percent=0.01)

        print(f" {w1:<15.1f} | {w2:<15.1f} | {top1_prec * 100:<25.2f}%")
        grid_sweep_results.append({
            "w1_baseline": w1,
            "w2_model": w2,
            "top_1pct_precision": round(top1_prec, 4)
        })

        if top1_prec > best_top1_prec:
            best_top1_prec = top1_prec
            best_w1, best_w2 = w1, w2

    print("-" * 65)
    print(f"[+] Optimal Fusion Weights Selected: w1 = {best_w1}, w2 = {best_w2} (Top-1% Precision: {best_top1_prec * 100:.2f}%)")
    print("=" * 65 + "\n")

    # -------------------------------------------------------------------------
    # Final Evaluation on 15% Held-Out Test Set
    # -------------------------------------------------------------------------
    print("[*] Evaluating optimal hybrid pipeline on 15% Held-Out Test Set...")
    test_probs = clf.predict_proba(X_test)
    test_p_attack = 1.0 - test_probs[:, normal_idx]
    test_baseline_scores = X_test["baseline_anomaly_score"].values

    test_hybrid_scores = best_w1 * test_baseline_scores + best_w2 * test_p_attack
    test_preds_encoded = clf.predict(X_test)
    test_preds_binary = (test_preds_encoded != normal_idx).astype(int)

    # Binary metrics (Attack vs Normal)
    test_prec = float(precision_score(y_test_binary, test_preds_binary, zero_division=0))
    test_rec = float(recall_score(y_test_binary, test_preds_binary, zero_division=0))
    test_f1 = float(f1_score(y_test_binary, test_preds_binary, zero_division=0))
    test_pr_auc = compute_pr_auc(y_test_binary, test_hybrid_scores)
    test_top1_prec = compute_top_percent_precision(y_test_binary, test_hybrid_scores, top_percent=0.01)

    # Confusion matrix for binary
    tn, fp, fn, tp = confusion_matrix(y_test_binary, test_preds_binary).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Per-Class Precision / Recall / F1
    per_class_report = classification_report(
        y_test, test_preds_encoded, target_names=class_names, output_dict=True, zero_division=0
    )

    per_class_summary = {}
    for cname in class_names:
        if cname in per_class_report:
            per_class_summary[cname] = {
                "precision": round(float(per_class_report[cname]["precision"]), 4),
                "recall": round(float(per_class_report[cname]["recall"]), 4),
                "f1_score": round(float(per_class_report[cname]["f1-score"]), 4),
                "support": int(per_class_report[cname]["support"])
            }

    multi_cm = confusion_matrix(y_test, test_preds_encoded).tolist()

    # Save Model Artifacts
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    model_package = {
        "model": clf,
        "label_encoder": label_encoder,
        "prev_res_encoder": prev_res_encoder,
        "features": model_features,
        "class_names": class_names,
        "normal_idx": int(normal_idx),
        "best_w1": best_w1,
        "best_w2": best_w2
    }
    joblib.dump(model_package, model_out_path)
    print(f"[+] Saved trained model package -> {model_out_path}")

    # Build Evaluation Results Structure
    eval_results = {
        "dataset_split": {
            "total_samples": len(merged_df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        },
        "optimal_weights": {
            "w1_baseline": best_w1,
            "w2_model": best_w2
        },
        "validation_grid_sweep": grid_sweep_results,
        "test_set_overall_metrics": {
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "pr_auc": round(test_pr_auc, 4),
            "false_positive_rate": round(fpr, 6),
            "top_1pct_alert_precision": round(test_top1_prec, 4)
        },
        "per_class_metrics": per_class_summary,
        "multiclass_confusion_matrix": multi_cm
    }

    os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)
    with open(eval_out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[+] Saved evaluation metrics -> {eval_out_path}")

    # Print Summary Table
    print("\n" + "=" * 65)
    print(" HELD-OUT TEST SET EVALUATION BENCHMARK")
    print("=" * 65)
    print(f" Precision (Binary)          : {test_prec * 100:.2f}%")
    print(f" Recall (Binary)             : {test_rec * 100:.2f}%")
    print(f" F1-Score (Binary)           : {test_f1 * 100:.2f}%")
    print(f" PR-AUC                      : {test_pr_auc:.4f}")
    print(f" False Positive Rate (FPR)   : {fpr * 100:.4f}%")
    print(f" Top-1% Alert Precision      : {test_top1_prec * 100:.2f}%")
    print("-" * 65)
    print(" Per-Class Breakdown:")
    for cname, metrics in per_class_summary.items():
        print(f"   - {cname:<20}: Prec={metrics['precision']*100:>5.1f}% | Rec={metrics['recall']*100:>5.1f}% | F1={metrics['f1_score']*100:>5.1f}% (N={metrics['support']})")
    print("=" * 65)

# -----------------------------------------------------------------------------
# Main Execution CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 3 Training Engine")
    parser.add_argument("--features", type=str, default="data/features.csv", help="Path to features CSV")
    parser.add_argument("--baseline", type=str, default="data/baseline_scores.csv", help="Path to baseline scores CSV")
    parser.add_argument("--labels", type=str, default="data/labels.csv", help="Path to ground truth labels CSV")
    parser.add_argument("--model-out", type=str, default="models/lightgbm_detector.pkl", help="Output model path")
    parser.add_argument("--eval-out", type=str, default="data/eval_results.json", help="Output metrics path")

    args = parser.parse_args()

    run_training_pipeline(args.features, args.baseline, args.labels, args.model_out, args.eval_out)

if __name__ == "__main__":
    main()
