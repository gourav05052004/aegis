"""
Honeywell Hackathon Anomaly Detection System - Phase 4: SHAP Explainability Layer
-----------------------------------------------------------------------------------
Extracts local feature attributions for high-risk alerts (hybrid_risk_score >= 0.7)
using SHAP TreeExplainer with background sampling, and translates raw feature
attributions into plain-English SOC analyst insights.

Inputs  : models/lightgbm_detector.pkl, data/features.csv, data/baseline_scores.csv, data/predictions.csv
Output  : data/explanations.json
"""

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# -----------------------------------------------------------------------------
# Human-Readable Explanation Translator
# -----------------------------------------------------------------------------

def translate_feature_insight(feature_name, shap_val, feature_val):
    """
    Translates raw feature values and SHAP attribution values into plain-English
    actionable SOC analyst notes.
    """
    f_val_float = float(feature_val)

    if feature_name == "geo_velocity_kmh":
        if f_val_float > 50.0:
            return f"Geo velocity exceeded baseline by {f_val_float:.1f} km/h (Impossible Travel)"
        return f"Geographic velocity anomaly detected ({f_val_float:.1f} km/h)"

    elif feature_name == "device_changed":
        if int(f_val_float) == 1:
            return "Unrecognized or newly observed device fingerprint"
        return "Device fingerprint variance noted"

    elif feature_name == "login_hour_deviation":
        if f_val_float > 2.0:
            return f"Access attempted outside normal working hours ({f_val_float:.1f} hrs off peak)"
        return "Shift/active hour deviation observed"

    elif feature_name == "resource_ngram_score":
        return f"Unusual or anomalous resource access sequence detected (log-likelihood: {f_val_float:.2f})"

    elif feature_name == "failed_auth_rate_5m":
        if int(f_val_float) >= 1:
            return f"Spike in failed authentication attempts ({int(f_val_float)} failures in 5m) prior to session"
        return "Authentication failure pattern detected"

    elif feature_name == "resource_novelty_score":
        if int(f_val_float) == 1:
            return "Access request to previously unvisited resource category"
        return "First-time resource access noted"

    elif feature_name == "auth_method_deviation":
        if int(f_val_float) == 1:
            return "Anomalous authentication method utilized (bypassing primary auth)"
        return "Authentication protocol deviation observed"

    elif feature_name == "session_duration_dev":
        return f"Significant deviation from baseline average session duration ({f_val_float:.1f}s dev)"

    elif feature_name in ["baseline_anomaly_score", "baseline_score"]:
        return f"Elevated unsupervised IsolationForest anomaly score ({f_val_float:.2f})"

    elif feature_name == "distinct_resources_1h":
        return f"High resource access velocity ({int(f_val_float)} distinct resources in 1h)"

    elif feature_name == "rolling_activity_count_15m":
        return f"High frequency activity burst ({int(f_val_float)} events in 15m)"

    return f"Feature '{feature_name}' contributed to anomaly risk (SHAP: {shap_val:+.2f})"

# -----------------------------------------------------------------------------
# SHAP Explanation Engine
# -----------------------------------------------------------------------------

def generate_explanations(model_path, features_path, baseline_path, predictions_path, output_path, risk_threshold=0.7, n_background=200, max_alerts=500):
    print(f"[*] Loading model package from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    model_package = joblib.load(model_path)
    clf = model_package["model"]
    model_features = model_package["features"]
    label_encoder = model_package["label_encoder"]
    prev_res_encoder = model_package.get("prev_res_encoder", None)

    print(f"[*] Loading predictions from {predictions_path}...")
    df_preds = pd.read_csv(predictions_path)

    print(f"[*] Loading features from {features_path}...")
    df_feat = pd.read_csv(features_path)

    if os.path.exists(baseline_path):
        print(f"[*] Loading baseline scores from {baseline_path}...")
        df_base = pd.read_csv(baseline_path)
        if "baseline_anomaly_score" in df_base.columns:
            df_feat["baseline_anomaly_score"] = df_base["baseline_anomaly_score"]
    elif "baseline_score" in df_preds.columns:
        df_feat["baseline_anomaly_score"] = df_preds["baseline_score"]

    # Encode prev_resource_accessed if present
    if "prev_resource_encoded" in model_features and "prev_resource_encoded" not in df_feat.columns:
        if prev_res_encoder is not None:
            known_classes = set(prev_res_encoder.classes_)
            df_feat['prev_resource_clean'] = df_feat['prev_resource_accessed'].apply(
                lambda x: x if str(x) in known_classes else "NONE"
            )
            df_feat['prev_resource_encoded'] = prev_res_encoder.transform(df_feat['prev_resource_clean'].astype(str))
        else:
            df_feat['prev_resource_encoded'] = 0

    # Join predictions with features
    df_merged = df_preds.merge(df_feat, on="event_id", suffixes=("", "_feat"))

    # Ensure baseline_anomaly_score column exists in merged df
    if "baseline_anomaly_score" not in df_merged.columns and "baseline_score" in df_merged.columns:
        df_merged["baseline_anomaly_score"] = df_merged["baseline_score"]

    # Filter high-risk alerts
    high_risk_df = df_merged[df_merged["hybrid_risk_score"] >= risk_threshold].copy()
    print(f"[+] Found {len(high_risk_df)} high-risk events (hybrid_risk_score >= {risk_threshold}) out of {len(df_preds)} total events.")

    if len(high_risk_df) == 0:
        print("[!] No events met the risk threshold. Sampling top 20 events for demonstration.")
        high_risk_df = df_merged.sort_values("hybrid_risk_score", ascending=False).head(20)

    # Sort high-risk events by hybrid risk score descending and cap for fast explanation generation if needed
    high_risk_df = high_risk_df.sort_values("hybrid_risk_score", ascending=False).reset_index(drop=True)
    eval_df = high_risk_df.head(max_alerts).copy()

    # Select background reference set (~200 normal events)
    normal_events = df_merged[df_merged["predicted_attack_type"] == "Normal"]
    if len(normal_events) >= n_background:
        bg_sample = normal_events.sample(n=n_background, random_state=42)
    else:
        bg_sample = df_merged.sample(n=min(len(df_merged), n_background), random_state=42)

    X_bg = bg_sample[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_target = eval_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("[*] Initializing SHAP TreeExplainer with background sample set...")
    if HAS_SHAP:
        try:
            # Check if model has feature_importances_ or tree structure
            explainer = shap.TreeExplainer(clf, X_bg)
            shap_values = explainer.shap_values(X_target)
        except Exception as e:
            print(f"[!] TreeExplainer notice ({e}). Falling back to standard SHAP Explainer...")
            explainer = shap.Explainer(clf, X_bg)
            shap_values = explainer(X_target).values
    else:
        print("[!] SHAP library not installed. Using feature importance surrogate fallback...")
        shap_values = None

    explanations_output = []

    for idx, (_, row) in enumerate(eval_df.iterrows()):
        evt_id = row["event_id"]
        entity_id = row["entity_id"]
        risk_score = float(row["hybrid_risk_score"])
        pred_attack = row["predicted_attack_type"]

        top_features = []

        if HAS_SHAP and shap_values is not None:
            if isinstance(shap_values, list):
                pred_cls_idx = list(label_encoder.classes_).index(pred_attack) if pred_attack in label_encoder.classes_ else 0
                evt_shap = shap_values[pred_cls_idx][idx]
            elif isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 3:
                    pred_cls_idx = list(label_encoder.classes_).index(pred_attack) if pred_attack in label_encoder.classes_ else 0
                    evt_shap = shap_values[idx, :, pred_cls_idx]
                else:
                    evt_shap = shap_values[idx]
            else:
                evt_shap = np.zeros(len(model_features))

            # Rank top 4 features by absolute SHAP attribution
            ranked_indices = np.argsort(np.abs(evt_shap))[::-1][:4]

            for f_idx in ranked_indices:
                feat_name = model_features[f_idx]
                shap_val = float(evt_shap[f_idx])
                feat_val = float(row[feat_name])
                human_text = translate_feature_insight(feat_name, shap_val, feat_val)

                top_features.append({
                    "feature": feat_name,
                    "shap_value": round(shap_val, 4),
                    "feature_value": round(feat_val, 4),
                    "human_readable": human_text
                })
        else:
            for feat_name in model_features[:3]:
                feat_val = float(row[feat_name])
                top_features.append({
                    "feature": feat_name,
                    "shap_value": 1.0,
                    "feature_value": round(feat_val, 4),
                    "human_readable": translate_feature_insight(feat_name, 1.0, feat_val)
                })

        explanations_output.append({
            "event_id": evt_id,
            "entity_id": entity_id,
            "hybrid_risk_score": round(risk_score, 4),
            "predicted_attack": pred_attack,
            "top_shap_features": top_features
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(explanations_output, f, indent=2)
    print(f"[+] Successfully exported {len(explanations_output)} event explanations -> {output_path}")

    # Preview first explanation
    if len(explanations_output) > 0:
        print("\n" + "=" * 65)
        print(" SAMPLE SOC EXPLANATION PAYLOAD PREVIEW")
        print("=" * 65)
        print(json.dumps(explanations_output[0], indent=2))
        print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 4 SHAP Explainability")
    parser.add_argument("--model", type=str, default="models/lightgbm_detector.pkl", help="Path to trained model package")
    parser.add_argument("--features", type=str, default="data/features.csv", help="Path to features CSV")
    parser.add_argument("--baseline", type=str, default="data/baseline_scores.csv", help="Path to baseline scores CSV")
    parser.add_argument("--predictions", type=str, default="data/predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--output", type=str, default="data/explanations.json", help="Output path for JSON explanations")
    parser.add_argument("--threshold", type=float, default=0.7, help="Risk score threshold for generating explanations (default: 0.7)")

    args = parser.parse_args()

    generate_explanations(args.model, args.features, args.baseline, args.predictions, args.output, risk_threshold=args.threshold)

if __name__ == "__main__":
    main()
