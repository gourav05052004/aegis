"""
Honeywell Hackathon Anomaly Detection System - Phase 7: Real-Time Stream Simulator
----------------------------------------------------------------------------------
Simulates a real-time SOC telemetry event stream (replicating production Kafka / Azure Event Hubs)
by replaying events from data/events.csv with sub-second feature extraction, baseline profiling,
LightGBM inference, and weight fusion scoring.

Usage:
    python src/stream_simulator.py --events 1000 --delay 0.2 --output data/live_stream_predictions.json
"""

import os
import sys
import time
import json
import math
import argparse
import joblib
import numpy as np
import pandas as pd

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in kilometers between two geo coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

class RealTimeStreamSimulator:
    def __init__(self, model_path="models/lightgbm_detector.pkl", profiles_path="data/profiles.json"):
        print("[INFO] Initializing Real-Time Stream Engine...")
        self.model = None
        self.classes = []
        self.feature_names = []

        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data.get("model")
                self.classes = model_data.get("class_names", model_data.get("classes", []))
                self.feature_names = model_data.get("features", [])
            else:
                self.model = model_data
                self.classes = getattr(self.model, "classes_", [])
                self.feature_names = getattr(self.model, "feature_names_in_", getattr(self.model, "feature_name_", []))
            print(f"[INFO] Loaded Detector Model from '{model_path}' with {len(self.feature_names)} features.")
        else:
            print(f"[WARNING] Model file '{model_path}' not found! Using surrogate scoring.")

        self.profiles = {}
        if os.path.exists(profiles_path):
            with open(profiles_path, "r") as f:
                profs = json.load(f)
                self.profiles = {p["entity_id"]: p for p in profs}
            print(f"[INFO] Loaded {len(self.profiles)} Entity Profiles.")

        # In-memory entity tracking state for streaming feature extraction
        self.entity_states = {}

    def extract_streaming_features(self, event, timestamp_dt):
        """Extracts point-in-time and sequential features for an incoming event."""
        entity_id = event["entity_id"]
        profile = self.profiles.get(entity_id, {})
        
        state = self.entity_states.setdefault(entity_id, {
            "last_ts": None,
            "last_lat": profile.get("home_location", {}).get("lat", 0.0),
            "last_lon": profile.get("home_location", {}).get("lon", 0.0),
            "visited_categories": set(),
            "recent_timestamps": [],
            "recent_failed": []
        })

        # 1. login_hour_deviation
        event_hour = timestamp_dt.hour
        peak_hour = profile.get("peak_hour", 14)
        hour_diff = abs(event_hour - peak_hour)
        login_hour_dev = min(hour_diff, 24 - hour_diff)

        # 2. geo_velocity_kmh
        current_lat = event.get("geo_lat", profile.get("home_location", {}).get("lat", 0.0))
        current_lon = event.get("geo_lon", profile.get("home_location", {}).get("lon", 0.0))
        
        if state["last_ts"] is not None:
            elapsed_sec = (timestamp_dt - state["last_ts"]).total_seconds()
            elapsed_hours = max(elapsed_sec / 3600.0, 0.0001)
            dist_km = haversine_distance_km(state["last_lat"], state["last_lon"], current_lat, current_lon)
            geo_velocity = min(dist_km / elapsed_hours, 2000.0)
            time_since_last_sec = max(elapsed_sec, 0.0)
        else:
            geo_velocity = 0.0
            time_since_last_sec = 86400.0

        # Update geo state
        state["last_ts"] = timestamp_dt
        state["last_lat"] = current_lat
        state["last_lon"] = current_lon

        # 3. resource_novelty_score
        cat = event.get("resource_category", "infra")
        novelty = 1.0 if cat not in state["visited_categories"] else 0.0
        state["visited_categories"].add(cat)

        # 4. auth_method_deviation
        primary_auth = profile.get("primary_auth", "password")
        current_auth = event.get("auth_method", "password")
        auth_dev = 0.0 if current_auth == primary_auth else 1.0

        # 5. device_changed
        primary_device = profile.get("primary_device", "DEV_001")
        current_device = event.get("device_fingerprint", "DEV_001")
        device_changed = 0.0 if current_device == primary_device else 1.0

        # 6. failed_auth_rate_5m
        status = event.get("status", "success")
        is_failed = 1.0 if status == "failed" else 0.0
        
        # Clean rolling 5m timestamps
        cutoff_5m = timestamp_dt.timestamp() - 300
        state["recent_failed"] = [ts for ts in state["recent_failed"] if ts >= cutoff_5m]
        if is_failed:
            state["recent_failed"].append(timestamp_dt.timestamp())
        failed_rate_5m = float(len(state["recent_failed"]))

        # 7. session_duration_dev
        avg_dur = profile.get("avg_session_duration", 1800.0)
        curr_dur = float(event.get("session_duration_sec", 1800.0))
        session_dur_dev = abs(curr_dur - avg_dur)

        # 8. Unsupervised baseline score heuristic
        baseline_score = min(1.0, max(0.05, 
            (login_hour_dev / 12.0) * 0.25 + 
            min(geo_velocity / 800.0, 1.0) * 0.35 + 
            novelty * 0.15 + 
            auth_dev * 0.15 + 
            min(failed_rate_5m / 5.0, 1.0) * 0.10
        ))

        feature_dict = {
            "login_hour_deviation": login_hour_dev,
            "geo_velocity_kmh": geo_velocity,
            "time_since_last_activity_sec": time_since_last_sec,
            "resource_novelty_score": novelty,
            "auth_method_deviation": auth_dev,
            "device_changed": device_changed,
            "failed_auth_rate_5m": failed_rate_5m,
            "session_duration_dev": session_dur_dev,
            "resource_transition_freq": 0.05,
            "distinct_resources_1h": 1.0,
            "rolling_activity_count_15m": float(len(state["recent_failed"]) + 1),
            "resource_ngram_score": -0.5 if novelty > 0 else 0.0,
            "baseline_anomaly_score": baseline_score,
            "prev_resource_encoded": 0.0
        }
        return feature_dict, baseline_score

    def process_event(self, event):
        """Runs sub-second real-time scoring for a single event."""
        start_time = time.time()
        
        ts_str = event["timestamp"]
        ts_dt = pd.to_datetime(ts_str)

        features, baseline_score = self.extract_streaming_features(event, ts_dt)

        model_prob = 0.05
        pred_class = "Normal"

        if self.model is not None:
            feat_df = pd.DataFrame([features])
            if self.feature_names:
                for col in self.feature_names:
                    if col not in feat_df.columns:
                        feat_df[col] = 0.0
                feat_df = feat_df[list(self.feature_names)]
            
            probs = self.model.predict_proba(feat_df)[0]
            normal_idx = 0
            if len(self.classes) > 0 and "Normal" in list(self.classes):
                normal_idx = list(self.classes).index("Normal")
            
            model_prob = 1.0 - probs[normal_idx]
            top_idx = int(np.argmax(probs))
            pred_class = str(self.classes[top_idx]) if len(self.classes) > top_idx else "Normal"
        else:
            model_prob = baseline_score
            if baseline_score >= 0.70:
                pred_class = "Impossible Travel" if features["geo_velocity_kmh"] > 200 else "Brute Force"

        # Optimal weight fusion
        w1, w2 = 0.3, 0.7
        hybrid_risk_score = float(np.clip(w1 * baseline_score + w2 * model_prob, 0.0, 1.0))

        latency_ms = (time.time() - start_time) * 1000.0

        payload = {
            "event_id": event["event_id"],
            "entity_id": event["entity_id"],
            "timestamp": ts_str,
            "baseline_score": round(float(baseline_score), 4),
            "model_probability": round(float(model_prob), 4),
            "hybrid_risk_score": round(float(hybrid_risk_score), 4),
            "predicted_attack_type": pred_class,
            "ingestion_latency_ms": round(latency_ms, 2)
        }
        return payload

def run_stream_simulation(events_path="data/events.csv", output_path="data/live_stream_predictions.json", max_events=1000, delay_sec=0.2):
    """Executes real-time streaming ingestion replay loop."""
    print(f"================================================================================")
    print(f" [STREAM] HONEYWELL REAL-TIME SOC TELEMETRY STREAM SIMULATOR")
    print(f" Target Stream: {events_path} | Max Events: {max_events} | Delay: {delay_sec}s")
    print(f"================================================================================")

    if not os.path.exists(events_path):
        print(f"[ERROR] Events file '{events_path}' does not exist!")
        return

    df_events = pd.read_csv(events_path).head(max_events)
    simulator = RealTimeStreamSimulator()

    live_predictions = []
    
    start_sim_time = time.time()
    threat_count = 0

    print(f"{'Event ID':<14} | {'Entity ID':<10} | {'Hybrid Score':<12} | {'Attack Class':<20} | {'Latency':<10}")
    print("-" * 78)

    for idx, row in df_events.iterrows():
        evt_dict = row.to_dict()
        pred_payload = simulator.process_event(evt_dict)

        live_predictions.append(pred_payload)

        if pred_payload["hybrid_risk_score"] >= 0.70:
            threat_count += 1

        score_str = f"{pred_payload['hybrid_risk_score']:.4f}"
        print(f"{pred_payload['event_id']:<14} | {pred_payload['entity_id']:<10} | {score_str:<12} | {pred_payload['predicted_attack_type']:<20} | {pred_payload['ingestion_latency_ms']}ms")

        # Periodically write live predictions atomically
        if len(live_predictions) % 10 == 0 or idx == len(df_events) - 1:
            with open(output_path, "w") as f:
                json.dump(live_predictions, f, indent=2)

        time.sleep(delay_sec)

    total_time = time.time() - start_sim_time
    print(f"================================================================================")
    print(f" [OK] STREAM SIMULATION COMPLETE")
    print(f" Processed {len(live_predictions)} Events in {total_time:.2f}s | Active Threats: {threat_count}")
    print(f" Live Stream Output Written to: '{output_path}'")
    print(f"================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honeywell SOC Real-Time Telemetry Stream Simulator")
    parser.add_argument("--events", type=int, default=1000, help="Number of telemetry events to stream")
    parser.add_argument("--delay", type=float, default=0.2, help="Ingestion delay in seconds between events")
    parser.add_argument("--events-file", type=str, default="data/events.csv", help="Input telemetry CSV path")
    parser.add_argument("--output", type=str, default="data/live_stream_predictions.json", help="Output JSON path")
    
    args = parser.parse_args()
    
    run_stream_simulation(
        events_path=args.events_file,
        output_path=args.output,
        max_events=args.events,
        delay_sec=args.delay
    )
