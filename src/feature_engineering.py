"""
Honeywell Hackathon Anomaly Detection System - Phase 2: Feature Engineering Engine
----------------------------------------------------------------------------------
Transforms raw event streams (data/events.csv) and profiles (data/profiles.json)
into rich behavioral feature vectors (data/features.csv).
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime

# -----------------------------------------------------------------------------
# Mathematical Helper Functions
# -----------------------------------------------------------------------------

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculates great-circle distance between two points on Earth in kilometers.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def circular_hour_deviation(hour1, hour2):
    """
    Computes absolute distance on a 24-hour circular clock.
    """
    diff = abs(hour1 - hour2) % 24
    return min(diff, 24 - diff)

# -----------------------------------------------------------------------------
# Feature Extraction Engine
# -----------------------------------------------------------------------------

class FeatureEngineeringEngine:
    def __init__(self, profiles_dict):
        self.profiles = profiles_dict

    def extract_features(self, events_df):
        """
        Processes events DataFrame chronologically to extract point-in-time
        and sequential transition features.
        """
        # Ensure chronological ordering
        events_df['parsed_time'] = pd.to_datetime(events_df['timestamp'])
        events_df = events_df.sort_values('parsed_time').reset_index(drop=True)

        # Parse geo_location JSON string if needed
        parsed_geos = []
        for geo in events_df['geo_location']:
            if isinstance(geo, str):
                parsed_geos.append(json.loads(geo))
            elif isinstance(geo, dict):
                parsed_geos.append(geo)
            else:
                parsed_geos.append({"country": "UNKNOWN", "lat": 0.0, "lon": 0.0})
        events_df['geo_dict'] = parsed_geos

        # Precompute global N-gram transition counts for n-gram scoring
        print("[*] Precomputing global resource N-gram transition probabilities...")
        transition_2gram_counts = defaultdict(lambda: defaultdict(int))
        transition_3gram_counts = defaultdict(lambda: defaultdict(int))
        resource_counts = defaultdict(int)

        # Group resources by entity to build initial transition stats
        entity_resource_sequences = defaultdict(list)
        for _, row in events_df.iterrows():
            entity_resource_sequences[row['entity_id']].append(row['resource_accessed'])

        for entity_id, seq in entity_resource_sequences.items():
            for i in range(len(seq)):
                r_curr = seq[i]
                resource_counts[r_curr] += 1
                if i >= 1:
                    r_prev1 = seq[i-1]
                    transition_2gram_counts[r_prev1][r_curr] += 1
                if i >= 2:
                    r_prev2 = seq[i-2]
                    r_prev1 = seq[i-1]
                    transition_3gram_counts[(r_prev2, r_prev1)][r_curr] += 1

        total_resources = sum(resource_counts.values()) or 1
        vocab_size = len(resource_counts) or 1

        # Per-entity state tracking structures
        entity_last_event = {}          # entity_id -> (timestamp_sec, lat, lon)
        entity_seen_resources = defaultdict(set)
        entity_recent_window = defaultdict(deque) # entity_id -> deque of (timestamp_sec, status, resource_accessed)
        entity_history_resources = defaultdict(list)

        output_rows = []

        print("[*] Processing events and building vector representations...")
        for idx, row in events_df.iterrows():
            event_id = row['event_id']
            entity_id = row['entity_id']
            curr_time = row['parsed_time']
            curr_time_sec = curr_time.timestamp()

            profile = self.profiles.get(entity_id, {
                "peak_hour": 14,
                "primary_device": "",
                "primary_auth": "password",
                "avg_session_duration": 1800.0,
                "primary_category": "email"
            })

            # -----------------------------------------------------------------
            # 1. Standard Point-in-Time Features
            # -----------------------------------------------------------------
            # a. Login hour deviation
            event_hour = curr_time.hour
            login_hour_dev = circular_hour_deviation(event_hour, profile["peak_hour"])

            # b. Geo velocity (km/h) & Time since last activity
            curr_geo = row['geo_dict']
            curr_lat, curr_lon = curr_geo.get("lat", 0.0), curr_geo.get("lon", 0.0)

            if entity_id in entity_last_event:
                prev_time_sec, prev_lat, prev_lon = entity_last_event[entity_id]
                delta_time_sec = max(0.0, curr_time_sec - prev_time_sec)
                dist_km = haversine_distance_km(prev_lat, prev_lon, curr_lat, curr_lon)
                delta_time_hours = delta_time_sec / 3600.0
                geo_velocity = (dist_km / delta_time_hours) if delta_time_hours > 0.0001 else 0.0
            else:
                delta_time_sec = 0.0
                geo_velocity = 0.0

            # c. Resource novelty score
            curr_res = row['resource_accessed']
            resource_novelty = 1 if (curr_res not in entity_seen_resources[entity_id]) else 0
            entity_seen_resources[entity_id].add(curr_res)

            # d. Session duration deviation
            sess_duration = float(row['session_duration_sec'])
            session_duration_dev = abs(sess_duration - profile["avg_session_duration"])

            # e. Device changed & Auth method deviation
            device_changed = 1 if row['device_fingerprint'] != profile["primary_device"] else 0
            auth_method_dev = 1 if row['auth_method'] != profile["primary_auth"] else 0

            # -----------------------------------------------------------------
            # 2. Sequential & Rolling Window Features
            # -----------------------------------------------------------------
            rec_deque = entity_recent_window[entity_id]
            # Prune events older than 1 hour (3600s) from entity rolling window
            while rec_deque and (curr_time_sec - rec_deque[0][0] > 3600):
                rec_deque.popleft()

            # Add current event to rolling window
            rec_deque.append((curr_time_sec, row['status'], curr_res))

            # f. Failed auth rate in rolling 5m (300s)
            failed_auth_5m = sum(1 for t_sec, status, _ in rec_deque if (curr_time_sec - t_sec <= 300) and status == "failed")

            # g. Distinct resources accessed in rolling 1h (3600s)
            distinct_res_1h = len(set(res for _, _, res in rec_deque))

            # h. Rolling activity count in 15m (900s)
            rolling_act_15m = sum(1 for t_sec, _, _ in rec_deque if (curr_time_sec - t_sec <= 900))

            # i. Sequential Transition Features (Prev Resource, Transition Freq, N-gram)
            hist_seq = entity_history_resources[entity_id]
            if len(hist_seq) >= 1:
                prev_res = hist_seq[-1]
            else:
                prev_res = "NONE"

            # 2-gram transition probability P(R_t | R_{t-1})
            if prev_res != "NONE":
                count_prev = sum(transition_2gram_counts[prev_res].values())
                count_trans = transition_2gram_counts[prev_res][curr_res]
                prob_2gram = (count_trans + 1.0) / (count_prev + vocab_size)
            else:
                prob_2gram = 1.0 / vocab_size

            # 3-gram transition probability P(R_t | R_{t-1}, R_{t-2})
            if len(hist_seq) >= 2:
                prev2_res = hist_seq[-2]
                prev1_res = hist_seq[-1]
                count_3prev = sum(transition_3gram_counts[(prev2_res, prev1_res)].values())
                count_3trans = transition_3gram_counts[(prev2_res, prev1_res)][curr_res]
                prob_3gram = (count_3trans + 1.0) / (count_3prev + vocab_size)
            else:
                prob_3gram = prob_2gram

            ngram_score = math.log(prob_2gram) + math.log(prob_3gram)

            # Update history state
            hist_seq.append(curr_res)
            entity_last_event[entity_id] = (curr_time_sec, curr_lat, curr_lon)

            # Build feature row dictionary
            output_rows.append({
                "event_id": event_id,
                "timestamp": row['timestamp'],
                "entity_id": entity_id,
                "entity_type": row['entity_type'],
                "login_hour_deviation": round(login_hour_dev, 4),
                "geo_velocity_kmh": round(geo_velocity, 4),
                "time_since_last_activity_sec": round(delta_time_sec, 2),
                "resource_novelty_score": resource_novelty,
                "session_duration_dev": round(session_duration_dev, 2),
                "failed_auth_rate_5m": failed_auth_5m,
                "device_changed": device_changed,
                "auth_method_deviation": auth_method_dev,
                "prev_resource_accessed": prev_res,
                "resource_transition_freq": round(prob_2gram, 6),
                "distinct_resources_1h": distinct_res_1h,
                "rolling_activity_count_15m": rolling_act_15m,
                "resource_ngram_score": round(ngram_score, 6)
            })

        return pd.DataFrame(output_rows)

# -----------------------------------------------------------------------------
# Main Execution CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 2 Feature Engineering")
    parser.add_argument("--events", type=str, default="data/events.csv", help="Path to events CSV (default: data/events.csv)")
    parser.add_argument("--profiles", type=str, default="data/profiles.json", help="Path to profiles JSON (default: data/profiles.json)")
    parser.add_argument("--output", type=str, default="data/features.csv", help="Output path for feature vectors (default: data/features.csv)")

    args = parser.parse_args()

    print(f"[*] Loading profiles from {args.profiles}...")
    with open(args.profiles, "r") as f:
        profiles_list = json.load(f)
    profiles_dict = {p["entity_id"]: p for p in profiles_list}

    print(f"[*] Loading events from {args.events}...")
    events_df = pd.read_csv(args.events)

    print(f"[*] Extracting point-in-time and sequential features for {len(events_df)} events...")
    engine = FeatureEngineeringEngine(profiles_dict)
    features_df = engine.extract_features(events_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"[+] Successfully generated feature matrix with shape {features_df.shape} -> {args.output}")

if __name__ == "__main__":
    main()
