"""
Honeywell Hackathon Anomaly Detection System - Phase 1: Synthetic Data Generator
---------------------------------------------------------------------------------
Generates realistic SOC telemetry with controlled attack injections.

Outputs:
  - data/profiles.json : Entity baseline profiles (Users, Service Accounts, Edge Devices)
  - data/events.csv    : Synthetic event telemetry stream sorted chronologically
  - data/labels.csv    : Aligned ground-truth labels for anomaly detection
"""

import os
import json
import csv
import math
import random
import argparse
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------------------------
# Constant Reference Distributions & Definitions
# -----------------------------------------------------------------------------

COUNTRIES_GEO = [
    {"country": "US", "lat": 37.7749, "lon": -122.4194},
    {"country": "US", "lat": 40.7128, "lon": -74.0060},
    {"country": "UK", "lat": 51.5074, "lon": -0.1278},
    {"country": "DE", "lat": 52.5200, "lon": 13.4050},
    {"country": "IN", "lat": 12.9716, "lon": 77.5946},
    {"country": "JP", "lat": 35.6762, "lon": 139.6503},
    {"country": "AU", "lat": -33.8688, "lon": 151.2093},
    {"country": "BR", "lat": -23.5505, "lon": -46.6333},
]

ATTACK_GEO_LOCATIONS = [
    {"country": "RU", "lat": 55.7558, "lon": 37.6173},
    {"country": "CN", "lat": 39.9042, "lon": 116.4074},
    {"country": "KP", "lat": 39.0392, "lon": 125.7625},
    {"country": "IR", "lat": 35.6892, "lon": 51.3890},
]

CATEGORIES = ["email", "git", "payroll", "infra"]

RESOURCE_CATALOG = {
    "email": ["email/inbox", "email/outbox", "email/sent", "email/settings"],
    "git": ["git/repo_honeywell_core", "git/repo_sec_analytics", "git/pull_request", "git/commit_push"],
    "payroll": ["payroll/salary_portal", "payroll/tax_forms", "payroll/direct_deposit", "payroll/admin"],
    "infra": ["infra/k8s_cluster", "infra/admin_console", "infra/domain_controller", "infra/secrets_vault"]
}

AUTH_METHODS = ["password", "mfa_app", "hardware_key", "api_token"]

ATTACK_TYPES = [
    "Brute Force",
    "Impossible Travel",
    "Credential Stuffing",
    "Lateral Movement",
    "Device Spoofing",
    "Low-and-Slow",
    "Insider Drift"
]

# -----------------------------------------------------------------------------
# Entity Profile Generation
# -----------------------------------------------------------------------------

def generate_entity_profiles(num_profiles=1200, seed=42):
    """
    Generates entity profiles covering:
      - Users (75%)
      - Service Accounts (15%)
      - Edge Devices (10%)
    """
    random.seed(seed)
    profiles = []

    num_users = int(num_profiles * 0.75)
    num_svc = int(num_profiles * 0.15)
    num_dev = num_profiles - num_users - num_svc

    # 1. Users
    for i in range(1, num_users + 1):
        entity_id = f"USR_{i:04d}"
        home_loc = random.choice(COUNTRIES_GEO)
        peak_hour = int(random.gauss(14, 2.0)) % 24  # Gaussian peak around 2 PM
        primary_cat = random.choice(["email", "git", "payroll", "infra"])
        primary_device = f"DEV_FP_{entity_id}_{random.randint(1000, 9999):X}"
        primary_auth = random.choice(["password", "mfa_app", "hardware_key"])
        avg_session = round(random.uniform(900, 3600), 2)  # 15m to 1h

        profiles.append({
            "entity_id": entity_id,
            "entity_type": "user",
            "home_location": home_loc,
            "peak_hour": peak_hour,
            "primary_category": primary_cat,
            "primary_device": primary_device,
            "primary_auth": primary_auth,
            "avg_session_duration": avg_session
        })

    # 2. Service Accounts
    for i in range(1, num_svc + 1):
        entity_id = f"SVC_{i:04d}"
        home_loc = random.choice(COUNTRIES_GEO)
        peak_hour = random.randint(0, 23)  # Uniform 24/7 activity
        primary_cat = random.choice(["infra", "git"])
        primary_device = f"DEV_FP_{entity_id}_{random.randint(1000, 9999):X}"
        primary_auth = random.choice(["api_token", "hardware_key"])
        avg_session = round(random.uniform(60, 600), 2)  # 1m to 10m

        profiles.append({
            "entity_id": entity_id,
            "entity_type": "service_account",
            "home_location": home_loc,
            "peak_hour": peak_hour,
            "primary_category": primary_cat,
            "primary_device": primary_device,
            "primary_auth": primary_auth,
            "avg_session_duration": avg_session
        })

    # 3. Edge Devices
    for i in range(1, num_dev + 1):
        entity_id = f"DEV_{i:04d}"
        home_loc = random.choice(COUNTRIES_GEO)
        peak_hour = int(random.gauss(10, 3.0)) % 24
        primary_cat = random.choice(["infra", "email"])
        primary_device = f"DEV_FP_{entity_id}_{random.randint(1000, 9999):X}"
        primary_auth = random.choice(["api_token", "password"])
        avg_session = round(random.uniform(1800, 7200), 2)  # 30m to 2h

        profiles.append({
            "entity_id": entity_id,
            "entity_type": "edge_device",
            "home_location": home_loc,
            "peak_hour": peak_hour,
            "primary_category": primary_cat,
            "primary_device": primary_device,
            "primary_auth": primary_auth,
            "avg_session_duration": avg_session
        })

    return profiles

# -----------------------------------------------------------------------------
# Event & Attack Generation
# -----------------------------------------------------------------------------

def generate_telemetry_dataset(profiles, total_events=10000, anomaly_rate=0.025, seed=42):
    """
    Generates synthetic event telemetry over a 7-day timeline.
    Injects 7 distinct attack scenarios matching the target anomaly rate.
    """
    random.seed(seed)

    start_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    duration_days = 7

    num_anomalies = int(total_events * anomaly_rate)
    num_normal = total_events - num_anomalies

    # Split anomalies evenly across 7 attack types
    attack_counts = {atype: num_anomalies // len(ATTACK_TYPES) for atype in ATTACK_TYPES}
    remainder = num_anomalies - sum(attack_counts.values())
    for i in range(remainder):
        attack_counts[ATTACK_TYPES[i]] += 1

    events = []
    labels = []

    # -------------------------------------------------------------------------
    # 1. Normal Events Generation
    # -------------------------------------------------------------------------
    for _ in range(num_normal):
        profile = random.choice(profiles)
        
        # Timestamp based on profile peak hour
        offset_days = random.uniform(0, duration_days)
        # Time of day influenced by peak_hour
        hour_sample = int(random.gauss(profile["peak_hour"], 2.5)) % 24
        minute_sample = random.randint(0, 59)
        second_sample = random.randint(0, 59)

        evt_time = start_time + timedelta(days=int(offset_days), hours=hour_sample, minutes=minute_sample, seconds=second_sample)

        # Normal resource access (85% primary category, 15% secondary category)
        if random.random() < 0.85:
            res_cat = profile["primary_category"]
        else:
            res_cat = random.choice(CATEGORIES)
        
        res_accessed = random.choice(RESOURCE_CATALOG[res_cat])

        # Normal auth method & device
        auth_method = profile["primary_auth"]
        device_fp = profile["primary_device"]
        geo_loc = profile["home_location"]

        # Session duration near baseline
        sess_duration = max(5.0, round(random.gauss(profile["avg_session_duration"], profile["avg_session_duration"] * 0.2), 2))
        status = "success" if random.random() < 0.96 else "failed"

        events.append({
            "timestamp": evt_time,
            "entity_id": profile["entity_id"],
            "entity_type": profile["entity_type"],
            "resource_category": res_cat,
            "resource_accessed": res_accessed,
            "auth_method": auth_method,
            "geo_location": json.dumps(geo_loc),
            "device_fingerprint": device_fp,
            "session_duration_sec": sess_duration,
            "status": status,
        })
        labels.append({
            "is_attack": 0,
            "attack_type": "Normal"
        })

    # -------------------------------------------------------------------------
    # 2. Attack Events Injections
    # -------------------------------------------------------------------------
    for attack_type, count in attack_counts.items():
        for _ in range(count):
            profile = random.choice(profiles)
            offset_days = random.uniform(0, duration_days)
            hour_sample = random.randint(0, 23)
            evt_time = start_time + timedelta(days=int(offset_days), hours=hour_sample, minutes=random.randint(0, 59), seconds=random.randint(0, 59))

            res_cat = profile["primary_category"]
            res_accessed = random.choice(RESOURCE_CATALOG[res_cat])
            auth_method = profile["primary_auth"]
            device_fp = profile["primary_device"]
            geo_loc = profile["home_location"]
            sess_duration = profile["avg_session_duration"]
            status = "success"

            if attack_type == "Brute Force":
                # High frequency failed attempt, low session duration, password auth
                status = "failed"
                auth_method = "password"
                sess_duration = round(random.uniform(1.0, 5.0), 2)
                res_cat = "infra"
                res_accessed = "infra/admin_console"

            elif attack_type == "Impossible Travel":
                # High geographic velocity relative to entity's home location
                geo_loc = random.choice(ATTACK_GEO_LOCATIONS)
                status = "success"

            elif attack_type == "Credential Stuffing":
                # Off-baseline device fingerprint, password auth failure
                device_fp = f"FP_UNKNOWN_{random.randint(100000, 999999):X}"
                auth_method = "password"
                status = "failed"

            elif attack_type == "Lateral Movement":
                # Non-infra entity accessing high-privilege infra resources
                non_infra_profiles = [p for p in profiles if p["primary_category"] != "infra"]
                if non_infra_profiles:
                    profile = random.choice(non_infra_profiles)
                res_cat = "infra"
                res_accessed = random.choice(["infra/k8s_cluster", "infra/domain_controller", "infra/secrets_vault"])
                status = "success"

            elif attack_type == "Device Spoofing":
                # Anomalous device fingerprint with successful auth
                device_fp = f"FP_SPOOFED_{random.randint(100000, 999999):X}"
                status = "success"

            elif attack_type == "Low-and-Slow":
                # Subtle off-hour access targeting sensitive resources
                off_hour = random.choice([1, 2, 3, 4])  # Deep night
                evt_time = start_time + timedelta(days=int(offset_days), hours=off_hour, minutes=random.randint(0, 59))
                res_cat = random.choice(["payroll", "infra"])
                res_accessed = random.choice(["payroll/admin", "infra/secrets_vault"])
                status = "success"

            elif attack_type == "Insider Drift":
                # Categorical drift away from entity's primary category
                other_cats = [c for c in CATEGORIES if c != profile["primary_category"]]
                res_cat = random.choice(other_cats)
                res_accessed = random.choice(RESOURCE_CATALOG[res_cat])
                status = "success"

            events.append({
                "timestamp": evt_time,
                "entity_id": profile["entity_id"],
                "entity_type": profile["entity_type"],
                "resource_category": res_cat,
                "resource_accessed": res_accessed,
                "auth_method": auth_method,
                "geo_location": json.dumps(geo_loc),
                "device_fingerprint": device_fp,
                "session_duration_sec": sess_duration,
                "status": status,
            })
            labels.append({
                "is_attack": 1,
                "attack_type": attack_type
            })

    # -------------------------------------------------------------------------
    # 3. Chronological Sorting & ID Assignment
    # -------------------------------------------------------------------------
    combined = list(zip(events, labels))
    combined.sort(key=lambda x: x[0]["timestamp"])

    final_events = []
    final_labels = []

    for idx, (evt, lbl) in enumerate(combined, start=1):
        event_id = f"EVT_{idx:08d}"
        evt["event_id"] = event_id
        evt["timestamp"] = evt["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")

        # Format reordering for events.csv
        event_row = {
            "event_id": event_id,
            "timestamp": evt["timestamp"],
            "entity_id": evt["entity_id"],
            "entity_type": evt["entity_type"],
            "resource_category": evt["resource_category"],
            "resource_accessed": evt["resource_accessed"],
            "auth_method": evt["auth_method"],
            "geo_location": evt["geo_location"],
            "device_fingerprint": evt["device_fingerprint"],
            "session_duration_sec": evt["session_duration_sec"],
            "status": evt["status"]
        }
        label_row = {
            "event_id": event_id,
            "is_attack": lbl["is_attack"],
            "attack_type": lbl["attack_type"]
        }

        final_events.append(event_row)
        final_labels.append(label_row)

    return final_events, final_labels

# -----------------------------------------------------------------------------
# Main CLI Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 1 Telemetry Generator")
    parser.add_argument("--events", type=int, default=10000, help="Total events to generate (default: 10000)")
    parser.add_argument("--profiles", type=int, default=1200, help="Number of entity profiles to generate (default: 1200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for generated dataset (default: data)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Generating {args.profiles} entity profiles (Seed: {args.seed})...")
    profiles = generate_entity_profiles(num_profiles=args.profiles, seed=args.seed)
    
    profiles_path = os.path.join(args.output_dir, "profiles.json")
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"[+] Saved profiles to {profiles_path}")

    print(f"[*] Generating {args.events} synthetic events with ~2.5% anomaly injection...")
    events, labels = generate_telemetry_dataset(profiles, total_events=args.events, anomaly_rate=0.025, seed=args.seed)

    events_path = os.path.join(args.output_dir, "events.csv")
    events_fields = [
        "event_id", "timestamp", "entity_id", "entity_type", "resource_category",
        "resource_accessed", "auth_method", "geo_location", "device_fingerprint",
        "session_duration_sec", "status"
    ]
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=events_fields)
        writer.writeheader()
        writer.writerows(events)
    print(f"[+] Saved events to {events_path}")

    labels_path = os.path.join(args.output_dir, "labels.csv")
    labels_fields = ["event_id", "is_attack", "attack_type"]
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=labels_fields)
        writer.writeheader()
        writer.writerows(labels)
    print(f"[+] Saved labels to {labels_path}")

    # Print summary statistics
    num_attacks = sum(1 for l in labels if l["is_attack"] == 1)
    print("\n" + "=" * 50)
    print(f" DATA GENERATION SUMMARY ({args.events} events)")
    print("=" * 50)
    print(f" Total Profiles  : {len(profiles)}")
    print(f" Total Events    : {len(events)}")
    print(f" Total Anomalies : {num_attacks} ({num_attacks/len(events)*100:.2f}%)")
    
    attack_counts = {}
    for l in labels:
        atype = l["attack_type"]
        attack_counts[atype] = attack_counts.get(atype, 0) + 1
    
    print("\n Breakdown by Class:")
    for atype, cnt in sorted(attack_counts.items()):
        print(f"   - {atype:<20}: {cnt:>6} ({cnt/len(events)*100:.3f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
