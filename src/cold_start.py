"""
Honeywell Hackathon Anomaly Detection System - Phase 5: Cold Start Handling Engine
-----------------------------------------------------------------------------------
Implements peer-group baseline fallback hierarchies (entity_type : resource_category)
and threshold-based smooth transition (N_events = 5) for brand-new entities without
prior historical logs.

Outputs:
  - data/cold_start_demo.json : Simulated onboarding event streams for 3 new entities
"""

import os
import json
import math
import random
import argparse
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------------------------
# Peer-Group Baseline Hierarchy Definitions
# -----------------------------------------------------------------------------

PEER_GROUP_BASELINES = {
    "user:email": {
        "expected_peak_hour": 14,
        "allowed_auth_methods": ["password", "mfa_app", "hardware_key"],
        "expected_avg_session_sec": 1800.0,
        "peer_anomaly_baseline": 0.12
    },
    "user:git": {
        "expected_peak_hour": 14,
        "allowed_auth_methods": ["password", "mfa_app", "hardware_key"],
        "expected_avg_session_sec": 2400.0,
        "peer_anomaly_baseline": 0.14
    },
    "user:payroll": {
        "expected_peak_hour": 14,
        "allowed_auth_methods": ["mfa_app", "hardware_key"],
        "expected_avg_session_sec": 1800.0,
        "peer_anomaly_baseline": 0.15
    },
    "user:infra": {
        "expected_peak_hour": 14,
        "allowed_auth_methods": ["mfa_app", "hardware_key"],
        "expected_avg_session_sec": 1800.0,
        "peer_anomaly_baseline": 0.18
    },
    "service_account:infra": {
        "expected_peak_hour": 12,
        "allowed_auth_methods": ["api_token", "hardware_key"],
        "expected_avg_session_sec": 300.0,
        "peer_anomaly_baseline": 0.08
    },
    "service_account:git": {
        "expected_peak_hour": 12,
        "allowed_auth_methods": ["api_token", "hardware_key"],
        "expected_avg_session_sec": 300.0,
        "peer_anomaly_baseline": 0.08
    },
    "edge_device:infra": {
        "expected_peak_hour": 10,
        "allowed_auth_methods": ["api_token", "password"],
        "expected_avg_session_sec": 3600.0,
        "peer_anomaly_baseline": 0.10
    },
    "edge_device:email": {
        "expected_peak_hour": 10,
        "allowed_auth_methods": ["api_token", "password"],
        "expected_avg_session_sec": 3600.0,
        "peer_anomaly_baseline": 0.10
    }
}

DEFAULT_PEER_BASELINE = {
    "expected_peak_hour": 14,
    "allowed_auth_methods": ["password", "mfa_app", "api_token"],
    "expected_avg_session_sec": 1800.0,
    "peer_anomaly_baseline": 0.15
}

# -----------------------------------------------------------------------------
# Cold-Start Scoring Engine
# -----------------------------------------------------------------------------

class ColdStartEngine:
    def __init__(self, cold_threshold=5):
        self.cold_threshold = cold_threshold
        self.entity_history_counts = {}
        self.entity_learned_baselines = {}

    def get_peer_group_key(self, entity_type, resource_category):
        return f"{entity_type}:{resource_category}"

    def evaluate_event(self, event_data):
        entity_id = event_data["entity_id"]
        entity_type = event_data["entity_type"]
        res_cat = event_data["resource_category"]

        # Track event sequence number for entity
        n_events = self.entity_history_counts.get(entity_id, 0) + 1
        self.entity_history_counts[entity_id] = n_events

        peer_key = self.get_peer_group_key(entity_type, res_cat)
        peer_info = PEER_GROUP_BASELINES.get(peer_key, DEFAULT_PEER_BASELINE)

        is_cold_start = (n_events < self.cold_threshold)

        # 1. Peer-Group Baseline Score Calculation
        # Compute deviation relative to peer-group expectations
        event_hour = int(event_data.get("hour", 14))
        peak_dev = min(abs(event_hour - peer_info["expected_peak_hour"]) % 24,
                       24 - abs(event_hour - peer_info["expected_peak_hour"]) % 24)

        auth_method = event_data.get("auth_method", "password")
        auth_dev = 0.0 if auth_method in peer_info["allowed_auth_methods"] else 0.4

        geo_vel = float(event_data.get("geo_velocity_kmh", 0.0))
        geo_dev = min(1.0, geo_vel / 500.0) if geo_vel > 50.0 else 0.0

        failed_auth = int(event_data.get("failed_auth_rate_5m", 0))
        fail_dev = min(1.0, failed_auth * 0.3)

        peer_score = min(1.0, peer_info["peer_anomaly_baseline"] +
                              (peak_dev / 24.0) * 0.2 +
                              auth_dev + geo_dev + fail_dev)

        # 2. Update / Derive Learned Personal Profile Baseline
        if entity_id not in self.entity_learned_baselines:
            self.entity_learned_baselines[entity_id] = peer_score
        else:
            # EWMA update of personal learned baseline
            prev_learned = self.entity_learned_baselines[entity_id]
            self.entity_learned_baselines[entity_id] = 0.3 * peer_score + 0.7 * prev_learned

        personal_score = self.entity_learned_baselines[entity_id]

        # 3. Smooth Transition Weighting
        if is_cold_start:
            final_risk_score = peer_score
            transition_stage = f"Cold-Start (Event {n_events}/{self.cold_threshold-1}): Peer-Group Baseline Applied"
        else:
            # Transition weight beta: smoothly ramps from 0.2 at event 5 to 1.0 at event 9+
            beta = min(1.0, (n_events - 4) / 5.0)
            final_risk_score = (1.0 - beta) * peer_score + beta * personal_score
            transition_stage = f"Active Profile (Event {n_events}): Transition Weight beta={beta:.2f}"

        return {
            "event_id": event_data["event_id"],
            "entity_id": entity_id,
            "entity_type": entity_type,
            "resource_category": res_cat,
            "event_number": n_events,
            "is_cold_start": is_cold_start,
            "assigned_peer_group": peer_key,
            "peer_group_baseline_score": round(peer_score, 4),
            "individual_learned_score": round(personal_score, 4),
            "final_hybrid_risk_score": round(final_risk_score, 4),
            "transition_status": transition_stage
        }

# -----------------------------------------------------------------------------
# Simulation Runner for 3 New Entities
# -----------------------------------------------------------------------------

def run_cold_start_simulation(output_path="data/cold_start_demo.json"):
    print("[*] Initializing Cold-Start Handling Engine...")
    engine = ColdStartEngine(cold_threshold=5)

    simulated_entities = [
        {
            "entity_id": "USR_NEW_999",
            "entity_type": "user",
            "primary_category": "git",
            "description": "Newly onboarded Software Engineer"
        },
        {
            "entity_id": "SVC_NEW_888",
            "entity_type": "service_account",
            "primary_category": "infra",
            "description": "Newly provisioned CI/CD Service Account"
        },
        {
            "entity_id": "DEV_NEW_777",
            "entity_type": "edge_device",
            "primary_category": "infra",
            "description": "Newly deployed Edge Temperature Sensor"
        }
    ]

    simulation_results = []

    print("[*] Simulating 10 onboarding events per new entity...")
    event_counter = 1

    for entity_info in simulated_entities:
        e_id = entity_info["entity_id"]
        e_type = entity_info["entity_type"]
        p_cat = entity_info["primary_category"]

        start_time = datetime(2026, 7, 10, 9, 0, 0, tzinfo=timezone.utc)

        for seq in range(1, 11):
            evt_id = f"EVT_SIM_{event_counter:05d}"
            event_counter += 1

            # Event 8 is an injected anomaly
            if seq == 8:
                if e_type == "user":
                    # Injected Impossible Travel + Password Spray
                    evt = {
                        "event_id": evt_id,
                        "entity_id": e_id,
                        "entity_type": e_type,
                        "resource_category": p_cat,
                        "hour": 3,  # Deep night
                        "auth_method": "password",
                        "geo_velocity_kmh": 4500.0,
                        "failed_auth_rate_5m": 3,
                        "is_anomaly_injected": True,
                        "attack_type": "Impossible Travel"
                    }
                elif e_type == "service_account":
                    # Injected Lateral Movement to Payroll Admin
                    evt = {
                        "event_id": evt_id,
                        "entity_id": e_id,
                        "entity_type": e_type,
                        "resource_category": "payroll",
                        "hour": 2,
                        "auth_method": "password",
                        "geo_velocity_kmh": 0.0,
                        "failed_auth_rate_5m": 0,
                        "is_anomaly_injected": True,
                        "attack_type": "Lateral Movement"
                    }
                else:
                    # Device Spoofing
                    evt = {
                        "event_id": evt_id,
                        "entity_id": e_id,
                        "entity_type": e_type,
                        "resource_category": p_cat,
                        "hour": 10,
                        "auth_method": "password",
                        "geo_velocity_kmh": 1200.0,
                        "failed_auth_rate_5m": 2,
                        "is_anomaly_injected": True,
                        "attack_type": "Device Spoofing"
                    }
            else:
                # Normal onboarding event matching peer expectations
                evt = {
                    "event_id": evt_id,
                    "entity_id": e_id,
                    "entity_type": e_type,
                    "resource_category": p_cat,
                    "hour": 14 if e_type == "user" else 10,
                    "auth_method": "mfa_app" if e_type == "user" else "api_token",
                    "geo_velocity_kmh": 0.0,
                    "failed_auth_rate_5m": 0,
                    "is_anomaly_injected": False,
                    "attack_type": "Normal"
                }

            res = engine.evaluate_event(evt)
            res["injected_attack_type"] = evt["attack_type"]
            simulation_results.append(res)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(simulation_results, f, indent=2)

    print(f"[+] Saved cold-start simulation payload ({len(simulation_results)} events) -> {output_path}")

    # Display console summary for event transitions
    print("\n" + "=" * 80)
    print(" COLD-START SIMULATION TRANSITION DEMO")
    print("=" * 80)
    print(f" {'Entity ID':<13} | {'Evt #':<5} | {'ColdStart?':<10} | {'Peer Group':<22} | {'Risk Score':<10}")
    print("-" * 80)
    for r in simulation_results[:10]:  # Show USR_NEW_999 events 1..10
        cs_flag = "TRUE" if r["is_cold_start"] else "FALSE"
        print(f" {r['entity_id']:<13} | {r['event_number']:<5} | {cs_flag:<10} | {r['assigned_peer_group']:<22} | {r['final_hybrid_risk_score']:<10.4f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Honeywell Anomaly Detection - Phase 5 Cold Start Engine")
    parser.add_argument("--output", type=str, default="data/cold_start_demo.json", help="Output JSON path")

    args = parser.parse_args()

    run_cold_start_simulation(args.output)

if __name__ == "__main__":
    main()
