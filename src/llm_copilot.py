"""
LLM Context Formatting & SOC Copilot Engine for Honeywell Anomaly Detection.

This module extracts high-risk security alerts (hybrid_risk_score >= 0.70) from prediction
and explanation pipelines, formats them into structured LLM contexts, and queries Groq API
(llama-3.3-70b-versatile) to generate Tier-3 SOC Incident Briefs.
"""

import json
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def _generate_ip_from_entity(entity_id: str) -> str:
    """Generate a consistent mock IP address based on entity ID hash."""
    hash_val = abs(hash(str(entity_id)))
    octet3 = (hash_val % 250) + 1
    octet4 = ((hash_val // 250) % 250) + 1
    return f"192.168.{octet3}.{octet4}"


def extract_high_risk_context(
    predictions_path: str = "data/predictions.csv",
    explanations_path: str = "data/explanations.json",
    events_path: str = "data/events.csv",
    top_n: int = 5,
    event_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract high-risk alert context (hybrid_risk_score >= 0.70) formatted for LLM ingestion.

    Args:
        predictions_path: Path to predictions.csv
        explanations_path: Path to explanations.json
        events_path: Path to events.csv (for entity & resource details)
        top_n: Maximum number of top SHAP features to include
        event_id: Optional specific event_id to extract. If None, picks highest risk score event.

    Returns:
        Structured dict containing event_id, entity_info, detection_metrics, and top_shap_features.
    """
    # 1. Load predictions
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found at '{predictions_path}'")

    preds_df = pd.read_csv(predictions_path)

    # Ensure required columns exist
    risk_col = "hybrid_risk_score" if "hybrid_risk_score" in preds_df.columns else "hybrid_risk"
    if risk_col not in preds_df.columns:
        raise KeyError(f"Risk score column '{risk_col}' not found in predictions CSV.")

    # Filter high risk (>= 0.70)
    high_risk_df = preds_df[preds_df[risk_col] >= 0.70]
    if high_risk_df.empty:
        # Fallback to top risk score row if no events >= 0.70
        selected_row = preds_df.sort_values(by=risk_col, ascending=False).iloc[0]
    elif event_id is not None:
        matched = high_risk_df[high_risk_df["event_id"] == event_id]
        if not matched.empty:
            selected_row = matched.iloc[0]
        else:
            selected_row = high_risk_df.sort_values(by=risk_col, ascending=False).iloc[0]
    else:
        # Default to highest risk score event among high risk alerts
        selected_row = high_risk_df.sort_values(by=risk_col, ascending=False).iloc[0]

    target_event_id = str(selected_row["event_id"])
    target_entity_id = str(selected_row["entity_id"])
    hybrid_score = float(selected_row[risk_col])

    predicted_attack = str(
        selected_row.get("predicted_attack_type", selected_row.get("predicted_attack", "Unknown Anomaly"))
    )
    baseline_score = float(
        selected_row.get("baseline_score", selected_row.get("baseline_anomaly_score", 0.0))
    )

    # 2. Enrich with events.csv details if available
    entity_type = "user" if target_entity_id.startswith("USR") else (
        "service_account" if target_entity_id.startswith("SVC") else "edge_device"
    )
    resource_accessed = "sensitive_system"
    auth_method = "password"
    session_duration = 1800.0
    source_ip = _generate_ip_from_entity(target_entity_id)

    if os.path.exists(events_path):
        try:
            events_df = pd.read_csv(events_path)
            evt_row = events_df[events_df["event_id"] == target_event_id]
            if not evt_row.empty:
                r = evt_row.iloc[0]
                entity_type = str(r.get("entity_type", entity_type))
                resource_accessed = str(r.get("resource_accessed", resource_accessed))
                auth_method = str(r.get("auth_method", auth_method))
                session_duration = float(r.get("session_duration_sec", session_duration))
        except Exception as e:
            print(f"[Warning] Could not parse events details from '{events_path}': {e}")

    # 3. Extract SHAP explanations
    top_shap_features: Dict[str, Any] = {}
    if os.path.exists(explanations_path):
        try:
            with open(explanations_path, "r", encoding="utf-8") as f:
                explanations_data = json.load(f)

            shap_list = []
            if isinstance(explanations_data, list):
                for item in explanations_data:
                    if item.get("event_id") == target_event_id:
                        shap_list = item.get("top_shap_features", [])
                        break
            elif isinstance(explanations_data, dict):
                item = explanations_data.get(target_event_id, {})
                shap_list = item.get("top_shap_features", [])

            for item in shap_list[:top_n]:
                feat_name = item.get("feature", "unknown_feature")
                # Prefer shap_value attribution score or feature_value
                val = item.get("shap_value", item.get("feature_value", 0.0))
                top_shap_features[feat_name] = round(float(val), 4)
        except Exception as e:
            print(f"[Warning] Could not parse explanations from '{explanations_path}': {e}")

    if not top_shap_features:
        # Fallback default features if none found
        top_shap_features = {
            "baseline_anomaly_score": round(baseline_score, 4),
            "failed_auth_rate_5m": 1.0,
            "device_changed": 1.0
        }

    # 4. Construct payload
    context = {
        "event_id": target_event_id,
        "entity_info": {
            "entity_id": target_entity_id,
            "entity_type": entity_type,
            "source_ip": source_ip,
            "resource_accessed": resource_accessed,
            "auth_method": auth_method,
            "session_duration": session_duration
        },
        "detection_metrics": {
            "hybrid_risk_score": round(hybrid_score, 4),
            "predicted_attack": predicted_attack,
            "baseline_score": round(baseline_score, 4)
        },
        "top_shap_features": top_shap_features
    }

    return context


def generate_soc_incident_brief(
    event_context: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a Tier-3 SOC Incident Brief using Groq API (llama-3.3-70b-versatile).

    Args:
        event_context: Structured dictionary extracted via extract_high_risk_context()
        api_key: Groq API Key. If None, retrieves from GROQ_API_KEY environment variable.

    Returns:
        Structured dictionary containing executive_summary, mitre_attack_mapping,
        recommended_playbook, and containment_cli.
    """
    effective_api_key = api_key or os.environ.get("GROQ_API_KEY")

    if effective_api_key:
        try:
            from groq import Groq

            client = Groq(api_key=effective_api_key)

            system_prompt = (
                "You are an enterprise SOC Tier-3 Lead security analyst. "
                "Analyze the provided anomaly detection incident payload and generate actionable threat intelligence "
                "in strict JSON format."
            )

            user_prompt = f"""Given the following high-risk security event context:
{json.dumps(event_context, indent=2)}

Generate a strict JSON object adhering strictly to this structure:
{{
  "executive_summary": "Concise 2-sentence plain-English summary for CISO/management summarizing threat and risk.",
  "mitre_attack_mapping": ["Array of MITRE ATT&CK technique IDs and names (e.g., 'T1078 - Valid Accounts', 'T1021 - Remote Services')"],
  "recommended_playbook": ["Array of 3 step-by-step containment instructions for Tier-1 analysts"],
  "containment_cli": "A ready-to-run 1-line PowerShell/Bash command to isolate the entity or revoke tokens"
}}"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            response_content = response.choices[0].message.content
            brief = json.loads(response_content)
            return brief

        except Exception as e:
            print(f"[LLM Copilot Warning] Groq API execution failed: {e}. Falling back to structured mock response.")

    else:
        print("[LLM Copilot Warning] GROQ_API_KEY not set. Returning structured mock SOC Incident Brief.")

    # Structured Mock Fallback
    entity_info = event_context.get("entity_info", {})
    metrics = event_context.get("detection_metrics", {})
    entity_id = entity_info.get("entity_id", "UNKNOWN_ENTITY")
    entity_type = entity_info.get("entity_type", "user")
    attack = metrics.get("predicted_attack", "Suspicious Anomaly")
    risk_score = metrics.get("hybrid_risk_score", 0.85)

    if entity_type == "user":
        containment_cmd = f"Disable-LocalUser -Name '{entity_id}'"
    elif entity_type == "service_account":
        containment_cmd = f"Revoke-AzureADUserAllRefreshToken -ObjectId '{entity_id}'"
    else:
        containment_cmd = f"netsh advfirewall firewall add rule name='Block_{entity_id}' dir=in action=block"

    return {
        "executive_summary": (
            f"High-risk anomaly detected on {entity_type} '{entity_id}' with a hybrid risk score of {risk_score:.2f}. "
            f"The behavior indicates potential {attack} targeting enterprise infrastructure resource '{entity_info.get('resource_accessed', 'unknown')}'."
        ),
        "mitre_attack_mapping": [
            "T1078 - Valid Accounts",
            "T1021 - Remote Services",
            "T1110 - Brute Force"
        ],
        "recommended_playbook": [
            f"1. Immediately suspend active sessions and disable access credentials for {entity_id}.",
            f"2. Inspect network logs and authentication telemetry surrounding source IP {entity_info.get('source_ip', 'N/A')} and resource '{entity_info.get('resource_accessed', 'N/A')}'.",
            "3. Confirm isolation of affected endpoints and escalate findings to SOC Tier-3 Lead."
        ],
        "containment_cli": containment_cmd
    }


def query_soc_telemetry_rag(
    user_query: str,
    df_events: Optional[pd.DataFrame] = None,
    df_preds: Optional[pd.DataFrame] = None,
    profiles_dict: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> str:
    """
    RAG over SOC Telemetry: Answers natural language questions about telemetry logs,
    entity profiles, threat alerts, and cold-start entities using contextual retrieval
    combined with LLM inference (or intelligent analytical response generation).
    """
    if df_events is None or df_events.empty:
        if os.path.exists("data/events.csv"):
            df_events = pd.read_csv("data/events.csv")
        else:
            df_events = pd.DataFrame()

    if df_preds is None or df_preds.empty:
        if os.path.exists("data/predictions.csv"):
            df_preds = pd.read_csv("data/predictions.csv")
        else:
            df_preds = pd.DataFrame()

    if profiles_dict is None:
        if os.path.exists("data/profiles.json"):
            with open("data/profiles.json", "r") as f:
                profs = json.load(f)
                profiles_dict = {p["entity_id"]: p for p in profs}
        else:
            profiles_dict = {}

    query_lower = user_query.lower()
    retrieved_facts = []

    import re
    target_ids = re.findall(r'(EVT_\d+|USR_\d+|SVC_\d+|DEV_\d+|evt_\d+|usr_\d+|svc_\d+|dev_\d+)', user_query, re.IGNORECASE)
    target_ids = [t.upper() for t in target_ids]

    if target_ids:
        for tid in target_ids:
            if tid.startswith("EVT_"):
                if not df_preds.empty and "event_id" in df_preds.columns:
                    evt_preds = df_preds[df_preds["event_id"].astype(str).str.upper() == tid]
                    if not evt_preds.empty:
                        r = evt_preds.iloc[0]
                        risk_score = float(r.get("hybrid_risk_score", 0.0))
                        attack_type = str(r.get("predicted_attack_type", "Normal"))
                        baseline_score = float(r.get("baseline_score", 0.0))
                        model_prob = float(r.get("model_probability", 0.0))
                        entity_id = str(r.get("entity_id", "Unknown"))

                        severity = "CRITICAL" if risk_score >= 0.85 else ("HIGH" if risk_score >= 0.70 else ("MEDIUM" if risk_score >= 0.50 else "NORMAL"))

                        retrieved_facts.append(
                            f"Exact Event Record for {tid}: Associated Entity={entity_id}, Timestamp={r.get('timestamp')}, "
                            f"Hybrid Risk Score={risk_score:.4f}, Severity Level={severity}, "
                            f"Risk Type / Attack Vector='{attack_type}', "
                            f"Unsupervised Baseline Score={baseline_score:.4f}, Supervised Model Probability={model_prob:.4f}."
                        )
                    else:
                        retrieved_facts.append(f"Event Record for {tid}: Event ID not found in prediction database.")
            else:
                target_ent = tid
                prof = profiles_dict.get(target_ent, {})
                if prof:
                    retrieved_facts.append(
                        f"Profile Baseline for {target_ent}: Type={prof.get('entity_type')}, "
                        f"Peak Hour={prof.get('peak_hour')}:00 UTC, Primary Resource={prof.get('primary_category')}, "
                        f"Primary Device={prof.get('primary_device')}, Primary Auth={prof.get('primary_auth')}, "
                        f"Avg Session Duration={prof.get('avg_session_duration')}s."
                    )

                if not df_events.empty and "entity_id" in df_events.columns:
                    ent_evts = df_events[df_events["entity_id"].astype(str).str.upper() == target_ent]
                    if not ent_evts.empty:
                        res_categories = ["payroll", "email", "git", "infra"]
                        mentioned_res = [c for c in res_categories if c in query_lower]

                        if mentioned_res:
                            for m_res in mentioned_res:
                                matched_evts = ent_evts[ent_evts["resource_category"].astype(str).str.lower() == m_res]
                                if not matched_evts.empty:
                                    sample_recs = matched_evts[['event_id', 'timestamp', 'resource_accessed', 'status', 'auth_method']].head(5).to_dict(orient='records')
                                    retrieved_facts.append(
                                        f"Telemetry Evidence: Entity {target_ent} HAS accessed resource category '{m_res}' "
                                        f"{len(matched_evts)} time(s). Recent Access Events: {sample_recs}"
                                    )
                                else:
                                    retrieved_facts.append(
                                        f"Telemetry Evidence: Entity {target_ent} HAS NOT accessed resource category '{m_res}' in the telemetry logs."
                                    )
                        else:
                            retrieved_facts.append(
                                f"Telemetry History for {target_ent}: Total Events={len(ent_evts)}, "
                                f"Accessed Resources={ent_evts['resource_category'].value_counts().to_dict()}, "
                                f"Statuses={ent_evts['status'].value_counts().to_dict()}."
                            )
                    else:
                        retrieved_facts.append(f"Telemetry History for {target_ent}: No historical event logs found in database.")

                if not df_preds.empty and "entity_id" in df_preds.columns:
                    ent_preds = df_preds[df_preds["entity_id"].astype(str).str.upper() == target_ent]
                    high_risk = ent_preds[ent_preds["hybrid_risk_score"] >= 0.70] if "hybrid_risk_score" in ent_preds.columns else pd.DataFrame()
                    all_scores = ent_preds["hybrid_risk_score"].tolist() if "hybrid_risk_score" in ent_preds.columns else []
                    max_score = max(all_scores) if all_scores else 0.0
                    avg_score = float(np.mean(all_scores)) if all_scores else 0.0
                    top_attack = ent_preds["predicted_attack_type"].mode()[0] if "predicted_attack_type" in ent_preds.columns and not ent_preds.empty else "Normal"

                    severity_level = "CRITICAL" if max_score >= 0.85 else ("HIGH" if max_score >= 0.70 else ("MEDIUM" if max_score >= 0.50 else "NORMAL"))

                    retrieved_facts.append(
                        f"Risk Metrics for {target_ent}: Max Risk Score={max_score:.4f}, Avg Risk Score={avg_score:.4f}, "
                        f"Overall Severity Level={severity_level}, Dominant Risk Type/Attack Vector='{top_attack}', "
                        f"Total High Risk Alerts (Score >= 0.70)={len(high_risk)}. Flagged Attack Classes={ent_preds['predicted_attack_type'].value_counts().to_dict() if 'predicted_attack_type' in ent_preds.columns else {}}."
                    )

    if "cold" in query_lower or "start" in query_lower or "onboard" in query_lower:
        if os.path.exists("data/cold_start_demo.json"):
            with open("data/cold_start_demo.json", "r") as f:
                cs_list = json.load(f)
            df_cs = pd.DataFrame(cs_list)
            cs_alerts = df_cs[df_cs["final_hybrid_risk_score"] >= 0.70] if "final_hybrid_risk_score" in df_cs.columns else pd.DataFrame()
            cs_entities = df_cs["entity_id"].unique() if "entity_id" in df_cs.columns else []
            retrieved_facts.append(
                f"Cold-Start Onboarding Telemetry: Monitored Entities={list(cs_entities)}. "
                f"Total Cold-Start Events={len(df_cs)}, High-Risk Cold-Start Alerts={len(cs_alerts)}. "
                f"Target Peer Groups: {df_cs['assigned_peer_group'].value_counts().to_dict() if 'assigned_peer_group' in df_cs.columns else 'N/A'}."
            )

    if "alert" in query_lower or "threat" in query_lower or "high risk" in query_lower or "attack" in query_lower or "summary" in query_lower or "today" in query_lower or "service account" in query_lower or "user" in query_lower:
        if not df_preds.empty:
            high_risk_df = df_preds[df_preds["hybrid_risk_score"] >= 0.70] if "hybrid_risk_score" in df_preds.columns else pd.DataFrame()
            retrieved_facts.append(
                f"Overall Threat Telemetry Summary: Total Monitored Events={len(df_preds)}, Active High-Risk Alerts={len(high_risk_df)}. "
                f"Top Attack Vectors: {high_risk_df['predicted_attack_type'].value_counts().to_dict() if not high_risk_df.empty else 'None'}."
            )

    if not retrieved_facts:
        retrieved_facts.append(
            f"Telemetry Database Overview: {len(df_events)} events across {len(profiles_dict)} profiles. "
            f"Active High-Risk Threat Alerts: {len(df_preds[df_preds['hybrid_risk_score'] >= 0.70]) if not df_preds.empty and 'hybrid_risk_score' in df_preds.columns else 0}."
        )

    context_str = "\n".join(retrieved_facts)

    effective_api_key = api_key or os.environ.get("GROQ_API_KEY")
    if effective_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=effective_api_key)
            system_prompt = (
                "You are 'Ask My SOC', an expert Security Operations Center AI assistant. "
                "Answer the analyst's question accurately using only the provided SOC telemetry facts. "
                "Be concise, clear, and professional. Highlight key findings with markdown formatting."
            )
            user_prompt = f"""Analyst Question: "{user_query}"

Retrieved Telemetry Facts & Database Evidence:
{context_str}

Provide a direct, authoritative, plain-English response to the analyst's question based on the telemetry evidence:"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[RAG Warning] Groq API execution failed: {e}. Falling back to structured response.")

    response_lines = [f"### 🔍 Telemetry RAG Query Result"]
    response_lines.append(f"**Question:** *\"{user_query}\"*\n")
    response_lines.append("#### 📋 Retrieved Evidence & Telemetry Logs:")

    for fact in retrieved_facts:
        response_lines.append(f"- {fact}")

    response_lines.append("\n**SOC Analyst Verdict:** Telemetry database queried successfully based on exact historical records.")
    return "\n".join(response_lines)


if __name__ == "__main__":
    print("=== Extracting High-Risk Incident Context ===")
    payload = extract_high_risk_context()
    print(json.dumps(payload, indent=2))

    print("\n=== Generating SOC Incident Brief ===")
    soc_brief = generate_soc_incident_brief(payload)
    print(json.dumps(soc_brief, indent=2))

