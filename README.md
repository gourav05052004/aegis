# Honeywell Hackathon Anomaly Detection System

A high-performance, enterprise-grade Security Operations Center (SOC) telemetry simulator, feature engineering pipeline, unsupervised baseline profiler, and hybrid attack classification engine.

---

## 1. Executive Summary & Overview

The **Honeywell Anomaly Detection Engine** provides end-to-end synthetic SOC telemetry generation, feature engineering, profiling, and real-time threat identification.

In modern enterprise infrastructure, security teams process millions of event logs daily. Identifying sophisticated multi-stage threats—such as credential stuffing, impossible travel, or low-and-slow exfiltration—requires high-fidelity statistical profiling of entities (users, service accounts, edge devices) and accurate detection of subtle deviations from normal baseline behavior.

This submission implements a complete 3-phase production solution:
- **Phase 1: Synthetic Telemetry Generator**, producing realistic 7-day multi-entity event streams with embedded ground truth labels for 7 distinct cyber attack vectors at a target 2.5% anomaly rate.
- **Phase 2: Feature Engineering & Baseline Profiler**, transforming raw event logs into 13+ behavioral feature vectors, fitting unsupervised Isolation Forests, and adapting entity baselines dynamically using Exponentially Weighted Moving Average (EWMA) concept drift adaptation.
- **Phase 3: Hybrid Detection & Attack Classification Engine**, combining unsupervised baseline anomaly scores with supervised multi-class LightGBM probabilities using an optimized weight fusion algorithm ($w_1 = 0.3, w_2 = 0.7$).

---

## 2. End-to-End System Architecture

```
+---------------------------------------------------------------------------------+
|                        Entity Baseline Profiles (data/profiles.json)            |
|         Users (75%)    |    Service Accounts (15%)    |    Edge Devices (10%)     |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|                      Synthetic Event Stream Engine (src/generator.py)           |
|         - 7-Day Timeline Simulation          - Gaussian Peak Hours              |
|         - Categorical Resource Mapping       - Seedable Reproducibility         |
+---------------------------------------------------------------------------------+
               |                                               |
               v                                               v
+-----------------------------+                 +-----------------------------+
|      data/events.csv        |                 |      data/labels.csv        |
|   (100,000 Event Stream)    |                 |    (Ground Truth Labels)    |
+-----------------------------+                 +-----------------------------+
               |
               v
+---------------------------------------------------------------------------------+
|                    Feature Engineering Engine (src/feature_engineering.py)      |
|         - Haversine Geo Velocity             - Rolling 5m/15m/1h Windows        |
|         - Point-in-Time Deviations           - N-Gram Sequence Log-Likelihood   |
+---------------------------------------------------------------------------------+
               |
               v
+---------------------------------------------------------------------------------+
|                         data/features.csv (100,000 x 17 Matrix)                 |
+---------------------------------------------------------------------------------+
               |                                               |
               v                                               v
+---------------------------------------------+   +-------------------------------+
|     Unsupervised Baseline Profiler          |   |  Supervised Multi-Class Model |
|      (src/baseline_profiler.py)             |   |    (src/train_detector.py)    |
|   - Isolation Forest Anomaly Scoring        |   | - Balanced LightGBM Classifier|
|   - EWMA Concept Drift Adaptation           |   | - 8-Class Attack Type Target  |
+---------------------------------------------+   +-------------------------------+
               \                                               /
                \                                             /
                 v                                           v
+---------------------------------------------------------------------------------+
|                         Optimal Hybrid Risk Weight Fusion Engine                |
|                    Hybrid Risk Score = 0.3 * Baseline + 0.7 * P(Attack)         |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|                       data/predictions.csv & data/eval_results.json             |
|              (SOC Actionable Risk Scores & Multi-Class Classifications)         |
+---------------------------------------------------------------------------------+
```

---

## 3. Phase 2 Feature Matrix & Security Relevance

The **Feature Engineering Engine** (`src/feature_engineering.py`) converts raw SOC event logs into high-dimensional behavioral feature vectors:

| Feature Name | Feature Type | Mathematical / Logic Description | Security Relevance |
|--------------|--------------|----------------------------------|--------------------|
| `login_hour_deviation` | Point-in-Time | $\min(|h_{evt} - h_{peak}|, 24 - |h_{evt} - h_{peak}|)$ | Detects off-hours login spikes and out-of-shift activity. |
| `geo_velocity_kmh` | Point-in-Time | $\frac{\text{Haversine}(Loc_t, Loc_{t-1})}{\Delta t_{hours}}$ | Identifies Impossible Travel anomalies across logins. |
| `time_since_last_activity_sec` | Point-in-Time | $t_t - t_{t-1}$ in seconds | Flags unexpected burst activity or long dormant reactivations. |
| `resource_novelty_score` | Point-in-Time | $\mathbb{I}(R_t \notin \text{Seen}(Entity))$ | Flags first-time resource access by a given user/device. |
| `session_duration_dev` | Point-in-Time | $| \text{Duration}_t - \overline{\text{Duration}}_{entity} |$ | Detects truncated brute-force sessions or suspicious long holds. |
| `failed_auth_rate_5m` | Rolling Window | Count of `failed` logins in $[t-300\text{s}, t]$ | Primary indicator for password spraying and brute force. |
| `device_changed` | Point-in-Time | $\mathbb{I}(\text{Device}_t \neq \text{Device}_{primary})$ | Highlights credential theft or unapproved device usage. |
| `auth_method_deviation` | Point-in-Time | $\mathbb{I}(\text{Auth}_t \neq \text{Auth}_{primary})$ | Detects auth downgrade attacks (e.g. bypassing MFA). |
| `prev_resource_accessed` | Sequential | Encoded prior resource $R_{t-1}$ | Captures baseline workflow state for transition analysis. |
| `resource_transition_freq` | Sequential | Empirical $P(R_t \mid R_{t-1})$ | Identifies illegal workflow transitions (e.g. Email $\rightarrow$ Infra Admin). |
| `distinct_resources_1h` | Rolling Window | Count of unique resources in $[t-3600\text{s}, t]$ | Detects internal reconnaissance and lateral movement. |
| `rolling_activity_count_15m` | Rolling Window | Total event count in $[t-900\text{s}, t]$ | Identifies automated scripts, scanners, or bot activity. |
| `resource_ngram_score` | Sequential | $\log P(R_t \mid R_{t-1}) + \log P(R_t \mid R_{t-1}, R_{t-2})$ | Evaluates sequence plausibility using 2-gram and 3-gram log-likelihood. |

---

## 4. Mathematical Formulations

### 1. Haversine Geographic Velocity ($\text{km/h}$)
Given two consecutive events for entity $e$ at coordinates $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$ with timestamps $t_1, t_2$:

$$d = 2R \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$

$$v_{\text{geo}} = \frac{d}{(t_2 - t_1) / 3600}$$

where $R = 6371\text{ km}$.

### 2. Resource Sequence N-Gram Log-Likelihood
For a sequence of resource accesses $(R_{t-2}, R_{t-1}, R_t)$, the $N$-gram sequence log-likelihood score with Laplace smoothing is:

$$P(R_t \mid R_{t-1}) = \frac{\text{Count}(R_{t-1}, R_t) + 1}{\text{Count}(R_{t-1}) + |V|}$$

$$\text{resource\_ngram\_score} = \ln P(R_t \mid R_{t-1}) + \ln P(R_t \mid R_{t-1}, R_{t-2})$$

where $|V|$ is the total resource vocabulary size.

### 3. EWMA Concept Drift Adaptation
$$\bar{S}_{entity, t} = \alpha \cdot S_{raw, t} + (1 - \alpha) \cdot \bar{S}_{entity, t-1}$$

### 4. Fused Hybrid Risk Score Equation
$$\text{Hybrid Risk Score} = w_1 \times \text{Baseline Anomaly Score} + w_2 \times P(\text{Attack})$$

---

## 5. Phase 3 Weight Optimization & Benchmark Results

### 1. Validation Set Weight Fusion Grid Sweep
We conducted a grid sweep over weight combinations $(w_1, w_2)$ on the **15% Validation Set (15,000 events)**, optimizing for **Top-1% Alert Budget Precision**:

| $w_1$ (Unsupervised Baseline) | $w_2$ (Supervised Model $P$) | Top-1% Alert Budget Precision | Selection Status |
|-------------------------------|------------------------------|-------------------------------|------------------|
| **0.3** | **0.7** | **100.00%** | **Selected Optimal** |
| 0.4 | 0.6 | 100.00% | Valid Candidate |
| 0.5 | 0.5 | 100.00% | Valid Candidate |
| 0.6 | 0.4 | 100.00% | Valid Candidate |

**Justification for $w_1=0.3, w_2=0.7$:**
The $0.3 / 0.7$ weight pairing gives primary emphasis to the supervised multi-class model probability while retaining a 30% baseline anchor. This ensures that zero-day novel structural deviations detected by the unsupervised baseline profiler elevate overall risk scores even if unclassified by the supervised model.

---

### 2. Held-Out Test Set Performance (15,000 Held-Out Events)

| Metric | Benchmark Score |
|--------|-----------------|
| **Overall Precision (Binary)** | **100.00%** |
| **Overall Recall (Binary)** | **99.73%** |
| **F1-Score (Binary)** | **99.87%** |
| **PR-AUC** | **1.0000** |
| **False Positive Rate (FPR)** | **0.0000%** |
| **Top-1% Alert Budget Precision** | **100.00%** |

---

### 3. Per-Attack Vector Performance Breakdown

| Attack Vector | Label Category | Precision | Recall | F1-Score | Support (N) |
|---------------|----------------|-----------|--------|----------|-------------|
| **Brute Force** | Attack #1 | 100.00% | 98.15% | 99.07% | 54 |
| **Credential Stuffing** | Attack #2 | 100.00% | 100.00% | 100.00% | 53 |
| **Device Spoofing** | Attack #3 | 100.00% | 100.00% | 100.00% | 53 |
| **Impossible Travel** | Attack #4 | 94.55% | 96.30% | 95.41% | 54 |
| **Insider Drift** | Attack #5 | 79.55% | 64.81% | 71.43% | 54 |
| **Lateral Movement** | Attack #6 | 100.00% | 100.00% | 100.00% | 54 |
| **Low-and-Slow** | Attack #7 | 67.74% | 79.25% | 73.04% | 53 |
| **Normal** | Baseline | 99.99% | 100.00% | 100.00% | 14,625 |

---

## 6. How to Run & Verify

### Prerequisites
- Python 3.9+
- Dependencies in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Complete End-to-End Pipeline Execution

#### Step 1: Generate Synthetic SOC Telemetry (Phase 1)
```bash
python src/generator.py --events 100000 --seed 42
```

#### Step 2: Extract Behavioral Feature Matrix (Phase 2)
```bash
python src/feature_engineering.py --events data/events.csv --profiles data/profiles.json --output data/features.csv
```

#### Step 3: Train Unsupervised Baseline Profiler (Phase 2)
```bash
python src/baseline_profiler.py --features data/features.csv --output data/baseline_scores.csv
```

#### Step 4: Train Detector & Optimize Weight Fusion (Phase 3)
```bash
python src/train_detector.py --features data/features.csv --baseline data/baseline_scores.csv --labels data/labels.csv
```

#### Step 5: Execute Single-Pass Batch Inference (Phase 3)
```bash
python src/predict.py --events data/events.csv --profiles data/profiles.json --output data/predictions.csv
```

---
*Honeywell Hackathon - Anomaly Detection Systems Engine (Phases 1, 2, & 3 Fully Completed)*
