# A-Delay-Optimized-ATAS-Based-Vertical-Handover-Framework-
A Delay-Optimized ATAS-Based Vertical Handover Framework for Performance Enhancement in 6G Non-Terrestrial Networks
# DO-ATAS: Delay-Optimized Adaptive Threshold-Assisted Selection for Vertical Handover in 6G NTN



A research prototype implementing the **Delay-Optimized Adaptive Threshold-Assisted Selection (DO-ATAS)** framework for intelligent vertical handover management in integrated terrestrial and non-terrestrial 6G communication environments.

The framework targets seamless Unmanned Aerial Vehicle (UAV) connectivity across **Terrestrial Base Stations**, **High Altitude Platform Stations (HAPS)**, and **Low Earth Orbit (LEO) satellites**, minimizing handover delay, packet loss, and ping-pong rate while maximizing throughput, handover success rate, network stability, and communication reliability.

---

##  Overview

Maintaining stable communication during UAV mobility in 6G Non-Terrestrial Networks (NTNs) is challenging due to:

- Frequent signal fluctuations
- Excessive handover delay
- Unstable network conditions
- Repeated ping-pong handovers in multi-layer environments

Conventional **Adaptive Threshold-Assisted Selection (ATAS)** relies on *static* thresholds and cannot adapt to rapidly changing conditions. **DO-ATAS** solves this by introducing:

| Component | Purpose |
|---|---|
| **Adaptive EMA Smoothing** | Mitigates abrupt signal fluctuations |
| **Min-Max Normalization** | Consistent comparison across heterogeneous layers |
| **Delay-Aware Adaptive Threshold (DAT)** | Dynamically adjusts thresholds per delay, mobility, and QoS |
| **Entropy Weight Method (EWM)** | Dynamic QoS weight optimization |
| **Mobility-Aware Utility Model** | Ranks Terrestrial, HAPS, LEO links |
| **Dual-Window Stability Verification** | Short- and long-term stability gating |
| **Hysteresis + Time-to-Trigger (TTT)** | Ping-pong suppression |
| **Baseline ATAS** | Reference implementation for comparison |

---

##  Architecture Pipeline
