
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# ==============================================================================
# 0. GLOBAL CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)


DATA_PATH = None            #  DATA_PATH = "data/ntn_handover.csv"

# Output directory for plots
OUTPUT_DIR = "."
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "doatas_results.png")

# Network layers
LAYERS = ["Terrestrial", "HAPS", "LEO"]
N_LAYERS = len(LAYERS)

# Simulation horizon (time steps) & sampling interval
N_STEPS = 1200
DT = 1.0  # seconds per step

# Handover control parameters
HYSTERESIS_MARGIN = 0.05
TTT_STEPS = 5
SHORT_WINDOW = 10
LONG_WINDOW = 30
STABILITY_THRESHOLD = 0.6

# EMA smoothing
EMA_ALPHA_BASE = 0.3

# Parameters used in EWM & utility
PARAM_COLS = ["rss", "snr", "throughput", "latency", "velocity", "elevation"]
BENEFIT_COLS = {"rss", "snr", "throughput", "elevation"}
COST_COLS    = {"latency", "velocity"}


# ==============================================================================
# 1. DATASET GENERATION / LOADING
# ==============================================================================
def generate_ntn_dataset(n_steps=N_STEPS):
    """Simulate heterogeneous communication parameters from Terrestrial/HAPS/LEO."""
    t = np.arange(n_steps)

    base_velocity = 20 + 10 * np.sin(2 * np.pi * t / 500) + 3 * np.sin(2 * np.pi * t / 97)
    base_velocity += np.random.normal(0, 1.0, n_steps)
    base_velocity = np.clip(base_velocity, 5, 45)

    elev_terr = np.clip(20 + 15 * np.sin(2 * np.pi * t / 400) + np.random.normal(0, 2, n_steps), 5, 60)
    elev_haps = np.clip(45 + 10 * np.sin(2 * np.pi * t / 600) + np.random.normal(0, 1.5, n_steps), 25, 80)
    elev_leo  = np.clip(60 + 8  * np.sin(2 * np.pi * t / 800) + np.random.normal(0, 1.0, n_steps), 40, 90)

    rss_terr = 80 - 0.35 * t % 60 + 6 * np.sin(2 * np.pi * t / 150) + np.random.normal(0, 2.5, n_steps)
    snr_terr = 18 + 6 * np.sin(2 * np.pi * t / 130) + np.random.normal(0, 1.5, n_steps)
    thr_terr = np.clip(25 + 10 * np.sin(2 * np.pi * t / 200) + np.random.normal(0, 2, n_steps), 5, 60)
    lat_terr = np.clip(15 + 5 * np.sin(2 * np.pi * t / 110) + np.random.normal(0, 1.5, n_steps), 5, 40)

    rss_hap = 70 + 5 * np.sin(2 * np.pi * t / 300) + np.random.normal(0, 1.8, n_steps)
    snr_hap = 22 + 4 * np.sin(2 * np.pi * t / 250) + np.random.normal(0, 1.2, n_steps)
    thr_hap = np.clip(35 + 8 * np.sin(2 * np.pi * t / 320) + np.random.normal(0, 1.8, n_steps), 10, 80)
    lat_hap = np.clip(25 + 4 * np.sin(2 * np.pi * t / 200) + np.random.normal(0, 1.2, n_steps), 10, 50)

    rss_leo = 60 + 7 * np.sin(2 * np.pi * t / 450) + np.random.normal(0, 2.0, n_steps)
    snr_leo = 25 + 5 * np.sin(2 * np.pi * t / 380) + np.random.normal(0, 1.4, n_steps)
    thr_leo = np.clip(45 + 12 * np.sin(2 * np.pi * t / 500) + np.random.normal(0, 2.2, n_steps), 15, 120)
    lat_leo = np.clip(40 + 8 * np.sin(2 * np.pi * t / 400) + np.random.normal(0, 2.0, n_steps), 20, 90)

    def pkt_loss(snr, lat):
        base = 0.02 + 0.004 * np.clip(15 - snr, 0, None) + 0.0015 * np.clip(lat - 20, 0, None)
        base += np.random.uniform(0, 0.005, size=snr.shape)
        return np.clip(base, 0, 0.4)

    df_terr = pd.DataFrame({
        "t": t, "layer": "Terrestrial", "rss": rss_terr, "snr": snr_terr,
        "throughput": thr_terr, "latency": lat_terr,
        "velocity": base_velocity, "elevation": elev_terr,
        "packet_loss": pkt_loss(snr_terr, lat_terr)
    })
    df_hap = pd.DataFrame({
        "t": t, "layer": "HAPS", "rss": rss_hap, "snr": snr_hap,
        "throughput": thr_hap, "latency": lat_hap,
        "velocity": base_velocity, "elevation": elev_haps,
        "packet_loss": pkt_loss(snr_hap, lat_hap)
    })
    df_leo = pd.DataFrame({
        "t": t, "layer": "LEO", "rss": rss_leo, "snr": snr_leo,
        "throughput": thr_leo, "latency": lat_leo,
        "velocity": base_velocity, "elevation": elev_leo,
        "packet_loss": pkt_loss(snr_leo, lat_leo)
    })

    return {"Terrestrial": df_terr, "HAPS": df_hap, "LEO": df_leo}


def load_dataset_from_csv(path):
    """
    Load a long-format CSV with columns:
      t, layer, rss, snr, throughput, latency, velocity, elevation, packet_loss
    Returns dict: layer -> DataFrame indexed by t.
    """
    df = pd.read_csv(path)
    required = {"t", "layer", "rss", "snr", "throughput", "latency",
                "velocity", "elevation", "packet_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    data = {}
    for lyr in LAYERS:
        sub = df[df["layer"] == lyr].sort_values("t").reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"No rows found for layer '{lyr}' in CSV.")
        data[lyr] = sub
    return data


def load_dataset(path):
    if path is None:
        print("[DATA] Using built-in synthetic dataset.")
        return generate_ntn_dataset(N_STEPS)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH '{path}' not found.")
    print(f"[DATA] Loading dataset from: {path}")
    return load_dataset_from_csv(path)


# ==============================================================================
# 2. ADAPTIVE EMA SIGNAL SMOOTHING
# ==============================================================================
class AdaptiveEMA:
    def __init__(self, alpha_base=EMA_ALPHA_BASE, var_window=20):
        self.alpha_base = alpha_base
        self.var_window = var_window
        self.state = None
        self.history = deque(maxlen=var_window)

    def update(self, x):
        self.history.append(x)
        if self.state is None:
            self.state = x
            return x
        var = np.var(self.history) if len(self.history) > 2 else 0.0
        scale = 1.0 / (1.0 + np.exp(-var))
        alpha = np.clip(self.alpha_base + 0.5 * scale, 0.1, 0.9)
        self.state = alpha * x + (1 - alpha) * self.state
        return self.state

    def reset(self):
        self.state = None
        self.history.clear()


def smooth_all(df):
    smoothed = df.copy()
    for col in ["rss", "snr", "throughput", "latency", "elevation"]:
        ema = AdaptiveEMA()
        smoothed[col + "_s"] = np.array([ema.update(v) for v in df[col].values])
    smoothed["velocity_s"] = df["velocity"].values
    return smoothed


# ==============================================================================
# 3. MIN-MAX NORMALIZATION
# ==============================================================================
def normalize_snapshot(raw_vals):
    out = {l: {} for l in raw_vals}
    for p in PARAM_COLS:
        vals = np.array([raw_vals[l][p] for l in LAYERS], dtype=float)
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx - mn > 1e-9 else 1.0
        for l in LAYERS:
            v = raw_vals[l][p]
            norm = (v - mn) / rng
            if p in COST_COLS:
                norm = 1.0 - norm
            out[l][p] = norm
    return out


# ==============================================================================
# 4. ENTROPY WEIGHT METHOD (EWM)
# ==============================================================================
def entropy_weights(norm_hist):
    n_steps, n_layers, n_params = norm_hist.shape
    eps = 1e-12
    X = norm_hist.reshape(-1, n_params)
    X = (X - X.min(0)) / (X.ptp(0) + eps)
    P = (X + eps) / (X.sum(0) + eps * n_params)
    E = -np.sum(P * np.log(P + eps), axis=0) / np.log(P.shape[0] + eps)
    D = 1.0 - E
    if D.sum() < 1e-12:
        return np.ones(n_params) / n_params
    return D / D.sum()


# ==============================================================================
# 5. DELAY-AWARE ADAPTIVE THRESHOLD (DAT)
# ==============================================================================
def compute_dat(base_thr, delay_ms, velocity, qos):
    lat_norm = np.clip(delay_ms / 100.0, 0, 1)
    vel_norm = np.clip(velocity / 50.0, 0, 1)
    k1, k2 = 0.35, 0.25
    qos_factor = 0.5 + 0.5 * np.clip(qos, 0, 1)
    dat = base_thr * (1 + k1 * lat_norm) * (1 + k2 * vel_norm) / qos_factor
    return float(np.clip(dat, 0.3, 0.95))


# ==============================================================================
# 6. DUAL-WINDOW STABILITY
# ==============================================================================
class DualWindowStability:
    def __init__(self, short=SHORT_WINDOW, long=LONG_WINDOW):
        self.short = deque(maxlen=short)
        self.long = deque(maxlen=long)

    def update(self, utility):
        self.short.append(utility)
        self.long.append(utility)

    def score(self):
        if len(self.long) < 3:
            return 0.0
        s_std = np.std(self.short) if len(self.short) > 1 else 0.0
        l_std = np.std(self.long)
        s_score = 1.0 / (1.0 + 10 * s_std)
        l_score = 1.0 / (1.0 + 5 * l_std)
        return float(np.clip(0.5 * s_score + 0.5 * l_score, 0, 1))

    def reset(self):
        self.short.clear()
        self.long.clear()


# ==============================================================================
# 7. DO-ATAS CONTROLLER
# ==============================================================================
class DOATASController:
    def __init__(self, hysteresis=HYSTERESIS_MARGIN, ttt=TTT_STEPS):
        self.hysteresis = hysteresis
        self.ttt = ttt
        self.current_layer = LAYERS[0]
        self.ttt_counter = 0
        self.candidate = None
        self.stability = {l: DualWindowStability() for l in LAYERS}
        self.norm_hist = deque(maxlen=60)

    def decide(self, t, raw_snapshot, ema_snapshot, qos_baseline=0.8):
        norm_input = {l: {p: ema_snapshot[l][p + "_s"] for p in PARAM_COLS} for l in LAYERS}
        normed = normalize_snapshot(norm_input)
        self.norm_hist.append(np.array([[normed[l][p] for p in PARAM_COLS] for l in LAYERS]))

        if len(self.norm_hist) >= 10:
            weights = entropy_weights(np.array(self.norm_hist))
        else:
            weights = np.ones(len(PARAM_COLS)) / len(PARAM_COLS)
        w = dict(zip(PARAM_COLS, weights))

        utilities = {}
        for l in LAYERS:
            u = sum(w[p] * normed[l][p] for p in PARAM_COLS)
            self.stability[l].update(u)
            stab = self.stability[l].score()
            utilities[l] = float(u * (0.7 + 0.3 * stab))

        cur_raw = raw_snapshot[self.current_layer]
        dat = compute_dat(
            base_thr=0.55,
            delay_ms=cur_raw["latency"],
            velocity=cur_raw["velocity"],
            qos=qos_baseline
        )

        best_layer = max(utilities, key=utilities.get)
        best_util = utilities[best_layer]
        cur_util = utilities[self.current_layer]
        best_stab = self.stability[best_layer].score()
        allow_stability = best_stab >= STABILITY_THRESHOLD

        decision = self.current_layer
        ho_executed = False
        if best_layer != self.current_layer:
            gain = best_util - cur_util
            if gain > self.hysteresis and best_util > dat and allow_stability:
                if self.candidate == best_layer:
                    self.ttt_counter += 1
                else:
                    self.candidate = best_layer
                    self.ttt_counter = 1
                if self.ttt_counter >= self.ttt:
                    decision = best_layer
                    ho_executed = True
                    self.current_layer = best_layer
                    self.candidate = None
                    self.ttt_counter = 0
            else:
                self.candidate = None
                self.ttt_counter = 0
        else:
            self.candidate = None
            self.ttt_counter = 0

        return decision, {
            "utilities": utilities,
            "dat": dat,
            "best_layer": best_layer,
            "ho_executed": ho_executed,
            "weights": w,
            "stability": {l: self.stability[l].score() for l in LAYERS}
        }


# ==============================================================================
# 8. BASELINE ATAS
# ==============================================================================
class ATASBaseline:
    def __init__(self, static_thr=0.6):
        self.thr = static_thr
        self.current = LAYERS[0]

    def decide(self, raw_snapshot, ema_snapshot):
        norm_input = {l: {p: ema_snapshot[l][p + "_s"] for p in PARAM_COLS} for l in LAYERS}
        normed = normalize_snapshot(norm_input)
        w = {p: 1.0 / len(PARAM_COLS) for p in PARAM_COLS}
        utils = {l: sum(w[p] * normed[l][p] for p in PARAM_COLS) for l in LAYERS}
        best = max(utils, key=utils.get)
        if best != self.current and utils[best] > self.thr + 0.05:
            self.current = best
            ho = True
        else:
            ho = False
        return self.current, utils, ho


# ==============================================================================
# 9. SIMULATION LOOP
# ==============================================================================
def run_simulation(data, controller, is_baseline=False):
    n = len(data["Terrestrial"])
    smoothed = {l: smooth_all(data[l]) for l in LAYERS}

    selected = []
    ho_count = 0
    ho_attempts = 0
    ho_success = 0
    delays = []
    throughputs = []
    pkt_losses = []
    rss_sel, snr_sel, lat_sel = [], [], []
    util_history = {l: [] for l in LAYERS}
    dat_history = []
    weight_history = []

    prev_layer = LAYERS[0]

    for t in range(n):
        raw = {l: {p: data[l].iloc[t][p] for p in PARAM_COLS} for l in LAYERS}
        ema = {l: {p + "_s": smoothed[l].iloc[t][p + "_s"] for p in PARAM_COLS} for l in LAYERS}

        if is_baseline:
            chosen, utils, ho = controller.decide(raw, ema)
            dat_val = controller.thr
            weights = {p: 1.0 / len(PARAM_COLS) for p in PARAM_COLS}
        else:
            chosen, info = controller.decide(t, raw, ema)
            utils = info["utilities"]
            ho = info["ho_executed"]
            dat_val = info["dat"]
            weights = info["weights"]

        selected.append(chosen)
        for l in LAYERS:
            util_history[l].append(utils[l])
        dat_history.append(dat_val)
        weight_history.append(weights)

        if chosen != prev_layer:
            ho_attempts += 1
            ho_delay = 60 + 30 * (1 - utils[chosen]) + np.random.uniform(5, 25)
            delays.append(ho_delay)
            ho_count += 1
            if utils[chosen] > 0.4:
                ho_success += 1
        prev_layer = chosen

        throughputs.append(data[chosen].iloc[t]["throughput"])
        pkt_losses.append(data[chosen].iloc[t]["packet_loss"])
        rss_sel.append(data[chosen].iloc[t]["rss"])
        snr_sel.append(data[chosen].iloc[t]["snr"])
        lat_sel.append(data[chosen].iloc[t]["latency"])

    # Ping-pong (A->B->A within any 2-step window)
    pp = 0
    for i in range(2, len(selected)):
        if selected[i] != selected[i - 1] and selected[i] == selected[i - 2]:
            pp += 1

    cross_layer_util_std = float(np.std(
        [np.mean([util_history[l][i] for l in LAYERS]) for i in range(n)]
    ))

    results = {
        "selected": selected,
        "utilities": util_history,
        "dat": dat_history,
        "weights": weight_history,
        "handover_count": ho_count,
        "handover_attempts": ho_attempts,
        "handover_success_rate": (ho_success / ho_attempts * 100) if ho_attempts > 0 else 100.0,
        "avg_handover_delay_ms": float(np.mean(delays)) if delays else 0.0,
        "total_handover_delay_ms": float(np.sum(delays)) if delays else 0.0,
        "avg_throughput_mbps": float(np.mean(throughputs)),
        "avg_packet_loss": float(np.mean(pkt_losses)),
        "avg_rss_dbm": float(np.mean(rss_sel)),
        "avg_snr_db": float(np.mean(snr_sel)),
        "avg_latency_ms": float(np.mean(lat_sel)),
        "ping_pong_rate": pp / len(selected) * 100,
        "qos_score": float(np.mean([
            np.mean(throughputs) / 100.0,
            1 - np.mean(pkt_losses),
            1 - np.mean(lat_sel) / 100.0
        ])),
        "network_stability": cross_layer_util_std,
        "communication_reliability": (1 - np.mean(pkt_losses)) *
                                      (ho_success / max(ho_attempts, 1)),
    }
    return results


# ==============================================================================
# 10. FULL METRICS PRINTOUT
# ==============================================================================
def summarize_full(name, res):
    stability = 1.0 / (1.0 + res["network_stability"])
    print(f"\n{'='*78}")
    print(f"  {name}")
    print(f"{'='*78}")
    print(f"  {'Metric':<38}{'Value':>18}{'Unit':>16}")
    print(f"  {'-'*72}")
    print(f"  {'1. Handover Delay (average)':<38}{res['avg_handover_delay_ms']:>18.3f}{'ms':>16}")
    print(f"  {'   Handover Delay (total)':<38}{res['total_handover_delay_ms']:>18.3f}{'ms':>16}")
    print(f"  {'2. Throughput':<38}{res['avg_throughput_mbps']:>18.3f}{'Mbps':>16}")
    print(f"  {'3. Packet Loss Ratio':<38}{res['avg_packet_loss']*100:>18.3f}{'%':>16}")
    print(f"  {'4. RSS':<38}{res['avg_rss_dbm']:>18.3f}{'dBm':>16}")
    print(f"  {'5. SNR':<38}{res['avg_snr_db']:>18.3f}{'dB':>16}")
    print(f"  {'6. Handover Success Rate':<38}{res['handover_success_rate']:>18.3f}{'%':>16}")
    print(f"  {'7. Ping-Pong Rate':<38}{res['ping_pong_rate']:>18.3f}{'%':>16}")
    print(f"  {'8. Latency':<38}{res['avg_latency_ms']:>18.3f}{'ms':>16}")
    print(f"  {'9. QoS Score':<38}{res['qos_score']:>18.4f}{'[0-1]':>16}")
    print(f"  {'10. Network Stability':<38}{stability:>18.4f}{'[0-1]':>16}")
    print(f"  {'11. Communication Reliability':<38}{res['communication_reliability']:>18.4f}{'[0-1]':>16}")
    print(f"  {'-'*72}")
    print(f"  {'Handovers Executed':<38}{res['handover_count']:>18d}{'count':>16}")
    print(f"  {'Handover Attempts':<38}{res['handover_attempts']:>18d}{'count':>16}")
    print(f"{'='*78}")


def print_full_comparison(doatas_res, atas_res):
    def stab(r):
        return 1.0 / (1.0 + r["network_stability"])

    table = [
        ("1. Handover Delay (avg)",      doatas_res["avg_handover_delay_ms"],
                                          atas_res["avg_handover_delay_ms"],      "ms",  True),
        ("   Handover Delay (total)",    doatas_res["total_handover_delay_ms"],
                                          atas_res["total_handover_delay_ms"],    "ms",  True),
        ("2. Throughput",                doatas_res["avg_throughput_mbps"],
                                          atas_res["avg_throughput_mbps"],        "Mbps",False),
        ("3. Packet Loss Ratio",         doatas_res["avg_packet_loss"]*100,
                                          atas_res["avg_packet_loss"]*100,        "%",   True),
        ("4. RSS",                       doatas_res["avg_rss_dbm"],
                                          atas_res["avg_rss_dbm"],                "dBm", False),
        ("5. SNR",                       doatas_res["avg_snr_db"],
                                          atas_res["avg_snr_db"],                 "dB",  False),
        ("6. Handover Success Rate",     doatas_res["handover_success_rate"],
                                          atas_res["handover_success_rate"],      "%",   False),
        ("7. Ping-Pong Rate",            doatas_res["ping_pong_rate"],
                                          atas_res["ping_pong_rate"],             "%",   True),
        ("8. Latency",                   doatas_res["avg_latency_ms"],
                                          atas_res["avg_latency_ms"],             "ms",  True),
        ("9. QoS Score",                 doatas_res["qos_score"],
                                          atas_res["qos_score"],                  "[0-1]",False),
        ("10. Network Stability",        stab(doatas_res),
                                          stab(atas_res),                         "[0-1]",False),
        ("11. Communication Reliability",doatas_res["communication_reliability"],
                                          atas_res["communication_reliability"],  "[0-1]",False),
        ("Handovers Executed",           doatas_res["handover_count"],
                                          atas_res["handover_count"],             "count",True),
    ]

    print(f"\n{'='*88}")
    print(f"  FULL METRIC COMPARISON:  DO-ATAS (Proposed)  vs  ATAS (Baseline)")
    print(f"{'='*88}")
    print(f"  {'#  Metric':<32}{'DO-ATAS':>13}{'ATAS':>13}{'Unit':>9}{'Δ Improvement':>17}")
    print(f"  {'-'*84}")

    for name, dv, av, unit, lower_better in table:
        if av == 0:
            impr = 0.0
        elif lower_better:
            impr = (av - dv) / av * 100.0
        else:
            impr = (dv - av) / av * 100.0
        arrow = "▲" if impr > 0 else ("▼" if impr < 0 else " ")
        if unit == "count":
            print(f"  {name:<32}{int(dv):>13d}{int(av):>13d}{unit:>9}{f'{arrow} {impr:+.2f}%':>17}")
        else:
            print(f"  {name:<32}{dv:>13.4f}{av:>13.4f}{unit:>9}{f'{arrow} {impr:+.2f}%':>17}")
    print(f"{'='*88}")
    print("  Note: ▲ = DO-ATAS better than ATAS,  ▼ = DO-ATAS worse than ATAS")
    print(f"{'='*88}")


# ==============================================================================
# 11. PLOTS
# ==============================================================================
def plot_results(doatas, atas, data):
    n = len(doatas["selected"])
    t = np.arange(n)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    def layer_to_int(sel):
        return [LAYERS.index(s) for s in sel]

    axs[0, 0].plot(t, layer_to_int(doatas["selected"]), label="DO-ATAS", lw=1.2)
    axs[0, 0].plot(t, layer_to_int(atas["selected"]), label="ATAS (baseline)", lw=1.0, alpha=0.7)
    axs[0, 0].set_yticks([0, 1, 2]); axs[0, 0].set_yticklabels(LAYERS)
    axs[0, 0].set_title("Selected Layer over Time")
    axs[0, 0].set_xlabel("Time step"); axs[0, 0].legend(); axs[0, 0].grid(alpha=0.3)

    for i, l in enumerate(LAYERS):
        axs[0, 1].plot(t, doatas["utilities"][l], label=l, alpha=0.8)
    axs[0, 1].set_title("DO-ATAS Utility per Layer")
    axs[0, 1].set_xlabel("Time step"); axs[0, 1].legend(); axs[0, 1].grid(alpha=0.3)

    axs[1, 0].plot(t, doatas["dat"], color="purple", label="DO-ATAS DAT")
    axs[1, 0].axhline(0.6, ls="--", color="gray", label="Static ATAS threshold")
    axs[1, 0].set_title("Delay-Aware Adaptive Threshold (DAT)")
    axs[1, 0].set_xlabel("Time step"); axs[1, 0].set_ylabel("Threshold")
    axs[1, 0].legend(); axs[1, 0].grid(alpha=0.3)

    labels = ["Handover\ndelay (ms)", "Ping-pong\n(%)", "Packet\nloss (%)"]
    do_vals = [doatas["avg_handover_delay_ms"], doatas["ping_pong_rate"], doatas["avg_packet_loss"]*100]
    at_vals = [atas["avg_handover_delay_ms"], atas["ping_pong_rate"], atas["avg_packet_loss"]*100]
    x = np.arange(len(labels)); w = 0.35
    axs[1, 1].bar(x - w/2, do_vals, w, label="DO-ATAS", color="#2b8cbe")
    axs[1, 1].bar(x + w/2, at_vals, w, label="ATAS", color="#fdae6b")
    axs[1, 1].set_xticks(x); axs[1, 1].set_xticklabels(labels)
    axs[1, 1].set_title("Key Performance Comparison (lower is better)")
    axs[1, 1].legend(); axs[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=130)
    plt.show()


# ==============================================================================
# 12. MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("  DO-ATAS Framework — Vertical Handover in 6G NTN")
    print("=" * 78)

    # ---------------- Load / generate dataset ----------------
    data = load_dataset(DATA_PATH)

    # Align lengths (in case CSV has a different number of steps per layer)
    min_len = min(len(data[l]) for l in LAYERS)
    for l in LAYERS:
        data[l] = data[l].iloc[:min_len].reset_index(drop=True)
    print(f"[DATA] Using {min_len} time steps per layer.")

    # ---------------- Run DO-ATAS ----------------
    print("\n[RUN] DO-ATAS (Proposed) ...")
    doatas_controller = DOATASController()
    doatas_res = run_simulation(data, doatas_controller, is_baseline=False)

    # ---------------- Run baseline ATAS ----------------
    print("[RUN] ATAS (Baseline) ...")
    atas_controller = ATASBaseline(static_thr=0.6)
    atas_res = run_simulation(data, atas_controller, is_baseline=True)

    # ---------------- FULL METRIC PRINTOUTS ----------------
    summarize_full("DO-ATAS  (Proposed Framework)", doatas_res)
    summarize_full("ATAS  (Baseline)", atas_res)
    print_full_comparison(doatas_res, atas_res)

    # ---------------- Plots ----------------
    plot_results(doatas_res, atas_res, data)
    print(f"\n[OUT] Saved figure -> {OUTPUT_FIG}")
