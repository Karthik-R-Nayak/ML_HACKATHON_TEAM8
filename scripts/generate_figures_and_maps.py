"""Generate interactive Folium maps and evaluation plots for the technical report."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import folium

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "pathsnap")]

from app.pathsnap.pipeline import PATHSNAP_PIPELINE
from app.pathsnap.member2_scorer import load_member2_scorer
from app.pathsnap.member3_decoder import load_member3_decoder
from app.pathsnap.interfaces import PathSnapModelRequest, PathSnapDecodeRequest
from app.pathsnap.preprocessing.align import align_route
from app.pathsnap.preprocessing.config import AlignConfig
from pathsnap.io import parse_nodes, parse_arcs, load_trip_files

FIG_DIR = ROOT / "docs/figures"
MAP_DIR = ROOT / "docs/maps"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = ROOT.parent / "experiments/member2/data/student_test/raw"
scorer = load_member2_scorer(ROOT / "backend/artifacts/member2")
decoder = load_member3_decoder(ROOT / "backend/artifacts/member3")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 15,
})


def generate_folium_map(trip_id="00000052"):
    tdir = RAW_DIR / trip_id
    nodes = parse_nodes(tdir / f"{trip_id}.nodes")
    arcs = parse_arcs(tdir / f"{trip_id}.arcs")
    track, route = load_trip_files(tdir / f"{trip_id}.track", tdir / f"{trip_id}.route")

    prep = PATHSNAP_PIPELINE.prepare(trip_id=trip_id, nodes=nodes, arcs=arcs, track=track, route_arc_ids=None)
    scored = scorer.score_candidates(PathSnapModelRequest(trip_id, prep.candidate_features, prep.candidate_feature_columns))
    decoded = decoder.decode_path(PathSnapDecodeRequest(trip_id, scored.candidate_scores, prep.observations, prep.network))

    net = prep.network
    center_lat = float(track["lat"].mean())
    center_lon = float(track["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # Layer 1: Road Network Candidates (subset near trip)
    fg_road = folium.FeatureGroup(name="Road Network Candidates (Gray)", show=True)
    seen_arcs = set()
    for row in scored.candidate_scores.itertuples():
        aid = int(row.arc_idx)
        if aid not in seen_arcs and aid < net.n_arcs:
            seen_arcs.add(aid)
            fn = int(net.arc_from[aid])
            tn = int(net.arc_to[aid])
            folium.PolyLine(
                [[float(nodes.iloc[fn].lat), float(nodes.iloc[fn].lon)],
                 [float(nodes.iloc[tn].lat), float(nodes.iloc[tn].lon)]],
                color="#94a3b8", weight=2, opacity=0.6,
                tooltip=f"Arc {row.arc_id}"
            ).add_to(fg_road)
    fg_road.add_to(m)

    # Layer 2: Reference Ground Truth Route (Purple)
    if route is not None:
        fg_gt = folium.FeatureGroup(name="Reference Route (Purple)", show=True)
        route_idx = net.route_to_idx(np.asarray(route, dtype=np.int64))
        for rid in route_idx:
            fn = int(net.arc_from[rid])
            tn = int(net.arc_to[rid])
            folium.PolyLine(
                [[float(nodes.iloc[fn].lat), float(nodes.iloc[fn].lon)],
                 [float(nodes.iloc[tn].lat), float(nodes.iloc[tn].lon)]],
                color="#8b5cf6", weight=5, opacity=0.7, dash_array="6, 6",
                tooltip=f"Ground Truth Arc {net.arc_ids[rid]}"
            ).add_to(fg_gt)
        fg_gt.add_to(m)

    # Layer 3: Matched Route (Emerald Green)
    fg_match = folium.FeatureGroup(name="PathSnap Matched Route (Emerald)", show=True)
    for row in decoded.decoded_path.itertuples():
        if pd.isna(row.arc_idx):
            continue
        aid = int(row.arc_idx)
        fn = int(net.arc_from[aid])
        tn = int(net.arc_to[aid])
        folium.PolyLine(
            [[float(nodes.iloc[fn].lat), float(nodes.iloc[fn].lon)],
             [float(nodes.iloc[tn].lat), float(nodes.iloc[tn].lon)]],
            color="#10b981", weight=6, opacity=0.9,
            tooltip=f"Matched Arc {row.arc_id} (Obs {row.obs_idx})"
        ).add_to(fg_match)
    fg_match.add_to(m)

    # Layer 4: GPS Observations (Blue Dots)
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (Blue Dots)", show=True)
    for idx, row in track.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4, color="#0284c7", fill=True, fillColor="#38bdf8", fillOpacity=0.8, weight=1,
            popup=f"Obs {idx}<br>Time: {row['t']}<br>Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html = MAP_DIR / f"trip_{trip_id}_folium_map.html"
    m.save(str(out_html))
    print(f"Saved Folium map to {out_html}")


def generate_plots():
    # 1. 5 Models Comparison Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = ["Logistic Reg.", "KNN (sampled)", "Decision Tree", "Random Forest (Sel.)", "Hist Gradient Boost"]
    cv_top1 = [82.59, 76.42, 82.10, 83.40, 83.16]
    test_top1 = [86.22, 81.93, 86.85, 87.81, 87.94]
    cv_f1 = [89.82, 85.12, 89.45, 90.61, 90.44]
    colors = ["#64748b", "#94a3b8", "#cbd5e1", "#0284c7", "#38bdf8"]

    # Subplot 1: CV Macro F1
    axes[0].bar(models, cv_f1, color=colors, edgecolor="#0f172a", linewidth=1.2)
    axes[0].set_title("Cross-Validation Macro F1 (%)")
    axes[0].set_ylim(80, 95)
    for i, v in enumerate(cv_f1):
        axes[0].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=10)
    axes[0].tick_params(axis="x", rotation=30)

    # Subplot 2: CV Top-1 Accuracy
    axes[1].bar(models, cv_top1, color=colors, edgecolor="#0f172a", linewidth=1.2)
    axes[1].set_title("CV Trip-Mean Top-1 Accuracy (%)")
    axes[1].set_ylim(70, 90)
    for i, v in enumerate(cv_top1):
        axes[1].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=10)
    axes[1].tick_params(axis="x", rotation=30)

    # Subplot 3: Student-Test Top-1 Accuracy
    axes[2].bar(models, test_top1, color=colors, edgecolor="#0f172a", linewidth=1.2)
    axes[2].set_title("Student-Test Top-1 Accuracy (%)")
    axes[2].set_ylim(75, 92)
    for i, v in enumerate(test_top1):
        axes[2].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=10)
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_model_comparison.png", dpi=300)
    plt.close()
    print("Saved 01_model_comparison.png")

    # 2. HMM vs Top-1 Accuracy & Continuity across 10 Student Trips
    trips = ["00", "07", "09", "40", "52", "56", "65", "76", "77", "91"]
    top1_acc = [93.97, 93.87, 86.37, 91.95, 95.42, 88.65, 86.85, 87.08, 68.78, 85.13]
    hmm_acc =  [95.96, 88.63, 91.80, 92.59, 100.0, 93.10, 88.67, 92.60, 70.01, 86.61]

    top1_cont = [96.24, 99.41, 87.79, 96.16, 95.02, 88.43, 91.40, 87.81, 91.64, 87.51]
    hmm_cont =  [99.60, 99.85, 95.04, 99.40, 96.55, 96.54, 98.17, 96.30, 97.85, 96.56]

    x = np.arange(len(trips))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Accuracy comparison
    ax1.bar(x - width/2, top1_acc, width, label="Classifier Top-1 (Mean: 87.81%)", color="#94a3b8", edgecolor="#334155")
    ax1.bar(x + width/2, hmm_acc, width, label="HMM Viterbi Decoded (Mean: 90.00%)", color="#0284c7", edgecolor="#0f172a")
    ax1.set_ylabel("Observation Accuracy (%)")
    ax1.set_title("A. Observation-Level Accuracy: Classifier Top-1 vs. HMM/Viterbi Route Decoder (+2.19% Mean Gain)")
    ax1.set_ylim(60, 105)
    ax1.legend(loc="lower right")

    # Continuity comparison
    ax2.bar(x - width/2, top1_cont, width, label="Classifier Top-1 (Mean: 92.14%)", color="#cbd5e1", edgecolor="#334155")
    ax2.bar(x + width/2, hmm_cont, width, label="HMM Viterbi Decoded (Mean: 97.59%)", color="#10b981", edgecolor="#0f172a")
    ax2.set_ylabel("Route Continuity (%)")
    ax2.set_xlabel("Student Evaluation Trip ID")
    ax2.set_title("B. Route Topological Continuity: Drastic Reduction in Disconnected Segments (+5.45% Mean Gain)")
    ax2.set_ylim(80, 102)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Trip {t}" for t in trips])
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_hmm_vs_top1_student_trips.png", dpi=300)
    plt.close()
    print("Saved 02_hmm_vs_top1_student_trips.png")

    # 3. Bayesian Optuna Convergence
    trials = np.arange(1, 13)
    hist_scores = [83.16, 83.18, 83.21, 83.20, 83.25, 83.22, 83.29, 83.31, 83.32, 83.30, 83.32, 83.32]
    best_so_far = np.maximum.accumulate(hist_scores)

    plt.figure(figsize=(9, 4.5))
    plt.plot(trials, hist_scores, "o--", color="#94a3b8", label="Individual Trial Top-1 CV (%)")
    plt.plot(trials, best_so_far, "s-", color="#0284c7", linewidth=2.5, label="Best Objective Value So Far")
    plt.axhline(83.40, color="#ef4444", linestyle=":", linewidth=2, label="Selected Untuned Random Forest (83.40%)")
    plt.xlabel("Optuna TPE Trial Number")
    plt.ylabel("CV Top-1 Accuracy (%)")
    plt.title("Bayesian Hyperparameter Optimisation Progression (12 Optuna TPE Trials)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_bayesian_optimization.png", dpi=300)
    plt.close()
    print("Saved 03_bayesian_optimization.png")

    # 4. Confusion Matrix Plot
    cm = np.array([[338902, 10121], [3214, 14396]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative (No Match)", "Positive (Match)"],
                yticklabels=["Negative (No Match)", "Positive (Match)"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Candidate Classification Confusion Matrix\n(Random Forest on Student Test Data)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_confusion_matrices.png", dpi=300)
    plt.close()
    print("Saved 04_confusion_matrices.png")


if __name__ == "__main__":
    generate_plots()
    generate_folium_map("00000052")
    generate_folium_map("00000000")
    print("All figures and Folium maps generated successfully!")
