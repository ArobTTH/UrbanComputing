from pathlib import Path
import json
import pandas as pd, numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# ---------- 0) Locate dataset via course helper ----------
from utils import airtraffic_helpers as h
PKG_DIR = Path(h.__file__).resolve().parents[1]
DATASET = PKG_DIR / "dataset" / "288798530_T_T100D_MARKET_ALL_CARRIER.csv"
OUT = Path("./q4_outputs"); OUT.mkdir(parents=True, exist_ok=True)
print("Using dataset:", DATASET)

# ---------- 1) Read edges & build DIRECTED graph (minimal cleaning) ----------
df = pd.read_csv(DATASET, usecols=["ORIGIN","DEST"]).dropna()
df = df[df["ORIGIN"] != df["DEST"]].drop_duplicates()

G = nx.DiGraph()
G.add_edges_from(df.itertuples(index=False, name=None))

N, M = G.number_of_nodes(), G.number_of_edges()
print(f"Nodes={N}, Edges={M}")

# ===================== Q4(a) Connected Components =====================
wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
wcc_sizes = np.array([len(c) for c in wccs])
scc_sizes = np.array([len(c) for c in sccs])
giant_wcc = int(wcc_sizes[0]) if len(wcc_sizes) else 0
giant_scc = int(scc_sizes[0]) if len(scc_sizes) else 0

summary_a = {
    "Nodes": N, "Edges": M,
    "Num_WCCs": len(wccs), "Giant_WCC_size": giant_wcc, "Giant_WCC_fraction": round(giant_wcc/N, 6) if N else 0.0,
    "Num_SCCs": len(sccs), "Giant_SCC_size": giant_scc, "Giant_SCC_fraction": round(giant_scc/N, 6) if N else 0.0
}
print(json.dumps(summary_a, indent=2))
(Path(OUT/"q4a_summary.json")).write_text(json.dumps(summary_a, indent=2))

# --- CCDF on log–log (primary for slides-style heavy-tail inspection) ---
def plot_ccdf(values, title, outfile):
    s = np.sort(np.asarray(values))
    # empirical CCDF: P(X >= x)
    y = 1.0 - np.arange(1, len(s)+1) / len(s) + (1.0/len(s))
    fig, ax = plt.subplots(figsize=(5.6,4.2))
    ax.loglog(s, y, marker='o', linestyle='none', markersize=3, alpha=0.85)
    ax.set_xlabel("Size"); ax.set_ylabel("CCDF  P(X ≥ x)")
    ax.set_title(title); ax.grid(True, which="both", ls=":")
    fig.tight_layout(); fig.savefig(OUT/outfile, dpi=200); plt.show()

plot_ccdf(wcc_sizes, "WCC size CCDF (log–log)", "q4a_wcc_ccdf.png")
plot_ccdf(scc_sizes, "SCC size CCDF (log–log)", "q4a_scc_ccdf.png")

# --- Discrete size–frequency excluding giant (secondary) ---
def plot_size_bar_excluding_giant(sizes, title, outfile):
    tail = [int(x) for x in sizes if x < sizes.max()]
    freq = Counter(tail)
    xs = sorted(freq)
    fig, ax = plt.subplots(figsize=(6,4))
    if xs:
        ys = [freq[x] for x in xs]
        ax.bar(xs, ys, width=0.8, edgecolor="black")
        step = max(1, (max(xs)-min(xs))//15) if len(xs)>30 else 1
        ax.set_xticks(range(min(xs), max(xs)+1, step))
    else:
        ax.text(0.5, 0.5, "No components besides the giant",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Component size"); ax.set_ylabel("Count")
    ax.set_title(title); ax.grid(axis="y", ls=":")
    fig.tight_layout(); fig.savefig(OUT/outfile, dpi=200); plt.show()

plot_size_bar_excluding_giant(wcc_sizes, "WCC sizes (excluding giant)", "q4a_wcc_hist.png")
plot_size_bar_excluding_giant(scc_sizes, "SCC sizes (excluding giant)", "q4a_scc_hist.png")

# ===================== Q4(b) Centrality =====================
# Directed degree
in_deg = dict(G.in_degree()); out_deg = dict(G.out_degree())

def plot_hist(vals, title, outfile, logy=False):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(vals, bins=40)
    if logy: ax.set_yscale("log")
    ax.set_title(title); ax.set_xlabel(title); ax.set_ylabel("Count")
    ax.grid(axis="y", ls=":"); fig.tight_layout(); fig.savefig(OUT/outfile, dpi=200); plt.show()

plot_hist(list(in_deg.values()),  "In-degree (directed)",  "q4b_in_degree_hist.png",  logy=True)
plot_hist(list(out_deg.values()), "Out-degree (directed)", "q4b_out_degree_hist.png", logy=True)

# CCDFs on log–log for degrees (primary in slides when examining heavy tails)
def plot_ccdf_from_counts(vals, title, outfile):
    arr = np.asarray(vals)
    s = np.sort(arr)
    y = 1.0 - np.arange(1, len(s)+1) / len(s) + (1.0/len(s))
    fig, ax = plt.subplots(figsize=(5.6,4.2))
    ax.loglog(s, y, marker='o', linestyle='none', markersize=3, alpha=0.85)
    ax.set_xlabel(title); ax.set_ylabel("CCDF  P(X ≥ x)")
    ax.set_title(title + " – CCDF (log–log)")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout(); fig.savefig(OUT/outfile, dpi=200); plt.show()

plot_ccdf_from_counts(list(in_deg.values()),  "In-degree",        "q4b_in_degree_ccdf.png")
plot_ccdf_from_counts(list(out_deg.values()), "Out-degree",       "q4b_out_degree_ccdf.png")

Gu = G.to_undirected()
undeg_vals = [d for _, d in Gu.degree()]
plot_ccdf_from_counts(undeg_vals,              "Undirected degree","q4b_undeg_ccdf.png")

# Undirected centralities (as in class)
deg_cent = nx.degree_centrality(Gu)
bet_cent = nx.betweenness_centrality(Gu, k=200, seed=42, normalized=True)

# Histogram with log-y for betweenness (secondary)
plot_hist(list(bet_cent.values()), "Betweenness (undirected, sampled)", "q4b_bet_hist_logy.png", logy=True)

# CCDF (log–log) for betweenness (primary)
plot_ccdf_from_counts(list(bet_cent.values()), "Betweenness (undirected, sampled)", "q4b_bet_ccdf.png")

# Top-10 betweenness bar (who are the transfer hubs?)
top10_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]
pd.DataFrame(top10_bet, columns=["airport","score"]).to_csv(OUT/"q4b_top_bet.csv", index=False)
fig, ax = plt.subplots(figsize=(7,4))
ax.bar([a for a,_ in top10_bet], [s for _,s in top10_bet], edgecolor="black")
ax.set_ylabel("Betweenness"); ax.set_title("Top-10 betweenness (undirected)")
ax.set_xticklabels([a for a,_ in top10_bet], rotation=45, ha="right")
ax.grid(axis="y", ls=":"); fig.tight_layout(); fig.savefig(OUT/"q4b_top_bet_bar.png", dpi=200); plt.show()

# ===================== Q4(c) Clustering =====================
local_C = nx.clustering(Gu)
avg_C = nx.average_clustering(Gu)
trans = nx.transitivity(Gu)
summary_c = {"average_clustering": round(avg_C, 4), "transitivity": round(trans, 4)}
print(json.dumps(summary_c, indent=2))
(Path(OUT/"q4c_summary.json")).write_text(json.dumps(summary_c, indent=2))

# Histogram for C_v > 0
fig, ax = plt.subplots(figsize=(6,4))
ax.hist([v for v in local_C.values() if v>0], bins=40)
ax.set_title("Clustering coefficient distribution (C_v>0)")
ax.set_xlabel("C_v"); ax.set_ylabel("Count")
ax.grid(axis="y", ls=":"); fig.tight_layout(); fig.savefig(OUT/"q4c_clustering_hist.png", dpi=200); plt.show()

# Degree vs clustering with log-x (hub-and-spoke signature clearer)
deg_u = dict(Gu.degree())
x = [deg_u[n] for n in Gu.nodes()]
y = [local_C[n] for n in Gu.nodes()]
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x, y, s=6, alpha=0.4)
ax.set_xscale("log")  # <— log-x makes the decay pattern more visible
ax.set_xlabel("Degree (undirected, log scale)"); ax.set_ylabel("Local clustering C_v")
ax.set_title("Degree vs. clustering (log-x)")
ax.grid(True, which="both", ls=":")
fig.tight_layout(); fig.savefig(OUT/"q4c_deg_vs_cluster_logx.png", dpi=200); plt.show()

print("All outputs saved to:", OUT.resolve())
