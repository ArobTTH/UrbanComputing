# Q4 – US Air Traffic (slides-aligned, minimal & helper-based)
# ------------------------------------------------------------
from pathlib import Path
import json
import pandas as pd, numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# ---------- 0) 通过课程 helper 定位数据 ----------
from utils import airtraffic_helpers as h
PKG_DIR = Path(h.__file__).resolve().parents[1]      # -> hw1codes/
DATASET = PKG_DIR / "dataset" / "288798530_T_T100D_MARKET_ALL_CARRIER.csv"
OUT = Path("./q4_outputs"); OUT.mkdir(parents=True, exist_ok=True)
print("Using dataset:", DATASET)

# ---------- 1) 读边并建有向图（最小清洗：去NA/自环/重复） ----------
df = pd.read_csv(DATASET, usecols=["ORIGIN","DEST"]).dropna()
df = df[df["ORIGIN"] != df["DEST"]].drop_duplicates()

G = nx.DiGraph()
G.add_edges_from(df.itertuples(index=False, name=None))

N, M = G.number_of_nodes(), G.number_of_edges()
print(f"Nodes={N}, Edges={M}")

# ===================== Q4(a) Connected Components =====================
# WCC（忽略方向）、SCC（尊重方向）
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

# 图1/图2：去巨分量后的“规模→频次”柱状图（用来说明“几乎全连通 + 少量边缘”）
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

plot_size_bar_excluding_giant(wcc_sizes, "WCC size frequencies (excluding giant)", "q4a_wcc_hist.png")
plot_size_bar_excluding_giant(scc_sizes, "SCC size frequencies (excluding giant)", "q4a_scc_hist.png")

# ===================== Q4(b) Centrality =====================
# 有向的 in/out-degree 分布（slides 有向网络常规）
in_deg = dict(G.in_degree()); out_deg = dict(G.out_degree())

def plot_degree_hist(d, title, outfile):
    vals = list(d.values())
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(vals, bins=40)
    ax.set_title(title); ax.set_xlabel(title); ax.set_ylabel("Count")
    ax.grid(axis="y", ls=":"); fig.tight_layout(); fig.savefig(OUT/outfile, dpi=200); plt.show()

plot_degree_hist(in_deg, "In-degree (directed)", "q4b_in_degree_hist.png")
plot_degree_hist(out_deg, "Out-degree (directed)", "q4b_out_degree_hist.png")

# 无向化后计算度中心性与介数中心性（slides 常用做法）
Gu = G.to_undirected()
deg_cent = nx.degree_centrality(Gu)
bet_cent = nx.betweenness_centrality(Gu, k=200, seed=42, normalized=True)  # 采样提速

def hist_and_top(d, title, outfile, k=10):
    vals = list(d.values())
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(vals, bins=40)
    ax.set_title(title); ax.set_xlabel(title); ax.set_ylabel("Count")
    ax.grid(axis="y", ls=":"); fig.tight_layout(); fig.savefig(OUT/outfile, dpi=200); plt.show()
    topk = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
    # 保存/打印 top-k
    csv_path = OUT / (outfile.replace(".png", "_top.csv"))
    pd.DataFrame(topk, columns=["airport","score"]).to_csv(csv_path, index=False)
    print(f"[Top {k}] {title}:")
    for i,(node,score) in enumerate(topk, 1):
        print(f"  {i:>2}. {node}: {score:.4f}")

hist_and_top(deg_cent, "Degree centrality (undirected)", "q4b_deg_cent_hist.png")
hist_and_top(bet_cent, "Betweenness centrality (undirected, sampled)", "q4b_bet_cent_hist.png")

# 画一个 Top-10 betweenness 的条形图，直接展示“关键中转”
top10_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]
fig, ax = plt.subplots(figsize=(7,4))
ax.bar([a for a,_ in top10_bet], [s for _,s in top10_bet], edgecolor="black")
ax.set_ylabel("Betweenness"); ax.set_title("Top-10 betweenness (undirected)")
ax.set_xticklabels([a for a,_ in top10_bet], rotation=45, ha="right")
ax.grid(axis="y", ls=":")
fig.tight_layout(); fig.savefig(OUT/"q4b_top_bet_bar.png", dpi=200); plt.show()

# ===================== Q4(c) Clustering =====================
local_C = nx.clustering(Gu)            # C_v
avg_C = nx.average_clustering(Gu)      # <C_v>
trans = nx.transitivity(Gu)            # global transitivity

summary_c = {"average_clustering": round(avg_C, 4), "transitivity": round(trans, 4)}
print(json.dumps(summary_c, indent=2))
(Path(OUT/"q4c_summary.json")).write_text(json.dumps(summary_c, indent=2))

# 图：C_v 分布（只画 >0 的节点，避免大量 0 堵在左端）
fig, ax = plt.subplots(figsize=(6,4))
ax.hist([v for v in local_C.values() if v>0], bins=40)
ax.set_title("Clustering coefficient distribution (C_v>0)")
ax.set_xlabel("C_v"); ax.set_ylabel("Count")
ax.grid(axis="y", ls=":"); fig.tight_layout(); fig.savefig(OUT/"q4c_clustering_hist.png", dpi=200); plt.show()

# 图（可选但很有解释力）：degree vs clustering（显示 hub-and-spoke 下 C_v 下降趋势）
deg_u = dict(Gu.degree())
x = [deg_u[n] for n in Gu.nodes()]
y = [local_C[n] for n in Gu.nodes()]
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x, y, s=6, alpha=0.4)
ax.set_xlabel("Degree (undirected)"); ax.set_ylabel("Local clustering C_v")
ax.set_title("Degree vs. clustering (hub-and-spoke signature)")
ax.grid(True, ls=":")
fig.tight_layout(); fig.savefig(OUT/"q4c_deg_vs_cluster.png", dpi=200); plt.show()

print("All outputs saved to:", OUT.resolve())
