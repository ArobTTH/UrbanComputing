# q6_air_traffic.py  — US Air Traffic Network: communities, PageRank, HITS
# ---------------------------------------------------------------
# Dependencies: pandas, networkx, matplotlib
# Data: uses hw2/dataset/{T_T100D_MARKET_ALL_CARRIER.csv, T_MASTER_CORD.csv}

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------- 0) Locate dataset ----------
PKG = Path(__file__).resolve().parent        # -> .../hw2/
DATA = PKG / "dataset"

T100   = DATA / "288798530_T_T100D_MARKET_ALL_CARRIER.csv"   # flows
MASTER = DATA / "288804893_T_MASTER_CORD.csv"                # airport meta
OUT = PKG / "q6_outputs"; OUT.mkdir(parents=True, exist_ok=True)

# ---------- small utils ----------
def plot_ccdf(values, title, fname):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return
    s = np.sort(vals)
    y = 1.0 - np.arange(1, len(s)+1)/len(s)
    plt.figure(figsize=(6,4.5))
    plt.loglog(s, y, 'o', ms=3, alpha=0.7)
    plt.xlabel("Value"); plt.ylabel("CCDF  P(X ≥ x)")
    plt.title(title)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(OUT/fname, dpi=220)
    plt.close()

print("Reading data ...")

# ---------- 1) Flows (make weight robust) ----------
flows_raw = pd.read_csv(T100)
flows_raw.columns = [c.upper() for c in flows_raw.columns]

if {"ORIGIN","DEST","PASSENGERS"}.issubset(flows_raw.columns):
    # ideal case: sum passengers
    flows = (flows_raw[["ORIGIN","DEST","PASSENGERS"]]
                .dropna()
                .groupby(["ORIGIN","DEST"], as_index=False)["PASSENGERS"].sum())
elif {"ORIGIN","DEST","MONTH"}.issubset(flows_raw.columns) or {"ORIGIN","DEST"}.issubset(flows_raw.columns):
    # fallback: use frequency as weight (count rows)
    tmp = flows_raw[["ORIGIN","DEST"]].dropna().copy()
    tmp["PASSENGERS"] = 1.0
    flows = (tmp.groupby(["ORIGIN","DEST"], as_index=False)["PASSENGERS"].sum())
else:
    raise ValueError(
        "Cannot find columns for flows. Available columns: "
        + ", ".join(flows_raw.columns)
    )

# ---------- 2) Airport metadata (robust CITY/STATE) ----------
meta = pd.read_csv(MASTER)
meta.columns = [c.upper() for c in meta.columns]

if {"AIRPORT","CITY","STATE"}.issubset(meta.columns):
    air = meta.rename(columns={"AIRPORT": "IATA"})[["IATA","CITY","STATE"]].dropna()
else:
    # Build CITY / STATE from available columns
    city_col  = None
    if "DISPLAY_AIRPORT_CITY_NAME_FULL" in meta.columns:
        city_col = meta["DISPLAY_AIRPORT_CITY_NAME_FULL"].astype(str).str.split(",").str[0]
    elif "DISPLAY_CITY_MARKET_NAME_FULL" in meta.columns:
        city_col = meta["DISPLAY_CITY_MARKET_NAME_FULL"].astype(str).str.split(",").str[0]
    else:
        city_col = pd.Series([""]*len(meta))

    state_col = meta["AIRPORT_STATE_CODE"] if "AIRPORT_STATE_CODE" in meta.columns else ""
    air = pd.DataFrame({
        "IATA":  meta["AIRPORT"],
        "CITY":  city_col,
        "STATE": state_col
    }).dropna(subset=["IATA"])

# ---------- 3) Build graphs ----------
print("Building directed graph ...")
G = nx.DiGraph()

# only keep airports that actually appear in flows (减少无用节点)
active_codes = set(flows["ORIGIN"]).union(set(flows["DEST"]))
air = air[air["IATA"].isin(active_codes)]

for _, r in air.iterrows():
    G.add_node(r["IATA"], city=r.get("CITY",""), state=r.get("STATE",""))

for _, r in flows.iterrows():
    o, d, w = r["ORIGIN"], r["DEST"], float(r["PASSENGERS"])
    if o in G and d in G and w > 0:
        if G.has_edge(o, d):
            G[o][d]["weight"] += w
        else:
            G.add_edge(o, d, weight=w)

print(f"Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

# undirected (for modularity)
UG = nx.Graph()
for _, r in flows.iterrows():
    u, v, w = r["ORIGIN"], r["DEST"], float(r["PASSENGERS"])
    if u in G and v in G and w > 0:
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)

# ================= (a) Communities =================
print("\n[a] Detecting communities (greedy modularity on undirected, weighted) ...")
comms = list(nx.algorithms.community.greedy_modularity_communities(UG, weight="weight"))
sizes = sorted([len(c) for c in comms], reverse=True)
print(f"  Found {len(comms)} communities.")
print("  Top community sizes:", sizes[:10])

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
k = min(20, len(sizes))
plt.bar(range(1, k+1), sizes[:k])
plt.xlabel("Community rank"); plt.ylabel("# of airports")
plt.title("Top community sizes (greedy modularity)")
plt.tight_layout()
plt.savefig(OUT/"a_top_community_sizes.png", dpi=220)
plt.close()

with open(OUT/"a_communities_summary.txt","w",encoding="utf-8") as f:
    f.write(f"Found {len(comms)} communities.\n")
    f.write("Top 10 sizes: " + ", ".join(map(str, sizes[:10])) + "\n")

# ================= (b) PageRank =================
print("\n[b] Computing PageRank ...")
pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200)
pr_values = np.array(list(pr.values()))
plot_ccdf(pr_values, "PageRank values (CCDF, log–log)", "b_pagerank_ccdf.png")

top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
print("  Top-10 PageRank airports:")
with open(OUT/"b_pagerank_top10.txt","w",encoding="utf-8") as f:
    for i, (a, val) in enumerate(top_pr, 1):
        meta = G.nodes[a]
        label = f"{a} ({meta.get('city','')}, {meta.get('state','')})"
        print(f"   {i:2d}. {label:30s}  PR={val:.5g}")
        f.write(f"{i:2d}. {label:30s}  PR={val:.6g}\n")

# ================= (c) HITS =================
print("\n[c] Computing HITS (hubs & authorities) ...")
hubs, auths = nx.hits(G, max_iter=300, normalized=True)
plot_ccdf(np.array(list(hubs.values()), dtype=float),  "Hub scores (CCDF, log–log)",       "c_hub_ccdf.png")
plot_ccdf(np.array(list(auths.values()), dtype=float), "Authority scores (CCDF, log–log)", "c_auth_ccdf.png")

top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]
top_auth = sorted(auths.items(), key=lambda x: x[1], reverse=True)[:10]

with open(OUT/"c_hits_top10.txt","w",encoding="utf-8") as f:
    f.write("Top-10 Hubs:\n")
    for i, (a, val) in enumerate(top_hubs, 1):
        meta = G.nodes[a]; label = f"{a} ({meta.get('city','')}, {meta.get('state','')})"
        print(f"   {i:2d}. {label:30s}  hub={val:.5g}")
        f.write(f"{i:2d}. {label:30s} hub={val:.6g}\n")
    f.write("\nTop-10 Authorities:\n")
    for i, (a, val) in enumerate(top_auth, 1):
        meta = G.nodes[a]; label = f"{a} ({meta.get('city','')}, {meta.get('state','')})"
        print(f"   {i:2d}. {label:30s}  auth={val:.5g}")
        f.write(f"{i:2d}. {label:30s} auth={val:.6g}\n")

print("\nSaved outputs to:", OUT.resolve())
