import pandas as pd, networkx as nx
import matplotlib.pyplot as plt

# 1) Load edges: a CSV with columns like ORIGIN, DEST (IATA codes)
df = pd.read_csv("us_air_traffic_edges.csv")  # <-- replace with provided file name
df = df[df["ORIGIN"].notna() & df["DEST"].notna()]

G = nx.DiGraph()
G.add_edges_from(df[["ORIGIN","DEST"]].itertuples(index=False, name=None))

# 2) Basic stats
N, M = G.number_of_nodes(), G.number_of_edges()
print(f"Nodes={N}, Edges={M}")

# 3) Components
wccs = list(nx.weakly_connected_components(G))
sccs = list(nx.strongly_connected_components(G))

wcc_sizes = sorted([len(c) for c in wccs], reverse=True)
scc_sizes = sorted([len(c) for c in sccs], reverse=True)

giant_wcc = wcc_sizes[0] if wcc_sizes else 0
giant_scc = scc_sizes[0] if scc_sizes else 0

print(f"WCCs={len(wccs)}, giant WCC={giant_wcc} ({giant_wcc/N:.1%})")
print(f"SCCs={len(sccs)}, giant SCC={giant_scc} ({giant_scc/N:.1%})")

# 4) Plot size distributions
plt.figure()
plt.hist([s for s in wcc_sizes if s < giant_wcc], bins=20)
plt.title("Weakly Connected Components (excluding giant)")
plt.xlabel("Component size"); plt.ylabel("Count"); plt.show()

plt.figure()
plt.hist([s for s in scc_sizes if s < giant_scc], bins=20)
plt.title("Strongly Connected Components (excluding giant)")
plt.xlabel("Component size"); plt.ylabel("Count"); plt.show()
