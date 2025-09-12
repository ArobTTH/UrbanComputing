

# geo_map_air_traffic.py  ——  US 航网地理可视化（用 Igismap）
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

# ---------- 0) 用课程 helper 定位路径 ----------
from utils import airtraffic_helpers as h
PKG = Path(h.__file__).resolve().parents[1]                  # -> hw1codes/
DATA = PKG / "dataset"
SHAPE_DIR = DATA / "Igismap"                                 # 这里放各州 shp
T100 = DATA / "288798530_T_T100D_MARKET_ALL_CARRIER.csv"
MASTER = DATA / "288804893_T_MASTER_CORD.csv"
OUT = Path("./q4_outputs"); OUT.mkdir(parents=True, exist_ok=True)

# ---------- 1) 底图：合并 Igismap 中所有州界 ----------
shps = sorted(SHAPE_DIR.glob("*_US_Poly.shp"))
if not shps:
    raise FileNotFoundError(f"No *_US_Poly.shp found in {SHAPE_DIR}")

states = gpd.GeoDataFrame(
    pd.concat([gpd.read_file(s) for s in shps], ignore_index=True),
    crs=gpd.read_file(shps[0]).crs
)
# 统一投影到北美常用的 Albers 等面积（线条更自然，便于叠加）
states = states.to_crs(epsg=5070)

# ---------- 2) 机场点：MASTER 中的经纬度 ----------
air = pd.read_csv(MASTER, usecols=["AIRPORT","LATITUDE","LONGITUDE"]).dropna()
g_air = gpd.GeoDataFrame(
    air, geometry=gpd.points_from_xy(air["LONGITUDE"], air["LATITUDE"]), crs="EPSG:4326"
).to_crs(states.crs)

# ---------- 3) 航线：聚合客流并选 Top-K ----------
K = 300   # 画多少条最繁忙航线（可调，避免太乱）
t100 = pd.read_csv(T100, usecols=["ORIGIN","DEST","PASSENGERS"]).dropna()
t100 = t100.groupby(["ORIGIN","DEST"], as_index=False)["PASSENGERS"].sum()
t100 = t100.sort_values("PASSENGERS", ascending=False).head(K)

# 为画线准备经纬度查找表
coord = air.set_index("AIRPORT")[["LONGITUDE","LATITUDE"]].dropna()

def make_line(row):
    if row["ORIGIN"] in coord.index and row["DEST"] in coord.index:
        x1,y1 = coord.loc[row["ORIGIN"], ["LONGITUDE","LATITUDE"]]
        x2,y2 = coord.loc[row["DEST"],   ["LONGITUDE","LATITUDE"]]
        return LineString([(x1,y1),(x2,y2)])
    return None

t100["geometry"] = t100.apply(make_line, axis=1)
t100 = t100.dropna(subset=["geometry"])
g_lines = gpd.GeoDataFrame(t100, geometry="geometry", crs="EPSG:4326").to_crs(states.crs)

# 线宽按客流归一，避免遮挡
wmax = g_lines["PASSENGERS"].max()
g_lines["lw"] = 0.2 + 2.8 * (g_lines["PASSENGERS"]/wmax)**0.5   # 开根号让差异更平滑

# ---------- 4) 作图 ----------
fig, ax = plt.subplots(figsize=(10, 7))
states.plot(ax=ax, facecolor="#f6f6f6", edgecolor="white", linewidth=0.4)
g_lines.plot(ax=ax, linewidth=g_lines["lw"], alpha=0.35, color="#2b6cb0")
g_air.plot(ax=ax, markersize=3, color="black", alpha=0.7)

ax.set_title(f"US Air Traffic — Top {K} Passenger Flows", fontsize=14)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(OUT/"q4_geo_topflows.png", dpi=250)
plt.show()

# ---------- 5)（可选）标注 Top-10 枢纽（按度数近似） ----------
# 用无向图近似识别度最高的机场，做小标签
import networkx as nx
Gu = nx.from_pandas_edgelist(t100, "ORIGIN", "DEST", create_using=nx.Graph())
top_hubs = [n for n,_ in sorted(Gu.degree(), key=lambda x:x[1], reverse=True)[:10]]
lab_pts = g_air[g_air["AIRPORT"].isin(top_hubs)]

fig, ax = plt.subplots(figsize=(10, 7))
states.plot(ax=ax, facecolor="#f6f6f6", edgecolor="white", linewidth=0.4)
g_lines.plot(ax=ax, linewidth=g_lines["lw"], alpha=0.35, color="#2b6cb0")
g_air.plot(ax=ax, markersize=2, color="black", alpha=0.6)
lab_pts.plot(ax=ax, markersize=14, color="#d9534f")  # 红点突出
for _, r in lab_pts.iterrows():
    ax.text(r.geometry.x, r.geometry.y, r["AIRPORT"], fontsize=8, ha="left", va="bottom")

ax.set_title(f"US Air Traffic — Top {K} Flows & Top Hubs", fontsize=14)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(OUT/"q4_geo_topflows_hubs.png", dpi=250)
plt.show()

print("Saved maps to:", (OUT/"q4_geo_topflows.png").resolve(), "and", (OUT/"q4_geo_topflows_hubs.png").resolve())
