import time
import heapq
from collections import deque

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from graph_data import graph as TEAM_GRAPH, locations as TEAM_LOCATIONS

from algorithms._common import make_admissible_heuristic
from algorithms.astar import astar_search
from algorithms.bfs import bfs_search
from algorithms.dfs import dfs_search
from algorithms.greedy import greedy_search
from algorithms.hill_climbing import hill_climbing_search
# ============================================================
# RouteIQ - Intelligent Route Finder  (v2 - illustrated map UI)
# Run:  streamlit run routeiq_ultra.py
# Needs: streamlit, networkx, pandas   (altair ships with streamlit)
# ============================================================

st.set_page_config(
    page_title="RouteIQ | Intelligent Route Finder",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def html(block):
    """Render raw HTML safely (strips indentation so markdown won't treat it as code)."""
    clean = "\n".join(l.strip() for l in block.splitlines() if l.strip())
    st.markdown(clean, unsafe_allow_html=True)


# ============================================================
# Styling
# ============================================================
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,1,0');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 15% 0%, #eaf2ff 0%, #f5f8fc 40%, #f5f8fc 100%); }
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1700px; transition: max-width .3s ease; }
header[data-testid="stHeader"] { background: transparent; }
.ms { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
      letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
      -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; vertical-align: middle; }

/* Hero */
.hero { position: relative; overflow: hidden; border-radius: 24px; padding: 30px 36px; color: #fff; margin-bottom: 20px;
  background: linear-gradient(120deg, #0b1f3a 0%, #123b70 45%, #1d7bc4 100%);
  box-shadow: 0 18px 40px rgba(16, 60, 110, .25); }
.hero::before { content:""; position:absolute; inset:0;
  background-image: linear-gradient(rgba(255,255,255,.06) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size: 34px 34px; mask-image: linear-gradient(90deg, transparent, #000 70%); }
.hero::after { content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px; border-radius:50%;
  background: radial-gradient(circle, rgba(56,189,248,.45), transparent 65%); }
.hero-row { position:relative; z-index:2; display:flex; align-items:center; gap:22px; }
.logo { width:68px; height:68px; border-radius:20px; background: rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.3); display:flex; align-items:center; justify-content:center; }
.hero h1 { margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-.5px; color:#fff; }
.hero h1 span { color:#7dd3fc; }
.hero p { margin:6px 0 0; color:#cfe3fb; font-size:1rem; }
.pills { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.pill { background: rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.22); padding:4px 12px;
  border-radius:999px; font-size:.78rem; font-weight:600; color:#e8f3ff; }

/* Stat cards */
.stat { display:flex; align-items:center; gap:14px; background:#fff; border:1px solid #e6edf6; border-radius:18px;
  padding:16px 18px; box-shadow:0 6px 18px rgba(20,60,110,.06); transition: transform .2s, box-shadow .2s; }
.stat:hover { transform: translateY(-3px); box-shadow:0 12px 26px rgba(20,60,110,.12); }
.stat .ico { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; color:#fff; flex-shrink:0; }
.stat .ico .ms { font-size:26px; }
.stat .lbl { color:#6b7c93; font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.6px; }
.stat .val { color:#0f2d52; font-size:1.3rem; font-weight:800; line-height:1.2; }

/* Section titles & panels */
.sec { display:flex; align-items:center; gap:10px; font-size:1.1rem; font-weight:750; color:#0f2d52; margin:6px 0 12px; }
.sec .ms { color:#1d7bc4; font-size:24px; }
.panel { background:#fff; border:1px solid #e6edf6; border-radius:20px; padding:18px 20px;
  box-shadow:0 8px 24px rgba(20,60,110,.06); margin-bottom:14px; }
.algo-badge { display:inline-flex; align-items:center; gap:8px; background:#eaf3ff; color:#145a99; font-weight:700;
  padding:6px 14px; border-radius:999px; font-size:.85rem; margin-bottom:14px; }
.mini { background:#f6f9fd; border:1px solid #e6edf6; border-radius:14px; padding:12px 14px; }
.mini .lbl { color:#6b7c93; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
.mini .val { color:#0f2d52; font-size:1.35rem; font-weight:800; }
.route-wrap { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-top:6px; }
.chip { padding:6px 12px; border-radius:10px; font-weight:700; font-size:.82rem; background:#e7f8f1; color:#0b7a55; border:1px solid #bdeedb; }
.chip.s { background:#dcfce7; color:#15803d; border-color:#86efac; }
.chip.g { background:#fee2e2; color:#b91c1c; border-color:#fca5a5; }
.chip.x { background:#fef3c7; color:#92400e; border-color:#fcd34d; }
.arrow { color:#94a3b8; font-size:18px; }
.empty { text-align:center; padding:26px 10px; }
.empty .ms { font-size:54px; color:#93c5fd; }
.empty h4 { margin:6px 0; color:#0f2d52; }
.empty p { color:#6b7c93; font-size:.92rem; margin:4px 0; }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin-top:10px; font-size:.82rem; color:#51637a; font-weight:600; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
.algo-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:12px; }
.algo-card { background:#f6f9fd; border:1px solid #e6edf6; border-radius:14px; padding:14px 16px; }
.algo-card b { color:#0f2d52; }
.algo-card p { margin:6px 0 0; color:#5a6b82; font-size:.86rem; }
.footer { text-align:center; color:#8795a8; font-size:.82rem; padding-top:24px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1c36 0%, #11305a 100%); }
section[data-testid="stSidebar"] * { color: #dbe9fb; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#fff !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input { color:#06152b !important; -webkit-text-fill-color:#06152b !important; background-color:transparent !important; background-image:none !important; opacity:1 !important; font-weight:600; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div { background-color:#9fd0ff !important; background-image:linear-gradient(135deg,#b9dcff,#8cc4fb) !important; border:1px solid #5eb0f5 !important; border-radius:12px; transition: border-color .2s, box-shadow .2s; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover { border-color:#38bdf8 !important; box-shadow:0 0 0 3px rgba(56,189,248,.25); }
section[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill:#0b2a4d !important; }
div[data-baseweb="popover"] ul { background:#e8f3ff !important; border-radius:12px; }
div[data-baseweb="popover"] li { color:#06152b !important; font-weight:600; transition: background .15s; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background:#bfe0ff !important; }

div.stButton > button { border-radius:12px; font-weight:700; min-height:2.7rem; transition: all .2s; }
div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #1d7bc4, #22b8cf); border:none; color:#fff; box-shadow:0 8px 18px rgba(29,123,196,.35); }
div.stButton > button[kind="primary"]:hover { transform: translateY(-2px); }
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] { background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.22); color:#fff; }
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover { background: rgba(255,255,255,.16); }

/* ---- Motion & polish ---- */
html { scroll-behavior: smooth; }
#MainMenu, footer { visibility: hidden; }
@keyframes fadeUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform:none; } }
@keyframes drift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.stat, .panel, .sec, .callout { animation: fadeUp .6s cubic-bezier(.2,.8,.2,1) both; }
.hero { background-size: 170% 170%; animation: fadeUp .6s cubic-bezier(.2,.8,.2,1) both, drift 14s ease-in-out infinite; }
.panel { transition: transform .25s, box-shadow .25s; }
.panel:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(20,60,110,.1); }
.mini { transition: transform .2s, background .2s; } .mini:hover { transform: translateY(-2px); background:#eef5fd; }
.chip { transition: transform .2s; } .chip:hover { transform: translateY(-2px) scale(1.05); }
div.stButton > button:active { transform: scale(.97); }
div[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; box-shadow:0 8px 24px rgba(20,60,110,.08); }
.tl { position:relative; margin:10px 0 0 6px; padding-left:24px; border-left:2px dashed #cfe0f3; }
.tl-item { position:relative; padding:8px 0; animation: fadeUp .5s cubic-bezier(.2,.8,.2,1) both; color:#0f2d52; }
.tl-item::before { content:""; position:absolute; left:-31px; top:14px; width:12px; height:12px; border-radius:50%;
  background:#10b981; border:3px solid #fff; box-shadow:0 0 0 2px #10b981; }
.tl-item span { color:#6b7c93; font-size:.82rem; }
.tl-cost { float:right; background:#e7f8f1; color:#0b7a55 !important; font-weight:800; border-radius:8px; padding:2px 10px; font-size:.8rem !important; }
.callout { display:flex; gap:12px; align-items:flex-start; background:linear-gradient(120deg,#eaf3ff,#f0fbf6);
  border:1px solid #cfe3f7; border-radius:16px; padding:14px 18px; color:#0f2d52; font-size:.93rem; margin-bottom:14px; }
.callout .ms { color:#1d7bc4; font-size:26px; }

/* ---- Sidebar v3: clean, grouped, smooth ---- */
section[data-testid="stSidebar"][aria-expanded="true"] { min-width: 330px; width: 330px; box-shadow: 6px 0 30px rgba(10,28,54,.18); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: .6rem 1.2rem 2rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { height: 2.2rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .6rem; }
section[data-testid="stSidebar"] label p { font-size: .84rem !important; font-weight: 600 !important; color: #e6f0fc !important; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div { transition: transform .2s; }
section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] { background:#38bdf8 !important; box-shadow:0 0 0 4px rgba(56,189,248,.25) !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1); }
.sb-brand { display:flex; align-items:center; gap:12px; padding:4px 0 16px; border-bottom:1px solid rgba(255,255,255,.1); margin-bottom:6px; animation: fadeUp .5s both; }
.sb-logo { width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#1d7bc4,#22b8cf); box-shadow:0 8px 18px rgba(34,184,207,.35); }
.sb-brand b { color:#fff !important; font-size:1.15rem; letter-spacing:-.3px; }
.sb-brand .sub { display:block; font-size:.74rem; color:#8fb2d9 !important; }
.sb-label { display:flex; align-items:center; gap:8px; font-size:.68rem; font-weight:800; letter-spacing:1.3px;
  text-transform:uppercase; color:#7fa6d4 !important; margin:18px 0 2px; }
.sb-label .ms { font-size:17px; color:#38bdf8 !important; }
.sb-note { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1); border-radius:12px;
  padding:10px 12px; font-size:.78rem; line-height:1.5; color:#b9d2ee !important; transition: background .2s; }
.sb-note:hover { background: rgba(255,255,255,.1); }
.st-key-swap_btn button { min-height:2rem !important; font-size:.78rem !important; font-weight:600 !important;
  background:transparent !important; border:1px dashed rgba(255,255,255,.3) !important; color:#9fc3ea !important; }
.st-key-swap_btn button:hover { border-color:#38bdf8 !important; color:#fff !important; background:rgba(56,189,248,.1) !important; }
.st-key-reset_btn button { background:transparent !important; border:none !important; color:#8fb2d9 !important; min-height:2rem !important; font-weight:600 !important; }
.st-key-reset_btn button:hover { color:#fff !important; background:rgba(255,255,255,.07) !important; }
/* Fix selected dropdown text */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #0f2d52 !important;
    -webkit-text-fill-color: #0f2d52 !important;
    opacity: 1 !important;
}

/* Fix dropdown background and border */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

/* Remove the unwanted red error-like border */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
}

/* Make dropdown arrows dark and visible */
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #0f2d52 !important;
    color: #0f2d52 !important;
}

/* Make dropdown menu text readable */
div[data-baseweb="popover"] li {
    color: #0f2d52 !important;
    background: #ffffff !important;
}

/* Highlight the selected menu option */
div[data-baseweb="popover"] li[aria-selected="true"] {
    background: #eaf3ff !important;
    color: #0f2d52 !important;
}
/* Readable selected values in sidebar dropdowns */
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #0f2d52 !important;
    -webkit-text-fill-color: #0f2d52 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* White dropdown fields with consistent borders */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

/* Visible dropdown arrows */
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #0f2d52 !important;
    fill: #0f2d52 !important;
}

/* Clean Swap button */
.st-key-swap_btn button {
    border: 1px solid #536b86 !important;
    border-radius: 12px !important;
    color: #e2edfa !important;
    background: #203b5c !important;
    transition: all 0.2s ease !important;
}

.st-key-swap_btn button:hover {
    background: #294d73 !important;
    border-color: #38bdf8 !important;
}

/* Consistent sidebar spacing */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    margin-bottom: 8px;
}

section[data-testid="stSidebar"] hr {
    margin: 16px 0;
}
</style>
""")


# ============================================================
# Sample map data (replace with team graph module when ready)
# ============================================================

# ============================================================
# Team graph data
# ============================================================
POSITIONS = {
    node: (details["x"] / 100, (400 - details["y"]) / 100)
    for node, details in TEAM_LOCATIONS.items()
}

G = nx.Graph()

for node, neighbors in TEAM_GRAPH.items():
    G.add_node(node)

    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

HEURISTIC = make_admissible_heuristic(TEAM_GRAPH)

# Display-friendly location names
LOCATION_NAMES = {
    node: details["name"]
    for node, details in TEAM_LOCATIONS.items()
}

# ============================================================
# Search algorithms (unchanged logic)
# ============================================================

# ============================================================
# Team search algorithms
# ============================================================

ALGORITHMS = {
    "Breadth-First Search (BFS)": bfs_search,
    "Depth-First Search (DFS)": dfs_search,
    "Greedy Best-First Search": greedy_search,
    "A* Search": astar_search,
    "Hill Climbing": hill_climbing_search,
}


def run_algorithm(name, start, goal):
    t0 = time.perf_counter()
    search = ALGORITHMS[name]

    # Greedy, A*, and Hill Climbing require a heuristic.
    if name in (
        "Greedy Best-First Search",
        "A* Search",
        "Hill Climbing",
    ):
        result = search(TEAM_GRAPH, start, goal, HEURISTIC)
    else:
        result = search(TEAM_GRAPH, start, goal)

    path = result["path"]
    found = bool(path) and path[0] == start and path[-1] == goal

    return {
        "found": found,
        "path": path,
        "cost": result["cost"],
        "edges": max(len(path) - 1, 0) if found else 0,
        "nodes_explored": result["nodes_explored"],
        "explored": [],
        "time_ms": (time.perf_counter() - t0) * 1000,
        "algorithm": name,
    }


# ============================================================
# Illustrated SVG map
# ============================================================
def R(x, y, w, h, f, s=None, rx=0, sw=1.2):
    st_ = f' stroke="{s}" stroke-width="{sw}"' if s else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}"{st_}/>'


def C(cx, cy, r, f, s=None, sw=1.2):
    st_ = f' stroke="{s}" stroke-width="{sw}"' if s else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{f}"{st_}/>'


def E(cx, cy, rx, ry, f, s=None, sw=1.2):
    st_ = f' stroke="{s}" stroke-width="{sw}"' if s else ""
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{f}"{st_}/>'


def P(pts, f, s=None, sw=1.2):
    st_ = f' stroke="{s}" stroke-width="{sw}" stroke-linejoin="round"' if s else ""
    return f'<polygon points="{pts}" fill="{f}"{st_}/>'


def L(x1, y1, x2, y2, s, sw=1.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{s}" stroke-width="{sw}" stroke-linecap="round"/>'


def icon_college():
    return (
        R(-36, -2, 72, 32, "#fde68a", "#b45309", 2) + R(-17, -22, 34, 52, "#fef3c7", "#b45309", 2)
        + P("-23,-22 0,-40 23,-22", "#b91c1c", "#7f1d1d") + C(0, -28, 5, "#fff", "#7f1d1d")
        + "".join(R(-13 + i * 9, -14, 4, 40, "#fff", "#d6a35c", 1, .8) for i in range(4))
        + R(-5, 12, 10, 18, "#92400e", rx=2)
        + "".join(R(x, 8, 5, 9, "#60a5fa", "#1d4ed8", 1, .8) for x in (-31, -23, 20, 28))
        + L(0, -40, 0, -52, "#7f1d1d") + P("0,-52 12,-48 0,-44", "#ef4444")
    )


def icon_library():
    return (
        R(-34, -8, 68, 36, "#ddd6fe", "#6d28d9", 2) + P("-40,-8 0,-34 40,-8", "#7c3aed", "#4c1d95")
        + R(-9, -22, 18, 11, "#fff", "#4c1d95", 2) + L(0, -21, 0, -12, "#7c3aed", 1.2)
        + "".join(R(x, -2, 8, 18, "#c4b5fd", "#5b21b6", 4, .9) for x in (-28, -14, 6, 20))
        + R(-5, 10, 10, 18, "#4c1d95", rx=2) + R(-38, 28, 76, 5, "#a78bfa", "#5b21b6", 1)
    )


def icon_hospital():
    return (
        R(-32, -22, 64, 52, "#ffffff", "#94a3b8", 3) + R(-32, -22, 64, 8, "#e2e8f0", "#94a3b8", 3)
        + R(-4, -12, 8, 22, "#ef4444", rx=1) + R(-11, -5, 22, 8, "#ef4444", rx=1)
        + "".join(R(x, 14, 7, 7, "#7dd3fc", "#0284c7", 1, .8) for x in (-26, -16, 9, 19))
        + R(-6, 16, 12, 14, "#38bdf8", "#0369a1", 2) + R(-14, -34, 28, 12, "#fef2f2", "#ef4444", 2)
        + '<text x="0" y="-25" text-anchor="middle" font-size="9" font-weight="800" fill="#ef4444">H</text>'
    )


def icon_park():
    def tree(x, y, r, c):
        return R(x - 2, y, 4, r + 4, "#92400e") + C(x, y, r, c, "#15803d", 1)
    return (
        E(0, 18, 36, 14, "#bbf7d0", "#4ade80") + E(-14, 20, 14, 6, "#7dd3fc", "#0ea5e9", 1)
        + tree(-20, -8, 14, "#22c55e") + tree(16, -14, 12, "#16a34a") + tree(6, 4, 13, "#4ade80")
        + R(14, 18, 20, 4, "#92400e", rx=1) + R(15, 22, 3, 5, "#78350f") + R(31, 22, 3, 5, "#78350f")
    )


def icon_station():
    return (
        R(-36, -4, 72, 34, "#e0f2fe", "#0369a1", 2) + R(-40, -12, 80, 10, "#dc2626", "#7f1d1d", 2)
        + R(-9, -38, 18, 28, "#bae6fd", "#0369a1", 2) + P("-11,-38 0,-48 11,-38", "#dc2626", "#7f1d1d")
        + C(0, -27, 6, "#fff", "#0369a1") + L(0, -27, 0, -31, "#0f172a", 1.2) + L(0, -27, 3, -27, "#0f172a", 1.2)
        + "".join(R(x, 6, 10, 16, "#38bdf8", "#0369a1", 5, 1) for x in (-28, -14, 4, 18))
        + L(-38, 34, 38, 34, "#64748b", 2) + L(-38, 38, 38, 38, "#64748b", 2)
    )


def icon_market():
    stripes = "".join(R(-36 + i * 14.4, -26, 14.4, 20, "#ef4444" if i % 2 == 0 else "#ffffff") for i in range(5))
    scallops = "".join(C(-28.8 + i * 14.4, -6, 7.2, "#ef4444" if i % 2 == 0 else "#ffffff") for i in range(5))
    return (
        R(-32, -6, 64, 36, "#fde68a", "#b45309", 2) + stripes + scallops
        + L(-36, -26, 36, -26, "#7f1d1d", 1.5)
        + R(-32, 12, 64, 6, "#b45309", rx=1)
        + C(-20, 8, 5, "#ef4444") + C(-8, 8, 5, "#f59e0b") + C(4, 8, 5, "#84cc16") + C(16, 8, 5, "#a855f7")
        + R(-14, -36, 28, 9, "#fff", "#b91c1c", 3)
        + '<text x="0" y="-29" text-anchor="middle" font-size="7" font-weight="800" fill="#b91c1c">MARKET</text>'
    )


def icon_airport():
    return (
        R(-38, 8, 76, 20, "#e2e8f0", "#64748b", 3) + "".join(R(x, 13, 8, 6, "#60a5fa", None, 1) for x in range(-32, 34, 12))
        + R(-5, -26, 10, 36, "#cbd5e1", "#64748b", 1) + R(-12, -38, 24, 12, "#7dd3fc", "#0369a1", 3)
        + L(0, -38, 0, -48, "#64748b") + C(0, -49, 2, "#ef4444")
        + '<g transform="translate(24,-18) rotate(-35)">'
        + E(0, 0, 14, 3.5, "#fff", "#475569", 1) + P("-2,0 -9,-12 -5,-12 4,0", "#94a3b8", "#475569", .8)
        + P("-2,0 -9,12 -5,12 4,0", "#94a3b8", "#475569", .8) + P("-12,0 -16,-6 -13,-6 -9,0", "#94a3b8", "#475569", .8) + "</g>"
    )


def icon_museum():
    return (
        P("-40,-14 0,-38 40,-14", "#fde68a", "#b45309") + R(-40, -14, 80, 7, "#fcd34d", "#b45309", 1)
        + C(0, -20, 4, "#b45309") + "".join(R(-32 + i * 16, -6, 7, 30, "#fff7ed", "#d6a35c", 1) for i in range(5))
        + R(-42, 24, 84, 5, "#e7d3a8", "#b45309", 1) + R(-46, 29, 92, 5, "#d6bd86", "#b45309", 1)
    )


def icon_stadium():
    lights = "".join(L(x, y, x, y - 14, "#475569", 2) + R(x - 4, y - 18, 8, 5, "#fde047", "#a16207", 1, .8)
                     for x, y in ((-34, -12), (34, -12), (-34, 14), (34, 14)))
    return (
        E(0, 4, 42, 30, "#e2e8f0", "#64748b", 2) + E(0, 4, 32, 21, "#cbd5e1", "#94a3b8", 1)
        + E(0, 4, 26, 16, "#22c55e", "#15803d", 1.5)
        + L(0, -12, 0, 20, "#bbf7d0", 1.5) + C(0, 4, 6, "none", "#bbf7d0", 1.5) + lights
    )


ICONS = {
    "A": icon_college,   # College Gate
    "B": icon_library,   # Library
    "C": icon_park,      # Main Road
    "D": icon_hospital,  # Canteen
    "E": icon_station,   # Bus Stop
    "F": icon_stadium,  # Hostel
}

W, H = 960, 620




def px(name):
    # Center the complete graph within the SVG map.
    x, y = POSITIONS[name]

    min_x = min(pos[0] for pos in POSITIONS.values())
    max_x = max(pos[0] for pos in POSITIONS.values())
    min_y = min(pos[1] for pos in POSITIONS.values())
    max_y = max(pos[1] for pos in POSITIONS.values())

    graph_width = max_x - min_x or 1
    graph_height = max_y - min_y or 1

    available_width = W - 240
    available_height = H - 220

    scale = min(
        available_width / graph_width,
        available_height / graph_height,
    )

    graph_pixel_width = graph_width * scale
    graph_pixel_height = graph_height * scale

    offset_x = (W - graph_pixel_width) / 2
    offset_y = (H - graph_pixel_height) / 2

    return (
        offset_x + (x - min_x) * scale,
        offset_y + (max_y - y) * scale,
    )




def build_map_html(start, goal, result=None, speed=1.0, show_order=True):
    path = result["path"] if result and result["found"] else []
    explored = result["explored"] if result else []
    path_edges = {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}
    step = 0.45 / speed  # seconds between explored nodes appearing
    explore_end = len(explored) * step

    s = []
    s.append('<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eaf6ec"/>'
             '<stop offset="1" stop-color="#e3f0fa"/></linearGradient>'
             '<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">'
             '<path d="M40 0H0V40" fill="none" stroke="#d3e4d6" stroke-width="1"/></pattern>'
             '<filter id="sh" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#0f2d52" flood-opacity=".25"/></filter>'
             '<filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    s.append(f'<rect width="{W}" height="{H}" rx="22" fill="url(#bg)"/><rect width="{W}" height="{H}" rx="22" fill="url(#grid)"/>')
    river = "M-10,330 C180,430 360,240 560,325 S850,430 980,350"
    s.append(f'<path d="{river}" fill="none" stroke="#a9d6f2" stroke-width="34" stroke-linecap="round"/>')
    s.append(f'<path d="{river}" fill="none" stroke="#cfeafb" stroke-width="22" stroke-linecap="round" stroke-dasharray="2 14"/>')
    for tx, ty in ((40, 40), (125, 590), (905, 70), (915, 585), (470, 36), (300, 598), (720, 592), (35, 300), (935, 300), (330, 230), (760, 440)):
        s.append(R(tx - 2, ty, 4, 9, "#92400e") + C(tx, ty - 2, 11, "#4ade80", "#16a34a", 1))

    # roads
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        (x1, y1), (x2, y2) = px(u), px(v)
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#94a3b8" stroke-width="22" stroke-linecap="round"/>')
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="17" stroke-linecap="round"/>')
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#fde68a" stroke-width="2" stroke-dasharray="10 9"/>')

    # selected route overlay (draws itself, synced with the car)
    path_d = "M" + " L".join(f"{px(n)[0]},{px(n)[1]}" for n in path) if path else ""
    dur = max(len(path) - 1, 1) * 1.1 / speed
    t0 = explore_end + 0.4
    if path:
        s.append(
            f'<path d="{path_d}" pathLength="1" fill="none" stroke="#10b981" stroke-width="12" stroke-linecap="round" stroke-linejoin="round" opacity=".6" filter="url(#glow)" class="draw" style="animation-duration:{dur:.2f}s;animation-delay:{t0:.2f}s"/>'
            f'<g class="fade" style="animation-delay:{t0 + dur:.2f}s"><path d="{path_d}" fill="none" stroke="#a7f3d0" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="14 10" class="flow"/></g>')

    # weight badges
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        (x1, y1), (x2, y2) = px(u), px(v)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        on = frozenset((u, v)) in path_edges
        fill = "#10b981" if on else "#ffffff"
        stroke = "#047857" if on else "#94a3b8"
        tcol = "#fff" if on else "#334155"
        s.append(f'<g transform="translate({mx:.1f},{my:.1f})">'
                 f'<rect x="-14" y="-10" width="28" height="20" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.2" filter="url(#sh)"/>'
                 f'<text y="4.5" text-anchor="middle" font-size="12" font-weight="800" fill="{tcol}">{w}</text></g>')

    # nodes (buildings)
    for idx, name in enumerate(G.nodes()):
        x, y = px(name)
        if name == start:
            ring, fill = "#16a34a", "#dcfce7"
        elif name == goal:
            ring, fill = "#dc2626", "#fee2e2"
        elif name in path:
            ring, fill = "#10b981", "#d1fae5"
        elif name in explored:
            ring, fill = "#f59e0b", "#fef3c7"
        else:
            ring, fill = "#60a5fa", "#ffffff"
        pulse = ""
        if name in (start, goal):
            pulse = f'<circle r="50" fill="none" stroke="{ring}" stroke-width="3" class="pulse"/>'
        ripple = ""
        if name in explored:
            ripple = f'<circle r="50" fill="none" stroke="#f59e0b" stroke-width="4" class="ripple" style="animation-delay:{explored.index(name) * step:.2f}s"/>'
        s.append(
                f'<g transform="translate({x:.1f},{y:.1f})">'
                f'<g class="node" style="animation-delay:{idx * 0.07:.2f}s">'
                f'<title>{LOCATION_NAMES.get(name, name)}</title>{pulse}{ripple}'
                 f'<circle r="50" fill="{fill}" stroke="{ring}" stroke-width="4" filter="url(#sh)"/>'
                 f'<g transform="scale(.9) translate(0,2)">{ICONS[name]()}</g>')
                # Display the actual location name instead of A, B, C...
        display_name = LOCATION_NAMES.get(name, name)

        # Keep long labels readable
        label = display_name if len(display_name) <= 16 else display_name[:15] + "…"

        # Calculate label width from the displayed name
        lw = max(len(label) * 7.5 + 20, 70)

        s.append(
            f'<rect x="{-lw / 2:.1f}" y="54" '
            f'width="{lw:.1f}" height="24" rx="11" fill="#0f2d52"/>'
            f'<text y="69" text-anchor="middle" font-size="10.5" '
            f'font-weight="700" fill="#fff">{label}</text>'
        )

        if name in (start, goal):
            tag, col = ("START", "#16a34a") if name == start else ("DEST", "#dc2626")
            s.append(
                f'<g transform="translate(0,-66)">'
                f'<rect x="-26" y="-11" width="52" height="22" rx="11" fill="{col}"/>'
                f'<polygon points="-5,10 5,10 0,17" fill="{col}"/>'
                f'<text y="4" text-anchor="middle" font-size="10" '
                f'font-weight="800" fill="#fff">{tag}</text></g>'
            )

        if show_order and name in explored:
            i = explored.index(name) + 1
            s.append(
                f'<g transform="translate(36,-36)">'
                f'<g class="pop" style="animation-delay:{(i - 1) * step:.2f}s">'
                f'<circle r="11" fill="#f59e0b" stroke="#fff" stroke-width="2"/>'
                f'<text y="4" text-anchor="middle" font-size="11" '
                f'font-weight="800" fill="#fff">{i}</text></g></g>'
            )

        s.append("</g></g>")

    # moving car + arrival burst
    if path:
        gx, gy = px(goal)
        begin = f"{t0:.2f}s"
        s.append(
            f'<g transform="translate({gx:.1f},{gy:.1f})"><circle r="50" fill="none" stroke="#10b981" stroke-width="5" class="ripple" style="animation-delay:{t0 + dur:.2f}s"/></g>'
            f'<g opacity="0"><set attributeName="opacity" to="1" begin="{begin}" fill="freeze"/>'
            f'<animateMotion dur="{dur:.2f}s" begin="{begin}" fill="freeze" rotate="auto" path="{path_d}" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines=".4 0 .2 1"/>'
            + R(-13, -7, 26, 14, "#f97316", "#9a3412", 5) + R(0, -5, 8, 10, "#bae6fd", None, 2)
            + C(-7, -8, 3, "#111827") + C(7, -8, 3, "#111827") + C(-7, 8, 3, "#111827") + C(7, 8, 3, "#111827")
            + "</g>"
        )

    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700;800&display=swap');
    body{margin:0;background:transparent}
    svg{display:block;width:100%;height:auto;font-family:Inter,system-ui,sans-serif}
    .node{animation:rise .7s cubic-bezier(.2,.8,.2,1) both;transform-box:fill-box;transform-origin:center;transition:transform .25s cubic-bezier(.2,.8,.2,1);cursor:pointer}
    .node:hover{transform:scale(1.08)}
    @keyframes rise{from{opacity:0;translate:0 16px}to{opacity:1;translate:0 0}}
    .pop{opacity:0;animation:pop .5s cubic-bezier(.3,1.5,.5,1) forwards;transform-box:fill-box;transform-origin:center}
    @keyframes pop{from{opacity:0;transform:scale(.2)}to{opacity:1;transform:scale(1)}}
    .fade{opacity:0;animation:fade .7s ease forwards}
    @keyframes fade{to{opacity:1}}
    .draw{stroke-dasharray:1;stroke-dashoffset:1;animation:draw linear forwards;animation-timing-function:cubic-bezier(.4,0,.2,1)}
    @keyframes draw{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}
    .flow{animation:flow .8s linear infinite} @keyframes flow{to{stroke-dashoffset:-24}}
    .pulse{animation:pulse 1.8s ease-out infinite;transform-box:fill-box;transform-origin:center}
    @keyframes pulse{0%{opacity:.7;transform:scale(.95)}100%{opacity:0;transform:scale(1.55)}}
    .ripple{opacity:0;animation:ripple 1s ease-out forwards;transform-box:fill-box;transform-origin:center}
    @keyframes ripple{0%{opacity:.9;transform:scale(.9)}100%{opacity:0;transform:scale(1.7)}}
    #rp{position:absolute;top:12px;right:12px;z-index:5;border:0;border-radius:999px;padding:8px 16px;font:700 12px Inter,sans-serif;
        background:rgba(15,45,82,.92);color:#fff;cursor:pointer;box-shadow:0 6px 16px rgba(15,45,82,.3);transition:transform .2s,background .2s}
    #rp:hover{transform:translateY(-2px);background:#1d7bc4}
    </style>"""
    svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{"".join(s)}</svg>'
    page = (css + '<div style="position:relative"><button id="rp" onclick="rp()">&#8635; Replay animation</button>'
            '<div id="m">__SVG__</div></div>'
            "<script>function rp(){var m=document.getElementById('m');var h=m.innerHTML;m.innerHTML='';void m.offsetWidth;m.innerHTML=h;}function fit(){try{window.frameElement.style.height=Math.ceil(document.getElementById('m').getBoundingClientRect().height)+'px';}catch(e){}}window.addEventListener('load',fit);window.addEventListener('resize',fit);if(window.ResizeObserver){new ResizeObserver(fit).observe(document.getElementById('m'));}fit();</script>")
    return page.replace("__SVG__", svg)


# ============================================================
# UI helpers
# ============================================================
def stat_card(icon, label, value, grad):
    return (f'<div class="stat"><div class="ico" style="background:{grad}"><span class="ms">{icon}</span></div>'
            f'<div><div class="lbl">{label}</div><div class="val">{value}</div></div></div>')


def mini(label, value):
    return f'<div class="mini"><div class="lbl">{label}</div><div class="val">{value}</div></div>'


def route_chips(path, explored_style=False):
    out = []

    for i, node in enumerate(path):
        display_name = LOCATION_NAMES.get(node, node)

        if explored_style:
            cls = "x"
        else:
            cls = "s" if i == 0 else ("g" if i == len(path) - 1 else "")

        out.append(f'<span class="chip {cls}">{display_name}</span>')

        if i < len(path) - 1:
            out.append('<span class="ms arrow">arrow_forward</span>')

    return '<div class="route-wrap">' + "".join(out) + "</div>"


ALGO_NOTES = {
    "Breadth-First Search (BFS)": "<b>BFS</b> fans out level by level, so it finds the route with the <b>fewest road segments</b>. It ignores road costs, so it may not be the cheapest.",
    "Depth-First Search (DFS)": "<b>DFS</b> commits to one direction and backtracks only when stuck. It finds <i>a</i> route quickly, but rarely the best one.",
    "Greedy Best-First Search": "<b>Greedy</b> always heads towards whichever neighbour looks closest to the goal. It is fast, but it ignores the cost already spent.",
    "A* Search": "<b>A*</b> balances the cost travelled so far with an estimate of what remains, which makes it the smartest all-rounder on this map.",
    "Hill Climbing": "<b>Hill Climbing</b> only ever steps to a neighbour that is closer to the goal. It is very light, but can get stuck on a dead end.",
}


def story_html(result):
    path = result["path"]
    running_cost = 0
    items = []

    for i, (a, b) in enumerate(zip(path, path[1:])):
        weight = G[a][b]["weight"]
        running_cost += weight

        from_name = LOCATION_NAMES.get(a, a)
        to_name = LOCATION_NAMES.get(b, b)

        items.append(
            f'<div class="tl-item" style="animation-delay:{i * 0.12:.2f}s">'
            f'<span class="tl-cost">+{weight}</span>'
            f'<b>{from_name} → {to_name}</b><br>'
            f'<span>Running total: {running_cost}</span></div>'
        )

    return (
        '<div class="panel">'
        '<div style="font-weight:700;color:#0f2d52;font-size:.9rem">'
        'Journey breakdown</div>'
        f'<div class="tl">{"".join(items)}</div></div>'
    )


PALETTE = ["#1d7bc4", "#22b8cf", "#8b5cf6", "#10b981", "#f59e0b"]


def bar_chart(df, col):
    return (
        alt.Chart(df).mark_bar(cornerRadiusEnd=8, height=26)
        .encode(
            y=alt.Y("Algorithm:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X(f"{col}:Q", title=None),
            color=alt.Color("Algorithm:N", legend=None, scale=alt.Scale(range=PALETTE)),
            tooltip=["Algorithm", col],
        ).properties(height=230).configure_view(strokeWidth=0)
    )


# ============================================================
# Session state
# ============================================================
st.session_state.setdefault("route_result", None)
st.session_state.setdefault("comparison_results", None)
st.session_state.setdefault("start_sel", "A")
st.session_state.setdefault("dest_sel", "F")
st.session_state.setdefault("last_key", None)


def reset_all():
    st.session_state.route_result = None
    st.session_state.comparison_results = None
    st.session_state.last_key = None
    st.session_state.live_mode = False


def swap_locations():
    st.session_state.start_sel, st.session_state.dest_sel = st.session_state.dest_sel, st.session_state.start_sel
    st.session_state.route_result = None
    st.session_state.comparison_results = None


# ============================================================
# Hero
# ============================================================
html("""
<div class="hero"><div class="hero-row">
  <div class="logo">
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
      <circle cx="20" cy="20" r="17" stroke="#fff" stroke-width="2.5" opacity=".9"/>
      <path d="M12 27 L17 17 L23 22 L29 11" stroke="#7dd3fc" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="12" cy="27" r="3.2" fill="#4ade80"/><circle cx="29" cy="11" r="3.2" fill="#f87171"/>
    </svg>
  </div>
  <div>
    <h1>Route<span>IQ</span></h1>
    <p>Intelligent Route Finder &mdash; watch AI search algorithms think, explore and drive through a living city map.</p>
    <div class="pills"><span class="pill">BFS</span><span class="pill">DFS</span><span class="pill">Greedy</span><span class="pill">A*</span><span class="pill">Hill Climbing</span></div>
  </div>
</div></div>
""")

# ============================================================
# Sidebar
# ============================================================
ALGO_SHORT = {
    "Breadth-First Search (BFS)": "Explores level by level. Fewest road segments.",
    "Depth-First Search (DFS)": "Dives deep into one branch before backtracking.",
    "Greedy Best-First Search": "Heads for whichever neighbour looks closest to the goal.",
    "A* Search": "Balances cost so far with the distance still to go.",
    "Hill Climbing": "Only steps closer to the goal. Can get stuck.",
}

with st.sidebar:
    html("""
    <div class="sb-brand">
      <div class="sb-logo"><svg width="26" height="26" viewBox="0 0 40 40" fill="none">
        <circle cx="20" cy="20" r="16" stroke="#fff" stroke-width="2.6"/>
        <path d="M12 27 L17 17 L23 22 L29 11" stroke="#fff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg></div>
      <div><b>RouteIQ</b><span class="sub">Intelligent Route Finder</span></div>
    </div>
    <div class="sb-label"><span class="ms">route</span>Journey</div>
    """)
    locations = sorted(G.nodes())

    start = st.selectbox(
        "From",
        locations,
        format_func=lambda node: LOCATION_NAMES.get(node, node),
        key="start_sel"
    )

    dest_options = [node for node in locations if node != start]

    if st.session_state.dest_sel not in dest_options:
        st.session_state.dest_sel = dest_options[0]

    st.button(
        "⇅  Swap",
        key="swap_btn",
        on_click=swap_locations,
        use_container_width=True
    )

    destination = st.selectbox(
        "To",
        dest_options,
        format_func=lambda node: LOCATION_NAMES.get(node, node),
        key="dest_sel"
    )

    html('<div class="sb-label"><span class="ms">psychology</span>Strategy</div>')
    algorithm_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()), label_visibility="collapsed")
    html(f'<div class="sb-note" >{ALGO_SHORT[algorithm_name]}</div>')

    html('<div class="sb-label"><span class="ms">tune</span>Playback</div>')
    speed = float(st.select_slider("Animation speed", options=["0.5×", "1×", "1.5×", "2×"], value="1×").replace("×", ""))
    show_order = st.toggle("Show exploration order", value=True)
    live = st.toggle("Live mode", value=True, key="live_mode", help="Re-run automatically whenever you change a setting.")

    html('<div class="sb-label"><span class="ms">bolt</span>Actions</div>')
    find_clicked = st.button("Find Route", type="primary", use_container_width=True, icon=":material/route:")
    compare_clicked = st.button("Compare all algorithms", use_container_width=True, icon=":material/leaderboard:")
    st.button("Reset", key="reset_btn", on_click=reset_all, use_container_width=True, icon=":material/restart_alt:")
    html('<div class="sb-note">Road badges show illustrative travel costs. Hover a building to see its name.</div>')

key = (start, destination, algorithm_name)
if find_clicked or (live and not compare_clicked and st.session_state.last_key != key):
    st.session_state.route_result = run_algorithm(algorithm_name, start, destination)
    st.session_state.comparison_results = None
    st.session_state.last_key = key
    r = st.session_state.route_result
    if find_clicked:
        if r["found"]:
            st.toast(f"{algorithm_name.split(' (')[0]} found a route · cost {r['cost']:g} · {r['time_ms']:.2f} ms", icon="✅")
        else:
            st.toast("No route found - this algorithm got stuck", icon="⚠️")
elif compare_clicked:
    st.session_state.comparison_results = [run_algorithm(n, start, destination) for n in ALGORITHMS]
    st.session_state.route_result = None
    st.session_state.last_key = key
    st.toast("Comparison ready - scroll down", icon="📊")

# ============================================================
# Overview cards
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.markdown(stat_card("location_city", "Locations", G.number_of_nodes(), "linear-gradient(135deg,#1d7bc4,#38bdf8)"), unsafe_allow_html=True)
c2.markdown(stat_card("alt_route", "Road segments", G.number_of_edges(), "linear-gradient(135deg,#8b5cf6,#a78bfa)"), unsafe_allow_html=True)
c3.markdown(stat_card("psychology", "Algorithms", len(ALGORITHMS), "linear-gradient(135deg,#10b981,#34d399)"), unsafe_allow_html=True)
start_name = LOCATION_NAMES.get(start, start)
destination_name = LOCATION_NAMES.get(destination, destination)

c4.markdown(
    stat_card(
        "navigation",
        "Current route",
        f"{start_name} → {destination_name}",
        "linear-gradient(135deg,#f59e0b,#fbbf24)"
    ),
    unsafe_allow_html=True
)
st.write("")

# ============================================================
# Map + result
# ============================================================
map_col, result_col = st.columns([1.7, 1], gap="large")
result = st.session_state.route_result

with map_col:
    html('<div class="sec"><span class="ms">map</span>Live City Map</div>')
    components.html(build_map_html(start, destination, result, speed, show_order), height=680, scrolling=False)
    html("""
    <div class="legend">
      <span><i style="background:#16a34a"></i>Start</span>
      <span><i style="background:#dc2626"></i>Destination</span>
      <span><i style="background:#10b981"></i>Selected route</span>
      <span><i style="background:#f59e0b"></i>Explored (numbered in order)</span>
      <span><i style="background:#60a5fa"></i>Other location</span>
    </div>""")

with result_col:
    html('<div class="sec"><span class="ms">pin_drop</span>Route Result</div>')
    if result is not None:
        badge = f'<div class="algo-badge"><span class="ms">psychology</span>{result["algorithm"]}</div>'
        if result["found"]:
            cost_txt = f"{result['cost']:g}"
            time_txt = f"{result['time_ms']:.3f} ms"
            html(
                f'<div class="panel">{badge}'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
                f'{mini("Total cost", cost_txt)}{mini("Road segments", result["edges"])}'
                f'{mini("Nodes explored", result["nodes_explored"])}{mini("Time", time_txt)}</div>'
                f'<div style="margin-top:16px;font-weight:700;color:#0f2d52;font-size:.9rem">Route sequence</div>'
                f'{route_chips(result["path"])}</div>'
            )
            html(
                '<div class="panel"><div style="font-weight:700;color:#0f2d52;font-size:.9rem">Exploration order</div>'
                f'{route_chips(result["explored"], explored_style=True)}</div>'
            )
            html(story_html(result))
            html(f'<div class="callout"><span class="ms">lightbulb</span><div>{ALGO_NOTES[result["algorithm"]]}</div></div>')
        else:
            n_exp = result["nodes_explored"]
            html(
                f'<div class="panel">{badge}'
                '<div class="empty"><span class="ms" style="color:#f87171">error</span><h4>No route found</h4>'
                f'<p>This algorithm got stuck before reaching the destination after exploring {n_exp} locations.</p></div></div>'
            )
    elif st.session_state.comparison_results is None:
        html("""<div class="panel empty"><span class="ms">explore</span><h4>Ready to find a route?</h4>
        <p>Pick locations and an algorithm in the sidebar. With <b>Live mode</b> on, the route appears instantly.</p>
        <p>Or use <b>Compare All Algorithms</b> to see how each strategy performs.</p></div>""")
    else:
        html("""<div class="panel empty"><span class="ms" style="color:#34d399">task_alt</span><h4>Comparison complete</h4>
        <p>Scroll down for the table and charts.</p></div>""")

# ============================================================
# Comparison
# ============================================================
comparison = st.session_state.comparison_results
if comparison:
    st.write("")
    html('<div class="sec"><span class="ms">leaderboard</span>Algorithm Performance Comparison</div>')
    found = [x for x in comparison if x["found"]]
    best_cost = min((x["cost"] for x in found), default=None)
    best_nodes = min((x["nodes_explored"] for x in found), default=None)
    rows = []
    for x in comparison:
        tags = []
        if x["found"] and x["cost"] == best_cost:
            tags.append("★ Lowest cost")
        if x["found"] and x["nodes_explored"] == best_nodes:
            tags.append("⚡ Fewest nodes")
        rows.append({
            "Algorithm": x["algorithm"], "Route found": "✅ Yes" if x["found"] else "❌ No",
            "Route cost": x["cost"] if x["found"] else None,
            "Road segments": x["edges"] if x["found"] else None,
            "Nodes explored": x["nodes_explored"], "Time (ms)": round(x["time_ms"], 4),
            "Highlights": "  ".join(tags),
            "Route": " → ".join(
                LOCATION_NAMES.get(node, node) for node in x["path"]
            ) if x["found"] else "No route",
        })
    df = pd.DataFrame(rows)
    if found:
        top = min(found, key=lambda x: (x["cost"], x["nodes_explored"]))
        html(f'<div class="callout"><span class="ms">emoji_events</span><div><b>{top["algorithm"]}</b> delivers the cheapest route '
             f'(cost {top["cost"]:g}) while exploring only {top["nodes_explored"]} locations.</div></div>')
    st.dataframe(df, use_container_width=True, hide_index=True)

    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        html('<div class="sec"><span class="ms">travel_explore</span>Nodes explored</div>')
        st.altair_chart(bar_chart(df[["Algorithm", "Nodes explored"]], "Nodes explored"), use_container_width=True)
    with ch2:
        html('<div class="sec"><span class="ms">payments</span>Route cost</div>')
        cdf = df.dropna(subset=["Route cost"])[["Algorithm", "Route cost"]]
        if len(cdf):
            st.altair_chart(bar_chart(cdf, "Route cost"), use_container_width=True)
        else:
            st.info("No algorithm found a route for this pair.")

    st.download_button("Download comparison as CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="routeiq_algorithm_comparison.csv", mime="text/csv",
                       icon=":material/download:")

# ============================================================
# Algorithm notes
# ============================================================
st.write("")
with st.expander("Learn about the algorithms", icon=":material/menu_book:"):
    html("""
    <div class="algo-grid">
      <div class="algo-card"><b>🌊 BFS</b><p>Explores level by level. Fewest road segments, not always lowest cost.</p></div>
      <div class="algo-card"><b>🕳️ DFS</b><p>Dives down one branch before backtracking. Route depends on neighbor order.</p></div>
      <div class="algo-card"><b>🎯 Greedy Best-First</b><p>Always heads for the neighbor that looks closest to the goal.</p></div>
      <div class="algo-card"><b>⭐ A* Search</b><p>Combines cost so far with a distance estimate. Fast and cost-aware.</p></div>
      <div class="algo-card"><b>⛰️ Hill Climbing</b><p>Keeps moving to a closer neighbor; can get stuck before the goal.</p></div>
    </div>
    <p style="color:#6b7c93;font-size:.85rem;margin-top:12px">This is a demonstration app. Map and costs are sample data; heuristics use the drawn map coordinates.</p>
    """)

html('<div class="footer">RouteIQ • AI Search Algorithms • Student project demonstration</div>')