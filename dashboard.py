import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="Exhauster Monitor", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
    --bg:#f4f2ee; --surface:#ffffff; --surface2:#faf9f7;
    --border:#e6e2dc; --border2:#d4cfc8; --text:#181816; --text2:#5a574f;
    --muted:#9a968e; --accent:#bf5630; --accent-lt:#f9ede8;
    --green:#276843; --green-lt:#eaf4ee; --red:#b02a2a; --red-lt:#fdf0f0;
    --amber:#96620f; --amber-lt:#fdf5e8; --blue:#2558a3; --purple:#6b52a8;
}
.stApp { background-color:var(--bg) !important; font-family:'Sora',sans-serif !important; }
.block-container { padding-top:1.2rem !important; padding-bottom:1rem !important; }
div[data-testid="stAppViewBlockContainer"] { padding-top:1.2rem !important; }
section[data-testid="stSidebar"] { background-color:var(--surface) !important; border-right:1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { padding-top:0 !important; }
section[data-testid="stSidebar"] * { font-family:'Sora',sans-serif !important; color:var(--text) !important; }
h1,h2,h3,h4,p,li,span,label,div { font-family:'Sora',sans-serif !important; color:var(--text) !important; }
div[data-testid="metric-container"] {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    border-left:3px solid var(--accent) !important; border-radius:10px !important;
    padding:13px 15px !important; margin-bottom:8px;
    box-shadow:0 1px 4px rgba(0,0,0,0.04); transition:box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover { box-shadow:0 3px 10px rgba(0,0,0,0.08); }
div[data-testid="metric-container"] label {
    font-family:'JetBrains Mono',monospace !important; font-size:0.58rem !important;
    letter-spacing:0.1em !important; color:var(--muted) !important; text-transform:uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family:'Sora',sans-serif !important; font-size:1.3rem !important;
    font-weight:600 !important; color:var(--text) !important; letter-spacing:-0.02em;
}
.banner { padding:13px 20px; border-radius:10px; margin-bottom:22px; font-family:'Sora',sans-serif;
    font-size:0.84rem; font-weight:500; display:flex; align-items:center; gap:10px; }
.banner-ok  { background:var(--green-lt); border:1px solid #a8d9bc; color:var(--green); }
.banner-err { background:var(--red-lt);   border:1px solid #eab5b5; color:var(--red); }
.pill { display:inline-block; padding:4px 14px; border-radius:20px;
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:500;
    letter-spacing:0.08em; text-transform:uppercase; }
.pill-ok   { background:var(--green-lt); color:var(--green); border:1px solid #a8d9bc; }
.pill-warn { background:var(--amber-lt); color:var(--amber); border:1px solid #e8c97a; }
.pill-err  { background:var(--red-lt);   color:var(--red);   border:1px solid #eab5b5; }
.label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.12em;
    color:var(--muted); text-transform:uppercase; margin:20px 0 10px;
    padding-bottom:7px; border-bottom:1px solid var(--border); }
.log-box { background:var(--surface2); border:1px solid var(--border); border-radius:10px;
    padding:14px 16px; height:215px; overflow-y:auto; font-family:'JetBrains Mono',monospace;
    font-size:0.68rem; line-height:2.1; color:var(--muted); }
.log-box::-webkit-scrollbar { width:3px; }
.log-box::-webkit-scrollbar-thumb { background:var(--border2); border-radius:2px; }
.log-ok { color:var(--green); } .log-warn { color:var(--amber); } .log-err { color:var(--red); }
hr { border-color:var(--border) !important; margin:16px 0 !important; }
#MainMenu,header[data-testid="stHeader"],div[data-testid="stToolbar"],
div[data-testid="stDecoration"],.stDeployButton,footer { display:none !important; visibility:hidden; }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.dot-green { background:var(--green); } .dot-red { background:var(--red); }
div[data-testid="stSelectbox"] > label { display:none !important; }
div[data-testid="stSelectbox"] > div > div {
    background:#f9ede8 !important; border:2px solid #bf5630 !important;
    border-radius:7px !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.88rem !important; font-weight:700 !important; color:#bf5630 !important;
    letter-spacing:0.06em !important; box-shadow:0 2px 8px rgba(191,86,48,0.13) !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    box-shadow:0 0 0 3px rgba(191,86,48,0.2) !important;
}
[data-baseweb="popover"],[data-baseweb="popover"] > div,
[data-baseweb="menu"],[data-baseweb="select"] [role="listbox"] {
    background:#ffffff !important; border:1px solid #e6e2dc !important;
    border-radius:8px !important; box-shadow:0 4px 20px rgba(0,0,0,0.10) !important;
}
li[role="option"] {
    background:#ffffff !important; color:#181816 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.82rem !important;
    font-weight:500 !important; padding:10px 16px !important;
    border-bottom:1px solid #f4f2ee !important;
}
li[role="option"]:hover { background:#f9ede8 !important; color:#bf5630 !important; }
li[role="option"][aria-selected="true"] { background:#f9ede8 !important; color:#bf5630 !important; font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
C = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
         font=dict(family="Sora, sans-serif", color="#9a968e", size=11),
         margin=dict(l=62, r=24, t=42, b=40),
         title_font=dict(family="JetBrains Mono, monospace", color="#181816", size=10.5))
GRID   = dict(gridcolor="#edeae5", zerolinecolor="#d4cfc8", showgrid=True, linecolor="#e6e2dc")
ACCENT="#bf5630"; GREEN="#276843"; RED="#b02a2a"; AMBER="#96620f"
BLUE="#2558a3";   PURPLE="#6b52a8"; TEAL="#1a7a6e"; INDIGO="#3d4f8a"

# 8 distinct colours for the 8 vibration sensors
VIB_COLORS = ["#6b52a8","#2558a3","#1a7a6e","#276843",
               "#96620f","#bf5630","#b02a2a","#7a2d6e"]

EXHAUSTER_UNITS = ["EXH-1", "EXH-2"]

# ── All candidate features (every useful sensor column) ────────
ALL_FEATURES = [
    "HT_Motor_Current", "Rotor_Field_Current",
    "Air_Flow", "Outlet_Temp",
    "Vibration_101","Vibration_102","Vibration_103","Vibration_104",
    "Vibration_105","Vibration_106","Vibration_107","Vibration_108",
    "Water_Flow", "Water_Pressure",
    "FB2_Temp", "MB1_Temp", "MB2_Temp",
    "Dis_Header_Pressure", "Diff_Pressure",
    "Outlet_Oil_Temp", "Main_Oil_Tank",
]
target = "FB1_Temp"
VIB_COLS = [c for c in ALL_FEATURES if c.startswith("Vibration_")]

@st.cache_data
def load_raw():
    return pd.read_excel("master_exhauster_dataset.xlsx")

full_df = load_raw()

# ── Page header with inline unit selector ─────────────────────
hL, hM, hR = st.columns([3.2, 1.5, 1.3])
with hL:
    st.markdown("""
    <div style="padding:6px 0 18px;border-bottom:1px solid #e6e2dc;margin-bottom:4px">
        <div style="font-size:1.5rem;font-weight:700;color:#181816;letter-spacing:-0.03em;line-height:1.1">
            Exhauster Health Monitor</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#9a968e;
                    letter-spacing:0.1em;text-transform:uppercase;margin-top:5px">
            Machine intelligence · Anomaly detection · Shift analytics</div>
    </div>""", unsafe_allow_html=True)
with hM:
    st.markdown("""
    <div style="padding-top:8px;border-bottom:1px solid #e6e2dc;padding-bottom:10px">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.14em;
                    color:#9a968e;text-transform:uppercase;margin-bottom:7px">⚙ &nbsp;Exhauster Unit</div>""",
    unsafe_allow_html=True)
    selected_exh = st.selectbox("Exhauster Unit", EXHAUSTER_UNITS,
                                key="exh_selector", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
with hR:
    st.markdown(f"""
    <div style="padding:10px 0 18px;border-bottom:1px solid #e6e2dc;text-align:right;
                font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#9a968e;line-height:2.1">
        <span class="dot dot-green"></span>Live simulation<br>
        Sinter Plant · <strong style="color:#bf5630">{selected_exh}</strong>
    </div>""", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom:22px'></div>", unsafe_allow_html=True)

# ── Pipeline: filter → select active features → train ─────────
if "Exhauster" in full_df.columns:
    _filt = full_df[full_df["Exhauster"] == selected_exh]
    df = (_filt if len(_filt) > 0 else full_df).copy()
else:
    df = full_df.copy()

# Only use features that actually exist AND have enough non-null values
features = [f for f in ALL_FEATURES
            if f in df.columns and df[f].notna().sum() > 10]

df = df.dropna(subset=features + [target]).reset_index(drop=True)
X, y  = df[features], df[target]
model = LinearRegression().fit(X, y)

df["Predicted_FB1"]      = model.predict(X)
df["Residual"]           = df["FB1_Temp"] - df["Predicted_FB1"]
threshold                = df["Residual"].mean() + 2 * df["Residual"].std()
df["Anomaly"]            = np.where(abs(df["Residual"]) > threshold, 1, 0)
df["Health_Score"]       = (100 - (abs(df["Residual"]) / (threshold * 2)) * 100).clip(0, 100)
df["Airflow_Efficiency"] = (df["Air_Flow"] / df["Outlet_Temp"].replace(0, np.nan)) * 100

mae = mean_absolute_error(y, df["Predicted_FB1"])
r2  = r2_score(y, df["Predicted_FB1"])

# Per-unit vib thresholds (primary sensor = 101)
vib_mean = df["Vibration_101"].mean()
vib_std  = df["Vibration_101"].std()
vib_warn = vib_mean + vib_std
vib_crit = vib_mean + 2 * vib_std

# Max vibration across all sensors for composite alert
active_vib_cols = [c for c in VIB_COLS if c in df.columns]
df["Max_Vib"] = df[active_vib_cols].max(axis=1)
max_vib_mean  = df["Max_Vib"].mean()
max_vib_warn  = max_vib_mean + df["Max_Vib"].std()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:#bf5630;padding:18px 20px 16px;margin:-1px -1px 0">
        <div style="font-size:1rem;font-weight:700;color:#fff;letter-spacing:-0.01em">Exhauster Monitor</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                    color:rgba(255,255,255,0.65);letter-spacing:0.1em;margin-top:4px">
            SINTER PLANT · AI SYSTEM</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#f9ede8;border:1px solid rgba(191,86,48,0.3);border-radius:8px;
                padding:10px 14px;margin:14px 4px 0;font-family:'JetBrains Mono',monospace">
        <div style="font-size:0.52rem;letter-spacing:0.12em;color:#9a968e;text-transform:uppercase;margin-bottom:4px">Active Unit</div>
        <div style="font-size:1.1rem;font-weight:700;color:#bf5630;letter-spacing:0.04em">{selected_exh}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.8rem;color:#5a574f;line-height:1.75;padding:14px 4px 4px">
    Operational anomaly detection and machine health intelligence with shift-wise efficiency analysis.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="label">Model info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#9a968e;line-height:2.3">
    Algorithm &nbsp; Linear Regression<br>Unit &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {selected_exh}<br>
    Features &nbsp;&nbsp; {len(features)} (all sensors)<br>
    Records &nbsp;&nbsp;&nbsp; {len(df):,}<br>MAE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {mae:.2f}<br>
    R² &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {r2:.4f}
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="label">Health bands</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.78rem;line-height:2.4;color:#5a574f">
    <span class="dot dot-green"></span>Healthy &nbsp; &lt; 50% threshold<br>
    <span class="dot" style="background:#96620f"></span>Warning &nbsp; 50–100%<br>
    <span class="dot dot-red"></span>Critical &nbsp; &gt; threshold
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="label">Vib 101 thresholds</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#9a968e;line-height:2.3">
    Mean &nbsp;&nbsp;&nbsp;&nbsp; {vib_mean:.3f}<br>Warn &nbsp;&nbsp;&nbsp;&nbsp; {vib_warn:.3f} (μ+1σ)<br>
    Critical &nbsp; {vib_crit:.3f} (μ+2σ)
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="label">Chart index</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.75rem;line-height:2.3;color:#5a574f">
    ① FB1 Actual vs Predicted<br>② Residual trend<br>③ Vibration 101 + anomaly<br>
    ④ Shift residual<br>⑤ Airflow efficiency<br>⑥ Bearing temps (FB+MB)<br>
    ⑦ Residual distribution<br>⑧ All vibration sensors<br>⑨ Pressure monitoring<br>
    ⑩ Oil health<br>⑪ Motor currents
    </div>""", unsafe_allow_html=True)

# ── Live simulation loop ───────────────────────────────────────
placeholder = st.empty()

_state_key = f"sim_idx_{selected_exh}"
if _state_key not in st.session_state:
    st.session_state[_state_key] = 20
st.session_state[_state_key] = min(st.session_state[_state_key], len(df)-1)

i = st.session_state[_state_key]
while i < len(df):
    st.session_state[_state_key] = i
    cur    = df.iloc[:i+1]
    latest = cur.iloc[-1]
    res    = abs(latest["Residual"])
    health = latest["Health_Score"]
    anoms  = cur[cur["Anomaly"] == 1]

    status, pill_cls = (("Healthy","pill-ok") if res < threshold*0.5
                        else ("Warning","pill-warn") if res < threshold
                        else ("Critical","pill-err"))
    gc = GREEN if health > 70 else (AMBER if health > 40 else RED)

    cur_vib   = latest["Vibration_101"]
    vib_label = ("Critical" if cur_vib > vib_crit else "Warning" if cur_vib > vib_warn else "Normal")
    cur_max_vib = latest["Max_Vib"]

    with placeholder.container():

        # Banner
        if latest["Anomaly"] == 1:
            st.markdown(f'<div class="banner banner-err"><strong>⚠ Anomaly · {selected_exh}</strong>'
                        f' &nbsp;—&nbsp; Residual exceeded. Abnormal thermal behaviour on FB1 bearing.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner banner-ok"><strong>✓ {selected_exh} nominal</strong>'
                        f' &nbsp;—&nbsp; All parameters within learned healthy behaviour envelope.</div>',
                        unsafe_allow_html=True)

        left, right = st.columns([1, 3])

        # ════════ LEFT PANEL ════════
        with left:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=health,
                number=dict(suffix="%", font=dict(family="Sora,sans-serif", size=30, color=gc)),
                title=dict(text="Machine Health",
                           font=dict(family="JetBrains Mono,monospace", size=10, color="#9a968e")),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor="#e6e2dc", ticklen=4, tickwidth=1,
                              tickfont=dict(size=9,color="#9a968e"), showticklabels=True),
                    bar=dict(color=gc, thickness=0.2), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[dict(range=[0,40],color="#fdf0f0"),
                           dict(range=[40,70],color="#fdf5e8"),
                           dict(range=[70,100],color="#eaf4ee")])))
            gauge.update_layout(**C, height=215)
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown(f'<div style="text-align:center;margin:-12px 0 18px">'
                        f'<span class="pill {pill_cls}">{status}</span></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Residual",  f"{latest['Residual']:.2f}")
                st.metric("MAE",       f"{mae:.2f}")
                st.metric("Threshold", f"{threshold:.2f}")
            with c2:
                st.metric("Anomalies", int(cur["Anomaly"].sum()))
                st.metric("R²",        f"{r2:.3f}")
                st.metric("Records",   f"{i+1}")

            # Vibration badge — shows max across all sensors
            vib_bg  = "#fdf0f0" if cur_vib>vib_crit else ("#fdf5e8" if cur_vib>vib_warn else "#eaf4ee")
            vib_col = RED if cur_vib>vib_crit else (AMBER if cur_vib>vib_warn else GREEN)
            st.markdown(f"""
            <div style="background:{vib_bg};border:1px solid {vib_col}33;
                        border-radius:8px;padding:10px 14px;margin:10px 0 4px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                            letter-spacing:0.1em;color:{vib_col};text-transform:uppercase;margin-bottom:2px">
                    Vibration · 101</div>
                <div style="font-size:1.1rem;font-weight:600;color:{vib_col};letter-spacing:-0.02em">
                    {cur_vib:.3f}
                    <span style="font-size:0.65rem;font-weight:400;margin-left:4px">{vib_label}</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                            color:#9a968e;margin-top:3px">
                    Max all sensors: {cur_max_vib:.3f}</div>
            </div>""", unsafe_allow_html=True)

            # ── System status mini-panel ─────────────────────────
            st.markdown('<div class="label">System status</div>', unsafe_allow_html=True)

            def _status_row(label, value, unit, good, warn_val, crit_val, fmt=".2f", higher_is_bad=True):
                if higher_is_bad:
                    col = RED if value > crit_val else (AMBER if value > warn_val else GREEN)
                    bg  = "#fdf0f0" if value > crit_val else ("#fdf5e8" if value > warn_val else "#eaf4ee")
                    tag = "CRIT" if value > crit_val else ("WARN" if value > warn_val else "OK")
                else:
                    col = RED if value < crit_val else (AMBER if value < warn_val else GREEN)
                    bg  = "#fdf0f0" if value < crit_val else ("#fdf5e8" if value < warn_val else "#eaf4ee")
                    tag = "CRIT" if value < crit_val else ("WARN" if value < warn_val else "OK")
                return (
                    f'<div style="display:flex;align-items:center;justify-content:space-between;' +
                    f'background:{bg};border-radius:6px;padding:5px 10px;margin-bottom:4px">' +
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#5a574f">{label}</span>' +
                    f'<span style="display:flex;align-items:center;gap:6px">' +
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;font-weight:600;color:#181816">' +
                    f'{value:{fmt}} {unit}</span>' +
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.52rem;font-weight:600;' +
                    f'color:{col};letter-spacing:0.06em">{tag}</span>' +
                    f'</span></div>'
                )

            vib_mean_cur = cur[active_vib_cols].mean(axis=1).iloc[-1]
            status_html  = (
                _status_row("Motor Current",  latest["HT_Motor_Current"],   "A",    True,  cur["HT_Motor_Current"].mean()*1.1,  cur["HT_Motor_Current"].mean()*1.25) +
                _status_row("Max Vibration",  cur_max_vib,                  "mm/s", True,  max_vib_warn, max_vib_warn*1.5, ".3f") +
                _status_row("Outlet Temp",    latest["Outlet_Temp"],        "°C",   True,  cur["Outlet_Temp"].mean()*1.05, cur["Outlet_Temp"].mean()*1.15, ".1f") +
                _status_row("Disc Pressure",  latest["Dis_Header_Pressure"],"mbar", True,  cur["Dis_Header_Pressure"].mean()*1.1, cur["Dis_Header_Pressure"].mean()*1.2, ".2f") +
                _status_row("Oil Outlet",     latest["Outlet_Oil_Temp"],    "°C",   True,  cur["Outlet_Oil_Temp"].mean()*1.1, cur["Outlet_Oil_Temp"].mean()*1.2, ".1f") +
                _status_row("Water Flow",     latest["Water_Flow"],         "m³/h", False, cur["Water_Flow"].mean()*0.9, cur["Water_Flow"].mean()*0.75, ".2f")
            )
            st.markdown(f'<div style="margin-bottom:8px">{status_html}</div>', unsafe_allow_html=True)

            # Live sensor feed — comprehensive
            st.markdown('<div class="label">Live sensor feed</div>', unsafe_allow_html=True)
            sensor_groups = [
                ("Motor Current",   f"{latest['HT_Motor_Current']:.2f}",  "A"),
                ("Rotor Current",   f"{latest['Rotor_Field_Current']:.2f}","A"),
                ("Air Flow",        f"{latest['Air_Flow']:.1f}",           "m³/h"),
                ("Outlet Temp",     f"{latest['Outlet_Temp']:.1f}",        "°C"),
                ("Water Flow",      f"{latest['Water_Flow']:.2f}",         "m³/h"),
                ("Water Pressure",  f"{latest['Water_Pressure']:.2f}",     "bar"),
                ("Vib 101",         f"{latest['Vibration_101']:.4f}",      "mm/s"),
                ("Max Vib",         f"{cur_max_vib:.4f}",                  "mm/s"),
                ("Disc Pressure",   f"{latest['Dis_Header_Pressure']:.2f}","mbar"),
                ("Diff Pressure",   f"{latest['Diff_Pressure']:.2f}",      "mbar"),
                ("Oil Outlet Temp", f"{latest['Outlet_Oil_Temp']:.1f}",    "°C"),
                ("Oil Tank",        f"{latest['Main_Oil_Tank']:.1f}",      "°C"),
            ]
            rows = "".join(
                f'<tr style="background:{"#faf9f7" if k%2==0 else "#fff"}">'
                f'<td style="padding:6px 10px;font-family:JetBrains Mono,monospace;font-size:0.65rem;'
                f'color:#5a574f;border-bottom:1px solid #f0ede8">{n}</td>'
                f'<td style="padding:6px 10px;font-family:JetBrains Mono,monospace;font-size:0.68rem;'
                f'font-weight:500;color:#181816;text-align:right;border-bottom:1px solid #f0ede8">{v}</td>'
                f'<td style="padding:6px 8px;font-family:JetBrains Mono,monospace;font-size:0.58rem;'
                f'color:#9a968e;border-bottom:1px solid #f0ede8;white-space:nowrap">{u}</td></tr>'
                for k,(n,v,u) in enumerate(sensor_groups))
            st.markdown(
                '<table style="width:100%;border-collapse:collapse;border:1px solid #e6e2dc;'
                'border-radius:10px;overflow:hidden;margin-bottom:4px">'
                '<thead><tr style="background:#f4f2ee">'
                '<th style="padding:6px 10px;font-family:JetBrains Mono,monospace;font-size:0.55rem;'
                'letter-spacing:0.08em;color:#9a968e;text-transform:uppercase;text-align:left;'
                'border-bottom:1px solid #e6e2dc;font-weight:500">Sensor</th>'
                '<th style="padding:6px 10px;font-family:JetBrains Mono,monospace;font-size:0.55rem;'
                'letter-spacing:0.08em;color:#9a968e;text-transform:uppercase;text-align:right;'
                'border-bottom:1px solid #e6e2dc;font-weight:500">Value</th>'
                '<th style="padding:6px 8px;font-family:JetBrains Mono,monospace;font-size:0.55rem;'
                'letter-spacing:0.08em;color:#9a968e;text-transform:uppercase;text-align:left;'
                'border-bottom:1px solid #e6e2dc;font-weight:500">Unit</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

            st.markdown('<div class="label">Alert log</div>', unsafe_allow_html=True)
            logs = [f'<span class="{"log-err" if abs(r["Residual"])>threshold*1.5 else "log-warn"}">'
                    f'↑ res {r["Residual"]:.2f} · health {r["Health_Score"]:.1f}%</span>'
                    for _,r in anoms.tail(7).iterrows()] or ['<span class="log-ok">— No active anomalies</span>']
            st.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)

        # ════════ RIGHT PANEL ════════
        with right:
            n_obs    = len(cur)
            x_obs    = list(range(n_obs))
            anom_pos = [cur.index.get_loc(idx) for idx in anoms.index if idx in cur.index]
            anom_fb1 = [cur["FB1_Temp"].iloc[p] for p in anom_pos]
            anom_res = [cur["Residual"].iloc[p]  for p in anom_pos]
            _xt = dict(tickfont=dict(family="JetBrains Mono,monospace",size=9,color="#5a574f"),
                       tickformat=",d")
            _yt = dict(tickfont=dict(family="JetBrains Mono,monospace",size=9,color="#5a574f"))

            # ① FB1 Actual vs Predicted
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=x_obs,y=cur["Predicted_FB1"].values,mode="lines",
                line=dict(width=0),showlegend=False,hoverinfo="skip",name="_l"))
            fig1.add_trace(go.Scatter(x=x_obs,y=cur["FB1_Temp"].values,mode="lines",fill="tonexty",
                fillcolor="rgba(191,86,48,0.07)",line=dict(width=0),showlegend=False,hoverinfo="skip",name="_u"))
            fig1.add_trace(go.Scatter(x=x_obs,y=cur["Predicted_FB1"].values,mode="lines",name="Predicted",
                line=dict(color=BLUE,width=1.4,dash="dot"),
                hovertemplate="Obs %{x}<br>Predicted: %{y:.1f}°C<extra></extra>"))
            fig1.add_trace(go.Scatter(x=x_obs,y=cur["FB1_Temp"].values,mode="lines",name="Actual",
                line=dict(color=ACCENT,width=2.2),
                hovertemplate="Obs %{x}<br>Actual: %{y:.1f}°C<extra></extra>"))
            fig1.add_trace(go.Scatter(x=anom_pos,y=anom_fb1,mode="markers",name="Anomaly",
                marker=dict(color="white",size=8,symbol="circle",line=dict(color=RED,width=2.5)),
                hovertemplate="⚠ Obs %{x} · %{y:.1f}°C<extra></extra>"))
            fig1.add_trace(go.Scatter(x=[n_obs-1],y=[latest["FB1_Temp"]],mode="markers",name="Now",
                marker=dict(color=ACCENT,size=10,symbol="circle",line=dict(color="white",width=2)),
                hovertemplate="Now · %{y:.1f}°C<extra></extra>"))
            y_vals=pd.concat([cur["FB1_Temp"],cur["Predicted_FB1"]]); y_pad=(y_vals.max()-y_vals.min())*0.12
            fig1.update_layout(**C,height=300,
                title_text=f"① FB1 Bearing Temperature — Actual vs Predicted · {selected_exh}",
                xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                yaxis=dict(**GRID,**_yt,range=[y_vals.min()-y_pad,y_vals.max()+y_pad],
                    title="Temperature",title_font=dict(size=10,color="#9a968e"),
                    tickformat=".1f",ticksuffix="°C"),
                legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=10,color="#9a968e"),
                    bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.34),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                    font=dict(family="JetBrains Mono",size=11,color="#181816")))
            st.plotly_chart(fig1, use_container_width=True)

            # ② Residual trend
            fig2 = go.Figure()
            fig2.add_hrect(y0=-threshold,y1=threshold,fillcolor="rgba(39,104,67,0.05)",line_width=0)
            fig2.add_trace(go.Scatter(x=x_obs,y=cur["Residual"].values,mode="lines",name="Residual",
                line=dict(color=PURPLE,width=1.6),fill="tozeroy",fillcolor="rgba(107,82,168,0.06)",
                hovertemplate="Obs %{x}<br>Residual: %{y:.2f}<extra></extra>"))
            fig2.add_hline(y=threshold, line_dash="dash",line_color="rgba(176,42,42,0.45)",line_width=1)
            fig2.add_hline(y=-threshold,line_dash="dash",line_color="rgba(176,42,42,0.45)",line_width=1)
            fig2.add_trace(go.Scatter(x=anom_pos,y=anom_res,mode="markers",
                marker=dict(color=RED,size=7,symbol="circle-open",line=dict(width=2)),
                showlegend=False,hovertemplate="⚠ Obs %{x} · %{y:.2f}<extra></extra>"))
            fig2.update_layout(**C,height=200,title_text="② Residual trend — threshold bands",
                xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                yaxis=dict(**GRID,**_yt,title="Residual",title_font=dict(size=9,color="#9a968e"),tickformat="+.2f"),
                showlegend=False,hovermode="x unified",
                hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                    font=dict(family="JetBrains Mono",size=11,color="#181816")))
            st.plotly_chart(fig2, use_container_width=True)

            # ③ Vibration 101 primary trend
            vib_s = cur["Vibration_101"].reset_index(drop=True)
            fig_v = go.Figure()
            fig_v.add_hrect(y0=0,y1=vib_warn,fillcolor="rgba(39,104,67,0.04)",line_width=0)
            fig_v.add_hrect(y0=vib_warn,y1=vib_crit,fillcolor="rgba(150,98,15,0.06)",line_width=0)
            fig_v.add_hrect(y0=vib_crit,y1=vib_s.max()*1.15,fillcolor="rgba(176,42,42,0.05)",line_width=0)
            _vt=vib_s.max()*1.15; _lc=dict(xref="paper",showarrow=False,x=0.995,xanchor="right")
            fig_v.add_annotation(**_lc,y=vib_warn*0.5,text="NORMAL",
                font=dict(family="JetBrains Mono,monospace",size=9,color=GREEN))
            fig_v.add_annotation(**_lc,y=(vib_warn+vib_crit)/2,text="WARN",
                font=dict(family="JetBrains Mono,monospace",size=9,color=AMBER))
            fig_v.add_annotation(**_lc,y=(vib_crit+_vt)/2,text="CRITICAL",
                font=dict(family="JetBrains Mono,monospace",size=9,color=RED))
            fig_v.add_trace(go.Scatter(y=vib_s,mode="none",fill="tozeroy",
                fillcolor="rgba(107,82,168,0.05)",showlegend=False,hoverinfo="skip"))
            fig_v.add_trace(go.Scatter(y=vib_s,mode="lines",name="Vibration 101",
                line=dict(color=PURPLE,width=2),hovertemplate="Vib 101: %{y:.4f}<extra></extra>"))
            fig_v.add_hline(y=vib_warn,line_dash="dash",line_color="rgba(150,98,15,0.55)",line_width=1.2)
            fig_v.add_hline(y=vib_crit,line_dash="dash",line_color="rgba(176,42,42,0.55)",line_width=1.2)
            if anom_pos:
                fig_v.add_trace(go.Scatter(x=anom_pos,y=[vib_s.iloc[p] for p in anom_pos],
                    mode="markers",name="Anomaly",
                    marker=dict(color="white",size=9,symbol="circle",line=dict(color=RED,width=2.2)),
                    hovertemplate="⚠ %{y:.4f}<extra></extra>"))
            lvc=RED if cur_vib>vib_crit else (AMBER if cur_vib>vib_warn else GREEN)
            fig_v.add_trace(go.Scatter(x=[len(vib_s)-1],y=[cur_vib],mode="markers",name="Now",
                marker=dict(color=lvc,size=11,symbol="circle",line=dict(color="white",width=2.5)),
                hovertemplate=f"Now: {cur_vib:.4f}<extra></extra>"))
            fig_v.update_layout(**C,height=230,
                title_text="③ Vibration Trend — 101 sensor · anomaly correlation · severity bands",
                xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                yaxis=dict(**GRID,**_yt,title="Vibration",title_font=dict(size=9,color="#9a968e"),
                    tickformat=".3f",ticksuffix=" mm/s",nticks=5),
                legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                    bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.36),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                    font=dict(family="JetBrains Mono",size=11,color="#181816")))
            st.plotly_chart(fig_v, use_container_width=True)

            # ④ Shift residual + ⑤ Airflow efficiency
            g3, g4 = st.columns(2)
            with g3:
                if "Shift" in cur.columns:
                    sd=cur.groupby("Shift")[["Residual","FB1_Temp","Vibration_101"]].mean().reset_index()
                    bc=[GREEN if abs(v)<threshold*0.5 else (AMBER if abs(v)<threshold else RED) for v in sd["Residual"]]
                    fig4=go.Figure(go.Bar(x=sd["Shift"],y=sd["Residual"],marker_color=bc,
                        marker_line=dict(color="rgba(0,0,0,0.06)",width=1),width=0.5,
                        hovertemplate="Shift %{x}<br>Avg residual: %{y:.2f}<extra></extra>"))
                    fig4.update_layout(**C,height=250,title_text="④ Shift-wise residual",
                        xaxis=dict(gridcolor="#edeae5",zerolinecolor="#d4cfc8",showgrid=False,
                            linecolor="#e6e2dc",title="Shift",title_font=dict(size=9,color="#9a968e"),
                            tickfont=dict(family="JetBrains Mono",size=10,color="#5a574f")),
                        yaxis=dict(**GRID,**_yt,title="Avg Residual",
                            title_font=dict(size=9,color="#9a968e"),tickformat="+.2f"),bargap=0.35)
                    st.plotly_chart(fig4, use_container_width=True)

            with g4:
                if "Shift" in cur.columns:
                    es=cur.groupby("Shift")[["Air_Flow","Outlet_Temp","Airflow_Efficiency"]].mean().reset_index()
                    em=es["Airflow_Efficiency"].max(); en=es["Airflow_Efficiency"].min()
                    er=em-en if em!=en else 1
                    es["Eff_Pct"]=((es["Airflow_Efficiency"]-en)/er*100).round(1)
                    bec,beb,edc=[],[],[]
                    for v in es["Eff_Pct"]:
                        if v>=70:   bec.append("rgba(39,104,67,0.22)");  beb.append(GREEN); edc.append(GREEN)
                        elif v>=40: bec.append("rgba(150,98,15,0.18)"); beb.append(AMBER); edc.append(AMBER)
                        else:       bec.append("rgba(176,42,42,0.15)"); beb.append(RED);   edc.append(RED)
                    fe=go.Figure()
                    fe.add_trace(go.Bar(x=es["Shift"],y=es["Air_Flow"],name="Air Flow",
                        marker_color=bec,marker_line=dict(color=beb,width=1.5),width=0.45,yaxis="y",
                        hovertemplate="<b>Shift %{x}</b><br>Air Flow: %{y:,.0f}<extra></extra>"))
                    fe.add_trace(go.Scatter(x=es["Shift"],y=es["Outlet_Temp"],mode="lines+markers",
                        name="Outlet Temp",line=dict(color=ACCENT,width=2,dash="dot"),
                        marker=dict(color=ACCENT,size=7,line=dict(color="white",width=1.5)),yaxis="y",
                        hovertemplate="<b>Shift %{x}</b><br>Outlet: %{y:.1f}°C<extra></extra>"))
                    fe.add_trace(go.Scatter(x=es["Shift"],y=es["Eff_Pct"],mode="lines+markers+text",
                        name="Efficiency %",line=dict(color=TEAL,width=2.2),
                        marker=dict(color=edc,size=11,line=dict(color="white",width=2)),
                        text=[f"{v:.0f}%" for v in es["Eff_Pct"]],textposition="top center",
                        textfont=dict(family="JetBrains Mono",size=9,color=TEAL),yaxis="y2",
                        hovertemplate="<b>Shift %{x}</b><br>Efficiency: %{y:.1f}%<extra></extra>"))
                    fe.update_layout(**C,height=250,title_text="⑤ Airflow / pressure efficiency by shift",
                        xaxis=dict(showgrid=False,linecolor="#e6e2dc",
                            tickfont=dict(family="JetBrains Mono",size=10)),
                        yaxis=dict(**GRID,title="Air Flow | Temp",title_font=dict(size=9,color="#9a968e"),
                            tickfont=dict(family="JetBrains Mono,monospace",size=9,color="#9a968e"),tickformat=","),
                        yaxis2=dict(overlaying="y",side="right",range=[0,115],title="Efficiency %",
                            title_font=dict(size=9,color=TEAL),ticksuffix="%",showgrid=False,
                            tickfont=dict(size=9,color=TEAL)),
                        legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                            bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.44),
                        bargap=0.4,hovermode="x unified",
                        hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                            font=dict(family="JetBrains Mono",size=11,color="#181816")))
                    st.plotly_chart(fe, use_container_width=True)

            # ⑥ All 4 bearing temps + ⑦ Residual histogram
            g5, g6 = st.columns(2)
            with g5:
                # FB1, FB2, MB1, MB2 all on same chart
                bearing_cols = ["FB1_Temp","FB2_Temp","MB1_Temp","MB2_Temp"]
                bearing_colors = [ACCENT, BLUE, TEAL, PURPLE]
                bearing_names  = ["FB1 Bearing","FB2 Bearing","MB1 Bearing","MB2 Bearing"]
                fig6 = go.Figure()
                # Fill between FB1 and FB2
                fig6.add_trace(go.Scatter(x=x_obs,y=cur["FB2_Temp"].values,mode="lines",
                    line=dict(width=0),showlegend=False,hoverinfo="skip",name="_base"))
                fig6.add_trace(go.Scatter(x=x_obs,y=cur["FB1_Temp"].values,fill="tonexty",
                    fillcolor="rgba(191,86,48,0.05)",mode="none",showlegend=False,
                    hoverinfo="skip",name="_fill"))
                for col, color, name in zip(bearing_cols, bearing_colors, bearing_names):
                    if col in cur.columns:
                        fig6.add_trace(go.Scatter(x=x_obs,y=cur[col].values,mode="lines",name=name,
                            line=dict(color=color,width=1.8,dash=("solid" if "FB1" in name else
                                      "dot" if "FB2" in name else "dash" if "MB1" in name else "dashdot")),
                            hovertemplate=f"Obs %{{x}}<br>{name}: %{{y:.1f}}°C<extra></extra>"))
                        fig6.add_trace(go.Scatter(x=[n_obs-1],y=[latest[col]],mode="markers",
                            name=f"{name} Now",showlegend=False,
                            marker=dict(color=color,size=8,symbol="circle",
                                        line=dict(color="white",width=1.5)),
                            hovertemplate=f"{name} Now: %{{y:.1f}}°C<extra></extra>"))
                all_b = pd.concat([cur[c] for c in bearing_cols if c in cur.columns])
                b_pad = (all_b.max()-all_b.min())*0.12
                fig6.update_layout(**C,height=250,
                    title_text="⑥ All bearing temps — FB1 · FB2 · MB1 · MB2",
                    xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                    yaxis=dict(**GRID,**_yt,range=[all_b.min()-b_pad,all_b.max()+b_pad],
                        title="Temperature",title_font=dict(size=9,color="#9a968e"),
                        tickformat=".1f",ticksuffix="°C",nticks=5),
                    legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                        bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.40),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                        font=dict(family="JetBrains Mono",size=11,color="#181816")))
                st.plotly_chart(fig6, use_container_width=True)

            with g6:
                fig7=go.Figure(go.Histogram(x=cur["Residual"],nbinsx=30,
                    marker=dict(color="rgba(191,86,48,0.3)",line=dict(color="rgba(191,86,48,0.6)",width=0.5))))
                fig7.add_vline(x=threshold, line_dash="dash",line_color="rgba(176,42,42,0.45)",line_width=1)
                fig7.add_vline(x=-threshold,line_dash="dash",line_color="rgba(176,42,42,0.45)",line_width=1)
                fig7.update_layout(**C,height=250,title_text="⑦ Residual distribution",
                    xaxis=dict(**GRID,**_yt,title="Residual",title_font=dict(size=9,color="#9a968e"),tickformat=".1f"),
                    yaxis=dict(**GRID,**_yt,title="Count",title_font=dict(size=9,color="#9a968e"),
                        tickformat=",d",ticksuffix=" ct"))
                st.plotly_chart(fig7, use_container_width=True)

            # ⑧ ALL 8 vibration sensors comparison — full width
            fig_allvib = go.Figure()
            for idx, vc in enumerate(active_vib_cols):
                sensor_num = vc.split("_")[1]
                fig_allvib.add_trace(go.Scatter(
                    x=x_obs, y=cur[vc].values, mode="lines",
                    name=f"Vib {sensor_num}",
                    line=dict(color=VIB_COLORS[idx % len(VIB_COLORS)], width=1.5),
                    hovertemplate=f"Obs %{{x}}<br>Vib {sensor_num}: %{{y:.4f}} mm/s<extra></extra>"
                ))
            # Warn/crit lines based on sensor 101 thresholds as reference
            fig_allvib.add_hline(y=vib_warn,line_dash="dash",
                line_color="rgba(150,98,15,0.4)",line_width=1)
            fig_allvib.add_hline(y=vib_crit,line_dash="dash",
                line_color="rgba(176,42,42,0.4)",line_width=1)
            fig_allvib.update_layout(**C,height=260,
                title_text=f"⑧ All vibration sensors — 101 to {active_vib_cols[-1].split('_')[1]} · {selected_exh}",
                xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                yaxis=dict(**GRID,**_yt,title="Vibration",title_font=dict(size=9,color="#9a968e"),
                    tickformat=".3f",ticksuffix=" mm/s",nticks=5),
                legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                    bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.38),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                    font=dict(family="JetBrains Mono",size=11,color="#181816")))
            fig_allvib.update_layout(margin=dict(l=62, r=24, t=42, b=72))
            st.plotly_chart(fig_allvib, use_container_width=True)

            # ⑨ Pressure monitoring + ⑩ Oil health
            g7, g8 = st.columns(2)

            with g7:
                press_cols = [c for c in ["Dis_Header_Pressure","Diff_Pressure"] if c in cur.columns]
                if press_cols:
                    fig_p = go.Figure()
                    pcols_cfg = [
                        ("Dis_Header_Pressure", "Discharge Pressure", BLUE,   "solid", "y"),
                        ("Diff_Pressure",        "Diff Pressure",      PURPLE, "dot",  "y2"),
                    ]
                    for col, name, color, dash, yax in pcols_cfg:
                        if col in cur.columns:
                            fig_p.add_trace(go.Scatter(x=x_obs,y=cur[col].values,mode="lines",
                                name=name,line=dict(color=color,width=1.8,dash=dash),yaxis=yax,
                                hovertemplate=f"Obs %{{x}}<br>{name}: %{{y:.2f}} mbar<extra></extra>"))
                            fig_p.add_trace(go.Scatter(x=[n_obs-1],y=[latest[col]],mode="markers",
                                showlegend=False,yaxis=yax,
                                marker=dict(color=color,size=8,symbol="circle",line=dict(color="white",width=1.5)),
                                hovertemplate=f"{name} Now: %{{y:.2f}}<extra></extra>"))
                    # Anomaly markers on pressure
                    if anom_pos and "Dis_Header_Pressure" in cur.columns:
                        fig_p.add_trace(go.Scatter(
                            x=anom_pos,
                            y=[cur["Dis_Header_Pressure"].iloc[p] for p in anom_pos],
                            mode="markers",name="Anomaly",showlegend=False,yaxis="y",
                            marker=dict(color="white",size=7,symbol="circle",line=dict(color=RED,width=2)),
                            hovertemplate="⚠ Obs %{x}<extra></extra>"))
                    # dual axis — each series auto-ranges on its own axis
                    fig_p.update_layout(**C,height=250,title_text="⑨ Pressure monitoring — discharge & differential",
                        xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                        yaxis=dict(**GRID,**_yt,
                            title="Discharge (mbar)",title_font=dict(size=9,color=BLUE),
                            tickformat=".2f",nticks=5),
                        yaxis2=dict(overlaying="y",side="right",
                            title="Diff Pressure (mbar)",title_font=dict(size=9,color=PURPLE),
                            tickformat=".3f",nticks=5,showgrid=False,
                            tickfont=dict(family="JetBrains Mono,monospace",size=9,color=PURPLE)),
                        legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                            bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.28),
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                            font=dict(family="JetBrains Mono",size=11,color="#181816")))
                    st.plotly_chart(fig_p, use_container_width=True)

            with g8:
                oil_cols = [c for c in ["Outlet_Oil_Temp","Main_Oil_Tank"] if c in cur.columns]
                if oil_cols:
                    fig_oil = go.Figure()
                    oil_cfg = [
                        ("Outlet_Oil_Temp", "Oil Outlet Temp", AMBER,  "solid"),
                        ("Main_Oil_Tank",   "Oil Tank Temp",   ACCENT, "dot"),
                    ]
                    for col, name, color, dash in oil_cfg:
                        if col in cur.columns:
                            fig_oil.add_trace(go.Scatter(x=x_obs,y=cur[col].values,mode="lines",
                                name=name,line=dict(color=color,width=1.8,dash=dash),
                                fill="tozeroy" if col=="Outlet_Oil_Temp" else "none",
                                fillcolor="rgba(150,98,15,0.05)" if col=="Outlet_Oil_Temp" else None,
                                hovertemplate=f"Obs %{{x}}<br>{name}: %{{y:.1f}}°C<extra></extra>"))
                            fig_oil.add_trace(go.Scatter(x=[n_obs-1],y=[latest[col]],mode="markers",
                                showlegend=False,
                                marker=dict(color=color,size=8,symbol="circle",line=dict(color="white",width=1.5)),
                                hovertemplate=f"{name} Now: %{{y:.1f}}°C<extra></extra>"))
                    oval=pd.concat([cur[c] for c in oil_cols]); op=(oval.max()-oval.min())*0.12
                    fig_oil.update_layout(**C,height=250,title_text="⑩ Oil health — outlet & tank temperature",
                        xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                        yaxis=dict(**GRID,**_yt,range=[oval.min()-op,oval.max()+op],
                            title="Temperature",title_font=dict(size=9,color="#9a968e"),
                            tickformat=".1f",ticksuffix="°C"),
                        legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                            bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.28),
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                            font=dict(family="JetBrains Mono",size=11,color="#181816")))
                    st.plotly_chart(fig_oil, use_container_width=True)

            # ⑪ Motor currents — HT + Rotor field
            curr_cols = [c for c in ["HT_Motor_Current","Rotor_Field_Current"] if c in cur.columns]
            if curr_cols:
                fig_curr = go.Figure()
                curr_cfg = [
                    ("HT_Motor_Current",   "HT Motor Current",   INDIGO, "solid"),
                    ("Rotor_Field_Current", "Rotor Field Current", TEAL,   "dot"),
                ]
                for col, name, color, dash in curr_cfg:
                    if col in cur.columns:
                        fig_curr.add_trace(go.Scatter(x=x_obs,y=cur[col].values,mode="lines",
                            name=name,line=dict(color=color,width=1.8,dash=dash),
                            fill="tozeroy" if col=="HT_Motor_Current" else "none",
                            fillcolor="rgba(61,79,138,0.05)" if col=="HT_Motor_Current" else None,
                            hovertemplate=f"Obs %{{x}}<br>{name}: %{{y:.2f}} A<extra></extra>"))
                        fig_curr.add_trace(go.Scatter(x=[n_obs-1],y=[latest[col]],mode="markers",
                            showlegend=False,
                            marker=dict(color=color,size=8,symbol="circle",line=dict(color="white",width=1.5)),
                            hovertemplate=f"{name} Now: %{{y:.2f}} A<extra></extra>"))
                if anom_pos and "HT_Motor_Current" in cur.columns:
                    fig_curr.add_trace(go.Scatter(
                        x=anom_pos,
                        y=[cur["HT_Motor_Current"].iloc[p] for p in anom_pos],
                        mode="markers",name="Anomaly",showlegend=False,
                        marker=dict(color="white",size=7,symbol="circle",line=dict(color=RED,width=2)),
                        hovertemplate="⚠ Obs %{x}<extra></extra>"))
                cval=pd.concat([cur[c] for c in curr_cols]); cp=(cval.max()-cval.min())*0.12
                fig_curr.update_layout(**C,height=230,
                    title_text=f"⑪ Motor currents — HT motor & rotor field · {selected_exh}",
                    xaxis=dict(**GRID,**_xt,showspikes=True,spikecolor="#e6e2dc",spikethickness=1,spikedash="dot"),
                    yaxis=dict(**GRID,**_yt,range=[cval.min()-cp,cval.max()+cp],
                        title="Current",title_font=dict(size=9,color="#9a968e"),
                        tickformat=".1f",ticksuffix=" A",nticks=5),
                    legend=dict(orientation="h",font=dict(family="JetBrains Mono",size=9,color="#9a968e"),
                        bgcolor="rgba(255,255,255,0.9)",bordercolor="#e6e2dc",borderwidth=1,x=0.01,y=-0.22),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="white",bordercolor="#e6e2dc",
                        font=dict(family="JetBrains Mono",size=11,color="#181816")))
                st.plotly_chart(fig_curr, use_container_width=True)

    i += 1
    time.sleep(3)