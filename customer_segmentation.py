import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Banking Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "Churn_Modelling.csv"

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
:root {
    --bg:#070A12;
    --panel:#0D1220;
    --panel2:#111827;
    --border:#202A3D;
    --text:#F4F7FB;
    --muted:#8792A8;
    --accent:#6EA8FE;
    --success:#42D392;
    --warning:#F5C451;
    --danger:#FF6B7A;
}
.stApp { background:var(--bg); color:var(--text); }
.block-container { padding:1.4rem 2rem 2rem; max-width:1600px; }
[data-testid="stSidebar"] { background:#090D17; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:#DCE4F2; }
h1,h2,h3 { letter-spacing:-.02em; }
.hero {
    padding:24px 26px; border:1px solid var(--border); border-radius:18px;
    background:linear-gradient(135deg,#101829 0%,#0B101B 65%,#101726 100%);
    margin-bottom:18px;
}
.eyebrow { color:var(--accent); font-size:11px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-size:30px; font-weight:800; margin:5px 0 3px; }
.hero-sub { color:var(--muted); font-size:13px; }
.kpi {
    background:linear-gradient(180deg,#101624,#0C111C); border:1px solid var(--border);
    border-radius:15px; padding:16px 17px; min-height:112px;
}
.kpi-label { color:var(--muted); font-size:10px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.kpi-value { color:var(--text); font-size:25px; font-weight:800; margin-top:8px; }
.kpi-note { color:#65728A; font-size:10px; margin-top:5px; }
.section {
    color:#E9EEF8; font-size:13px; font-weight:800; letter-spacing:.03em;
    margin:22px 0 9px; padding-bottom:8px; border-bottom:1px solid var(--border);
}
.insight {
    background:#0D1421; border:1px solid var(--border); border-radius:14px;
    padding:15px 16px; min-height:112px;
}
.insight-title { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
.insight-value { font-size:19px; font-weight:800; margin-top:7px; }
.insight-copy { font-size:11px; color:#78859B; margin-top:5px; line-height:1.45; }
.badge {
    display:inline-block; padding:4px 8px; border-radius:999px;
    font-size:9px; font-weight:800; letter-spacing:.05em;
    background:#18243A; color:#9FC0FF; border:1px solid #263A5C;
}
div[data-testid="stMetric"] { background:transparent; }
.stButton button { border-radius:10px; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_data():
    encodings = ["utf-8-sig","utf-8","latin1","cp1252"]
    last = None
    for enc in encodings:
        try:
            d = pd.read_csv(RAW_FILE, encoding=enc)
            if not d.empty:
                return d
        except Exception as e:
            last = e
    raise last or ValueError("Unable to read dataset.")

raw = load_data()

required = [
    "CreditScore","Age","Tenure","Balance","NumOfProducts",
    "HasCrCard","IsActiveMember","EstimatedSalary","Exited"
]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

data = raw.copy()
for c in required:
    data[c] = pd.to_numeric(data[c], errors="coerce")
    data[c] = data[c].fillna(data[c].median())

# ============================================================
# MODEL
# ============================================================
features = [
    "CreditScore","Age","Tenure","Balance","NumOfProducts",
    "HasCrCard","IsActiveMember","EstimatedSalary"
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

@st.cache_data
def build_model(X):
    scores = {}
    for k in range(2,9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(X)
    return scores, best_k, labels, final_model

silhouette_scores, best_k, labels, model = build_model(X_scaled)
data["Cluster"] = labels

# ============================================================
# BUSINESS PERSONAS
# Deterministic naming based on observed cluster behaviour.
# Technical cluster IDs stay hidden from the user.
# ============================================================
profile = data.groupby("Cluster").agg(
    Customers=("Cluster","size"),
    Avg_Age=("Age","mean"),
    Avg_CreditScore=("CreditScore","mean"),
    Avg_Balance=("Balance","mean"),
    Avg_Salary=("EstimatedSalary","mean"),
    Avg_Products=("NumOfProducts","mean"),
    Active_Rate=("IsActiveMember","mean"),
    Churn_Rate=("Exited","mean")
).reset_index()

profile["Active_Rate"] *= 100
profile["Churn_Rate"] *= 100

# Assign names using behavioural signals.
# This avoids displaying "Segment 1/2/3..." in the business UI.
def persona_name(row, med_balance, med_churn):
    if row.Avg_Products >= 1.9 and row.Active_Rate >= 75:
        return "Product Power Users"
    if row.Avg_Products >= 1.9 and row.Active_Rate < 75:
        return "Multi-Product Opportunity"
    if row.Churn_Rate >= med_churn * 1.25 and row.Active_Rate < 75:
        return "Retention Watchlist"
    if row.Avg_Balance >= med_balance * 1.25 and row.Active_Rate >= 75:
        return "High-Value Active"
    if row.Churn_Rate >= med_churn * 1.15:
        return "Re-Engagement Priority"
    if row.Active_Rate >= 75 and row.Churn_Rate < med_churn:
        return "Loyal Core"
    if row.Active_Rate < 75 and row.Avg_Balance >= med_balance:
        return "Dormant Value"
    return "Growth Potential"

med_balance = profile["Avg_Balance"].median()
med_churn = profile["Churn_Rate"].median()
profile["Persona"] = profile.apply(lambda r: persona_name(r, med_balance, med_churn), axis=1)

# Ensure unique labels if two clusters receive same label.
used = {}
names = []
for name in profile["Persona"]:
    used[name] = used.get(name, 0) + 1
    names.append(name if used[name] == 1 else f"{name} — Cohort {used[name]}")
profile["Persona"] = names

persona_map = dict(zip(profile["Cluster"], profile["Persona"]))
data["Customer Persona"] = data["Cluster"].map(persona_map)

# ============================================================
# FILTERS
# ============================================================
st.sidebar.markdown("## Customer Intelligence")
st.sidebar.caption("Interactive analysis controls")

geo_options = sorted(data["Geography"].dropna().unique().tolist()) if "Geography" in data.columns else []
gender_options = sorted(data["Gender"].dropna().unique().tolist()) if "Gender" in data.columns else []

selected_geo = st.sidebar.multiselect("Market", geo_options, default=geo_options)
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
status = st.sidebar.radio("Customer status", ["All customers","Active only","Inactive only","Exited only"], index=0)

age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

credit_min, credit_max = int(data["CreditScore"].min()), int(data["CreditScore"].max())
credit_range = st.sidebar.slider("Credit score", credit_min, credit_max, (credit_min, credit_max))

persona_options = sorted(data["Customer Persona"].unique().tolist())
selected_personas = st.sidebar.multiselect("Customer persona", persona_options, default=persona_options)

filtered = data.copy()
if geo_options:
    filtered = filtered[filtered["Geography"].isin(selected_geo)]
if gender_options:
    filtered = filtered[filtered["Gender"].isin(selected_gender)]
filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]
filtered = filtered[(filtered["CreditScore"] >= credit_range[0]) & (filtered["CreditScore"] <= credit_range[1])]
filtered = filtered[filtered["Customer Persona"].isin(selected_personas)]

if status == "Active only":
    filtered = filtered[filtered["IsActiveMember"] == 1]
elif status == "Inactive only":
    filtered = filtered[filtered["IsActiveMember"] == 0]
elif status == "Exited only":
    filtered = filtered[filtered["Exited"] == 1]

# ============================================================
# KPIs
# ============================================================
total = len(filtered)
exited = int(filtered["Exited"].sum()) if total else 0
active = int(filtered["IsActiveMember"].sum()) if total else 0
churn = filtered["Exited"].mean()*100 if total else 0
active_rate = filtered["IsActiveMember"].mean()*100 if total else 0
balance = filtered["Balance"].mean() if total else 0

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
  <div class="eyebrow">BANKING CUSTOMER INTELLIGENCE • ANALYTICS</div>
  <div class="hero-title">Customer Portfolio Command Center</div>
  <div class="hero-sub">
    Behavioural segmentation, retention signals and portfolio health — powered by K-Means clustering.
  </div>
</div>
""", unsafe_allow_html=True)

k = st.columns(5)
cards = [
    ("CUSTOMER BASE", f"{total:,}", "Filtered portfolio"),
    ("RETENTION RISK", f"{churn:.1f}%", "Observed exit rate"),
    ("ACTIVE RATE", f"{active_rate:.1f}%", "Engagement indicator"),
    ("AVG. BALANCE", f"${balance:,.0f}", "Portfolio average"),
    ("ACTIVE CUSTOMERS", f"{active:,}", "Currently active")
]
for col, (label, value, note) in zip(k, cards):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview", "Customer Personas", "Retention Intelligence", "Model Insights"
])

# Common plot theme
def polish(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=8,r=8,t=32,b=8),
        paper_bgcolor="#0D1220", plot_bgcolor="#0D1220",
        font=dict(color="#DCE4F2", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111827", font_color="#F4F7FB"),
        xaxis=dict(gridcolor="#202A3D", zeroline=False),
        yaxis=dict(gridcolor="#202A3D", zeroline=False)
    )
    return fig

with tab1:
    st.markdown('<div class="section">PORTFOLIO HEALTH</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)

    with a:
        counts = filtered["Customer Persona"].value_counts().reset_index()
        counts.columns = ["Persona","Customers"]
        fig = px.bar(counts, x="Customers", y="Persona", orientation="h",
                     title="Portfolio composition", text="Customers")
        fig.update_traces(textposition="outside")
        st.plotly_chart(polish(fig, 360), use_container_width=True, config={"displayModeBar":False})

    with b:
        if total:
            ret = pd.DataFrame({
                "Status":["Retained","Exited"],
                "Customers":[total-exited, exited]
            })
            fig = px.pie(ret, names="Status", values="Customers", hole=.68,
                         title="Retention health")
            fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>")
            st.plotly_chart(polish(fig, 360), use_container_width=True, config={"displayModeBar":False})

    with c:
        by_geo = filtered.groupby("Geography").agg(
            Customers=("CustomerId","size"),
            Churn=("Exited","mean")
        ).reset_index()
        by_geo["Churn"] *= 100
        fig = px.bar(by_geo, x="Geography", y="Churn", text="Churn",
                     title="Retention risk by market")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(polish(fig, 360), use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="section">BEHAVIOURAL SIGNALS</div>', unsafe_allow_html=True)
    x1,x2 = st.columns(2)
    with x1:
        fig = px.scatter(filtered, x="Age", y="Balance", size="EstimatedSalary",
                         color="Exited" if total else None,
                         hover_data=["CreditScore","NumOfProducts","IsActiveMember","Customer Persona"],
                         title="Customer value landscape", opacity=.72)
        st.plotly_chart(polish(fig, 390), use_container_width=True, config={"displayModeBar":False})
    with x2:
        age_bins = pd.cut(filtered["Age"], bins=[17,25,35,45,55,100],
                          labels=["18–25","26–35","36–45","46–55","56+"])
        age_churn = filtered.assign(AgeBand=age_bins).groupby("AgeBand", observed=False)["Exited"].mean().reset_index()
        age_churn["Exited"] *= 100
        fig = px.line(age_churn, x="AgeBand", y="Exited", markers=True,
                      title="Churn trend across age bands")
        fig.update_traces(line_width=3)
        st.plotly_chart(polish(fig, 390), use_container_width=True, config={"displayModeBar":False})

with tab2:
    st.markdown('<div class="section">CUSTOMER PERSONA INTELLIGENCE</div>', unsafe_allow_html=True)
    persona_profile = filtered.groupby("Customer Persona").agg(
        Customers=("CustomerId","size"),
        Avg_Age=("Age","mean"),
        Avg_Credit=("CreditScore","mean"),
        Avg_Balance=("Balance","mean"),
        Active_Rate=("IsActiveMember","mean"),
        Churn_Rate=("Exited","mean")
    ).reset_index()
    persona_profile["Active_Rate"] *= 100
    persona_profile["Churn_Rate"] *= 100

    p1,p2 = st.columns([1.1,1])
    with p1:
        fig = px.bar(persona_profile.sort_values("Churn_Rate"), x="Churn_Rate",
                     y="Customer Persona", orientation="h", text="Churn_Rate",
                     title="Retention exposure by persona")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(polish(fig, 410), use_container_width=True, config={"displayModeBar":False})
    with p2:
        fig = px.scatter(persona_profile, x="Avg_Balance", y="Churn_Rate",
                         size="Customers", hover_name="Customer Persona",
                         title="Value vs. retention exposure")
        st.plotly_chart(polish(fig, 410), use_container_width=True, config={"displayModeBar":False})

    table = persona_profile.rename(columns={
        "Customer Persona":"Persona","Customers":"Customers","Avg_Age":"Avg Age",
        "Avg_Credit":"Avg Credit","Avg_Balance":"Avg Balance",
        "Active_Rate":"Active %","Churn_Rate":"Churn %"
    }).copy()
    for c in ["Avg Age","Avg Credit","Avg Balance","Active %","Churn %"]:
        table[c] = table[c].round(1)
    st.dataframe(table, use_container_width=True, hide_index=True, height=330)

with tab3:
    st.markdown('<div class="section">RETENTION INTELLIGENCE</div>', unsafe_allow_html=True)
    r1,r2,r3 = st.columns(3)

    with r1:
        risk_count = int((filtered["Exited"]==1).sum())
        st.markdown(f'<div class="insight"><div class="insight-title">Observed exits</div><div class="insight-value">{risk_count:,}</div><div class="insight-copy">Customers marked as exited in the selected portfolio.</div></div>', unsafe_allow_html=True)
    with r2:
        high_risk = persona_profile.loc[persona_profile["Churn_Rate"].idxmax(),"Customer Persona"] if len(persona_profile) else "—"
        st.markdown(f'<div class="insight"><div class="insight-title">Highest exposure persona</div><div class="insight-value">{high_risk}</div><div class="insight-copy">Highest observed churn rate among the filtered personas.</div></div>', unsafe_allow_html=True)
    with r3:
        inactive = int((filtered["IsActiveMember"]==0).sum())
        st.markdown(f'<div class="insight"><div class="insight-title">Inactive customers</div><div class="insight-value">{inactive:,}</div><div class="insight-copy">Potential re-engagement population in the selected view.</div></div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        prod = filtered.groupby("NumOfProducts")["Exited"].mean().reset_index()
        prod["Exited"] *= 100
        prod["NumOfProducts"] = prod["NumOfProducts"].astype(str)
        fig = px.bar(prod, x="NumOfProducts", y="Exited", text="Exited",
                     title="Churn by product relationship")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(polish(fig, 350), use_container_width=True, config={"displayModeBar":False})
    with c2:
        tenure = filtered.groupby("Tenure")["Exited"].mean().reset_index()
        tenure["Exited"] *= 100
        fig = px.line(tenure, x="Tenure", y="Exited", markers=True,
                      title="Churn pattern across tenure")
        st.plotly_chart(polish(fig, 350), use_container_width=True, config={"displayModeBar":False})

with tab4:
    st.markdown('<div class="section">MODEL PERFORMANCE & CUSTOMER SPACE</div>', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    best_score = silhouette_scores[best_k]
    m1.metric("Optimal cluster count", best_k)
    m2.metric("Silhouette score", f"{best_score:.3f}")
    m3.metric("Features used", len(features))

    left,right = st.columns([1,1.35])
    with left:
        sdf = pd.DataFrame({"Clusters":list(silhouette_scores.keys()),
                            "Silhouette Score":list(silhouette_scores.values())})
        fig = px.line(sdf, x="Clusters", y="Silhouette Score", markers=True,
                      title="Clustering validation")
        st.plotly_chart(polish(fig, 380), use_container_width=True, config={"displayModeBar":False})

    with right:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame({
            "PC1":coords[:,0], "PC2":coords[:,1],
            "Persona":data["Customer Persona"].values,
            "Exited":data["Exited"].map({0:"Retained",1:"Exited"}).values
        })
        # Sample for browser performance while retaining full model.
        plot_df = pca_df.sample(min(3000,len(pca_df)), random_state=42)
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Persona",
                         symbol="Exited", hover_data=["Persona","Exited"],
                         title="Customer behavioural space")
        st.plotly_chart(polish(fig, 380), use_container_width=True, config={"displayModeBar":False})

# ============================================================
# CUSTOMER EXPLORER
# ============================================================
st.markdown('<div class="section">CUSTOMER EXPLORER</div>', unsafe_allow_html=True)
show_cols = [c for c in [
    "CustomerId","Surname","Geography","Gender","Age","CreditScore",
    "Balance","NumOfProducts","IsActiveMember","EstimatedSalary",
    "Customer Persona","Exited"
] if c in filtered.columns]
explorer = filtered[show_cols].copy()
explorer = explorer.rename(columns={
    "CustomerId":"Customer ID","CreditScore":"Credit Score",
    "NumOfProducts":"Products","IsActiveMember":"Active",
    "EstimatedSalary":"Est. Salary","Customer Persona":"Persona",
    "Exited":"Exited"
})
st.dataframe(explorer, use_container_width=True, hide_index=True, height=340)

st.caption(
    f"Banking Customer Intelligence • {len(data):,} records • "
    f"K-Means + PCA + Silhouette Validation • Interactive portfolio analysis"
)
