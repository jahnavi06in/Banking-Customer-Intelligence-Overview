# ============================================================
# BANKING CUSTOMER SEGMENTATION & BEHAVIOURAL ANALYTICS
# MACHINE LEARNING + STREAMLIT DASHBOARD
# ============================================================

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Customer Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM DASHBOARD STYLE
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #080914;
    color: #ffffff;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

.dashboard-header {
    background-color: #0d0f1c;
    border: 1px solid #1d2132;
    border-radius: 8px;
    padding: 13px 18px;
    margin-bottom: 10px;
}

.dashboard-title {
    color: #ffffff;
    font-size: 21px;
    font-weight: 700;
}

.dashboard-subtitle {
    color: #85899e;
    font-size: 11px;
    margin-top: 3px;
}

.metric-card {
    background-color: #0d0f1c;
    border: 1px solid #1d2132;
    border-radius: 7px;
    padding: 13px;
    min-height: 90px;
}

.metric-title {
    color: #85899e;
    font-size: 10px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
    margin-top: 7px;
}

.metric-small {
    color: #27d17f;
    font-size: 10px;
    margin-top: 4px;
}

.section-title {
    background-color: #0d0f1c;
    border: 1px solid #1d2132;
    border-bottom: none;
    border-radius: 7px 7px 0px 0px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
    padding: 8px 10px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1d2132;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "Churn_Modelling.csv"

RESULT_FILE = (
    BASE_DIR /
    "Banking_Customer_Segmentation_Result.csv"
)

PROFILE_FILE = (
    BASE_DIR /
    "Customer_Segment_Profile.csv"
)


# ============================================================
# SAFE CSV READER
# ============================================================

def read_csv_safe(file_path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin1",
        "cp1252"
    ]

    last_error = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=encoding
            )

            if df.empty:
                continue

            return df

        except Exception as error:

            last_error = error

    if last_error is not None:
        raise last_error

    raise ValueError(
        f"The file {file_path.name} is empty."
    )


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

if not RAW_FILE.exists():

    st.error(
        "Churn_Modelling.csv was not found."
    )

    st.write(
        "Expected location:"
    )

    st.code(
        str(RAW_FILE)
    )

    st.stop()


try:

    raw = read_csv_safe(RAW_FILE)

except Exception as error:

    st.error(
        "Unable to read Churn_Modelling.csv"
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# CHECK DATASET
# ============================================================

required_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

missing_columns = [
    column
    for column in required_columns
    if column not in raw.columns
]

if missing_columns:

    st.error(
        "Required columns are missing from Churn_Modelling.csv"
    )

    st.write(
        missing_columns
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(raw.columns)
    )

    st.stop()


# ============================================================
# COPY DATA
# ============================================================

data = raw.copy()


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for column in numeric_columns:

    if data[column].isna().any():

        data[column] = data[column].fillna(
            data[column].median()
        )


# ============================================================
# FEATURES FOR CLUSTERING
# ============================================================

features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


X = data[features].copy()


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

silhouette_scores = {}

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score


# ============================================================
# SELECT BEST K
# ============================================================

best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

data["Segment"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# CUSTOMER SEGMENT PROFILE
# ============================================================

profile = (
    data
    .groupby("Segment")
    .agg(
        Customers=("Segment", "size"),
        Avg_Age=("Age", "mean"),
        Avg_CreditScore=("CreditScore", "mean"),
        Avg_Balance=("Balance", "mean"),
        Avg_Salary=("EstimatedSalary", "mean"),
        Avg_Products=("NumOfProducts", "mean"),
        Active_Rate=("IsActiveMember", "mean"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)


# Convert rates to percentages

profile["Active_Rate"] = (
    profile["Active_Rate"] * 100
)

profile["Churn_Rate"] = (
    profile["Churn_Rate"] * 100
)


# ============================================================
# SAVE RESULTS
# ============================================================

try:

    data.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    profile.to_csv(
        PROFILE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

except Exception:
    pass


# ============================================================
# PCA
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(
    X_scaled
)

pca_data = pd.DataFrame({

    "PC1": pca_result[:, 0],

    "PC2": pca_result[:, 1],

    "Segment": data["Segment"].astype(str)

})


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(data)

total_exited = int(
    data["Exited"].sum()
)

total_active = int(
    data["IsActiveMember"].sum()
)

overall_churn = (
    data["Exited"].mean() * 100
)

overall_active = (
    data["IsActiveMember"].mean() * 100
)

average_balance = (
    data["Balance"].mean()
)

average_credit = (
    data["CreditScore"].mean()
)

average_salary = (
    data["EstimatedSalary"].mean()
)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown("""
<div class="dashboard-header">

<div class="dashboard-title">
● Banking Customer Intelligence Overview
</div>

<div class="dashboard-subtitle">
Customer Segmentation • Behavioural Analytics • Churn Insights
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP KPI CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    TOTAL CUSTOMERS
    </div>

    <div class="metric-value">
    {total_customers:,}
    </div>

    <div class="metric-small">
    Customer records
    </div>

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    CUSTOMER SEGMENTS
    </div>

    <div class="metric-value">
    {best_k}
    </div>

    <div class="metric-small">
    K-Means clusters
    </div>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    CHURN RATE
    </div>

    <div class="metric-value">
    {overall_churn:.1f}%
    </div>

    <div class="metric-small">
    Observed churn
    </div>

    </div>
    """, unsafe_allow_html=True)


with c4:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    ACTIVE CUSTOMERS
    </div>

    <div class="metric-value">
    {overall_active:.1f}%
    </div>

    <div class="metric-small">
    Active membership
    </div>

    </div>
    """, unsafe_allow_html=True)


with c5:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    AVG BALANCE
    </div>

    <div class="metric-value">
    ${average_balance:,.0f}
    </div>

    <div class="metric-small">
    Customer average
    </div>

    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(
    [1.25, 1, 1]
)


# ============================================================
# CUSTOMER SEGMENT DISTRIBUTION
# ============================================================

with col1:

    st.markdown(
        '<div class="section-title">'
        '● Customer Segment Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    segment_counts = (
        data["Segment"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    fig = px.bar(
        segment_counts,
        x="Segment",
        y="Customers",
        text="Customers"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        plot_bgcolor="#0d0f1c",
        font=dict(
            color="#d9dce8",
            size=10
        ),
        xaxis=dict(
            title="Segment",
            gridcolor="#1c2032"
        ),
        yaxis=dict(
            title="Customers",
            gridcolor="#1c2032"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# CHURN GAUGE
# ============================================================

with col2:

    st.markdown(
        '<div class="section-title">'
        '● Customer Churn Rate'
        '</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall_churn,
            number={
                "suffix": "%",
                "font": {
                    "size": 28
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#27d17f"
                },
                "bgcolor": "#171a29",
                "borderwidth": 0
            }
        )
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        font={
            "color": "#ffffff"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# ACTIVE CUSTOMER GAUGE
# ============================================================

with col3:

    st.markdown(
        '<div class="section-title">'
        '● Active Customer Rate'
        '</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall_active,
            number={
                "suffix": "%",
                "font": {
                    "size": 28
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#27d17f"
                },
                "bgcolor": "#171a29",
                "borderwidth": 0
            }
        )
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        font={
            "color": "#ffffff"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# ROW 2
# ============================================================

col4, col5, col6 = st.columns(
    [1.25, 1, 1]
)


# ============================================================
# CHURN BY SEGMENT
# ============================================================

with col4:

    st.markdown(
        '<div class="section-title">'
        '● Churn Behaviour by Segment'
        '</div>',
        unsafe_allow_html=True
    )

    churn_data = profile[
        [
            "Segment",
            "Churn_Rate"
        ]
    ].copy()

    fig = px.bar(
        churn_data,
        x="Segment",
        y="Churn_Rate",
        text="Churn_Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        plot_bgcolor="#0d0f1c",
        font=dict(
            color="#d9dce8",
            size=10
        ),
        xaxis=dict(
            title="Segment",
            gridcolor="#1c2032"
        ),
        yaxis=dict(
            title="Churn Rate (%)",
            gridcolor="#1c2032"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# BALANCE BY SEGMENT
# ============================================================

with col5:

    st.markdown(
        '<div class="section-title">'
        '● Average Balance by Segment'
        '</div>',
        unsafe_allow_html=True
    )

    balance_data = profile[
        [
            "Segment",
            "Avg_Balance"
        ]
    ].copy()

    fig = px.bar(
        balance_data,
        x="Segment",
        y="Avg_Balance"
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        plot_bgcolor="#0d0f1c",
        font=dict(
            color="#d9dce8",
            size=10
        ),
        xaxis=dict(
            title="Segment",
            gridcolor="#1c2032"
        ),
        yaxis=dict(
            title="Average Balance",
            gridcolor="#1c2032"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# RETENTION STATUS
# ============================================================

with col6:

    st.markdown(
        '<div class="section-title">'
        '● Customer Retention Status'
        '</div>',
        unsafe_allow_html=True
    )

    retention = pd.DataFrame({
        "Status": [
            "Retained",
            "Exited"
        ],
        "Customers": [
            total_customers - total_exited,
            total_exited
        ]
    })

    fig = px.pie(
        retention,
        names="Status",
        values="Customers",
        hole=0.60
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        font=dict(
            color="#d9dce8",
            size=10
        ),
        legend=dict(
            orientation="h",
            y=-0.05
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# ROW 3
# ============================================================

col7, col8 = st.columns(
    [1.65, 1]
)


# ============================================================
# CUSTOMER SEGMENT PROFILE TABLE
# ============================================================

with col7:

    st.markdown(
        '<div class="section-title">'
        '● Customer Segment Profile'
        '</div>',
        unsafe_allow_html=True
    )

    table = profile.copy()

    table = table.rename(
        columns={
            "Segment": "Segment",
            "Customers": "Customers",
            "Avg_Age": "Avg Age",
            "Avg_CreditScore": "Avg Credit",
            "Avg_Balance": "Avg Balance",
            "Avg_Salary": "Avg Salary",
            "Avg_Products": "Products",
            "Active_Rate": "Active %",
            "Churn_Rate": "Churn %"
        }
    )

    table["Avg Age"] = (
        table["Avg Age"].round(1)
    )

    table["Avg Credit"] = (
        table["Avg Credit"].round(0)
    )

    table["Avg Balance"] = (
        table["Avg Balance"].round(0)
    )

    table["Avg Salary"] = (
        table["Avg Salary"].round(0)
    )

    table["Products"] = (
        table["Products"].round(2)
    )

    table["Active %"] = (
        table["Active %"].round(1)
    )

    table["Churn %"] = (
        table["Churn %"].round(1)
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=285
    )


# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

with col8:

    st.markdown(
        '<div class="section-title">'
        '● Clustering Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    silhouette_df = pd.DataFrame({
        "Clusters": list(
            silhouette_scores.keys()
        ),
        "Silhouette Score": list(
            silhouette_scores.values()
        )
    })

    fig = px.line(
        silhouette_df,
        x="Clusters",
        y="Silhouette Score",
        markers=True
    )

    fig.update_layout(
        height=285,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="#0d0f1c",
        plot_bgcolor="#0d0f1c",
        font=dict(
            color="#d9dce8",
            size=10
        ),
        xaxis=dict(
            title="Clusters",
            gridcolor="#1c2032"
        ),
        yaxis=dict(
            title="Silhouette Score",
            gridcolor="#1c2032"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# PCA SECTION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">'
    '● Customer Segment Behaviour — PCA Visualization'
    '</div>',
    unsafe_allow_html=True
)

fig = px.scatter(
    pca_data,
    x="PC1",
    y="PC2",
    color="Segment",
    hover_data=["Segment"],
    opacity=0.65
)

fig.update_layout(
    height=420,
    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10
    ),
    paper_bgcolor="#0d0f1c",
    plot_bgcolor="#0d0f1c",
    font=dict(
        color="#d9dce8",
        size=10
    ),
    xaxis=dict(
        title="Principal Component 1",
        gridcolor="#1c2032"
    ),
    yaxis=dict(
        title="Principal Component 2",
        gridcolor="#1c2032"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ============================================================
# BOTTOM KPI ROW
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)


with b1:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    AVG CREDIT SCORE
    </div>

    <div class="metric-value">
    {average_credit:.0f}
    </div>

    </div>
    """, unsafe_allow_html=True)


with b2:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    AVG SALARY
    </div>

    <div class="metric-value">
    ${average_salary:,.0f}
    </div>

    </div>
    """, unsafe_allow_html=True)


with b3:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    EXITED CUSTOMERS
    </div>

    <div class="metric-value">
    {total_exited:,}
    </div>

    </div>
    """, unsafe_allow_html=True)


with b4:

    best_score = silhouette_scores[best_k]

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    BEST SILHOUETTE SCORE
    </div>

    <div class="metric-value">
    {best_score:.3f}
    </div>

    <div class="metric-small">
    {best_k} clusters selected
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    text-align:center;
    color:#656a7e;
    font-size:10px;
    padding:15px;
">

Banking Customer Segmentation & Behavioural Analytics
&nbsp; • &nbsp;
K-Means Clustering
&nbsp; • &nbsp;
Machine Learning

</div>
""", unsafe_allow_html=True)
