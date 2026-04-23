import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Workload Imbalance Dashboard", layout="wide")

st.title("Workload Imbalance Detection Dashboard")
st.markdown("This dashboard presents agent-level workload indicators and compares rule-based classification with ML predictions.")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("cleaned_helpdesk_tickets.csv")

    # Fix missing resolution times
    df["resolution_time_hrs"] = df["resolution_time_hrs"].fillna(df["resolution_time_hrs"].median())

    # Controlled agent assignment
    np.random.seed(42)

    def assign_agent(team):
        if team == "Desktop Support":
            return np.random.choice(["D1", "D2", "D3", "D4", "D5"], p=[0.4, 0.25, 0.2, 0.1, 0.05])
        elif team == "Application Support":
            return np.random.choice(["A1", "A2", "A3", "A4", "A5"], p=[0.35, 0.25, 0.2, 0.1, 0.1])
        elif team == "Network Team":
            return np.random.choice(["N1", "N2", "N3", "N4", "N5"], p=[0.3, 0.25, 0.2, 0.15, 0.1])
        elif team == "Security Team":
            return np.random.choice(["S1", "S2", "S3", "S4", "S5"], p=[0.3, 0.25, 0.2, 0.15, 0.1])
        return "Unknown"

    df["agent_id"] = df["assigned_team"].apply(assign_agent)

    # Feature creation
    df["is_backlog"] = df["status"] != "Resolved"
    df["is_high_priority"] = df["priority"].isin(["High", "Critical"])

    # Agent-level metrics
    tickets_per_agent = df.groupby("agent_id").size().rename("ticket_count")
    backlog_per_agent = df.groupby("agent_id")["is_backlog"].sum().rename("backlog_count")
    avg_resolution = df.groupby("agent_id")["resolution_time_hrs"].mean().rename("avg_resolution_time")
    priority_ratio = df.groupby("agent_id")["is_high_priority"].mean().rename("high_priority_ratio")

    metrics_df = pd.concat(
        [tickets_per_agent, backlog_per_agent, avg_resolution, priority_ratio],
        axis=1
    ).reset_index()

    # Rule-based classification
    ticket_threshold_high = metrics_df["ticket_count"].quantile(0.75)
    ticket_threshold_low = metrics_df["ticket_count"].quantile(0.25)
    backlog_threshold = metrics_df["backlog_count"].quantile(0.75)

    def classify_workload(row):
        if row["ticket_count"] > ticket_threshold_high or row["backlog_count"] > backlog_threshold:
            return "Overloaded"
        elif row["ticket_count"] < ticket_threshold_low:
            return "Underloaded"
        else:
            return "Balanced"

    metrics_df["rule_based_status"] = metrics_df.apply(classify_workload, axis=1)

    # ML model
    X = metrics_df[["ticket_count", "backlog_count", "avg_resolution_time", "high_priority_ratio"]]
    y = metrics_df["rule_based_status"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    metrics_df["ml_predicted_status"] = le.inverse_transform(model.predict(X))

    return metrics_df

metrics_df = load_and_prepare_data()

# Top metrics
st.subheader("Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Agents", len(metrics_df))
col2.metric("Overloaded Agents", (metrics_df["rule_based_status"] == "Overloaded").sum())
col3.metric("Balanced Agents", (metrics_df["rule_based_status"] == "Balanced").sum())
col4.metric("Underloaded Agents", (metrics_df["rule_based_status"] == "Underloaded").sum())

# Filter
st.subheader("Filter Agents")
selected_status = st.selectbox(
    "Select rule-based workload status",
    ["All"] + sorted(metrics_df["rule_based_status"].unique().tolist())
)

if selected_status == "All":
    filtered_df = metrics_df.copy()
else:
    filtered_df = metrics_df[metrics_df["rule_based_status"] == selected_status].copy()

# Agent table
st.subheader("Agent Workload Summary")


if filtered_df.empty:
    st.warning("No data to display")
else:
    st.dataframe(filtered_df, use_container_width=True)

# Charts
st.markdown("Insights")
st.write("The charts below illustrate how workload is distributed across agents based on ticket volume and backlog levels.")

st.subheader("Ticket Count by Agent")
st.bar_chart(filtered_df.set_index("agent_id")["ticket_count"])

st.subheader("Backlog Count by Agent")
st.bar_chart(filtered_df.set_index("agent_id")["backlog_count"])

st.write(
    "Agents with significantly higher ticket and backlog counts are classified as overloaded, "
    "while those with lower values are considered underloaded."
)

# Comparison
st.write(
    "This section compares rule-based workload classification with predictions from the decision tree model, "
    "highlighting areas of agreement and disagreement."
)

st.subheader("Comparison of rules & ML classifications")
comparison_df = filtered_df[["agent_id", "rule_based_status", "ml_predicted_status"]].copy()
comparison_df["match"] = comparison_df["rule_based_status"] == comparison_df["ml_predicted_status"]
st.dataframe(comparison_df, use_container_width=True)
