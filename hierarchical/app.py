import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering

# -------------------------------
# App Title
# -------------------------------
st.title("Customer Segmentation App")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Mall_Customers.csv")

# -------------------------------
# User Input Section
# -------------------------------
st.header("Enter New Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=1, max_value=200, value=50)
spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):

    # -------------------------------
    # Preprocessing
    # -------------------------------
    df_model = df.drop("CustomerID", axis=1)

    # Encode Gender
    le = LabelEncoder()
    df_model["Genre"] = le.fit_transform(df_model["Genre"])
    # Convert user input
    user_gender = 1 if gender == "Male" else 0
    new_customer = pd.DataFrame([[user_gender, age, income, spending]],
        columns=df_model.columns
    )
    # Combine old + new data
    final_data = pd.concat([df_model, new_customer], ignore_index=True)
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(final_data)
    # -------------------------------
    # Hierarchical Clustering
    # -------------------------------
    hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    final_data["Cluster"] = labels
    # Get cluster of new customer
    new_customer_cluster = final_data.iloc[-1]["Cluster"]
    st.success(f"New Customer belongs to Cluster: {new_customer_cluster}")

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    ax.scatter(
        final_data["Annual Income (k$)"],
        final_data["Spending Score (1-100)"],
        c=final_data["Cluster"]
    )
    # Highlight new customer
    ax.scatter(
        income, spending,
        s=200, marker="X"
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation")
    st.pyplot(fig)
