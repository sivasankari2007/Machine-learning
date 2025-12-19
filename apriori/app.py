import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori Algorithm App", layout="centered")

st.title("🛒 Apriori Algorithm – Association Rule Mining")
st.write("Upload a CSV file with transactional data (1 = Yes, 0 = No)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    min_support = st.slider("Minimum Support", 0.01, 1.0, 0.2)
    min_confidence = st.slider("Minimum Confidence", 0.01, 1.0, 0.5)

    if st.button("Run Apriori Algorithm"):
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found. Try lowering support.")
        else:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            st.subheader("✅ Frequent Itemsets")
            st.dataframe(frequent_itemsets)

            st.subheader("🔗 Association Rules")
            st.dataframe(
                rules[["antecedents", "consequents", "support", "confidence", "lift"]]
            )
