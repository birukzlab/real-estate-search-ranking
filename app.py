# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from src.simulate_queries import simulate_user_queries
from src.train_model import train_ranker

st.set_page_config(page_title="Real Estate Search Ranking", layout="centered")
st.title("🏠 Real Estate Search Ranking App")
st.markdown("Simulate a home search and view the top ranked listings based on your preferences.")

# User input
city = st.text_input("City", value="Orlando")
state = st.text_input("State", value="Florida")
min_bed = st.slider("Minimum Bedrooms", 0, 10, 3)
min_bath = st.slider("Minimum Bathrooms", 0, 10, 2)
min_house_size = st.number_input("Minimum House Size (sqft)", value=1000)
max_price = st.number_input("Maximum Price", value=500000)

st.markdown("⬇️ Click below to run the search and rank listings")

# Add a small empty space for spacing
st.write(" ")

run_search = st.button("🔍 Rank Listings")


if run_search:
    with st.spinner("Ranking listings... Please wait."):
        # Load and preprocess data
        df = pd.read_csv("./data/processed/data.csv")

        # Simulate query
        query = [{
            'city': city,
            'state': state,
            'min_bed': min_bed,
            'min_bath': min_bath,
            'min_house_size': min_house_size,
            'max_price': max_price
        }]
        query_df = simulate_user_queries(df, query, n_negative=200)

        # Feature engineering for queries
        query_df['query_dict'] = query_df['query'].apply(ast.literal_eval)
        query_df['query_min_bed'] = query_df['query_dict'].apply(lambda x: x.get('min_bed', np.nan))
        query_df['query_max_price'] = query_df['query_dict'].apply(lambda x: x.get('max_price', np.nan))
        query_df['query_min_bath'] = query_df['query_dict'].apply(lambda x: x.get('min_bath', np.nan))
        query_df['query_min_house_size'] = query_df['query_dict'].apply(lambda x: x.get('min_house_size', np.nan))
        query_df['query_city_encoded'] = LabelEncoder().fit_transform(query_df['query_dict'].apply(lambda x: str(x.get('city', 'NaN'))))
        query_df['query_state_encoded'] = LabelEncoder().fit_transform(query_df['query_dict'].apply(lambda x: str(x.get('state', 'NaN'))))

        features = [
            'bed', 'bath', 'house_size', 'lot_sqft', 'price',
            'query_min_bed', 'query_min_bath', 'query_min_house_size', 'query_max_price',
            'query_city_encoded', 'query_state_encoded'
        ]
        X = query_df[features]
        y = query_df['relevance']

        model, _, _, _ = train_ranker(X, y)
        query_df['relevance_score'] = model.predict_proba(X)[:, 1]

        # Show top 10 results
        top_results = query_df.sort_values(by='relevance_score', ascending=False).head(10)
        st.subheader("🔝 Top 10 Ranked Listings")
        st.dataframe(top_results[['city', 'state', 'bed', 'bath', 'house_size', 'price', 'relevance_score']].reset_index(drop=True))
