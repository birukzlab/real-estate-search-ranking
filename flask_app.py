# flask_app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.simulate_queries import simulate_user_queries
from src.train_model import train_ranker
import ast
import os

app = Flask(__name__)

# Load and preprocess data once
data_path = "./data/processed/data.csv"
df = pd.read_csv(data_path)


@app.route("/", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/rank", methods=["POST"])
def rank():
    city = request.form.get("city")
    state = request.form.get("state")
    min_bed = int(request.form.get("min_bed"))
    min_bath = int(request.form.get("min_bath"))
    min_house_size = int(request.form.get("min_house_size"))
    max_price = int(request.form.get("max_price"))

    query = [{
        'city': city,
        'state': state,
        'min_bed': min_bed,
        'min_bath': min_bath,
        'min_house_size': min_house_size,
        'max_price': max_price
    }]

    query_df = simulate_user_queries(df, query, n_negative=200)
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

    top_results = query_df.sort_values(by='relevance_score', ascending=False).head(10)
    listings = top_results[['city', 'state', 'bed', 'bath', 'house_size', 'price', 'relevance_score']].to_dict(orient='records')

    return render_template("results.html", listings=listings)

if __name__ == "__main__":
    app.run(debug=True)
