# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.simulate_queries import simulate_user_queries
from src.train_model import train_ranker
import ast

app = FastAPI(title="Real Estate Search Ranking API")

# Load and preprocess data once on startup
df = pd.read_csv("./data/processed/data.csv")


# Request body schema
class QueryRequest(BaseModel):
    city: str
    state: str
    min_bed: int
    min_bath: int
    min_house_size: int
    max_price: int
    top_n: int = 10

@app.post("/rank")
def rank_listings(request: QueryRequest):
    query = [{
        'city': request.city,
        'state': request.state,
        'min_bed': request.min_bed,
        'min_bath': request.min_bath,
        'min_house_size': request.min_house_size,
        'max_price': request.max_price
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

    top_results = query_df.sort_values(by='relevance_score', ascending=False).head(request.top_n)
    response = top_results[['city', 'state', 'bed', 'bath', 'house_size', 'price', 'relevance_score']].to_dict(orient='records')
    return response
