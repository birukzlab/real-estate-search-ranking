import pandas as pd
import numpy as np
import ast
from src.simulate_queries import simulate_user_queries
from src.train_model import train_ranker
from src.evaluate_ranking import evaluate_grouped_metrics
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset (replace with your actual path)
df = pd.read_csv("./data/processed/data.csv")



# Define sample queries
queries = [
    {'city': 'Orlando', 'min_bed': 3, 'max_price': 500000},
    {'state': 'Florida', 'min_bath': 2, 'max_price': 600000},
    {'city': 'San Juan', 'min_house_size': 1500},
    {'state': 'Texas', 'max_price': 400000},
    {'city': 'Ponce', 'min_bed': 4, 'min_bath': 3, 'max_price': 750000}
]

# Simulate queries and build dataset
query_df = simulate_user_queries(df, queries)

# Extract features from query

query_df['query_dict'] = query_df['query'].apply(ast.literal_eval)
query_df['query_min_bed'] = query_df['query_dict'].apply(lambda x: x.get('min_bed', np.nan))
query_df['query_max_price'] = query_df['query_dict'].apply(lambda x: x.get('max_price', np.nan))
query_df['query_min_bath'] = query_df['query_dict'].apply(lambda x: x.get('min_bath', np.nan))
query_df['query_min_house_size'] = query_df['query_dict'].apply(lambda x: x.get('min_house_size', np.nan))
query_df['query_city_encoded'] = LabelEncoder().fit_transform(query_df['query_dict'].apply(lambda x: str(x.get('city', 'NaN'))))
query_df['query_state_encoded'] = LabelEncoder().fit_transform(query_df['query_dict'].apply(lambda x: str(x.get('state', 'NaN'))))

# Prepare training data
features = [
    'bed', 'bath', 'house_size', 'lot_sqft', 'price',
    'query_min_bed', 'query_min_bath', 'query_min_house_size', 'query_max_price',
    'query_city_encoded', 'query_state_encoded'
]
X = query_df[features]
y = query_df['relevance']

# Train model
model, X_test, y_test, y_proba = train_ranker(X, y)

# Attach query ID for ranking evaluation
query_df['query_id'] = query_df['query_city_encoded'].astype(str) + '-' + query_df['query_state_encoded'].astype(str)
results_df = X_test.copy()
results_df['true_relevance'] = y_test.values
results_df['pred_proba'] = y_proba
results_df['query_id'] = query_df.loc[X_test.index, 'query_id'].values

# Evaluate
ndcg, precision = evaluate_grouped_metrics(results_df)
print(f"Final NDCG@5: {ndcg:.4f}")
print(f"Final Precision@5: {precision:.4f}")