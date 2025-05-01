# Real Estate Search Ranking Model

This project builds a machine learning–powered search ranking engine for real estate listings using simulated user queries. It predicts the relevance of listings and ranks results using XGBoost and neural networks.

## Dataset
- 2.2 M property listings (location, price, size, bed/bath, etc.)

## Key Features
- Simulated 5+ user search queries (e.g. "3 beds under $500k in Austin")
- Feature engineering for both listing and query attributes
- XGBoost + Neural Network rankers trained to predict relevance
- Evaluation using NDCG@5 and Precision@5

## Results
- **NDCG@5**: 0.94  
- **Precision@5**: 0.97

## Repo Structure
- `src/`: All source code and ML logic
- `notebooks/`: Exploratory development
- `figures/`: Charts and visualizations
- `results/`: Final metrics

## Usage
Run:
```bash
python src/train_model.py
