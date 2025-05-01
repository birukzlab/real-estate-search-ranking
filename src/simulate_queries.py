import pandas as pd
import numpy as np

def simulate_user_queries(df, queries, n_negative=300):
    all_rows = []
    for q in queries:
        relevant = df.copy()

        if 'city' in q:
            relevant = relevant[relevant['city'].str.lower() == q['city'].lower()]
        if 'state' in q:
            relevant = relevant[relevant['state'].str.lower() == q['state'].lower()]
        if 'min_bed' in q:
            relevant = relevant[relevant['bed'] >= q['min_bed']]
        if 'min_bath' in q:
            relevant = relevant[relevant['bath'] >= q['min_bath']]
        if 'min_house_size' in q:
            relevant = relevant[relevant['house_size'] >= q['min_house_size']]
        if 'max_price' in q:
            relevant = relevant[relevant['price'] <= q['max_price']]

        relevant = relevant.copy()
        relevant['relevance'] = 1
        relevant['query'] = str(q)

        irrelevant = df[~df.index.isin(relevant.index)].sample(n=min(n_negative, len(df)), random_state=42)
        irrelevant = irrelevant.copy()
        irrelevant['relevance'] = 0
        irrelevant['query'] = str(q)

        all_rows.append(relevant)
        all_rows.append(irrelevant)

    return pd.concat(all_rows, ignore_index=True)
