"""
Reduce AmEx train_data.csv (16GB, time-series) to one row per customer_ID.
Approach:
- Read in chunks (data is sorted by customer_ID, so chunk boundaries don't
  split a customer's rows across chunks... mostly. We handle the edge case
  by carrying over any "incomplete" trailing customer to the next chunk).
- Downcast numeric columns to float32 to save memory.
- For each customer_ID, keep only their LAST statement (most recent row).
- Merge with train_labels.csv on customer_ID at the end.
- Save as parquet (~150-300MB expected).

"""

import pandas as pd
import numpy as np


def reduce_train_data(train_data_path: str, train_labels_path: str, output_path: str,
                       chunksize: int = 500_000):

    # Categorical columns per AmEx data dictionary (low cardinality)
    cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
                 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

    leftover = pd.DataFrame()  # rows belonging to a customer split across chunk boundary
    aggregated_chunks = []

    reader = pd.read_csv(train_data_path, chunksize=chunksize)

    for i, chunk in enumerate(reader):
        # prepend leftover rows from previous chunk
        if not leftover.empty:
            chunk = pd.concat([leftover, chunk], ignore_index=True)

        # downcast numeric columns (skip customer_ID, S_2 date, and categoricals for now)
        numeric_cols = [c for c in chunk.columns
                         if c not in ['customer_ID', 'S_2'] + cat_cols]
        for col in numeric_cols:
            chunk[col] = chunk[col].astype('float32')

        # find the last customer_ID in this chunk — its rows might continue
        # into the next chunk, so hold them back
        last_cust = chunk['customer_ID'].iloc[-1]
        is_last_cust = chunk['customer_ID'] == last_cust

        complete_part = chunk[~is_last_cust]
        leftover = chunk[is_last_cust].copy()

        if not complete_part.empty:
            # take last row per customer (most recent statement = max S_2 date,
            # but since sorted, .tail(1) per group is equivalent and faster)
            last_rows = complete_part.groupby('customer_ID', sort=False).tail(1)
            aggregated_chunks.append(last_rows)

        print(f"Processed chunk {i+1}, rows so far aggregated: "
              f"{sum(len(c) for c in aggregated_chunks)}")

    # handle final leftover (last customer in the file)
    if not leftover.empty:
        last_rows = leftover.groupby('customer_ID', sort=False).tail(1)
        aggregated_chunks.append(last_rows)

    df = pd.concat(aggregated_chunks, ignore_index=True)
    print(f"Total customers after reduction: {len(df)}")

    # merge labels
    labels = pd.read_csv(train_labels_path)
    df = df.merge(labels, on='customer_ID', how='left')

    print(f"Final shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    reduce_train_data(
        train_data_path="../data/train_data.csv",      
        train_labels_path="../data/train_labels.csv",  
        output_path="amex_train_last_statement.parquet",
        chunksize=500_000
    )
