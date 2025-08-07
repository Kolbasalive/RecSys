import pickle
import pandas as pd
import matplotlib.pyplot as plt
import catboost
import time
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from custom_roc_auc import custom_roc_auc

# Загрузка данных
file_path_train_interactions = "../data/train_interactions.parquet"
file_path_users_meta = "../data/users_meta.parquet.parquet"
file_path_items_meta = "../data/items_meta.parquet.parquet"
file_path_test_pair = "../data/test_pairs.csv.csv"

data = pd.read_parquet(file_path_train_interactions).head(10_000_000)
items_meta = pd.read_parquet(file_path_items_meta)
users_meta = pd.read_parquet(file_path_users_meta)
test_pairs = pd.read_csv(file_path_test_pair)

data = data.merge(items_meta.drop('embeddings', axis=1), how='left', on='item_id', sort=False)
data = data.merge(users_meta, how='left', on='user_id', sort=False)
test_pairs = test_pairs.merge(items_meta.drop('embeddings', axis=1), how='left', on='item_id')
test_pairs = test_pairs.merge(users_meta, how='left', on='user_id')
data['target'] = data['like'].astype('int32') - data['dislike']

def compute_als_embeddings(df_history, factors=16):
    # Кодируем user_id и item_id в числовые категории
    user_ids = df_history['user_id'].astype('category')
    item_ids = df_history['item_id'].astype('category')

    df_history['user_cat'] = user_ids.cat.codes
    df_history['item_cat'] = item_ids.cat.codes

    # Обратные отображения
    user_cat_to_user_id = pd.Series(user_ids.cat.categories).reset_index().rename(columns={'index': 'user_cat', 0: 'user_id'})
    item_cat_to_item_id = pd.Series(item_ids.cat.categories).reset_index().rename(columns={'index': 'item_cat', 0: 'item_id'})

    # Матрица взаимодействий: item x user
    matrix = coo_matrix(
        (df_history["like"].astype(float), (df_history["item_cat"], df_history["user_cat"]))
    )

    # Обучаем ALS
    model = AlternatingLeastSquares(factors=factors, regularization=0.1, iterations=15)
    model.fit(matrix)

    # Эмбеддинги
    user_df = pd.DataFrame(model.user_factors, columns=[f"user_emb_{i}" for i in range(factors)])
    user_df["user_cat"] = range(len(user_df))
    user_df = user_df.merge(user_cat_to_user_id, on="user_cat", how="left").drop(columns=["user_cat"])

    item_df = pd.DataFrame(model.item_factors, columns=[f"item_emb_{i}" for i in range(factors)])
    item_df["item_cat"] = range(len(item_df))
    item_df = item_df.merge(item_cat_to_item_id, on="item_cat", how="left").drop(columns=["item_cat"])

    return user_df, item_df

def split_data(data):
    size = 3_000_000
    df = data[-size:].copy()
    df_history = data[:-size].copy()
    df = df[df['user_id'].isin(df_history['user_id'])]
    # Исключаем пользователей с однообразной реакцией(менее двух таргетов).
    # Которые уже учавствовали во взаимодействиях в history.🧠
    user_unique_target = df.groupby('user_id')['target'].nunique()
    target_users = user_unique_target[user_unique_target > 1].index
    df = df[df['user_id'].isin(target_users)]
    df = df.sort_values(['user_id', 'item_id'])
    return df_history, df

test_history, test = split_data(data)
train_history, train = split_data(test_history)

# Балансировка нагрузки на каждого пользователя, верхний предел
rank = train.groupby('user_id').cumcount() + 1
train = train[rank <= 1023]
group_indices = test['user_id'].value_counts().sort_index().values
metric = ROCAUC(group_indices=group_indices)

def _get_group(ser1, ser2):
    coef = {
        'user_id': 10 ** 6,
        'item_id': 10 ** 6,
        'source_id': 10 ** 5,
        'gender': 10,
        'age': 10 ** 2,
    }.get(ser2.name, 10 ** 8)
    return ser1.astype('int64') * coef + ser2

def get_group(df, cols):
    if len(cols) == 1:
        return df[cols[0]]
    return _get_group(df[cols[0]], get_group(df, cols[1:]))

def create_features(df_history, df):
    start = time.time()
    als_features = [col for col in df.columns if col.startswith("user_emb_") or col.startswith("item_emb_")]
    features = df[als_features + ['gender', 'age', 'duration', 'source_id']].copy()

    df_history['timespent_share'] = (df_history['timespent'] / df_history['duration']).clip(upper=2)

    bool_cols = ['like', 'dislike', 'share', 'bookmarks']
    all_new_features = {}

    print('Part 1: ', time.time() - start)
    for col in ['user_id', 'item_id', 'source_id', 'gender', 'age']:
        counts = df_history[col].value_counts()
        all_new_features[f'{col}_counts'] = df[col].map(counts).fillna(0)

        for bool_col in bool_cols:
            col_sum = df_history.groupby(col)[bool_col].sum()
            key = f'{col}_{bool_col}'
            all_new_features[f'{key}_sum'] = df[col].map(col_sum).fillna(0).astype('float32')
            all_new_features[f'{key}_mean'] = (
                all_new_features[f'{key}_sum'] / all_new_features[f'{col}_counts']
            ).astype('float32')

    print('Part 2: ', time.time() - start)
    for col1, col2 in [
        ('user_id', 'source_id'),
        ('item_id', 'gender'),
        ('item_id', 'age'),
        ('source_id', 'gender'),
        ('source_id', 'age')
    ]:
        col_prefix = f'{col1}_{col2}'
        group = get_group(df, (col1, col2))
        group_history = get_group(df_history, (col1, col2))
        counts = group_history.value_counts()
        all_new_features[f'{col_prefix}_counts'] = group.map(counts).fillna(0).astype('float32')

        for bool_col in bool_cols:
            col_sum = df_history.groupby(group_history)[bool_col].sum()
            key = f'{col_prefix}_{bool_col}'
            all_new_features[f'{key}_sum'] = group.map(col_sum).fillna(0).astype('float32')
            all_new_features[f'{key}_mean'] = (
                all_new_features[f'{key}_sum'] / all_new_features[f'{col_prefix}_counts']
            ).astype('float32')

        for col in ['timespent', 'timespent_share']:
            mean_col = df_history[col].groupby(group_history).mean()
            all_new_features[f'{col_prefix}_{col}_mean'] = group.map(mean_col).fillna(-1).astype('float32')

    print("Part 3: ", time.time() - start)
    # 👇 Одним махом добавляем все новые колонки
    features = pd.concat([features] + [pd.Series(v, name=k) for k, v in all_new_features.items()], axis=1)
    return features

# ALS
user_embs, item_embs = compute_als_embeddings(train_history)
train = train.merge(user_embs, on='user_id', how='left')
train = train.merge(item_embs, on='item_id', how='left')
train_f = train.copy()
test = test.merge(user_embs, on='user_id', how='left')
test = test.merge(item_embs, on='item_id', how='left')
test_f = test.copy()

# Создание фичей
train_features = create_features(train_history, train_f)
test_features = create_features(train_history, test_f)

