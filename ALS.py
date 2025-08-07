from implicit.als import AlternatingLeastSquares
import scipy.sparse as sp
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np

file_path_train_interactions = "data/train_interactions.parquet"
file_path_users_meta = "data/users_meta.parquet.parquet"
file_path_items_meta = "data/items_meta.parquet.parquet"
file_path_test_pair = "data/test_pairs.csv.csv"

print("Загружаем данные...")
train = pl.scan_parquet(file_path_train_interactions)
# Первые N строк
N = 1_000_000
train = train.slice(0, N)

def split_data(data):
    """
    Делит данные на две части:
      - df_history — все записи, кроме последних size строк;
      - df — последние size строк, отфильтрованные по условиям:
            * у пользователя должна быть история (в df_history);
            * у пользователя должно быть больше одного уникального значения target.
      Затем сортирует df по user_id и item_id.
    """
    size = 300_000
    # Если данных меньше, чем size, используем все записи
    if len(data) < size:
        df = data.copy()
        df_history = data.iloc[0:0].copy()  # пустой DataFrame
    else:
        df = data.iloc[-size:].copy()
        df_history = data.iloc[:-size].copy()

    df = df[df['user_id'].isin(df_history['user_id'])]
    user_unique_target = df.groupby('user_id')['target'].nunique()
    target_users = user_unique_target[user_unique_target > 1].index
    df = df[df['user_id'].isin(target_users)]
    df = df.sort_values(['user_id', 'item_id'])
    return df_history, df

train_history, train = split_data(train)

print("Обрабатываем данные...")
train_i = pl.scan_parquet(file_path_items_meta)
train_i = train_i.drop('embeddings').drop('source_id')
train = train.join(train_i, on='item_id')
train = train.with_columns((pl.col('timespent') / pl.col('duration')).alias('engagement_ratio'))

# Веса
train = train.collect().with_columns([
    ((pl.col("like") * 165)
        - (pl.col("dislike") * 70)
        + (pl.col("share") * 100)
        + (pl.col("bookmarks") * 115)
        + (pl.col("engagement_ratio") * 200)
        ).alias("weight")
        ])
train = train.drop('timespent').drop('like').drop('dislike').drop('share').drop('bookmarks').drop('duration').drop('engagement_ratio')

# Строим разреженную матрицу (user_id, item_id, weight)
user_item_matrix = sp.coo_matrix((
    train["weight"],
    (train["user_id"], train["item_id"])
))

# Обучаем ALS
model = AlternatingLeastSquares(factors=16, iterations=10, regularization=1)
model.fit(user_item_matrix)

def predict_like_probability(model, user_id, item_id):
    score = model.user_factors[user_id] @ model.item_factors[item_id].T
    return score  # Чем больше, тем вероятнее лайк

# Пример предсказания:
user_id = 123
item_id = 456
probability = predict_like_probability(model, user_id, item_id)
print(f"Вероятность лайка (от -1 до 1): {probability:.4f}")
