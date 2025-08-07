import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np

file_path_train_interactions = "../data/train_interactions.parquet"
file_path_users_meta = "../data/users_meta.parquet.parquet"
file_path_items_meta = "../data/items_meta.parquet.parquet"
file_path_test_pair = "../data/test_pairs.csv.csv"

print("Загружаем данные...")
train = pl.scan_parquet(file_path_train_interactions)
# Первые N строк
N = 3_000_000
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
    size = 100_000
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

# Матрица корреляций
# correlation_matrix = train.collect().select(
#     ["like", "dislike", "share", "bookmarks", "timespent", "duration", "engagement_ratio"]
# ).corr()
# print(correlation_matrix)

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
print("Полная парашa", train.filter(train["item_id"] == 21))

# Преобразуем в разреженную матрицу
users = train["user_id"].to_numpy()
items = train["item_id"].to_numpy()
weights = train["weight"].to_numpy()

# Определяем размеры
n_users = users.max() + 1
n_items = items.max() + 1

# Создаем разреженную матрицу user-item
user_item_matrix = csr_matrix((weights, (users, items)), shape=(n_users, n_items))

def get_similar_items(item_id, top_n=5):
    """Функция вычисляет косинусное сходство с другими айтемами на лету"""
    item_vector = user_item_matrix[:, item_id]  # Берем столбец (все юзеры, 1 айтем)
    similarity = cosine_similarity(user_item_matrix.T, item_vector.T)  # Считаем схожесть
    similarity = similarity.flatten()  # Делаем 1D массив

    # Получаем top-N похожих айтемов (исключая сам item_id) 56814, 46990, 21, 8811, 102739]
    similar_items = np.argsort(similarity)[::-1][:top_n+1]
    return similar_items.tolist()

# Пример: ищем похожие айтемы на item_id=123
similar_items = get_similar_items(21, top_n=5)
print("Похожие айтемы:", similar_items)

print("Полная параша я заебался это делать", train.filter(train["item_id"] == 21))

for similar_item in similar_items:
    print(train.filter(train['item_id'] == similar_item))