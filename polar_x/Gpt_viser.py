import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_roc_auc import custom_roc_auc
from catboost import CatBoostRanker, Pool

# === Пути к данным ===
file_path_train_interactions = "../data/train_interactions.parquet"
file_path_users_meta = "../data/users_meta.parquet.parquet"
file_path_items_meta = "../data/items_meta.parquet.parquet"

# === Загрузка данных ===
train = pd.read_parquet(file_path_train_interactions)
train = train.head(2_000)  # Ограничение по объему

train_i = pd.read_parquet(file_path_items_meta).drop(columns='embeddings', errors='ignore')
train_u = pd.read_parquet(file_path_users_meta)

# === Объединение данных ===
train = train.merge(train_i, how='left', on='item_id')
train = train.merge(train_u, how='left', on='user_id')

# === Фичи пользователя ===
user_stats = train.groupby('user_id').agg(
    views=('item_id', 'count'),
    total_likes=('like', 'sum'),
    total_shares=('share', 'sum'),
    total_bookmarks=('bookmarks', 'sum'),
    total_timespent=('timespent', 'sum')
).reset_index()

user_stats['avg_timespent_per_view'] = user_stats['total_timespent'] / user_stats['views']
user_stats['activity_score'] = user_stats[['total_likes', 'total_shares', 'total_bookmarks']].sum(axis=1)

train = train.merge(user_stats, on='user_id', how='left')

# === Фичи item-ов ===
item_stats = train.groupby('item_id').agg(
    item_views=('user_id', 'count'),
    item_likes=('like', 'sum')
).reset_index()
item_stats['item_ctr'] = item_stats['item_likes'] / item_stats['item_views']
train = train.merge(item_stats[['item_id', 'item_ctr']], on='item_id', how='left')

# === Дополнительные фичи ===
train['engagement_ratio'] = train['timespent'] / train['duration']
avg_duration = train.groupby('user_id')['duration'].transform('mean')
train['timespent_to_user_avg_duration'] = train['timespent'] / avg_duration

# === Целевая переменная ===
df = train.copy()
df['target'] = np.select(
    [df['like'] == 1, df['dislike'] == 1],
    [1, -1],
    default=0
)

# === Выбор фичей ===
features = ['timespent', 'share', 'bookmarks', 'source_id', 'duration',
            'age', 'engagement_ratio', 'timespent_to_user_avg_duration', 'item_ctr']

user_ids = df['user_id'].values
X = df[features]
y = df['target'].values

# === Трен/тест сплит ===
X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
    X, y, user_ids, test_size=0.2, random_state=42
)

# === Группировка и сортировка ===
def prepare_catboost_ranker_data(X, y, group_id):
    sorted_idx = np.argsort(group_id)
    X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
    y_sorted = np.array(y)[sorted_idx]
    group_id_sorted = np.array(group_id)[sorted_idx]
    group_sizes = pd.Series(group_id_sorted).value_counts().sort_index().values
    return X_sorted, y_sorted, group_id_sorted, group_sizes

X_train_sorted, y_train_sorted, group_id_train_sorted, group_sizes_train = prepare_catboost_ranker_data(X_train, y_train, user_ids_train)
X_test_sorted, y_test_sorted, group_id_test_sorted, group_sizes_test = prepare_catboost_ranker_data(X_test, y_test, user_ids_test)

# === Создание пулов ===
train_pool = Pool(X_train_sorted, y_train_sorted, group_id=group_id_train_sorted)
val_pool = Pool(X_test_sorted, y_test_sorted, group_id=group_id_test_sorted)

# === Обучение модели ===
model = CatBoostRanker(
    loss_function='YetiRank',
    task_type='CPU',
    iterations=1,
    reg_lambda=24,
    depth=5,
    min_child_samples=1,
    random_state=17,
    learning_rate=0.05,
    early_stopping_rounds=50,
    verbose=100
)

model.fit(train_pool, eval_set=val_pool)

# === Предсказания и метрики ===
y_pred_train = model.predict(X_train_sorted)
y_pred_test = model.predict(X_test_sorted)

print("Roc Auc Train: ", custom_roc_auc(y_pred_train, y_train_sorted, group_sizes_train))
print("Roc Auc Test:  ",  custom_roc_auc(y_pred_test, y_test_sorted, group_sizes_test))

