from scipy.sparse import csr_matrix
import polars as pl
import implicit


train = pl.scan_parquet("data/train_interactions.parquet")
train = train.filter((pl.col("like") + pl.col("dislike")) >= 1)
train = train.with_columns(weight=pl.col("like") - pl.col("dislike"))
train = train.select("user_id", "item_id", "weight")
train = train.collect()
items_meta = pl.read_parquet("data/items_meta.parquet.parquet")
users_meta = pl.read_parquet("data/users_meta.parquet.parquet")
n_items = items_meta["item_id"].max() + 1
n_users = users_meta["user_id"].max() + 1
train = csr_matrix((train["weight"],
                    (train["user_id"].to_numpy(),
                     train["item_id"].to_numpy())),
                   shape=(n_users, n_items))
model = implicit.als.AlternatingLeastSquares(factors=16,
                                             iterations=10,
                                             regularization=1,
                                             alpha=100,
                                             calculate_training_loss=True)
model.fit(train)
test_pairs = pl.read_csv('data/test_pairs.csv.csv')
print(model.user_factors[test_pairs['user_id']])
print(model.item_factors[test_pairs['item_id']])

als_predict = (model.user_factors[test_pairs['user_id']] *
               model.item_factors[test_pairs['item_id']]).sum(axis=1)
test_pairs.with_columns(predict=als_predict).write_csv('sample_submission_my.csv')

# # ROC AUC:
# from sklearn.metrics import roc_auc_score
#
# # Загружаем тестовый набор
# test_pairs = pl.read_csv('data/test_pairs.csv.csv')
#
# # Получаем предсказания модели
# als_predict = (model.user_factors[test_pairs['user_id']] *
#                model.item_factors[test_pairs['item_id']]).sum(axis=1)
#
# # Загружаем реальные оценки (например, из тестового датасета)
# test_ground_truth = pl.read_parquet("data/train_interactions.parquet")
# test_ground_truth = test_ground_truth.select("user_id", "item_id", "like")
#
# # Объединяем с предсказаниями
# test_data = test_pairs.with_columns(predict=als_predict).join(test_ground_truth,
#                                                               on=["user_id", "item_id"],
#                                                               how="left")
#
# # Заполняем NaN нулями (если у какого-то пользователя нет лайка)
# test_data = test_data.fill_null(0)
#
# # ROC AUC
# roc_auc = roc_auc_score(test_data["like"], test_data["predict"])
# print(f"ROC AUC: {roc_auc:.4f}")
