#!/usr/bin/env python3
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import catboost
from catboost import Pool, CatBoost
from catboost import CatBoostRanker, Pool
from custom_roc_auc import custom_roc_auc
from sklearn.metrics import roc_auc_score, roc_curve

def split_data(data):
    """
    Делит данные на две части:
      - df_history — все записи, кроме последних size строк;
      - df — последние size строк, отфильтрованные по условиям:
            * у пользователя должна быть история (в df_history);
            * у пользователя должно быть больше одного уникального значения target.
      Затем сортирует df по user_id и item_id.
    """
    size = 3_000_000
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

def main():
    # Пути к файлам (обновите пути в соответствии с расположением файлов)
    file_path_train_interactions = "data/train_interactions.parquet"
    file_path_users_meta = "data/users_meta.parquet.parquet"
    file_path_items_meta = "data/items_meta.parquet.parquet"
    file_path_test_pair = "data/test_pairs.csv.csv"

    print("Загружаем данные...")
    # data = pd.read_parquet(file_path_train_interactions).sort_values(by = "user_id").head(10_000_000)
    data = pd.read_parquet(file_path_train_interactions).head(10_000_000)
    items_meta = pd.read_parquet(file_path_items_meta)
    users_meta = pd.read_parquet(file_path_users_meta)
    test_pairs = pd.read_csv(file_path_test_pair)

    print("Null - значения", data.isnull().sum())

    # Объединяем данные с метаданными товаров и пользователей
    print("Объединяем таблицы...")
    # Если в items_meta есть колонка embeddings, отбрасываем её
    items_meta_drop = items_meta.drop(columns=['embeddings'], errors='ignore')
    data = data.merge(items_meta_drop, how='left', on='item_id', sort=False)
    data = data.merge(users_meta, how='left', on='user_id', sort=False)
    test_pairs = test_pairs.merge(items_meta_drop, how='left', on='item_id')
    test_pairs = test_pairs.merge(users_meta, how='left', on='user_id')

    # Создаем целевую переменную: разница между лайками и дизлайками
    print("Формируем таргет...")
    # data['target'] = data['like'].astype('int32') - data['dislike']
    # data['ignore'] = data['ignore'].fillna(0)
    data['target'] = (data['like'] * 165
    - data['dislike'] * 100
    + data['share'] * 100
    + data['bookmarks'] * 115
    + data['timespent']/data['duration'] * 100)

    min_target = data['target'].min()
    max_target = data['target'].max()
    # data['target'] = -1 + ((data['target'] - min_target) * 2) / (max_target - min_target)
    plt.scatter(data['user_id'], data['target'])
    print(data.describe())
    print("Что за валуе сонст")
    print(data['target'].min())
    print(data['target'].max())
    # 0.6596677446632122 2000
    # 0.6596021763505648 20000
    # 0.6588744420831325 200
    # Разбиваем данные (сначала на тестовую историю и тест, затем тестовую историю на тренировочную историю и тренировку)
    print("Разбиваем данные...")
    test_history, test = split_data(data)
    train_history, train = split_data(test_history)

    # Ограничиваем число записей для каждого пользователя (например, до 1023 записей)
    print("Ограничиваем число записей на пользователя в тренировочном наборе...")
    train['rank'] = train.groupby('user_id').cumcount() + 1
    train = train[train['rank'] <= 256].copy()
    train.drop(columns=['rank'], inplace=True)

    # Ограничиваем число записей для каждого пользователя в тестовом наборе
    test['rank'] = test.groupby('user_id').cumcount() + 1
    test = test[test['rank'] <= 256].copy()
    test.drop(columns=['rank'], inplace=True)
    print(test[test['dislike'] == 1])

    print(test)

    # Для простоты выберем базовый набор признаков (обновите список, если требуется)
    feature_columns = ['gender', 'age', 'duration', 'source_id']
    missing = [col for col in feature_columns if col not in train.columns]
    if missing:
        print(f"Внимание: отсутствуют признаки: {missing}")

    train_features = train[feature_columns].copy()
    test_features = test[feature_columns].copy()

    # Подготавливаем объекты Pool для CatBoost (используем user_id как group_id для ранжирования)
    print("Формируем CatBoost Pool...")
    train_pool = Pool(
        data=train_features,
        label=train['target'],
        group_id=train['user_id']
    )
    test_pool = Pool(
        data=test_features,
        label=test['target'],
        group_id=test['user_id']
    )

    params = {
        'iterations': 7,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'PairLogit',  # Функция потерь для ранжирования
        'task_type': 'CPU',
        'thread_count': -1,
        'verbose': 100
    }

    # 1-2 ...0.5765908 на 500 при depth 7 roc auc = ~0.5323
    #     ...0.5771    на 500 при depth 9 roc auc = 0.5294
    #     params = {
    #     'iterations': 100,
    #     'depth': 7,
    #     'learning_rate': 0.03,
    #     #'eval_metric': metric,
    #     'loss_function': 'YetiRank',
    #     'task_type': 'GPU',
    #     'devices': '0',
    # }
    print("Обучаем модель CatBoost на CPU...")
    model = CatBoost(params)

    #  model = CatBoostRanker(
    #     iterations=700,  # Количество итераций
    #     depth=6,  # Глубина дерева
    #     learning_rate=0.1,
    #     loss_function='YetiRank',  # Функция потерь YetiRank
    #     custom_metric=[custom_roc_auc],  # Кастомная метрика ROC AUC
    #     early_stopping_rounds=50,  # Остановка при отсутствии улучшений
    #     task_type="CPU"  # Обучение на процессоре
    # )

        # iterations=700,  # Уменьшаем итерации для скорости
        # depth=6,  # Меньше глубина дерева
        # learning_rate=0.1,
        # loss_function='Logloss',  # Основная функция потерь
        # # custom_metric=['AUC'],  # Кастомная метрика ROC AUC
        # early_stopping_rounds=50,  # Остановка при отсутствии улучшений
        # task_type="CPU"  # Обучение на процессоре
    model.fit(train_pool, eval_set=test_pool, verbose=500)

    group_indices = test['user_id'].value_counts().sort_index().values
    # metric = ROCAUC(group_indices=group_indices)

    y_train_pred = model.predict(train_pool)
    y_test_pred = model.predict(test_pool)
    print("Min y_train:", y_train_pred.min(), "Max y_train:", y_train_pred.max())
    print("Min y_test:", y_test_pred.min(), "Max y_test:", y_test_pred.max())

    roc_auc = roc_auc_score(test['like'].values, y_test_pred)
    print(f"РОК BLYAD: {roc_auc}")

    fpr, tpr, thresholds = roc_curve(test['like'].values, y_test_pred)
    # Строим график ROC-кривой
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    y_true_binary = (test['target'] >= 0).astype(int)
    # roc_auc = roc_auc_score(y_true_binary, y_test_pred)
    print(f"РОК BLYAD2: {roc_auc}")
    fpr, tpr, thresholds = roc_curve(test['like'].values, y_test_pred)
    # Строим график ROC-кривой
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    print(custom_roc_auc(y_test_pred, test['target'].values, group_indices))
    print(y_test_pred, test['target'].values)
    print(custom_roc_auc(y_train_pred, test['target'].values, group_indices))
    # test_pairs.with_columns(predict=als_predict).write_csv('sample_submission_my.csv')
    # Вычисление метрик
    # train_accuracy = accuracy_score(y_train, y_train_pred)
    # test_accuracy = accuracy_score(y_test, y_test_pred)

    # train_f1 = f1_score(y_train, y_train_pred)
    # test_f1 = f1_score(y_test, y_test_pred)

    # print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # print(f"Train F1-score: {train_f1:.4f}, Test F1-score: {test_f1:.4f}")

    # Сохраняем модель (при необходимости)
    model.save_model("catboost_model.cbm")

    # Делаем предсказания и выводим несколько значений
    print("Делаем предсказания на тестовом наборе...")
    predictions = model.predict(test_pool)
    print("Первые 10 предсказаний:")
    print(predictions[:10])

if __name__ == "__main__":
    start_time = time.time()
    main()
    # custom_roc_auc(predict, test['target'].values, group_indices)
    end_time = time.time()
    print("Времени всего прошло:", end_time - start_time)
