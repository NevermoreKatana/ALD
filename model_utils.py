import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import build_feature_matrix

def train_anomaly_model(X: pd.DataFrame, model_path: str = None) -> IsolationForest:
    """
    Обучает модель IsolationForest на признаках X. При необходимости сохраняет.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=500,
        max_samples='auto',
        contamination=0.01,
        random_state=42
    )
    model.fit(X_scaled)

    if model_path:
        joblib.dump((model, scaler), model_path)
        print(f"Модель сохранена в {model_path}")
    return model

def load_anomaly_model(model_path: str):
    """
    Загружает модель IsolationForest и scaler из файла.
    """
    model, scaler = joblib.load(model_path)
    return model, scaler

def infer_anomalies(df: pd.DataFrame, model: IsolationForest, scaler: StandardScaler) -> pd.DataFrame:
    """
    Прогоняет инференс на новом DataFrame, возвращает исходный df с отметками аномалий.
    """
    X = build_feature_matrix(df)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df['anomaly'] = preds
    df['anomaly_score'] = scores
    return df

def analyze_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Постобработка результатов: анализ аномалий, генерация причины.
    """
    anomalies_df = df[df['anomaly'] == -1].copy()

    mean_requests_per_minute = np.percentile(df['requests_per_minute'].dropna(), 98)
    mean_error_rate_5min = np.percentile(df['error_rate_5min'].dropna(), 98)

    p98_request_time = np.percentile(df['request_time'].dropna(), 98)

    reasons = []
    for idx, row in anomalies_df.iterrows():
        detail = []

        if row['request_time'] > p98_request_time:
            detail.append(
                f"Слишком долгое время ответа (request_time={row['request_time']:.3f} > p98={p98_request_time:.3f})"
            )

        if row['requests_per_minute'] > mean_requests_per_minute:
            detail.append(
                f"Частота запросов выше p98 (requests_per_minute={row['requests_per_minute']:.2f} > {mean_requests_per_minute:.2f})"
            )

        if row['response_status'] >= 400 and row['error_rate_5min'] > mean_error_rate_5min:
            detail.append(
                f"Слишком много ошибочных ответов (error_rate_5min={row['error_rate_5min']:.2f} > {mean_error_rate_5min:.2f})"
            )

        if row.get('is_rare_ua', 0) == 1:
            detail.append("Редкий User-Agent (is_rare_ua=1)")

        if abs(row.get('endpoint_zscore', 0)) > 2:
            detail.append(f"Аномальный Z-score (endpoint_zscore={row['endpoint_zscore']:.2f})")

        if row.get('is_suspicious_endpoint', 0) == 1:
            detail.append("Подозрительный Endpoint (is_suspicious_endpoint=1)")

        if row.get('is_redirect', 0) == 1:
            detail.append("Редирект (is_redirect=1)")

        if row.get('redirect_rate_5min', 0) > 0.3:
            detail.append(f"Высокая частота редиректов (redirect_rate_5min={row['redirect_rate_5min']:.2f})")

        if row.get('unique_ips_10min', 0) > 10:
            detail.append(f"Слишком много уникальных IP (unique_ips_10min={row['unique_ips_10min']})")

        if not detail:
            detail.append("Аномальное отклонение без явных конкретных метрик")

        reasons.append("; ".join(detail))

    anomalies_df['anomaly_reason'] = reasons

    anomalies_df = anomalies_df[anomalies_df['anomaly_reason'] != 'Аномальное отклонение без явных конкретных метрик']

    return anomalies_df