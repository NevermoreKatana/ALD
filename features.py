import pandas as pd
import numpy as np

def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая предобработка и извлечение ключевых признаков.
    """
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT

    def extract_endpoint(request: str) -> str:
        try:
            parts = request.split()
            return parts[1] if len(parts) > 1 else 'unknown'
        except:
            return 'unknown'

    if 'request' in df.columns:
        df['endpoint'] = df['request'].apply(extract_endpoint)
    else:
        df['endpoint'] = 'unknown'

    df['request_time'] = pd.to_numeric(df['request_time'], errors='coerce')
    df['body_bytes_sent'] = pd.to_numeric(df['body_bytes_sent'], errors='coerce')
    df['response_status'] = pd.to_numeric(df['response_status'], errors='coerce')
    df.fillna({'request_time': 0, 'body_bytes_sent': 0, 'response_status': 0}, inplace=True)

    df = df.sort_values(by='timestamp').set_index('timestamp')

    df['requests_per_minute'] = (
        df.groupby(pd.Grouper(freq='1min'))['remote_addr']
          .transform('count')
    )

    df['is_error'] = df['response_status'].apply(lambda x: 1 if x >= 400 else 0)
    df['error_rate_5min'] = (
        df.groupby(pd.Grouper(freq='5min'))['is_error']
          .transform('mean')
    )

    df['endpoint_variance_5min'] = (
        df.groupby(pd.Grouper(freq='5min'))['endpoint']
          .transform(lambda x: len(set(x)))
    )

    if 'http_user_agent' not in df.columns:
        df['http_user_agent'] = 'unknown'
    ua_counts = df['http_user_agent'].value_counts()
    total_ua = len(df)
    threshold_freq = 0.005 * total_ua
    rare_ua_set = set(ua_counts[ua_counts < threshold_freq].index)
    df['is_rare_ua'] = df['http_user_agent'].apply(lambda ua: 1 if ua in rare_ua_set else 0)

    endpoint_stats = df.groupby('endpoint')['request_time'].agg(['mean','std']).rename(columns={'mean':'mean_rt','std':'std_rt'})
    endpoint_stats.fillna(0, inplace=True)
    df = df.reset_index().merge(endpoint_stats, how='left', on='endpoint').set_index('timestamp')

    def zscore(row):
        std_rt = row['std_rt']
        mean_rt = row['mean_rt']
        if std_rt == 0:
            return 0
        return (row['request_time'] - mean_rt) / std_rt

    df['endpoint_zscore'] = df.apply(zscore, axis=1)

    SUSPICIOUS_ENDPOINTS = {'/phpmyadmin','/admin','/shell','/wp-login.php'} #!!!!
    df['is_suspicious_endpoint'] = df['endpoint'].apply(lambda ep: 1 if ep.lower() in SUSPICIOUS_ENDPOINTS else 0)

    df['is_redirect'] = df['response_status'].apply(lambda x: 1 if 300 <= x < 400 else 0)
    df['redirect_rate_5min'] = (
        df.groupby(pd.Grouper(freq='5min'))['is_redirect']
          .transform('mean')
    )

    df['unique_ips_10min'] = (
        df.groupby(pd.Grouper(freq='10min'))['remote_addr']
          .transform(lambda x: len(set(x)))
    )

    df = df.reset_index()
    df.drop(columns=['is_error'], inplace=True, errors='ignore')

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует готовую матрицу признаков для обучения модели.
    """
    feature_cols = [
        'request_time',
        'body_bytes_sent',
        'response_status',
        'requests_per_minute',
        'error_rate_5min',
        'endpoint_variance_5min',
        'is_rare_ua',
        'endpoint_zscore',
        'is_suspicious_endpoint',
        'is_redirect',
        'redirect_rate_5min',
        'unique_ips_10min'
    ]
    X = df[feature_cols].fillna(0)
    return X