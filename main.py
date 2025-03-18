import logging
from model_utils import (
    train_anomaly_model,
    infer_anomalies,
    analyze_anomalies,
    load_anomaly_model
)
from data_utils import collect_log_files, load_logs_to_dataframe
from features import preprocess_logs, build_feature_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from report_generator import generate_html_report

from config import Settings
settings = Settings()



def main_train(base_dir: str, model_path: str = None):
    """
    Основная функция для обучения модели на всех логах.
    """
    logging.info("Собираем список лог-файлов...")
    files = collect_log_files(base_dir)
    logging.info(f"Найдено {len(files)} файлов с логами.")

    logging.info("Считываем логи в DataFrame...")
    df = load_logs_to_dataframe(files)
    logging.info(f"Общее количество записей в логах: {len(df)}")

    logging.info("Предобрабатываем логи и создаём признаки...")
    df_preproc = preprocess_logs(df)

    logging.info("Формируем матрицу признаков для обучения...")
    X = build_feature_matrix(df_preproc)

    logging.info("Обучаем модель обнаружения аномалий...")
    model = train_anomaly_model(X, model_path=model_path)

    logging.info("Модель успешно обучена.")
    return model

def main_detect_one_day(model_path: str, one_day_log_path: str, html_report: bool = False):
    """
    Функция для детекции аномалий в логах за один день.
    """
    model, scaler = load_anomaly_model(model_path)

    logging.info("Читаем логи за один день...")
    daily_df = load_logs_to_dataframe([one_day_log_path])
    if daily_df.empty:
        logging.info("Файл с логами пуст или нет корректных записей.")
        return

    logging.info("Предобрабатываем и вычисляем признаки для детекции...")
    daily_df_preproc = preprocess_logs(daily_df)

    logging.info("Делаем предсказание аномалий...")
    result_df = infer_anomalies(daily_df_preproc, model, scaler)

    logging.info("Анализируем аномалии...")
    anomalies_df = analyze_anomalies(result_df)
    if html_report:
        generate_html_report(anomalies_df, "my_anomalies_report.html")

    logging.info(f"Всего найдено аномалий: {len(anomalies_df)}")
    for idx, row in anomalies_df.iterrows():
        log_str = (
            f"Время: {row.get('timestamp')} | "
            f"Endpoint: {row.get('endpoint')} | "
            f"Status: {row.get('response_status')} | "
            f"remote_addr: {row.get('remote_addr')} | "
            f"body_bytes_sent: {row.get('body_bytes_sent')} | "
            f"request_time: {row.get('request_time')} | "
            f"request: {row.get('request')} | "
            f"request_method: {row.get('request_method')} | "
            f"upstream_addr: {row.get('upstream_addr')} | "
            f"http_x_real_ip: {row.get('http_x_real_ip')} | "
            f"http_x_forwarded_for: {row.get('http_x_forwarded_for')} | "
            f"http_referrer: {row.get('http_referrer')} | "
            f"http_user_agent: {row.get('http_user_agent')} | "
            f"http_version: {row.get('http_version')} | "
            f"nginx_access: {row.get('nginx_access')} | "
            f"requests_per_minute: {row.get('requests_per_minute')} | "
            f"is_rare_ua: {row.get('is_rare_ua')} | "
            f"endpoint_zscore: {row.get('endpoint_zscore'):.2f} | "
            f"is_suspicious_endpoint: {row.get('is_suspicious_endpoint')} | "
            f"is_redirect: {row.get('is_redirect')} | "
            f"redirect_rate_5min: {row.get('redirect_rate_5min'):.2f} | "
            f"error_rate_5min: {row.get('error_rate_5min')} | "
            f"endpoint_variance_5min: {row.get('endpoint_variance_5min')} | "
            f"std_rt: {row.get('std_rt')} | "
            f"unique_ips_10min: {row.get('unique_ips_10min')} | "
            f"anomaly: {row.get('anomaly')} | "
            f"anomaly_score: {row.get('anomaly_score')} | "
            f"Причина: {row.get('anomaly_reason')}"
        )
        print(log_str)
