import pandas as pd
import os
from datetime import datetime

def generate_html_report(anomalies_df: pd.DataFrame, output_path: str = "anomalies_report.html"):
    """
    Генерирует HTML-отчет по аномалиям и сохраняет его в файл output_path.

    Ожидается, что anomalies_df может содержать следующие столбцы:
      - timestamp
      - endpoint
      - response_status
      - remote_addr
      - body_bytes_sent
      - request_time
      - request
      - request_method
      - upstream_addr
      - http_x_real_ip
      - http_x_forwarded_for
      - http_referrer
      - http_user_agent
      - http_version
      - nginx_access
      - requests_per_minute
      - is_rare_ua
      - endpoint_zscore
      - is_suspicious_endpoint
      - is_redirect
      - redirect_rate_5min
      - error_rate_5min
      - endpoint_variance_5min
      - std_rt
      - unique_ips_10min
      - anomaly
      - anomaly_score
      - anomaly_reason

    Если каких-то столбцов в DataFrame нет, они автоматически пропускаются в таблице.
    """

    desired_columns = [
        "timestamp", "endpoint", "response_status", "remote_addr", "body_bytes_sent",
        "request_time", "request", "request_method", "upstream_addr", "http_x_real_ip",
        "http_x_forwarded_for", "http_referrer", "http_user_agent", "http_version",
        "nginx_access", "requests_per_minute", "is_rare_ua", "endpoint_zscore",
        "is_suspicious_endpoint", "is_redirect", "redirect_rate_5min",
        "error_rate_5min", "endpoint_variance_5min", "std_rt", "unique_ips_10min",
        "anomaly", "anomaly_score", "anomaly_reason"
    ]

    existing_columns = [col for col in desired_columns if col in anomalies_df.columns]

    if anomalies_df.empty or not existing_columns:
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8"/>
            <title>Отчет по аномалиям</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .header {{
                    margin-bottom: 20px;
                }}
                .no-anomalies {{
                    color: #2E86C1;
                    font-size: 18px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет по аномалиям</h1>
                <p>Дата генерации: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            <div class="no-anomalies">Аномалий не обнаружено или отсутствуют необходимые поля.</div>
        </body>
        </html>
        """
    else:
        table_html = anomalies_df[existing_columns].to_html(
            index=False,
            justify="left",
            border=0
        )

        html_content = f"""
<html>
<head>
    <meta charset="utf-8"/>
    <title>Отчет по аномалиям</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f7f7f7;
        }}
        .header {{
            margin-bottom: 20px;
            padding: 20px;
            background-color: #ececec;
            border-radius: 5px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            color: #333333;
        }}
        .header p {{
            margin: 5px 0 0;
            color: #666666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            background: #ffffff;
            border-radius: 5px;
            overflow: hidden;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #fafafa;
            color: #333333;
        }}
        tr:nth-child(even) {{
            background-color: #fcfcfc;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .summary {{
            margin-top: 20px;
            font-weight: bold;
            color: #333333;
        }}
        .anomaly-row {{
            background-color: #ffeded;
        }}
        .highlight {{
            color: #d9534f; /* bootstrap's danger color */
            font-weight: bold;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 12px;
            color: #999999;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Отчет по аномалиям</h1>
        <p>Дата генерации: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    <table>
        <thead>
            <tr>
                {"".join(f"<th>{col}</th>" for col in existing_columns)}
            </tr>
        </thead>
        <tbody>
            {"".join(build_table_rows(anomalies_df, existing_columns))}
        </tbody>
    </table>
    <div class="summary">
        Всего аномалий: {len(anomalies_df)}
    </div>
    <div class="footer">
        &copy; ALD 2025
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML-отчет сохранён в: {os.path.abspath(output_path)}")


def build_table_rows(df: pd.DataFrame, columns: list[str]) -> list[str]:
    rows_html = []
    for _, row in df.iterrows():
        anomaly_class = ""
        if "anomaly" in df.columns and row["anomaly"]:
            anomaly_class = " anomaly-row"

        cells = []
        for col in columns:
            cell_value = row[col]
            if col in ["anomaly", "anomaly_reason"] and cell_value:
                cells.append(f"<td><span class='highlight'>{cell_value}</span></td>")
            else:
                cells.append(f"<td>{cell_value}</td>")

        row_html = f"<tr class='{anomaly_class}'>" + "".join(cells) + "</tr>"
        rows_html.append(row_html)
    return rows_html