import os
import json
import logging
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

def collect_log_files(base_dir: str) -> List[str]:
    """
    Рекурсивно находит все файлы с логами в заданной директории.
    
    :param base_dir: Путь к базовой директории с логами (в папках для разных стендов).
    :return: Список путей к лог-файлам.
    """
    log_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.log') or file.endswith('.txt'):
                log_files.append(os.path.join(root, file))
    return log_files

def parse_log_line(log_line: str) -> Dict[str, Any]:
    """
    Разбирает одну строку лога, предполагая, что полезные данные
    находятся после двоеточия ':' и являются JSON-объектом.
    """
    json_start = log_line.find('{')
    if json_start == -1:
        return {}
    
    json_part = log_line[json_start:].strip()
    try:
        data = json.loads(json_part)
    except json.JSONDecodeError:
        return {}
    return data

def load_logs_to_dataframe(log_files: List[str]) -> pd.DataFrame:
    """
    Считывает все логи из списка файлов, парсит и возвращает единый DataFrame.
    """
    rows = []
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = parse_log_line(line)
                if data:
                    rows.append(data)
    df = pd.DataFrame(rows)
    return df