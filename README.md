# ALD
Anomaly log detection

# Использование

## Установка зависимостей 

Python3.12!

```
pip install -r req.txt
```


## Подготовка 

Необходимо создать структуру 

```
Ai logs/<папки стендов>
model/
```

## Запуск

```
python manage.py
```

## Аргументы команды запуска

```
Options:
  --train BOOLEAN  Start train new model
  --test BOOLEAN   Start test model
  --html BOOLEAN   Generate HTML report
  --help           Show this message and exit.
```


## Фичи которые используются для определения аномалий

1. `response_status`
HTTP‑код ответа сервера (например, 200, 404, 500). Позволяет понять, чем закончился запрос: успешно, с ошибкой и т. д.
2.	`remote_addr`
IP‑адрес клиента, который обратился к серверу. Показывает, откуда пришёл запрос.
3.	`body_bytes_sent`
Количество байт, отправленных сервером в теле ответа. Полезно для оценки объёма данных, возвращаемых в ответе.
4.	`request_time`
Время выполнения запроса на стороне сервера (в секундах). Может помочь выяснить, насколько быстро обрабатывается запрос.
5.	`request`
Полная строка запроса, включая метод и путь (например, “GET /api/front/live_report/graph/data HTTP/1.1”). Содержит основной URI и версию протокола.
6.	`request_method`
HTTP‑метод запроса (GET, POST, PUT, DELETE и т. д.). Показывает, какое действие клиент хотел произвести.
7.	`upstream_addr`
Адрес (IP и порт) конечного бэкенда (upstream), к которому обратился сервер (Nginx) при проксировании запроса. Удобно для выяснения, какой сервис отвечал на запрос в случае распределённой системы.
8.	`http_x_real_ip`
Дополнительная информация о реальном IP‑адресе клиента, которую может проксировать Nginx. Если до этого запроса уже была цепочка прокси‑серверов, этот заголовок может хранить реальный внешний IP.
9.	`http_x_forwarded_for`
Список IP‑адресов, через которые прошёл запрос (добавляется каждым промежуточным прокси). Это даёт историю проксирования запроса.
10.	`http_referrer`
Заголовок Referrer (или Referer), который указывает адрес веб‑страницы, с которой перешли на текущий ресурс. Может использоваться для аналитики переходов.
11.	`http_user_agent`
Заголовок User‑Agent, идентифицирующий тип, версию и прочую информацию о браузере или другом клиенте (например, Mozilla/5.0, curl/7.74.0 и т. д.).
12.	`http_version`
Версия HTTP‑протокола, по которому происходит соединение (HTTP/1.0, HTTP/1.1, HTTP/2 и т. п.).
13.	`nginx_access`
Флаг, показывающий, что данные поступили из логов доступа Nginx. Может использоваться для быстрой фильтрации строк, относящихся к HTTP‑запросам, в отличие от других типов логов (ошибок, системных и т. д.).
14.	`requests_per_minute`
Метрика, показывающая, сколько запросов происходит на конкретный endpoint или весь сервис в минуту. Чаще всего рассчитывается либо за скользящее окно времени, либо агрегируется в реальном времени.
15.	`is_rare_ua`
Логический признак, свидетельствующий о том, что в поле User-Agent находится редкая или нетипичная строка. Это может указывать на подозрительного бота или необычную программу.
16.	`endpoint_zscore`
Z‑оценка (статистический показатель), отражающая, насколько данный endpoint (или время ответа по нему) отклоняется от среднего значения в рамках наблюдаемого периода. Помогает выявлять аномальное поведение (например, слишком много ошибок или слишком долгие ответы).
17.	`is_suspicious_endpoint`
Булевый признак, указывающий, что endpoint посчитан «подозрительным» по совокупности метрик (например, неожиданно большая нагрузка, всплеск ошибок, необычная частота запросов).
18.	`is_redirect`
Флаг, показывающий, выполняется ли для данного запроса редирект (обычно проверяется по коду ответа 3xx или иным признакам).
19.	`redirect_rate_5min`
Доля (или количество) запросов, которые возвращали коды редиректа (3xx), за последние 5 минут. Позволяет заметить нетипичные всплески редиректов.
20.	`error_rate_5min`
Доля (или количество) запросов с ошибочными кодами ответа (4xx, 5xx) за последние 5 минут. Помогает видеть аномальные пики ошибок.
21.	`endpoint_variance_5min`
Метрика рассеяния (дисперсия) количества обращений к endpoint’ам в 5‑минутном интервале. Высокая вариативность может указывать на «взрывы» трафика.
22.	`std_rt`
Стандартное отклонение (standard deviation) времени ответа (request_time). Если оно слишком велико, значит, время ответа сильно скачет (неустойчиво).
23.	`unique_ips_10min`
Счётчик уникальных IP‑адресов, обращавшихся к сервису или к конкретному endpoint’у за последние 10 минут. Помогает заметить подозрительную активность (например, массовый трафик с множества IP).
24.	`anomaly`
Логический признак, показывающий, что эта строка (этот запрос) была классифицирована как аномальная по совокупности признаков.
25.	`anomaly_score`
Числовая оценка «степени аномальности» (обычно от 0 до 1 или выше), где более высокое значение означает большую уверенность в том, что событие — аномалия.
26.	`anomaly_reason`
Описание причины, по которой запрос или событие признано аномальным. Обычно содержит краткое текстовое объяснение — например, «Много ошибок за короткое время» или «Слишком высокая задержка ответа».