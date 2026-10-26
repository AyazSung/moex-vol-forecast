"""Telegram alerts about failed DAG runs."""

# alerts.py — оповещения в Telegram, когда запуск DAG-а упал (этап 6).
#
# Зачем. Красную задачу Airflow видно только в его интерфейсе, а туда нужно
# зайти самому. Ночью проверка качества может упасть, а узнаешь ты об этом,
# когда случайно откроешь Airflow. Оповещение приходит само, сообщением в
# Telegram, через несколько секунд после того, как запуск закончился неудачей.
#
# Где выполняется. Функцию вызывает не задача, а та часть Airflow, что читает
# DAG-файлы (dag-processor), — в отдельном процессе. Её сообщения в лог
# попадают не в логи задач, а в ~/airflow/logs/dag_processor/<дата>/.
#
# Как это работает. У DAG-а есть параметр on_failure_callback — функция,
# которую Airflow вызывает, когда запуск DAG-а целиком закончился неудачей: все
# задачи доделаны, и хотя бы одна упала после всех повторных попыток. В функцию
# Airflow передаёт context — словарь со сведениями о запуске: какой DAG, какая
# задача упала последней, когда был срок запуска. notify_dag_failure собирает
# из него короткое сообщение и отправляет его в Telegram.
#
# Почему оповещение на весь запуск, а не на каждую задачу. Если ночью не
# отвечает биржа, падают все 21 загрузка свечей. Оповещение на задачу прислало
# бы 21 сообщение подряд, а на запуск — одно: какой DAG, какая задача упала
# последней и ссылка на её лог. Остальное видно в Airflow по ссылке.
#
# Куда отправлять. Сообщения шлёт твой Telegram-бот:
#   TELEGRAM_BOT_TOKEN — токен бота: по нему Telegram понимает, от какого бота
#                        сообщение;
#   TELEGRAM_CHAT_ID   — номер чата с тобой: кому его доставить.
# Токен — секрет: с ним кто угодно может писать от имени бота. Поэтому в Git
# его нет: оба значения лежат в файле .env.alerts в корне проекта, а
# scripts/airflow.sh при запуске передаёт их Airflow переменными окружения.
# Значения не заданы — функция пишет предупреждение в лог и ничего не
# отправляет: DAG-и работают как раньше, просто без оповещений.
#
# Модуль не импортирует Airflow — только стандартную библиотеку Python. Поэтому
# его можно проверить обычными тестами проекта (tests/test_alerts.py) без
# установленного Airflow, а Airflow сможет его выполнить: библиотек проекта в
# его Python нет. Файл лежит в папке dags, чтобы DAG-и могли его импортировать,
# а .airflowignore говорит Airflow не искать в нём DAG-и.

import json
import logging
import os
import urllib.request
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Адрес Telegram Bot API. Переменной TELEGRAM_API_URL его можно заменить —
# например, на свой сервер Bot API или на заглушку для проверки без интернета.
TELEGRAM_API = "https://api.telegram.org"

# Сроки запусков Airflow хранит в UTC; в сообщении показываем московское время,
# как в расписании DAG-ов.
MOSCOW = ZoneInfo("Europe/Moscow")


def failure_message(context) -> str:
    """Short text about a failed DAG run, built from the Airflow callback context."""
    # context["dag"] есть всегда; остальное Airflow кладёт, если знает, какая
    # задача упала (у нас — всегда, но лишний раз не падаем, если нет).
    lines = [f"🔴 Airflow: запуск {context['dag'].dag_id} упал"]

    # ti (task instance) — упавшая задача этого запуска; если упало несколько,
    # Airflow передаёт ту, что упала последней. map_index — номер копии у
    # динамической задачи: у fetch_candles их 21, по одной на тикер (-1 — задача
    # без копий).
    ti = context.get("ti")
    if ti is not None:
        task = ti.task_id
        if ti.map_index is not None and ti.map_index >= 0:
            task = f"{task} [{ti.map_index}]"
        lines.append(f"Задача: {task}")

    # dag_run.run_after — срок запуска по расписанию (у запуска кнопкой — время
    # нажатия), в UTC. astimezone(MOSCOW) переводит его в московское время.
    dag_run = context.get("dag_run")
    if dag_run is not None and dag_run.run_after is not None:
        run_after = dag_run.run_after.astimezone(MOSCOW)
        lines.append(f"Срок запуска: {run_after:%d.%m.%Y %H:%M} МСК")

    # reason — почему запуск помечен упавшим. Обычно task_failure, «упала
    # задача»; другие причины, например all_tasks_deadlocked, пишем в сообщение.
    reason = context.get("reason")
    if reason and reason != "task_failure":
        lines.append(f"Причина: {reason}")

    # Ссылка на лог упавшей задачи — в нём строки ERROR с подробностями. Без
    # задачи — хотя бы номер запуска, чтобы найти его в Airflow.
    if ti is not None:
        lines.append(f"Лог: {ti.log_url}")
    else:
        lines.append(f"Запуск: {context['run_id']}")
    return "\n".join(lines)


def open_url(request: urllib.request.Request):
    """Open the request without reading the macOS system proxy settings."""
    # Обычный urllib.request.urlopen перед каждым запросом спрашивает у macOS
    # системные настройки прокси. Делает он это через системные библиотеки Apple,
    # а их нельзя трогать в процессе, созданном копированием другого процесса
    # (fork). Airflow вызывает оповещение именно в таком процессе, и на macOS тот
    # падал: «objc: … may have been in progress in another thread when fork() was
    # called … Crashing instead» — сообщение не уходило.
    # Поэтому собираем свой «открыватель» адресов: прокси он берёт только из
    # переменных окружения (HTTPS_PROXY и подобных), а системные настройки не
    # читает. Нет таких переменных — запрос идёт напрямую.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies_environment())
    )
    return opener.open(request, timeout=10)


def send_telegram(text: str) -> bool:
    """Send a Telegram message; False if not configured or not delivered."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set: alert not sent"
        )
        return False

    # Метод sendMessage Bot API: POST на адрес с токеном бота, в теле — JSON с
    # номером чата и текстом. Отвечает Telegram тоже JSON-ом: {"ok": true, ...}.
    api = os.environ.get("TELEGRAM_API_URL", TELEGRAM_API)
    request = urllib.request.Request(
        f"{api}/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Ошибку отправки не пробрасываем дальше: оповещение — дополнение, и его
    # сбой не должен ломать работу Airflow. OSError ловит и сетевые ошибки, и
    # ответы Telegram с ошибкой (HTTPError — её частный случай). Адрес с токеном
    # в лог не пишем: токен — секрет.
    try:
        with open_url(request) as response:
            json.load(response)
    except OSError as err:
        logger.error("cannot send alert to Telegram: %s", err)
        return False
    logger.info("alert sent to Telegram")
    return True


def notify_dag_failure(context) -> None:
    """on_failure_callback for DAGs: send a Telegram message about the failed run."""
    send_telegram(failure_message(context))
