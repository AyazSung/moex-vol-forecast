# Тесты оповещений (dags/alerts.py). Ни Airflow, ни Telegram, ни интернет им не
# нужны:
#   - context, который Airflow передаёт в функцию оповещения, подменяем
#     простыми объектами SimpleNamespace с теми же полями;
#   - отправку в Telegram подменяем подделкой open_url: она не ходит в сеть, а
#     запоминает запрос, чтобы проверить адрес и текст.
# Модуль alerts лежит в папке dags; pytest находит его благодаря настройке
# pythonpath в pyproject.toml.

import io
import json
import urllib.error
from datetime import UTC, datetime
from types import SimpleNamespace

import alerts

LOG_URL = "http://localhost:8080/dags/moex_vol_daily/runs/scheduled/tasks/fetch_candles"


def make_context(task_id="evaluate_forecasts", map_index=-1):
    """What Airflow passes to on_failure_callback of a DAG, only the fields we use."""
    return {
        "dag": SimpleNamespace(dag_id="moex_vol_daily"),
        "run_id": "scheduled__2026-10-27T21:15:00+00:00",
        "reason": "task_failure",
        "ti": SimpleNamespace(task_id=task_id, map_index=map_index, log_url=LOG_URL),
        # Срок ночного запуска 28 октября в 00:15 по Москве — это 21:15 UTC
        # накануне: Airflow хранит время в UTC.
        "dag_run": SimpleNamespace(
            run_after=datetime(2026, 10, 27, 21, 15, tzinfo=UTC)
        ),
    }


def test_message_names_dag_task_moscow_time_and_log():
    text = alerts.failure_message(make_context())

    assert text.splitlines() == [
        "🔴 Airflow: запуск moex_vol_daily упал",
        "Задача: evaluate_forecasts",
        "Срок запуска: 28.10.2026 00:15 МСК",
        f"Лог: {LOG_URL}",
    ]


def test_copy_of_a_mapped_task_is_shown_with_its_number():
    text = alerts.failure_message(make_context("fetch_candles", map_index=17))

    # У динамической задачи fetch_candles 21 копия, по одной на тикер.
    assert "Задача: fetch_candles [17]" in text.splitlines()


def test_message_without_failed_task():
    # Бывает, что Airflow не знает упавшую задачу: тогда в context только DAG,
    # номер запуска и причина. Сообщение всё равно собирается.
    context = {
        "dag": SimpleNamespace(dag_id="moex_vol_retrain"),
        "run_id": "manual__2026-10-31T23:00:00+00:00",
        "reason": "all_tasks_deadlocked",
    }

    assert alerts.failure_message(context).splitlines() == [
        "🔴 Airflow: запуск moex_vol_retrain упал",
        "Причина: all_tasks_deadlocked",
        "Запуск: manual__2026-10-31T23:00:00+00:00",
    ]


def test_alert_is_not_sent_without_settings(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    def must_not_be_called(request):
        raise AssertionError("Telegram must not be called without settings")

    monkeypatch.setattr(alerts, "open_url", must_not_be_called)

    # Нет токена или чата — ничего не отправляем и не падаем.
    assert alerts.send_telegram("hello") is False


def test_message_goes_to_the_bot_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.delenv("TELEGRAM_API_URL", raising=False)
    sent = []

    def fake_open_url(request):
        sent.append(request)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(alerts, "open_url", fake_open_url)

    assert alerts.send_telegram("hello") is True
    # Метод sendMessage бота с токеном 123:abc, в теле — номер чата и текст.
    assert sent[0].full_url == "https://api.telegram.org/bot123:abc/sendMessage"
    assert json.loads(sent[0].data) == {"chat_id": "42", "text": "hello"}


def test_network_error_does_not_break_airflow(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def offline(request):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(alerts, "open_url", offline)

    # Ошибку сети пишем в лог и возвращаем False — без исключения: сбой
    # оповещения не должен ломать работу Airflow.
    assert alerts.send_telegram("hello") is False


def test_macos_system_proxy_settings_are_not_read(monkeypatch):
    # Ловушка этапа 6: чтение системных настроек прокси macOS в процессе,
    # созданном через fork, роняет процесс, и оповещение не уходит. open_url
    # берёт прокси только из переменных окружения. Проверяем: если бы кто-то
    # спросил системные настройки (urllib.request.getproxies), тест упал бы.
    def must_not_be_called():
        raise AssertionError("system proxy settings must not be read")

    monkeypatch.setattr(alerts.urllib.request, "getproxies", must_not_be_called)
    monkeypatch.setattr(
        alerts.urllib.request.OpenerDirector,
        "open",
        lambda self, request, timeout: io.BytesIO(b'{"ok": true}'),
    )

    request = alerts.urllib.request.Request("http://127.0.0.1:9/sendMessage")
    assert alerts.open_url(request).read() == b'{"ok": true}'
