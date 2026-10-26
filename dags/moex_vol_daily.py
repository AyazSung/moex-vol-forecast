"""Daily pipeline: fetch candles, forecast, check forecast quality and input drift."""

# Ежедневный конвейер проекта:
#
#   fetch_candles ×21 → predict → check_forecasts → evaluate_forecasts
#                     ↘ check_drift
#
# Каждая задача — команда docker compose в папке проекта. Airflow решает, что и
# когда запускать, а работу делают те же контейнеры, что мы запускали руками на
# этапе 3: loader (загрузка свечей) и predictor (пакетный прогноз). Две задачи
# этапа 6 — проверки: evaluate_forecasts сверяет прошлые прогнозы с тем, что
# случилось на самом деле, а check_drift — похожи ли свежие признаки на данные,
# на которых модель училась.
#
# Если запуск упал, приходит оповещение в Telegram: on_failure_callback ниже
# вызывает notify_dag_failure из alerts.py (этап 6).

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag
from alerts import notify_dag_failure

# Папка проекта: здесь лежит compose.yaml, из неё задачи запускают docker compose.
PROJECT_DIR = str(Path(__file__).resolve().parents[1])

# Тикеры, свечи которых лежат в data/raw.
TICKERS = [
    "AFLT", "ALRS", "CHMF", "GAZP", "GMKN", "HYDR", "IRAO",
    "LKOH", "MAGN", "MGNT", "MOEX", "MTSS", "NLMK", "NVTK",
    "PHOR", "PLZL", "ROSN", "SBER", "SNGS", "TATN", "VTBR",
]  # fmt: skip


@dag(
    # Расписание в формате cron (день 2): в 00:15 со вторника по субботу — после
    # торгов понедельника–пятницы, когда дневная свеча уже закрыта. Часовой пояс
    # расписания берётся из start_date: время — московское.
    schedule="15 0 * * 2-6",
    start_date=pendulum.datetime(2026, 10, 1, tz="Europe/Moscow"),
    # Не догонять все пропущенные дни: прогноз всегда строится по последним
    # свечам, и запуски «за прошлые даты» дали бы тот же самый результат.
    catchup=False,
    # Не начинать новый запуск, пока не закончился предыдущий.
    max_active_runs=1,
    # Настройки для всех задач: при сбое ещё две попытки с паузой в минуту.
    # Биржа или Docker могут ненадолго не ответить — повтор часто помогает.
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    # Параметры запуска: их можно поменять в форме при запуске кнопкой.
    # fetch = false — не ходить на биржу, посчитать прогноз по уже скачанным свечам.
    params={
        "fetch": Param(True, type="boolean", description="fetch candles from MOEX"),
    },
    # Запуск упал (хоть одна задача упала после всех попыток) — Airflow вызовет
    # эту функцию, и она пришлёт одно сообщение в Telegram на весь запуск.
    on_failure_callback=notify_dag_failure,
    tags=["moex"],
)
def moex_vol_daily():
    # Загрузка — отдельная задача на каждый тикер: из одного описания Airflow
    # сделает 21 задачу. .partial(...) — что общее у всех, .expand(...) — что
    # меняется: здесь переменная окружения TICKER.
    #   {{ ... }}           — шаблон: Airflow подставит значение перед запуском;
    #   params.fetch        — параметр запуска; false → exit 99, и благодаря
    #                         skip_on_exit_code=99 задача станет пропущенной
    #                         (skipped), а не упавшей;
    #   FROM_DATE           — неделя до времени запуска (dag_run.run_after):
    #                         последние дни перезаписываются окончательными
    #                         значениями, история в CSV сохраняется (часть 2);
    #   append_env=True     — к TICKER добавить обычное окружение (PATH и другое),
    #                         иначе задача не найдёт команду docker;
    #   max_active_tis_per_dag=3 — не больше трёх загрузок одновременно, чтобы
    #                         не перегружать ни Mac, ни сервер биржи.
    fetch_candles = BashOperator.partial(
        task_id="fetch_candles",
        cwd=PROJECT_DIR,
        bash_command=(
            '[ "{{ params.fetch }}" = "True" ] || exit 99; '
            "docker compose run --rm -e TICKER=$TICKER "
            "-e FROM_DATE={{ (dag_run.run_after - macros.timedelta(days=7)) | ds }} "
            "loader"
        ),
        skip_on_exit_code=99,
        append_env=True,
        max_active_tis_per_dag=3,
    ).expand(env=[{"TICKER": ticker} for ticker in TICKERS])

    # Пакетный прогноз чемпионом — в CSV и в таблицу forecasts (этап 3).
    # trigger_rule="none_failed" — запускать, если ни одна загрузка не упала;
    # пропущенные (fetch = false) не мешают. По умолчанию нужен успех всех.
    predict = BashOperator(
        task_id="predict",
        cwd=PROJECT_DIR,
        bash_command="docker compose run --rm predictor",
        trigger_rule="none_failed",
    )

    # Проверка результата: в этом запуске записан прогноз для всех 21 тикера.
    #   exec -T     — выполнить команду в работающем контейнере db без терминала:
    #                 у задачи Airflow терминала нет;
    #   -tAc        — psql печатает только число (этап 1, часть 5);
    #   dag_run.start_date — когда начался этот запуск DAG-а: считаем только
    #                 прогнозы, записанные после него. Прогнозы другого запуска,
    #                 например ручного получасом раньше, не засчитаются;
    #   [ ... -ge ] — «не меньше»: если тикеров меньше, команда вернёт 1, и задача
    #                 упадёт — сбой будет виден в Airflow, а не останется незамеченным.
    check_forecasts = BashOperator(
        task_id="check_forecasts",
        cwd=PROJECT_DIR,
        bash_command=(
            "n=$(docker compose exec -T db psql -U moex -d moex -tAc "
            '"SELECT count(DISTINCT ticker) FROM forecasts '
            "WHERE created_at >= '{{ dag_run.start_date }}'\"); "
            'echo "tickers with fresh forecasts: $n"; '
            f'[ "$n" -ge {len(TICKERS)} ]'
        ),
    )

    # Проверка качества (этап 6). Только что загрузились свечи последнего дня, а
    # это факт для прогноза, сделанного прошлой ночью. Команда сверяет все
    # прогнозы, для которых факт уже известен, записывает итог в таблицу
    # forecast_quality и проверяет версию модели, которая сейчас в работе. Если
    # за 20 последних дней её прогнозы хуже наивного «завтра как сегодня»,
    # команда вернёт код 1, и задача станет красной (failed).
    # retries=0 — без повторных попыток. Повторы из default_args выше помогают,
    # когда сеть или Docker ненадолго не ответили. Плохое качество повтор не
    # исправит: те же данные дадут то же число, поэтому задача падает сразу.
    evaluate_forecasts = BashOperator(
        task_id="evaluate_forecasts",
        cwd=PROJECT_DIR,
        bash_command="docker compose run --rm predictor evaluate-forecasts",
        retries=0,
    )

    # Проверка входных данных (этап 6, часть 2): не ушли ли признаки чемпиона от
    # данных его обучения — PSI за 20 дней и значения за границами обучения.
    # Идёт сразу после загрузки, параллельно с прогнозом, и прогноз не
    # останавливает: в необычный день прогноз как раз нужен, а красная задача —
    # повод посмотреть на данные. Подробно — в drift.py.
    #   trigger_rule="none_failed" — как у predict: загрузки могли быть
    #                  пропущены параметром fetch = false;
    #   retries=0    — как у проверки качества: те же данные дадут тот же ответ.
    check_drift = BashOperator(
        task_id="check_drift",
        cwd=PROJECT_DIR,
        bash_command="docker compose run --rm predictor check-drift",
        trigger_rule="none_failed",
        retries=0,
    )

    # Две ветки после загрузки: прогноз с проверками выхода и проверка входа.
    fetch_candles >> predict >> check_forecasts >> evaluate_forecasts
    fetch_candles >> check_drift


moex_vol_daily()
