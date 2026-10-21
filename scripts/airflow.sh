#!/bin/bash
# airflow.sh — Airflow на Mac с настройками проекта (этап 4).
#
# Запустить Airflow целиком:  ./scripts/airflow.sh
# Любая команда Airflow:      ./scripts/airflow.sh dags list
# Веб-интерфейс: http://localhost:8080; остановить — Ctrl+C в этом терминале.
#
# Airflow стоит отдельно от проекта, в папке ~/airflow со своим окружением .venv:
# у него больше сотни библиотек строго заданных версий, и с библиотеками проекта
# они не уживаются (например, Airflow нужен fastapi 0.136.3, а проекту — 0.143.1
# или новее). DAG-файлы при этом лежат в проекте, в папке dags, и хранятся в Git.

set -euo pipefail

# Папка проекта — на уровень выше папки scripts, где лежит этот скрипт (день 2).
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Домашняя папка Airflow: настройки (airflow.cfg), база запусков (airflow.db),
# логи задач и пароль администратора. ${X:-умолчание} — значение X, а если X
# не задана — умолчание.
export AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"

# Airflow не установлен — понятная ошибка сразу, а не «command not found» потом.
if [ ! -x "$AIRFLOW_HOME/.venv/bin/airflow" ]; then
    echo "ERROR: Airflow not found in $AIRFLOW_HOME/.venv - install it first" >&2
    exit 1
fi

# Любую настройку Airflow можно задать переменной AIRFLOW__РАЗДЕЛ__КЛЮЧ — она
# главнее файла airflow.cfg:
#   DAGS_FOLDER      — где искать DAG-файлы: папка dags нашего проекта;
#   LOAD_EXAMPLES    — не показывать сотню учебных DAG-ов самого Airflow;
#   REFRESH_INTERVAL — искать новые DAG-файлы каждые 30 секунд, а не раз в 5 минут;
#   HOST             — веб-интерфейс доступен только с этого Mac, а не всей сети Wi-Fi.
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_DIR/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL=30
export AIRFLOW__API__HOST=127.0.0.1

# «Активировать» окружение Airflow: его программы — первыми в PATH. Команда
# airflow standalone запускает свои части командой airflow, и они должны её найти.
export PATH="$AIRFLOW_HOME/.venv/bin:$PATH"

# Без аргументов — запустить все части Airflow сразу (standalone).
# set -- заменяет аргументы скрипта на перечисленные.
if [ "$#" -eq 0 ]; then
    set -- standalone
fi

# Любая команда, кроме standalone: exec заменяет этот скрипт программой airflow.
if [ "$1" != "standalone" ]; then
    exec airflow "$@"
fi

# standalone запускаем по-особому, чтобы Ctrl+C останавливал Airflow чисто.
# Ctrl+C терминал посылает не одной программе, а всей группе процессов
# (день 2, часть 5): сигнал разом получают и главный процесс Airflow, и его
# планировщик. Главный процесс при этом сам посылает планировщику SIGTERM, и в
# Airflow 3.3.2 на Mac второй сигнал обрывает остановку на полпути: терминал
# «висит», а рабочие процессы остаются в памяти. Поэтому:
#   set -m  — запустить Airflow в отдельной группе процессов: Ctrl+C из
#             терминала получает только этот скрипт;
#   trap    — когда скрипту пришёл Ctrl+C (INT), kill (TERM) или закрыли окно
#             терминала (HUP), передать один SIGINT главному процессу Airflow —
#             тот сам по очереди остановит свои части.
set -m
airflow "$@" &
airflow_pid=$!
trap 'kill -INT "$airflow_pid" 2>/dev/null || true' INT TERM HUP

# wait ждёт завершения Airflow, но прерывается, когда приходит сигнал. Поэтому
# ждём в цикле, пока процесс жив (kill -0 ничего не посылает, только проверяет).
# wait внутри if: так строгий режим (set -e) не остановит скрипт, если wait вернёт
# не 0, а в status попадёт код возврата самого Airflow.
status=0
while kill -0 "$airflow_pid" 2>/dev/null; do
    if wait "$airflow_pid"; then
        status=0
    else
        status=$?
    fi
done
exit "$status"
