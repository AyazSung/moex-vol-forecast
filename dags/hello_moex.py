"""First DAG: three small tasks run one after another."""

# Учебный DAG этапа 4. DAG (directed acyclic graph, «направленный граф без
# циклов») — описание конвейера: какие задачи выполнить и в каком порядке.
# Airflow сам запускает их — по расписанию или по кнопке, — повторяет при сбое
# и показывает всё в веб-интерфейсе.
#
# Этот файл читает и выполняет сам Airflow, своим Python из ~/airflow/.venv.
# Библиотек проекта (pandas, mlflow) там нет, поэтому здесь импортируем только
# Airflow и стандартную библиотеку Python. Работу проекта задачи будут запускать
# командами терминала (в части 2 — через docker compose).

from datetime import datetime
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag, task

# Папка проекта: этот файл лежит в <проект>/dags, значит проект — на уровень выше.
PROJECT_DIR = Path(__file__).resolve().parents[1]


# @dag превращает функцию в описание DAG; имя функции — имя DAG (dag_id).
#   schedule=None — запуск только вручную, кнопкой (расписание — в части 2);
#   start_date    — с какой даты DAG «существует» для планировщика;
#   catchup=False — не догонять запуски за прошлые даты;
#   tags          — метки для поиска в списке DAG-ов.
@dag(
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["moex", "learning"],
)
def hello_moex():
    # Задача-команда: BashOperator выполнит строку в bash и сохранит вывод в лог.
    # task_id — имя задачи, уникальное внутри DAG.
    say_hello = BashOperator(
        task_id="say_hello",
        bash_command="echo 'Привет из Airflow'; date; docker --version",
    )

    # Задача-функция: @task превращает обычную функцию Python в задачу.
    # То, что функция вернёт, Airflow сохранит в своей базе (это называется XCom),
    # и значение можно передать следующей задаче.
    @task
    def count_candle_files() -> int:
        files = sorted((PROJECT_DIR / "data" / "raw").glob("*.csv"))
        print(f"{len(files)} files: {', '.join(f.stem for f in files)}")
        return len(files)

    @task
    def report(n_files: int) -> None:
        # n_files приходит из XCom: это то, что вернула count_candle_files.
        print(f"candles are ready for {n_files} tickers")

    # Порядок задач. >> — «сначала левая, потом правая». Вызов report(files)
    # задаёт порядок сам: report ждёт результат count_candle_files.
    files = count_candle_files()
    say_hello >> files
    report(files)


# Вызов функции регистрирует DAG — без этой строки Airflow его не увидит.
hello_moex()
