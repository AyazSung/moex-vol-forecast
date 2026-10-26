"""Manual DAG that always fails: checks that failure alerts reach Telegram."""

# Проверка оповещений (этап 6). Как пожарная тревога, которую иногда включают
# нарочно: убедиться, что сигнал доходит, лучше заранее, а не в ночь, когда
# что-то сломалось по-настоящему.
#
# DAG запускается только руками, кнопкой Trigger. Его единственная задача
# намеренно падает, запуск становится красным, и Airflow вызывает
# notify_dag_failure — ту же функцию, что у рабочих DAG-ов. Если оповещения
# настроены, в Telegram приходит сообщение. DAG ничего не загружает, не считает
# и не меняет: задача только печатает строку и выходит с кодом 1.

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag

# alerts.py лежит рядом, в папке dags: Airflow добавляет эту папку в пути
# поиска модулей, поэтому импорт работает как у обычного модуля.
from alerts import notify_dag_failure


@dag(
    # Только вручную: по расписанию проверка не нужна.
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    # Функция, которую Airflow вызовет, когда запуск упадёт.
    on_failure_callback=notify_dag_failure,
    tags=["moex"],
)
def moex_alert_test():
    # retries=0 — без повторных попыток: падение нужно сразу, а не через минуты.
    BashOperator(
        task_id="always_fails",
        bash_command='echo "test failure: checking that alerts reach Telegram"; exit 1',
        retries=0,
    )


moex_alert_test()
