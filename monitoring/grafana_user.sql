-- grafana_user.sql — пользователь базы для Grafana, только для чтения (этап 6).
--
-- Grafana читает из базы итоги проверок: forecast_quality и feature_drift.
-- Писать ей не нужно, поэтому у неё отдельный пользователь, который может
-- только читать (SELECT). Это принцип наименьших прав: даже ошибка в запросе
-- дашборда не сможет ничего изменить или удалить.
--
-- Выполняется один раз:
--   docker compose exec -T db psql -U moex -d moex < monitoring/grafana_user.sql
-- При повторном запуске первая команда напишет ERROR: role "grafana" already
-- exists — пользователь уже есть. Остальные команды просто выдадут те же права
-- ещё раз, это безопасно.

-- Пользователь grafana, который может входить в базу (LOGIN). Пароль учебный,
-- как у moex: база открыта только для этого Mac.
CREATE ROLE grafana LOGIN PASSWORD 'grafana';

-- Право подключаться к базе moex и видеть объекты в схеме public — там лежат
-- все наши таблицы.
GRANT CONNECT ON DATABASE moex TO grafana;
GRANT USAGE ON SCHEMA public TO grafana;

-- Читать все таблицы, которые уже есть: свечи, прогнозы, качество, дрейф.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;

-- И таблицы, которые пользователь moex создаст потом: права на чтение у них
-- будут сразу, без новых GRANT.
ALTER DEFAULT PRIVILEGES FOR ROLE moex IN SCHEMA public GRANT SELECT ON TABLES TO grafana;
