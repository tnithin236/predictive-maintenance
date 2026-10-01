-- Loads REAL dataset rows only (no fabricated data). Steps:
--   1. python -m src.train                  (creates data/processed/ai4i2020_clean.csv)
--   2. python -m src.database export        (writes the two CSVs below)
--   3. psql "$DATABASE_URL" -f database/schema.sql
--   4. psql "$DATABASE_URL" -f database/seed.sql   (run from the project root)
-- Alternative: `python -m src.database seed` does the same through SQLAlchemy.
-- maintenance_records stays empty: the dataset has no maintenance history.

\copy machines (machine_id, machine_type) FROM 'data/processed/db_machines.csv' WITH (FORMAT csv, HEADER true)
\copy sensor_readings (machine_id, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min, machine_failure) FROM 'data/processed/db_sensor_readings.csv' WITH (FORMAT csv, HEADER true)
