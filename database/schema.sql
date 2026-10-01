-- PostgreSQL schema. Columns follow the AI4I 2020 data: it has no timestamps,
-- voltage, current, vibration or pressure, so those are not invented here.
-- `timestamp`, `installation_date`, `operating_hours` and `status` are nullable
-- because the dataset does not provide them.

CREATE TABLE IF NOT EXISTS machines (
    machine_id        VARCHAR(20) PRIMARY KEY,
    machine_type      CHAR(1)     NOT NULL CHECK (machine_type IN ('L', 'M', 'H')),
    installation_date DATE,
    operating_hours   NUMERIC(12, 2),
    status            VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id            BIGSERIAL PRIMARY KEY,
    machine_id            VARCHAR(20) NOT NULL REFERENCES machines (machine_id),
    "timestamp"           TIMESTAMPTZ,
    air_temperature_k     NUMERIC(7, 2),
    process_temperature_k NUMERIC(7, 2),
    rotational_speed_rpm  NUMERIC(8, 2),
    torque_nm             NUMERIC(7, 2),
    tool_wear_min         NUMERIC(8, 2),
    machine_failure       SMALLINT CHECK (machine_failure IN (0, 1))  -- recorded label
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       BIGSERIAL PRIMARY KEY,
    machine_id          VARCHAR(20) NOT NULL REFERENCES machines (machine_id),
    "timestamp"         TIMESTAMPTZ NOT NULL DEFAULT now(),
    failure_probability NUMERIC(7, 6) NOT NULL CHECK (failure_probability BETWEEN 0 AND 1),
    risk_level          VARCHAR(10) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    prediction          SMALLINT NOT NULL CHECK (prediction IN (0, 1)),
    anomaly_score       NUMERIC(8, 6),
    anomaly_status      VARCHAR(10) CHECK (anomaly_status IN ('NORMAL', 'ANOMALY')),
    model_version       VARCHAR(40)
);

CREATE TABLE IF NOT EXISTS maintenance_records (
    maintenance_id   BIGSERIAL PRIMARY KEY,
    machine_id       VARCHAR(20) NOT NULL REFERENCES machines (machine_id),
    maintenance_date DATE NOT NULL,
    maintenance_type VARCHAR(40),
    description      TEXT
);

CREATE INDEX IF NOT EXISTS idx_readings_machine    ON sensor_readings (machine_id);
CREATE INDEX IF NOT EXISTS idx_predictions_machine ON predictions (machine_id, "timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_maintenance_machine ON maintenance_records (machine_id, maintenance_date);
