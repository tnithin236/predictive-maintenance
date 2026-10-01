-- Analytics queries (PostgreSQL). Some spec examples need data AI4I lacks;
-- those are adapted and noted.

-- 1. Machines with highest predicted risk (latest prediction per machine)
WITH ranked AS (
    SELECT p.*,
           ROW_NUMBER() OVER (PARTITION BY p.machine_id
                              ORDER BY p."timestamp" DESC, p.prediction_id DESC) AS rn
    FROM predictions p
)
SELECT r.machine_id, m.machine_type, r.failure_probability, r.risk_level,
       r.anomaly_status, r.model_version
FROM ranked r
JOIN machines m USING (machine_id)
WHERE r.rn = 1
ORDER BY r.failure_probability DESC
LIMIT 20;

-- 2. Average torque and tool wear per machine, ranked within machine type
--    (adapted: AI4I has no vibration sensor)
SELECT m.machine_id, m.machine_type,
       AVG(s.torque_nm)     AS avg_torque_nm,
       AVG(s.tool_wear_min) AS avg_tool_wear_min,
       RANK() OVER (PARTITION BY m.machine_type ORDER BY AVG(s.tool_wear_min) DESC) AS wear_rank_in_type
FROM machines m
JOIN sensor_readings s USING (machine_id)
GROUP BY m.machine_id, m.machine_type
ORDER BY m.machine_type, wear_rank_in_type
LIMIT 100;

-- 3. Recorded failure count and rate by machine type
SELECT m.machine_type,
       COUNT(*)                                      AS readings,
       SUM(s.machine_failure)                        AS failures,
       ROUND(100.0 * AVG(s.machine_failure::numeric), 2) AS failure_pct
FROM sensor_readings s
JOIN machines m USING (machine_id)
GROUP BY m.machine_type
ORDER BY failures DESC;

-- 4. Monthly failure trend (returns rows only when readings carry a timestamp;
--    the AI4I dataset has none, so this applies to future time-stamped data)
SELECT date_trunc('month', s."timestamp") AS month,
       SUM(s.machine_failure)             AS failures,
       COUNT(*)                           AS readings,
       SUM(s.machine_failure) - LAG(SUM(s.machine_failure)) OVER (ORDER BY date_trunc('month', s."timestamp")) AS change_vs_prev_month
FROM sensor_readings s
WHERE s."timestamp" IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- 5. Machines with repeated anomalies (scored 2+ times as ANOMALY)
WITH anomalous AS (
    SELECT machine_id, COUNT(*) AS anomaly_count, MAX("timestamp") AS last_seen,
           AVG(anomaly_score) AS avg_anomaly_score
    FROM predictions
    WHERE anomaly_status = 'ANOMALY'
    GROUP BY machine_id
)
SELECT a.*, m.machine_type
FROM anomalous a
JOIN machines m USING (machine_id)
WHERE a.anomaly_count >= 2
ORDER BY a.anomaly_count DESC, a.last_seen DESC;

-- 6. Maintenance history with days since the previous maintenance
SELECT m.machine_id, m.machine_type, r.maintenance_date, r.maintenance_type, r.description,
       r.maintenance_date - LAG(r.maintenance_date) OVER (
           PARTITION BY r.machine_id ORDER BY r.maintenance_date) AS days_since_previous
FROM machines m
LEFT JOIN maintenance_records r USING (machine_id)
ORDER BY m.machine_id, r.maintenance_date;

-- 7. Average sensor values for failing vs non-failing readings, by machine type
--    (AI4I has no time axis, so "before failure" is approximated by label)
WITH labelled AS (
    SELECT m.machine_type, s.machine_failure, s.air_temperature_k, s.process_temperature_k,
           s.rotational_speed_rpm, s.torque_nm, s.tool_wear_min
    FROM sensor_readings s
    JOIN machines m USING (machine_id)
    WHERE s.machine_failure IS NOT NULL
)
SELECT machine_type,
       CASE machine_failure WHEN 1 THEN 'failure' ELSE 'no_failure' END AS outcome,
       COUNT(*)                        AS n,
       AVG(air_temperature_k)          AS avg_air_temp_k,
       AVG(process_temperature_k)      AS avg_process_temp_k,
       AVG(rotational_speed_rpm)       AS avg_speed_rpm,
       AVG(torque_nm)                  AS avg_torque_nm,
       AVG(tool_wear_min)              AS avg_tool_wear_min
FROM labelled
GROUP BY machine_type, machine_failure
ORDER BY machine_type, machine_failure DESC;
