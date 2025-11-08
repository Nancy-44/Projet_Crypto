DROP TABLE IF EXISTS dim_date;

CREATE TABLE dim_date (
    date DATE PRIMARY KEY,
    year INT,
    month INT,
    day INT,
    week INT,
    quarter INT,
    day_name VARCHAR(10),
    month_name VARCHAR(10),
    is_weekend BOOLEAN
);
