CREATE TABLE btc_prices_week (
    date DATE PRIMARY KEY,
    open DECIMAL(18,3),
    high DECIMAL(18,3),
    low DECIMAL(18,3),
    close DECIMAL(18,3),
    volume DECIMAL(18,3)
);