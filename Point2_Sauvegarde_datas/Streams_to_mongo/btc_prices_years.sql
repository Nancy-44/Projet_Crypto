CREATE TABLE btc_prices_years (
    date DATE PRIMARY KEY,
    open DECIMAL(18,3),
    high DECIMAL(18,3),
    low DECIMAL(18,3),
    close DECIMAL(18,3),
    volume DECIMAL(18,3)
);btc_prices_hours