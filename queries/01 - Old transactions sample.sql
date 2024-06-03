SELECT
  EXTRACT(YEAR FROM block_timestamp) as year,
  EXTRACT(MONTH FROM block_timestamp) as month,
  EXTRACT(DAY FROM block_timestamp) as day,
  from_address,
  to_address,
  COUNT(*) as transaction_count,
  SUM(IF(transaction_type <> 2, 1, 0)) as legacy_transaction_count,
  SUM(value) as total_value,
  SUM(gas) as total_gas,
  SUM(gas_price) as total_gas_values
FROM bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions
WHERE block_timestamp >= '2022-05-01'
  AND block_timestamp <= '2022-05-03'
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1, 2, 3