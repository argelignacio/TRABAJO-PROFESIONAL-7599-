SELECT
  EXTRACT(YEAR FROM block_timestamp) as year,
  EXTRACT(MONTH FROM block_timestamp) as month,
  EXTRACT(DAY FROM block_timestamp) as day,
  SUM(IF(transaction_type <> 2, 1, 0)) as sum_legacy,
  COUNT(*) as transaction_count
FROM bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions t
WHERE block_timestamp >= '2020-01-01'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3