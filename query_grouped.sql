SELECT
  t.from_address,
  t.to_address,
  COUNT(*) as count_transactions,
  SUM(t.value / 1E18) as sum_transactions
FROM bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions t
WHERE block_timestamp >= '2022-05-01'
  AND block_timestamp <= '2022-07-31'
  AND value > 0
GROUP BY from_address, to_address
HAVING count_transactions > 3 AND sum_transactions > 0.01
ORDER BY count_transactions DESC