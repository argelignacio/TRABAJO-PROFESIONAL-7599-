SELECT
  t.block_timestamp,
  t.transaction_index,
  t.nonce,
  t.from_address,
  t.to_address,
  t.value,
  t.value_lossless,
  t.gas,
  t.gas_price,
  t.max_fee_per_gas,
  t.max_priority_fee_per_gas,
  t.transaction_type,
FROM
  bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions as t
WHERE (block_timestamp >= "2022-05-01" AND block_timestamp <= "2022-05-04" AND value != 0)
