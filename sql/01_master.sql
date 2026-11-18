CREATE OR REPLACE TABLE master AS
WITH rev AS (
  SELECT order_id, AVG(review_score) AS review_score
  FROM read_csv_auto('data/olist_order_reviews_dataset.csv')
  GROUP BY order_id
),
items AS (
  SELECT order_id, SUM(price) AS revenue, SUM(freight_value) AS freight
  FROM read_csv_auto('data/olist_order_items_dataset.csv')
  GROUP BY order_id
)
SELECT o.order_id, c.customer_unique_id, c.customer_state,
       CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
       CAST(o.order_delivered_customer_date AS TIMESTAMP) AS delivered_ts,
       CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS estimated_ts,
       i.revenue, i.freight, r.review_score
FROM read_csv_auto('data/olist_orders_dataset.csv') o
JOIN read_csv_auto('data/olist_customers_dataset.csv') c ON o.customer_id = c.customer_id
JOIN items i ON o.order_id = i.order_id
LEFT JOIN rev r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL;