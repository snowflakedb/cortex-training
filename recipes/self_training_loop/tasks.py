# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic retail SQL tasks: the database, the questions with reference answers, and the tools.

The business rules exist only in the system prompt. Baseline tasks (ids ``h*``) are the
first traffic the agent sees; drift tasks (ids ``n*``) are four question types added after
the first trained version, standing in for traffic that changes after deployment. Every
task's reference answer is computed in SQL; it is used only to measure the judges.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import uuid

SYSTEM_PROMPT = (
    "You are a retail finance analytics agent. Answer the user's question using the tools. "
    "Use sql_query to read the SQLite database. Tables: customers(id, name, region, segment); "
    "products(id, name, category, list_price); orders(id, customer_id, order_date, status) with order_date "
    "'YYYY-MM-DD' and status in ('shipped','cancelled','pending'); order_items(order_id, product_id, qty, "
    "unit_price, discount_pct); returns(order_id, product_id, qty_returned, return_date). "
    "Business rules: only orders with status 'shipped' count. A line's net value is "
    "qty * unit_price * (1 - discount_pct/100). Net revenue is the sum of line net values minus the "
    "value of returned units, where returned value is qty_returned * unit_price * (1 - discount_pct/100) "
    "of the matching order line. A sale's region and segment come from the customer. Months are by order_date. "
    "Use calculator for arithmetic you do not do in SQL. Check your result before answering. "
    "End your reply with a line 'FINAL ANSWER: <value>' where value is a number (no units, percent signs or "
    "thousands separators) or a name."
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a read-only SQLite query and return up to 50 rows as JSON.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression with + - * / ( ) and decimals.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

CATEGORIES = ["Outdoor", "Kitchen", "Office", "Toys", "Garden"]
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Business"]
MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]
BASELINE_KINDS = [
    "net_rev_cat_region_month",
    "return_rate_cat_month",
    "mom_growth_segment",
    "second_customer_region_month",
    "aov_segment_region_month",
    "discount_gap_product_region",
    "churned_customers",
]
DRIFT_KINDS = [
    "top_category_region_month",
    "return_value_segment_month",
    "category_share_region_month",
    "repeat_buyers_cat_region",
]

NET_LINES = """
WITH lines AS (
  SELECT o.id AS order_id, o.order_date, substr(o.order_date,1,7) AS month, c.id AS customer_id, c.name AS customer,
         c.region, c.segment, p.id AS product_id, p.name AS product, p.category, p.list_price,
         oi.qty, oi.unit_price, oi.discount_pct,
         oi.qty * oi.unit_price * (1 - oi.discount_pct/100.0) AS gross_net,
         COALESCE((SELECT SUM(r.qty_returned) FROM returns r WHERE r.order_id=o.id AND r.product_id=p.id), 0) AS qty_ret
  FROM orders o JOIN customers c ON c.id=o.customer_id JOIN order_items oi ON oi.order_id=o.id
  JOIN products p ON p.id=oi.product_id WHERE o.status='shipped'
), net AS (
  SELECT *, gross_net - qty_ret * unit_price * (1 - discount_pct/100.0) AS net_value FROM lines
)
"""


def build_db(seed: int = 7) -> sqlite3.Connection:
    """The retail database. Deterministic for a seed, so stored runs can be replayed."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(
        """
      CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT, segment TEXT);
      CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, list_price REAL);
      CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
      CREATE TABLE order_items(order_id INTEGER, product_id INTEGER, qty INTEGER, unit_price REAL, discount_pct REAL);
      CREATE TABLE returns(order_id INTEGER, product_id INTEGER, qty_returned INTEGER, return_date TEXT);
    """
    )
    first = [
        "Acme",
        "Birch",
        "Cobalt",
        "Dune",
        "Echo",
        "Flint",
        "Grove",
        "Halo",
        "Iris",
        "Juno",
        "Kite",
        "Lumen",
    ]
    second = ["Labs", "Supply", "Works", "Co", "Group", "Partners"]
    for cid in range(1, 61):
        conn.execute(
            "INSERT INTO customers VALUES (?,?,?,?)",
            (
                cid,
                f"{rng.choice(first)} {rng.choice(second)} {cid}",
                rng.choice(REGIONS),
                rng.choice(SEGMENTS),
            ),
        )
    adjectives = [
        "Alpine",
        "Brisk",
        "Cedar",
        "Delta",
        "Ember",
        "Fjord",
        "Granite",
        "Harbor",
    ]
    pid = 0
    for cat in CATEGORIES:
        for adj in rng.sample(adjectives, 6):
            pid += 1
            name = f"{adj} {cat.removesuffix('s')} {pid}"
            conn.execute(
                "INSERT INTO products VALUES (?,?,?,?)",
                (pid, name, cat, round(rng.uniform(5, 250), 2)),
            )
    prices = {r[0]: r[1] for r in conn.execute("SELECT id, list_price FROM products")}
    for oid in range(1, 901):
        date = f"{rng.choice(MONTHS)}-{rng.randint(1, 28):02d}"
        status = rng.choices(["shipped", "cancelled", "pending"], [0.8, 0.12, 0.08])[0]
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?)",
            (oid, rng.randint(1, 60), date, status),
        )
        for prod in rng.sample(range(1, pid + 1), rng.randint(1, 4)):
            qty = rng.randint(1, 10)
            unit = round(prices[prod] * rng.uniform(0.85, 1.05), 2)
            disc = rng.choice([0, 0, 0, 5, 10, 15, 20])
            conn.execute(
                "INSERT INTO order_items VALUES (?,?,?,?,?)",
                (oid, prod, qty, unit, disc),
            )
            if status == "shipped" and rng.random() < 0.12:
                conn.execute(
                    "INSERT INTO returns VALUES (?,?,?,?)",
                    (oid, prod, rng.randint(1, qty), f"{date[:7]}-28"),
                )
            elif status != "shipped" and rng.random() < 0.05:
                # Returns on non-shipped orders are data noise that the rules exclude.
                conn.execute(
                    "INSERT INTO returns VALUES (?,?,?,?)",
                    (oid, prod, 1, f"{date[:7]}-28"),
                )
    conn.commit()
    return conn


def split_for(task_id: str, heldout_fraction: float) -> str:
    """Stable task-level split: every run of a held-out task stays out of training."""
    bucket = int(uuid.uuid5(uuid.NAMESPACE_URL, task_id).hex, 16) % 100
    return "heldout" if bucket < heldout_fraction * 100 else "train"


def _finish(
    tasks: list[dict],
    seen: set,
    prefix: str,
    kind: str,
    question: str,
    answer,
    heldout_fraction: float,
):
    if answer is None or question in seen:
        return
    if isinstance(answer, float):
        answer = round(answer, 2)
    seen.add(question)
    task_id = f"{prefix}{len(tasks):04d}"
    tasks.append(
        {
            "task_id": task_id,
            "kind": kind,
            "question": question,
            "answer": answer,
            "split": split_for(task_id, heldout_fraction),
        }
    )


def baseline_tasks(
    conn: sqlite3.Connection,
    n: int = 240,
    seed: int = 13,
    heldout_fraction: float = 0.2,
):
    rng = random.Random(seed)

    def q1(sql, params=()):
        return conn.execute(NET_LINES + sql, params).fetchone()[0]

    tasks: list[dict] = []
    seen: set = set()
    attempts = 0
    while len(tasks) < n and attempts < 20000:
        attempts += 1
        kind = rng.choice(BASELINE_KINDS)
        cat, reg, seg = (
            rng.choice(CATEGORIES),
            rng.choice(REGIONS),
            rng.choice(SEGMENTS),
        )
        mi = rng.randint(0, len(MONTHS) - 2)
        month, nxt = MONTHS[mi], MONTHS[mi + 1]
        if kind == "net_rev_cat_region_month":
            q = f"What was the net revenue from {cat} products sold to {reg} region customers in {month}? Round to 2 decimals."
            a = q1(
                "SELECT COALESCE(SUM(net_value),0) FROM net WHERE category=? AND region=? AND month=?",
                (cat, reg, month),
            )
        elif kind == "return_rate_cat_month":
            q = f"What percentage of {cat} units shipped in {month} were returned? Round to 2 decimals."
            a = q1(
                "SELECT 100.0*SUM(qty_ret)/NULLIF(SUM(qty),0) FROM net WHERE category=? AND month=?",
                (cat, month),
            )
        elif kind == "mom_growth_segment":
            q = (
                f"By what percent did net revenue from {seg} customers change from {month} to {nxt}? "
                "Negative if it fell. Round to 2 decimals."
            )
            a0 = q1(
                "SELECT SUM(net_value) FROM net WHERE segment=? AND month=?",
                (seg, month),
            )
            a1 = q1(
                "SELECT SUM(net_value) FROM net WHERE segment=? AND month=?", (seg, nxt)
            )
            a = None if not a0 else 100.0 * (a1 - a0) / a0
        elif kind == "second_customer_region_month":
            q = f"Which customer had the second-highest net revenue in the {reg} region in {month}? Give the customer name."
            a = q1(
                "SELECT customer FROM net WHERE region=? AND month=? GROUP BY customer_id "
                "ORDER BY SUM(net_value) DESC, customer_id LIMIT 1 OFFSET 1",
                (reg, month),
            )
        elif kind == "aov_segment_region_month":
            q = (
                f"What was the average net revenue per shipped order for {seg} customers in the {reg} region "
                f"in {month}? Round to 2 decimals."
            )
            a = q1(
                "SELECT SUM(net_value)/NULLIF(COUNT(DISTINCT order_id),0) FROM net WHERE segment=? AND region=? AND month=?",
                (seg, reg, month),
            )
        elif kind == "discount_gap_product_region":
            q = (
                f"Among {cat} products sold to {reg} region customers, which one has the largest gap between its "
                "list price and its average realized price per unit sold (after discount, shipped orders, weighted "
                "by quantity)? Give the product name."
            )
            a = q1(
                "SELECT product FROM net WHERE category=? AND region=? GROUP BY product_id "
                "ORDER BY MAX(list_price) - SUM(qty*unit_price*(1-discount_pct/100.0))/SUM(qty) DESC, product_id LIMIT 1",
                (cat, reg),
            )
        else:
            q = f"How many {reg} region customers had at least one shipped order in {month} but no shipped orders in {nxt}?"
            a = q1(
                "SELECT COUNT(*) FROM (SELECT DISTINCT customer_id FROM net WHERE region=? AND month=? "
                "EXCEPT SELECT DISTINCT customer_id FROM net WHERE month=?)",
                (reg, month, nxt),
            )
        _finish(tasks, seen, "h", kind, q, a, heldout_fraction)
    return tasks


def drift_tasks(
    conn: sqlite3.Connection,
    n: int = 160,
    seed: int = 29,
    heldout_fraction: float = 0.2,
):
    rng = random.Random(seed)

    def q1(sql, params=()):
        return conn.execute(NET_LINES + sql, params).fetchone()[0]

    tasks: list[dict] = []
    seen: set = set()
    attempts = 0
    while len(tasks) < n and attempts < 20000:
        attempts += 1
        kind = rng.choice(DRIFT_KINDS)
        cat, reg, seg, month = (
            rng.choice(CATEGORIES),
            rng.choice(REGIONS),
            rng.choice(SEGMENTS),
            rng.choice(MONTHS),
        )
        if kind == "top_category_region_month":
            q = f"Which product category had the highest net revenue from {reg} region customers in {month}? Give the category."
            a = q1(
                "SELECT category FROM net WHERE region=? AND month=? GROUP BY category "
                "ORDER BY SUM(net_value) DESC, category LIMIT 1",
                (reg, month),
            )
        elif kind == "return_value_segment_month":
            q = (
                f"What was the total value of units returned by {seg} customers on shipped orders placed in {month}, "
                "at the discounted price paid? Round to 2 decimals."
            )
            a = q1(
                "SELECT SUM(qty_ret*unit_price*(1-discount_pct/100.0)) FROM net WHERE segment=? AND month=?",
                (seg, month),
            )
        elif kind == "category_share_region_month":
            q = f"What percentage of the {reg} region's net revenue in {month} came from {cat} products? Round to 2 decimals."
            tot = q1(
                "SELECT SUM(net_value) FROM net WHERE region=? AND month=?",
                (reg, month),
            )
            part = q1(
                "SELECT COALESCE(SUM(net_value),0) FROM net WHERE region=? AND month=? AND category=?",
                (reg, month, cat),
            )
            a = None if not tot else 100.0 * part / tot
        else:
            q = f"How many {reg} region customers bought {cat} products on shipped orders in at least two different months?"
            a = q1(
                "SELECT COUNT(*) FROM (SELECT customer_id FROM net WHERE region=? AND category=? "
                "GROUP BY customer_id HAVING COUNT(DISTINCT month) >= 2)",
                (reg, cat),
            )
        _finish(tasks, seen, "n", kind, q, a, heldout_fraction)
    return tasks


def load_tasks(*, drift: bool = True) -> tuple[sqlite3.Connection, list[dict]]:
    """``(db, tasks)``: the baseline tasks, plus the drift tasks when ``drift``."""
    db = build_db()
    tasks = baseline_tasks(db)
    if drift:
        tasks += drift_tasks(db)
    return db, tasks


FINAL_ANSWER = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


def extract_final(text: str) -> str | None:
    found = FINAL_ANSWER.findall(text or "")
    return found[-1].strip().strip("*`'\". ") if found else None


def is_correct(final_text: str, expected) -> bool:
    """Final answer matches the reference: within 0.011 for numbers, case-insensitive for names."""
    got = extract_final(final_text)
    if got is None:
        return False
    if isinstance(expected, (int, float)):
        try:
            return (
                abs(float(got.replace(",", "").replace("$", "")) - float(expected))
                <= 0.011
            )
        except ValueError:
            return False
    return got.lower() == str(expected).lower()


def run_sql(conn: sqlite3.Connection, query: str) -> str:
    """The sql_query tool: SELECT or WITH only, up to 50 rows as JSON."""
    if not re.match(r"^\s*(select|with)\b", query, re.IGNORECASE):
        return "ERROR: only SELECT queries are allowed"
    try:
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description]
        return json.dumps([dict(zip(cols, r)) for r in cur.fetchmany(50)])
    except Exception as e:  # surfaced to the model
        return f"ERROR: {e}"


def run_calculator(expression: str) -> str:
    """The calculator tool: digits, + - * / ( ), decimals and exponents only."""
    if not re.fullmatch(r"[0-9.+\-*/() eE]+", expression):
        return "ERROR: unsupported characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: {e}"
