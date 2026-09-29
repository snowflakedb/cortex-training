# Authentication

Cortex Training authenticates through a Snowflake Programmatic Access Token
(PAT). This page walks you through three steps:

1. **Create a PAT** — required regardless of which configuration method you use
2. **Create the database** — one-time setup for the connection config
3. **Configure the client** — pick one of the methods below to supply your PAT

## Step 1: Create a PAT

1. Log in to your Snowflake account at `https://<your-account>.snowflakecomputing.com`
2. Click your **user icon** (bottom left)
3. Go to **Settings**
4. Select **Developer** → **Programmatic Access Tokens**
5. Click **Generate New Token**
6. Give it a name (e.g., `cortex-training`) and set an expiry
7. Copy the token value immediately — you cannot view it again

<!-- TODO: add screenshots for steps 2-7 -->

## Step 2: Create the Database

The connection config references a database and schema. This database must
exist in your Snowflake account — otherwise the client will fail with a
"database not found" error. It is only used for connection routing, not for
storing training data.

Run this in Snowsight before configuring the client:

```sql
CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB;
```

You can use any database name, but it must match the `database` field in your
connection config below.

## Step 3: Configure the Client

Once you have a PAT, pick **one** of the following methods to configure the
client. We recommend `connections.toml`.

### Option A: connections.toml (recommended)

This is Snowflake's standard configuration file, shared across all Snowflake
tools. Create or edit `~/.snowflake/connections.toml`:

```toml
[training]
account = "ORG-ACCOUNT"
host = "ACCOUNT.snowflakecomputing.com"
user = "YOUR_USERNAME"
authenticator = "programmatic_access_token"
token = "YOUR_PAT"
database = "CORTEX_TRAINING_DB"
schema = "PUBLIC"
```

Protect the file:

```bash
chmod 600 ~/.snowflake/connections.toml
```

#### Finding your account and host

- **account**: your org and account name joined by a hyphen, e.g.,
  `MYORG-MYACCOUNT`. You can find this in your Snowsight URL:
  `https://app.snowflake.com/MYORG/MYACCOUNT` → `MYORG-MYACCOUNT`.
- **host**: `ACCOUNT.snowflakecomputing.com`, e.g.,
  `myorg-myaccount.snowflakecomputing.com`. Use hyphens, not underscores —
  some Snowflake admin tools show underscores but the SSL certificate requires
  hyphens.

#### Setting the default connection

To avoid passing `--connection training` on every command, set a default using
any of these methods:

- Name the profile `[default]` instead of `[training]`
- Set the environment variable:
  ```bash
  export SNOWFLAKE_DEFAULT_CONNECTION_NAME=training
  ```
- Add `default_connection_name = "training"` to `~/.snowflake/config.toml`

#### Verify

```bash
cortex-training capacity
```

This should print your available GPU capacity.

### Option B: JSON config (legacy)

Create a JSON file outside the repository:

```json
{
  "host": "ACCOUNT.snowflakecomputing.com",
  "pat": "YOUR_PAT",
  "database": "CORTEX_TRAINING_DB",
  "schema": "PUBLIC"
}
```

Register it once so subsequent commands use it automatically:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

### Option C: Environment variables

Set credentials directly in your shell:

```bash
export CORTEX_TRAINING_HOST="ACCOUNT.snowflakecomputing.com"
export CORTEX_TRAINING_PAT="YOUR_PAT"
export CORTEX_TRAINING_DATABASE="CORTEX_TRAINING_DB"
export CORTEX_TRAINING_SCHEMA="PUBLIC"
```

Then run commands directly:

```bash
cortex-training capacity
```

Environment variables override both `connections.toml` and JSON config values.
This is useful for CI/CD pipelines or when you don't want secrets in files.

Snowflake-prefixed variants (`SNOWFLAKE_PAT`, `SNOWFLAKE_HOST`,
`SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`) are also accepted.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `SSL certificate verify failed` | Underscores in hostname | Replace underscores with hyphens in `host` |
| `no PAT found` | Missing token in config | Add `token` to connections.toml, `pat` to JSON, or set `CORTEX_TRAINING_PAT` |
| `401 Unauthorized` | PAT expired or invalid | Regenerate a new PAT in Snowsight |
