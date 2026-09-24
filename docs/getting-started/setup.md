# Setup

To run a training recipe, you need the following. If any are missing, follow
the detailed instructions below to set each one up.

1. **Python 3.10+** and **uv** package manager
2. **A programmatic access token (PAT)** for authentication
3. **A connection config** pointing to your Snowflake account
4. **Cortex Training endpoint** and **GPU capacity** available on your Snowflake account

## 1. Install Dependencies

Requires Python 3.10 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Verify your Python version:

```bash
python3 --version
```

See the [install instructions](../../README.md#install) to set up the client
and clone the repository. Make sure to activate the virtual environment before
continuing:

```bash
source .venv/bin/activate
```

## 2. Create a Programmatic Access Token (PAT)

You need a PAT to authenticate the CLI with your Snowflake account.

1. Log in to your Snowflake account in a browser
2. Click your user menu (bottom left) → **Settings**
3. Select **Authentication** in the left sidebar
4. Scroll down to **Programmatic access tokens**
5. Click **Generate token**
6. Copy the token — you will not be able to see it again

**Note**: If your account hostname contains underscores (e.g. `my_account`),
replace them with hyphens (e.g. `my-account`) when using it in the connection
config. The SSL certificate requires hyphens.

## 3. Create a Connection Config and Log In

If you cloned the repository, copy the template:

```bash
cp examples/config/connection.json.template ~/cortex-training-config.json
```

Otherwise create the file by hand:

```json
{
  "host": "ACCOUNT.snowflakecomputing.com",
  "pat": "YOUR_PAT",
  "database": "YOUR_DATABASE",
  "schema": "PUBLIC"
}
```

Fill in your account hostname and the PAT from step 2. The endpoint is
account-scoped — any database and schema on your account will work.

Register the config with the CLI:

```bash
cortex-training login ~/cortex-training-config.json
```

## 4. Verify the Endpoint and Capacity

Check that Cortex Training is enabled on your account and that GPU capacity
is available:

```bash
cortex-training capacity
```

If your endpoint is active, you should see output like:

```json
{
  "capacity_by_hardware": {
    "H200": {
      "available_gpus": 48,
      "has_reservation": false,
      "in_use_gpus": 0,
      ...
    }
  }
}
```

If you get a connection or authorization error, the endpoint may not be enabled
yet — contact your Snowflake account team to request access. If capacity shows
zero available GPUs across all hardware types, contact your account team to
request GPU capacity.

This is the quickest path to a working setup. For environment variables,
alternative authentication methods, and advanced configuration, see the
[CLI reference](../reference/cli.md).
