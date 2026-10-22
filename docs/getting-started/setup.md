# Set Up the Client

Install the client:

```bash
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

If you cloned the repository (needed to run the recipes, which are not part of
the installed package), install it in editable mode instead:

```bash
uv pip install -e .
```

The quickest setup is interactive:

```bash
cortex-training login
```

It picks one of your Snowflake connection profiles or creates one (account
host, user, and a programmatic access token), checks the database Cortex
Training will use, and remembers the profile in
`~/.config/cortex-training/config.json`. See [login](../reference/cli.md#login).

To set it up by hand instead, configure a standard Snowflake connection profile
in `~/.snowflake/connections.toml`:

```toml
[training]
account = "ORG-ACCOUNT"
host = "ACCOUNT.snowflakecomputing.com"
user = "USER"
authenticator = "programmatic_access_token"
token = "YOUR_PROGRAMMATIC_ACCESS_TOKEN"
database = "CORTEX_TRAINING_DB"
schema = "PUBLIC"
```

Protect the file and verify the connection:

```bash
chmod 600 ~/.snowflake/connections.toml
cortex-training --connection training capacity
```

Set `SNOWFLAKE_DEFAULT_CONNECTION_NAME=training`, configure
`default_connection_name = "training"` in Snowflake's `config.toml`, or name
the profile `[default]` to run `cortex-training capacity` without
`--connection`.

### Legacy JSON configuration

Existing JSON connection files remain supported with `--config`.
From a clone, copy the template:

```bash
cp examples/config/connection.json.template ~/cortex-training-config.json
```

Otherwise create it by hand, outside the repository:

```json
{
  "host": "ACCOUNT.snowflakecomputing.com",
  "pat": "YOUR_PROGRAMMATIC_ACCESS_TOKEN",
  "database": "CORTEX_TRAINING_DB",
  "schema": "PUBLIC"
}
```

Fill in the account host, programmatic access token, database, and schema. Keep
the file outside the repository and do not commit it.

Pass it to each command:

```bash
cortex-training --config ~/cortex-training-config.json capacity
```

`cortex-training login` no longer remembers a JSON file; run
`cortex-training login` with no arguments to set up a profile instead.

`capacity` with no `--hardware` lists every GPU type. See
[GPU hardware](../concepts/hardware.md).

See the [CLI reference](../reference/cli.md) for environment variables,
alternative server targets, and one-command overrides.
