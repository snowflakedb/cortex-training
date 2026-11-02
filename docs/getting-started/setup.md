# Set Up the Client

Install the client:

```bash
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

If you cloned the repository (needed to run the examples, which are not part of
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
`~/.snowflake/cortex-training/config.json`. See [login](../reference/cli.md#login).

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

To run commands without `--connection`, save the profile as the one Cortex
Training uses with `cortex-training login training`. Without a saved profile,
Cortex Training uses Snowflake's default profile: set
`SNOWFLAKE_DEFAULT_CONNECTION_NAME=training`, configure
`default_connection_name = "training"` in Snowflake's `config.toml`, or name
the profile `[default]`.

## JSON connection file

Have a JSON connection file from an older setup? Convert it into a profile with
`cortex-training login --convert-legacy-config FILE` (see
[authentication](authentication.md#old-users-who-log-in-with-configjson)).

`capacity` with no `--hardware` lists every GPU type. See
[GPU hardware](../concepts/hardware.md).

See the [CLI reference](../reference/cli.md) for environment variables,
alternative server targets, and one-command overrides.
