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

To run `cortex-training capacity` without `--connection`, run
`cortex-training login training` once, which saves the profile in
`config.json`. Without a saved profile, commands use your Snowflake default
profile (`SNOWFLAKE_DEFAULT_CONNECTION_NAME`, `default_connection_name` in
Snowflake's `config.toml`, or a profile named `[default]`).

Every command that connects starts with one line on stderr naming the profile,
database, and schema it uses, and where each came from.

### Have an old JSON config?

Commands no longer sign in with a JSON file (`--config`, `CORTEX_TRAINING_CONFIG`,
or a file remembered by an older `cortex-training login`). Convert it into a
profile once:

```bash
cortex-training login --convert-legacy-config ~/cortex-training-config.json
```

In a terminal, a command that finds an old JSON config offers to do this for
you. See [Option B](authentication.md#option-b-json-config-legacy).

### Upgrading

How to get a new version depends on how you installed it:

| Installed with | Upgrade with |
|---|---|
| `uv pip install -e .` in a clone (needed for recipes) | `git pull` in the clone. The recipes and the library update together. |
| `uv pip install cortex-training` or `git+https://...` | `uv pip install --upgrade cortex-training` (or the same `git+https://...` URL with `--upgrade`) |

Recipes come only from a clone. If the clone's library came from PyPI instead
of `-e .`, the two can differ; recipes then stop with a message to run
`uv pip install -e .` in the clone.

Before an upgrade's breaking changes, earlier releases warn first. For
example, 0.1.0 warns about the forms that stop working in 0.2.0. For jobs that
run unattended, pin the version so an upgrade can't change them without you,
for example `cortex-training>=0.1,<0.2`.

`capacity` with no `--hardware` lists every GPU type. See
[GPU hardware](../concepts/hardware.md).

See the [CLI reference](../reference/cli.md) for environment variables,
alternative server targets, and one-command overrides.
