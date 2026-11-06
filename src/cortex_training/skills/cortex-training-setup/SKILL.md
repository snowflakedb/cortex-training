---
name: cortex-training-setup
description: "Check and fix a user's Cortex Training connection with the cortex-training CLI. Use when: setting up Cortex Training, checking which Snowflake profile, account, role or database cortex-training uses, a cortex-training command can't connect (403, PAT invalid, not enabled, schema not found), converting an old JSON connection config, or a job is refused for USE CORTEX TRAINING or CREATE EXPERIMENT. Triggers: cortex-training login, cortex-training status, connections.toml, which account, can't connect, convert old config."
---

# Cortex Training setup

Cortex Training connects with a standard Snowflake connection profile (the
`connections.toml` file the Snowflake CLI and Python Connector use). Help the
user see what it uses, confirm it works, and fix what doesn't.

## Rules

- **Never ask for, print, or store a token or password.** Don't read the
  `token`/`password` lines of `connections.toml` aloud or paste them anywhere.
- **Run only these commands yourself:** `cortex-training --version`,
  `cortex-training status`, and `cortex-training capacity`. All are read-only.
- **Hand interactive commands to the user**, to run in their own terminal:
  `cortex-training login`, `cortex-training login NAME`, and
  `cortex-training login --convert-legacy-config FILE`. They ask questions and
  can take a token as hidden input.
- **Don't edit `connections.toml` or `config.json`** unless the user explicitly
  asks; show the change instead.
- **Don't create databases or grant privileges.** Show the SQL for an account
  admin to run.

## Step 1: Is the CLI installed?

```bash
cortex-training --version
```

If the command is missing, the user installs it:
`uv pip install git+https://github.com/snowflakedb/cortex-training.git`. If
`cortex-training status` (Step 2) is an unknown command, the installed version
is too old: reinstall the same way. If a command prints that this skill is out
of date, suggest `cortex-training skills install` to update it.

## Step 2: See what it uses (offline)

```bash
cortex-training status
```

It never signs in. It prints JSON with the same keys every time; read
`connection`, `profile`, `account`, `role`, `database`, `schema`, `usable` and
`warnings`.

| What you see | What to do |
|---|---|
| Exit code 2, "No Snowflake connection profile is set up" | Ask the user to run `cortex-training login` in their terminal. It sets up a profile, checks it, and saves it |
| `"connection": "profile"`, `"usable": true`, no `warnings` | Summarize the profile, account, role and database, then go to Step 3 |
| `"usable": false` | Explain the reason in `warnings` (for example "no user set"); suggest `cortex-training login` |
| `"connection": "config file"` or `"flags or environment"` | It uses an older connection method. Suggest the user converts it: `cortex-training login --convert-legacy-config FILE` (it keeps the old file) |
| A `warnings` entry about an environment variable | Explain which variable changes the result, and that `unset NAME` (bash/zsh) or `set -e NAME` (fish) removes it |

`--connection NAME`, `--database` and `--schema` work as with any command:
`cortex-training --connection NAME status`.

## Step 3: Confirm it connects

```bash
cortex-training capacity
```

Read-only. JSON with `capacity_by_hardware` means the profile signs in and
Cortex Training is enabled on the account. On an error, find it in
[troubleshooting.md](troubleshooting.md).

## Step 4: Can the role create jobs?

That check runs in `cortex-training login`. Ask the user to run
`cortex-training login NAME` (NAME from `status`) and share what it printed:

- `✓ Role R can create training jobs`: ready.
- `! Role R can't create training jobs yet`, with GRANT statements: an
  account admin runs those GRANTs, then the user runs `login` again.
- `! Role R doesn't have USE CORTEX TRAINING yet`: jobs still start for now;
  suggest the admin runs the GRANT shown before it's enforced.
- `? Could not check …`: not an error; the job's own message will say if a
  privilege is missing.

## More

- Errors and fixes: [troubleshooting.md](troubleshooting.md)
- Setup guide: <https://github.com/snowflakedb/cortex-training/blob/main/docs/getting-started/authentication.md>
- CLI reference: <https://github.com/snowflakedb/cortex-training/blob/main/docs/reference/cli.md>
