# Cortex Training connection errors

Match the error the user sees (from `cortex-training capacity`, `login`, or a
job) to the first row it fits. Never ask for the token to "check" it.

| Error contains | Cause | Fix |
|---|---|---|
| `403` on every call, or "Verify the account name is correct" | The network can't reach the account (a VPN that's off), or the user has no network policy, which a PAT needs | Check the VPN. Otherwise see [Network policy requirement](https://github.com/snowflakedb/cortex-training/blob/main/docs/getting-started/authentication.md#network-policy-requirement) |
| "Programmatic access token is invalid" | The PAT expired, was revoked, or was mistyped | The user creates a new PAT in Snowsight, then runs `cortex-training login` |
| `517600`, or "isn't enabled on account" | Cortex Training isn't enabled on that account | The user contacts their Snowflake account team, or picks a profile on another account |
| "Schema DB.SCHEMA is not found or not authorized" | The database or schema doesn't exist, or the role can't see it | `cortex-training login` offers to create a missing database (it shows the SQL and asks). For an existing one, an admin runs `GRANT USAGE ON DATABASE DB TO ROLE R` and `GRANT USAGE ON SCHEMA DB.SCHEMA TO ROLE R` |
| `003001` and "must have USE CORTEX TRAINING granted" | The role lacks the account privilege for creating jobs | An account admin runs `GRANT USE CORTEX TRAINING ON ACCOUNT TO ROLE R` |
| "CREATE EXPERIMENT privilege required on schema" | Each job creates an experiment; the role can't in that schema | An admin runs `GRANT CREATE EXPERIMENT ON SCHEMA DB.SCHEMA TO ROLE R` |
| "MODIFY privilege required on experiment" | The job joins an existing experiment the role can't change | Use a new experiment (omit `experiment_name`), or an admin grants `MODIFY` on that experiment |
| "has 2 profiles named NAME" | `connections.toml` names a profile twice, so Snowflake can't read the file | `cortex-training login` shows the fix and asks before renaming the extra copy |
| "profile 'NAME' … does not exist; known profiles: …" | A typo in `--connection`, or the profile file is in another folder | Check the name with `cortex-training status`; `profile_file` shows which file is read |
| "signing in with an old JSON config … is no longer supported" | `--config`, `CORTEX_TRAINING_CONFIG`, or the `login.json` an old `cortex-training login FILE` wrote | The user runs `cortex-training login --convert-legacy-config FILE` in their terminal (it keeps the file), or unsets `CORTEX_TRAINING_CONFIG` |
| "--host and --pat are no longer supported" | Signing in with flags instead of a profile | The user runs `cortex-training login`, which can create a profile from an account and a PAT. In CI, set `CORTEX_TRAINING_HOST` and `CORTEX_TRAINING_PAT` |
| "is set but ignored" | An old variable (`SNOWFLAKE_HOST`, `SNOWFLAKE_PAT`, `CORTEX_TRAINING_CONNECTION`, …) | Nothing is wrong; the warning names what to use instead. Unset the variable to stop it |
| "will be deprecated in a future release" | A recipe's `config=FILE`, or the SDK's `from_pat()`/`from_connection_name()` | It still works. Suggest `cortex-training login --convert-legacy-config FILE`, then `connection=NAME` |

The full list of privileges per operation is in
[Access control](https://github.com/snowflakedb/cortex-training/blob/main/docs/reference/rest-api.md#37-access-control).
