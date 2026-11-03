# Troubleshooting

For authentication, URL, and server error messages, see the
[CLI troubleshooting section](cli.md#troubleshooting).

## 403 when creating a job

`Insufficient privileges to operate on Account '<account>'. Your primary role
<role> must have USE CORTEX TRAINING granted on ACCOUNT <account>` means the
role has no `USE CORTEX TRAINING` privilege. An `ACCOUNTADMIN` fixes it with
`GRANT USE CORTEX TRAINING ON ACCOUNT TO ROLE <role>`; access returns
immediately. Other operations on a job are limited to the role that created it,
so a different role gets 404. See
[Access control](rest-api.md#37-access-control).

When an example fails, first capture:

- The example command with secrets removed
- Job and sub-job IDs
- The installed client version, from `python -c "import cortex_training; print(cortex_training.__version__)"`
- Model, precision, GPU count, and sequence length
- The Snowflake request ID. The CLI prints it on server errors. In the Python
  SDK, HTTP errors start with `[snowflake request id: <id>]` when the response
  carried one. A server-side failure with no response of its own (a request
  or job that ended `failed`, a poll that timed out, an undecodable response,
  or a connection error) logs a `WARNING` such as
  `[last snowflake request id: <id>] poll_request failed: ...`; that is the
  operation's most recent request, which did not necessarily fail. Other
  errors, such as client-side validation or a job that reached `done` or
  `cancelled`, log nothing.
- Downloaded execution logs, from `cortex-training download-log JOB_ID`

Example-specific failure modes are documented beside the example rather than
accumulated on this page.
