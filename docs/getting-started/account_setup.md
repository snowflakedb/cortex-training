# Account Setup

This guide walks you from a fresh Snowflake account to a completed training job.
Follow the steps in order.

## Step 1: Create a Snowflake Account

Cortex Training is in public preview and available in all commercial Amazon Web
Services (AWS) regions. Snowflake accounts with a capacity contract have access
by default. If your account has a capacity contract, skip
to [Step 3](#step-3-authenticate).

To try Cortex Training, create a free trial account at
[signup.snowflake.com](https://signup.snowflake.com). Fill in your details and
verify your email. No credit card is required. The cloud provider and region
are assigned automatically, so check in the next step that the region is a
commercial AWS region. Trial accounts need enablement, which is covered in the
next step.

## Step 2: Request Enablement on Trial Accounts

Trial accounts must contact the Snowflake team to enable Cortex Training and
reserve GPU capacity. You will need to provide:

1. **Account locator** — find it in Snowsight by running:
   ```sql
   SELECT CURRENT_ACCOUNT();
   ```
   Or check the URL in your browser: `https://app.snowflake.com/ORG/ACCOUNT` —
   the `ACCOUNT` part is your account locator.

2. **Organization name** — find it by running:
   ```sql
   SELECT CURRENT_ORGANIZATION_NAME();
   ```
   Or from the Snowsight URL: the `ORG` part.

3. **Region** — confirm which cloud region your account is in. It must be a
   commercial AWS region:
   ```sql
   SELECT CURRENT_REGION();
   ```

Share these three values with the Snowflake team. They will:

- Enable the Cortex Training feature on your account
- Reserve GPU capacity for your use

You cannot proceed until this is complete. Once the team confirms enablement,
continue to the next step.

## Step 3: Authenticate

Generate a Programmatic Access Token (PAT): follow
[Create a PAT](authentication.md#step-1-create-a-pat) in the authentication
guide. After installing the client in the next step, `cortex-training login`
uses it to set up your connection and database.

### Grant access to other roles

Run this on every account, including accounts with a capacity contract.
Creating a job requires the `USE CORTEX TRAINING` privilege on the account.
`ACCOUNTADMIN` has it. To let another role create jobs, an `ACCOUNTADMIN` runs:

```sql
GRANT USE CORTEX TRAINING ON ACCOUNT TO ROLE <role_name>;
```

If a role can already create jobs without `USE CORTEX TRAINING`, grant it the
privilege anyway, so the role keeps working when the requirement applies to
your account.

To let every role in the account create jobs, grant it to `PUBLIC` instead. Do
this only if the account does not need to limit who can start GPU training:

```sql
GRANT USE CORTEX TRAINING ON ACCOUNT TO ROLE PUBLIC;
```

Roles also need privileges on the database and schema you use, and
`CREATE EXPERIMENT` on that schema unless you pass an existing experiment. See
[Access control](../reference/rest-api.md#37-access-control) for every operation.

## Step 4: Install the Client and Example Dependencies

Requires [uv](https://docs.astral.sh/uv/) (or `pip`) and Python 3.10 or later
for the client, or 3.11 or later for the examples (tinker-cookbook needs it).

Clone the repository and install:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install the example dependency. The training examples and the inference
`generate` and `evaluate` commands need it for tokenization; `serve` and the
raw API examples do not:

```bash
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Set up your connection and database:

```bash
cortex-training login
```

It asks for the account, user, role and PAT, checks that the profile connects
to Cortex Training, and saves it as the one to use. It also checks the
profile's database and schema: if either is missing, it shows the `CREATE`
statement and runs it only if you say yes. See the
[authentication guide](authentication.md) for other ways to connect.

Verify your connection and GPU capacity:

```bash
cortex-training capacity
```

You should see your reserved GPU capacity listed. If you see zero capacity or
an error, confirm that your account has a capacity contract or, for a trial
account, that the Snowflake team has finished enablement and GPU reservation.

## Step 5: Run Your First Training Job

The conversational SFT example fine-tunes Qwen3-8B on a one-example chat dataset
("Who trained you?" → "Snowflake AI Research"). Start with a short run:

```bash
python -m recipes.sft_quickstart.conversational.train \
  max_steps=10
```

The example connects the same way the CLI does: the profile
`cortex-training login` saved, else your default Snowflake profile. Pass
`connection=<name>` to use another profile.

What happens during this run:

1. Creates a training job on the platform (allocates 4 GPUs, loads Qwen3-8B)
2. Tokenizes the chat dataset
3. Runs 10 forward-backward + optimizer steps
4. Saves a weights-only checkpoint
5. Cancels the job and releases GPUs

Watch for `train_nll` in the output — it should decrease, indicating the model
is learning.

## Step 6: Verify with Inference

The example prints a generate command, with the job and checkpoint IDs filled
in, after it saves the checkpoint. Run it to confirm the model learned the
expected answer:

```bash
python -m recipes.inference.generate \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=0 \
  prompt="Who trained you?"
```

The answer should be `Snowflake AI Research`.

## Step 7: Monitor and Manage Jobs

Useful commands while jobs are running or after they complete:

```bash
cortex-training list                  # list all your jobs
cortex-training get JOB_ID --raw      # detailed job info
cortex-training get JOB_ID            # its config, to submit again
cortex-training cancel JOB_ID         # cancel a job and release GPUs
cortex-training tui                   # terminal UI for browsing jobs and logs
```

## What's Next

You have a working setup. From here:

- **Try different training methods**: pass `lora=true` for
  [LoRA](../../recipes/sft_quickstart/README.md), `model=` for a different model from the
  [model catalog](../../model-catalog/README.md), or `lr=`,
  `train_batch_size=` and `max_seq_len=` for different hyperparameters
- **Try reinforcement learning**: run the
  [Math GRPO example](../../recipes/rl_quickstart/README.md)
- **Explore all examples**: browse the [example index](../../recipes/README.md)
- **Learn the CLI**: see the [CLI reference](../reference/cli.md) for all
  available commands
