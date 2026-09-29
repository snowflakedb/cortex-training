# Account Setup

This guide walks you from a fresh Snowflake account to a completed training job.
Follow the steps in order.

## Step 1: Create a Snowflake Account

If you already have a Snowflake account with Cortex Training enabled, skip to
[Step 3](#step-3-authenticate).

Create a free trial account at [signup.snowflake.com](https://signup.snowflake.com).
Fill in your details and verify your email. No credit card is required. The
cloud provider and region are assigned automatically.

## Step 2: Request Cortex Training Enablement

After creating your account, contact the Snowflake team to enable Cortex
Training and reserve GPU capacity. You will need to provide:

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

3. **Region** — confirm which cloud region your account is in:
   ```sql
   SELECT CURRENT_REGION();
   ```

Share these three values with the Snowflake team. They will:

- Enable the Cortex Training feature on your account
- Reserve GPU capacity for your use

You cannot proceed until this is complete. Once the team confirms enablement,
continue to the next step.

## Step 3: Authenticate

Generate a Programmatic Access Token (PAT) and configure your client connection.
Follow the [authentication guide](authentication.md) to complete this step.

Once done, verify your connection and GPU capacity:

```bash
cortex-training capacity
```

You should see your reserved GPU capacity listed. If you see zero capacity or
an error, confirm with the Snowflake team that enablement and GPU reservation
are complete.

## Step 4: Install the Client and Recipe Dependencies

Requires Python 3.10 or later and [uv](https://docs.astral.sh/uv/) (or `pip`).

Clone the repository and install:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
uv pip install -e .
```

Install the recipe dependency — this is required to run any recipe:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

## Step 5: Run Your First Training Job

The conversational SFT recipe fine-tunes Qwen3-8B on a one-example chat dataset
("Who trained you?" → "Snowflake AI Research"). Start with a short run:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/your-config.json \
  max_steps=10
```

If you used `connections.toml` with a default connection, the recipe picks it
up automatically. If you used a JSON config, pass the path as `config=`.

What happens during this run:

1. Creates a training job on the platform (allocates 4 GPUs, loads Qwen3-8B)
2. Tokenizes the chat dataset
3. Runs 10 forward-backward + optimizer steps
4. Saves a weights-only checkpoint
5. Cancels the job and releases GPUs

Watch for `train_nll` in the output — it should decrease, indicating the model
is learning.

## Step 6: Verify with Inference

The recipe prints a generate command after saving the checkpoint. Run it to
confirm the model learned the expected answer:

```bash
python -m recipes.inference.generate \
  config=/path/to/your-config.json \
  job_config=recipes/inference/configs/qwen3_8b_full.json \
  source_job_id=<training_job_id> \
  checkpoint_id=<checkpoint_id> \
  temperature=0 \
  prompt="Who trained you?"
```

The answer should be `Snowflake AI Research`.

## Step 7: Monitor and Manage Jobs

Useful commands while jobs are running or after they complete:

```bash
cortex-training list                  # list all your jobs
cortex-training get <job_id>          # detailed job info
cortex-training cancel <job_id>       # cancel a job and release GPUs
cortex-training tui                   # terminal UI for browsing jobs and logs
```

## What's Next

You have a working setup. From here:

- **Try different training methods**: adjust the job config JSON to use
  [LoRA](../../recipes/sft/conversational/README.md)
  (`job_config=configs/qwen3_8b_lora.json`), a different model, or different
  hyperparameters
- **Try reinforcement learning**: run the
  [Math GRPO recipe](../../recipes/rl/math_grpo/README.md)
- **Explore all recipes**: browse the [recipe catalog](../../recipes/README.md)
- **Learn the CLI**: see the [CLI reference](../reference/cli.md) for all
  available commands
