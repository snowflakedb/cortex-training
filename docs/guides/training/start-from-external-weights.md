# Start a Job from External Weights

Use an external-weight import (also called a BYO checkpoint) when you have a
complete Hugging Face model directory that was produced outside Cortex Training
and want to use its weights for a new training or sampling job.

This initializes model weights; it does not restore an optimizer, scheduler, or
training step. A training job starts with a fresh optimizer. External-weight
import must be enabled for your Snowflake account.

## Prepare the model directory

Save full model weights in safetensors format. For example:

```python
model.save_pretrained("/absolute/path/to/model", safe_serialization=True)
tokenizer.save_pretrained("/absolute/path/to/model")
```

The files must be flat at the directory root and include:

- `config.json`.
- Either one `model.safetensors`, or safetensors shards plus
  `model.safetensors.index.json`.
- The matching tokenizer and any processor configuration files the model uses.

The declared `model_name` must identify a supported base model whose architecture
matches the uploaded configuration and tensors.

LoRA adapters cannot be imported directly. Merge an adapter into its base model
first, then save and upload the resulting full model:

```python
merged = peft_model.merge_and_unload()
merged.save_pretrained("/absolute/path/to/model", safe_serialization=True)
tokenizer.save_pretrained("/absolute/path/to/model")
```

## Upload the weights to a stage

The source must be a readable, S3-backed Snowflake stage. For a named internal
stage, use Snowflake server-side encryption so Cortex Training can read it:

```sql
CREATE STAGE MY_DB.MY_SCHEMA.MODEL_IMPORT
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

Upload every file without compression. `PUT` compresses files by default, which
would turn safetensors into unsupported `.gz` objects.

```sql
PUT file:///absolute/path/to/model/*
  @MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7
  AUTO_COMPRESS = FALSE
  OVERWRITE = TRUE;

LIST @MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7;
```

`PUT` uploads only to internal stages. For an external S3 stage, upload the
uncompressed files with your object-storage tooling or copy them from an
internal stage while preserving the same flat layout.

The role that submits the job must be able to read the stage. Keep the objects
unchanged while the job is initializing.

## Create a training job

Set `source_checkpoint_info.external_stage_path` on the sub-job. Use a fully
qualified stage path including its leading `@`:

```json
{
  "sub_job_configs": [
    {
      "job_type": "training",
      "model_name": "Qwen/Qwen3-0.6B",
      "source_checkpoint_info": {
        "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
      },
      "training_config": {
        "optimizer": {"name": "AdamW", "lr": 0.00001},
        "max_seq_len": 128,
        "train_batch_size": 1,
        "n_gpus": 1
      }
    }
  ]
}
```

The repository contains this request as
[`examples/api/external-weights-training.json`](../../../examples/api/external-weights-training.json).
After replacing the stage path and adapting the supported model configuration,
submit it like any other job:

```bash
cortex-training submit examples/api/external-weights-training.json --wait
```

The Python SDK uses the same request field:

```python
from cortex_training import SubJobConfig

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-0.6B",
    optimizer={"name": "AdamW", "lr": 1e-5},
    max_seq_len=128,
    train_batch_size=1,
    n_gpus=1,
    source_checkpoint_info={
        "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
    },
)
job_id = client.create_job(sub_jobs=[training])
client.wait_for_job(job_id)
```

To initialize a sampling job instead, pass the same
`source_checkpoint_info` dictionary to `SubJobConfig.sampling_job(...)`.

## Constraints

- `external_stage_path` is mutually exclusive with `checkpoint_id` and
  `source_job_id`. Do not send `stage_info` or storage credentials; the service
  resolves the stage under the submitting role.
- Only S3-backed stages and safetensors weights are supported. Azure and GCS
  stages are not supported for external-weight imports.
- Pickled weight formats such as `.bin`, `.pt`, `.pth`, and `.ckpt`, executable
  Python files, and Hugging Face dynamic-code configuration are rejected.
- A LoRA adapter is not a complete model. Merge it into the base weights before
  uploading.
- External weights contain no optimizer state. To resume optimizer and training
  state, use a resumable checkpoint previously saved by Cortex Training.
