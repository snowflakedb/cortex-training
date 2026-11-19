-- Store full prompts and responses, and accept client spans (TruLens) on the gateway.
-- FROM SPECIFICATION replaces the whole specification: save the current one first with
--   SELECT SYSTEM$GET_AI_GATEWAY_SPECIFICATION('SNOWFLAKE');   -- or DESCRIBE AI GATEWAY
-- and restore it after the run to turn payload capture off.
ALTER AI GATEWAY SNOWFLAKE FROM SPECIFICATION $$
schema_version: 1
models:
  - name: '*'
logging:
  enabled: true
  enable_client_telemetry: true
  capture_payload:
    request_response: true
$$;
