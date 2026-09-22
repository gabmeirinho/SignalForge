# Local Langfuse

This is Langfuse's Docker Compose deployment, based on upstream commit
`5b59af472bbd5f8368d821b14f7a9e2a8701958c`. The web and MinIO ports
are bound to `127.0.0.1`, and the services use a separate Compose project.

The ignored `infra/langfuse/.env` holds the generated passwords, initial user,
and API keys. It must be present when starting this stack.

```bash
docker compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env up -d
```

Open **http://localhost:3000** and log in as `local@signalforge.test`.
Read the local password with:

```bash
awk -F= '$1 == "LANGFUSE_INIT_USER_PASSWORD" { print $2 }' infra/langfuse/.env
```

The `SignalForge Evaluation` project and its API key pair are initialized on
startup. The root `.env` sends traces from a locally run `make api` process to
`http://127.0.0.1:3000/api/public/otel/v1/traces`.

Check the local stack with:

```bash
docker compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env ps
```

To stop it while keeping its data:

```bash
docker compose -p signalforge-langfuse \
  -f infra/langfuse/compose.yaml \
  --env-file infra/langfuse/.env down
```

The separate SignalForge Docker Compose API cannot use `127.0.0.1:3000` as an
OTLP endpoint from inside its container. Use the local `make api` process with
this stack, or configure a container-reachable Langfuse endpoint.
