# A2A Agent Template

A minimal template for building [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) green agents compatible with the [AgentBeats](https://agentbeats.dev) platform.

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Your agent implementation goes here
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # Agent tests
Dockerfile        # Docker configuration
pyproject.toml    # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Getting Started

1. **Create your repository** - Click "Use this template" to create your own repository from this template

2. **Implement your agent** - Add your agent logic to [`src/agent.py`](src/agent.py)

3. **Configure your agent card** - Fill in your agent's metadata (name, skills, description) in [`src/server.py`](src/server.py)

4. **Write your tests** - Add custom tests for your agent in [`tests/test_agent.py`](tests/test_agent.py)

For a concrete example of implementing a green agent using this template, see this [draft PR](https://github.com/RDI-Foundation/green-agent-template/pull/3).

## Running Locally

```bash
# Install dependencies
uv sync
uv sync --extra tau2-evaluator

# Run the server
uv run src/server.py
```

## Running with Docker

> **Note:** Docker is optional for local development. Use the local setup above for development and testing.

First, ensure [Docker](https://www.docker.com/products/docker-desktop) is installed:

```bash
# Build the image
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).

## Partial Credits Scoring

The agent evaluation system implements a partial credits mechanism for tool calling accuracy. This allows for nuanced scoring when tool calls are partially correct.

### Scoring Logic

The scoring system evaluates predicted tool calls against gold standard (expected) tool calls using the following rules:

1. **Tool Name Mismatch** - If the predicted tool name doesn't match the gold standard tool name:
   - **Result:** No credit (`[False, 0.0]`)

2. **No Arguments to Compare** - If neither the predicted nor gold standard tool call has arguments to compare:
   - **Result:** Full credit (`[True, 1.0]`)

3. **Exact Argument Match** - If all required arguments match exactly between predicted and gold standard:
   - **Result:** Full credit (`[True, 1.0]`)

4. **Partial Argument Match** - If the tool call is correct but some arguments don't match:
   - **Result:** Partial credit (`[True, 0.5]`)

### Example Scenarios

- **Perfect Tool Call:** Agent calls `search_flights` with all correct arguments → Score: `1.0`
- **Wrong Tool:** Agent calls `book_hotel` instead of `search_flights` → Score: `0.0`
- **Correct Tool, Missing/Wrong Args:** Agent calls `search_flights` but with incorrect or incomplete arguments → Score: `0.5`
- **No Arguments Needed:** Agent calls `get_status` with no arguments (and none expected) → Score: `1.0`

### Configuration

The `compare_args` parameter in a tool call definition allows specifying which arguments should be evaluated. If `compare_args` is `None`, all arguments in the predicted tool call are evaluated against all arguments in the gold standard tool call.
