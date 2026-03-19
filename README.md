# Browser Agents (ChatGPT + Gemini)

This project provides a Python CLI agent that:
- Opens a visible browser via Playwright (Google Chrome stable by default)
- Navigates to AI chat sites (ChatGPT and Gemini)
- Submits a user prompt
- Extracts the assistant answer
- Appends results to JSONL files in `data/`
- Saves debug artifacts on failure (`debug/`)

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
uv pip install -e .
```

3. Install Playwright browser binaries:

```bash
uv run playwright install chromium
```

## Usage

Run ChatGPT:

```bash
uv run python -m agent.run --prompt "what are the best luxury bags i can buy for my wife?"
```

Run Gemini:

```bash
uv run python -m agent.run_gemini --prompt "what are the best luxury bags i can buy for my wife?"
```

Optional flags:
- `--output`: output JSONL path (ChatGPT default: `data/chatgpt_answers.jsonl`, Gemini default: `data/gemini_answers.jsonl`)
- `--samples`: number of repetitions for the same prompt (default: `1`)
- `--jitter-min` / `--jitter-max`: random delay range between samples in seconds (default: `2.0` / `6.0`)
- `--cooldown-every`: apply a longer cooldown every N samples (default: `10`, `0` disables)
- `--cooldown-seconds`: cooldown duration in seconds (default: `20.0`)
- `--ready-timeout`: seconds to wait for prompt input (default: `180`)
- `--response-timeout`: seconds to wait for response (default: `240`)
- `--stall-timeout`: seconds before failing stalled "in progress" generation (default: `45`)
- `--browser-channel`: `chrome` (default), `chromium`, or `auto`
- `--session-mode`: `ephemeral` (default, incognito-like) or `persistent`
- `--session-dir`: persistent profile directory used only when `--session-mode persistent` (default: `.state/chrome-profile`)
- `--debug-dir`: debug artifact directory on failures (default: `debug/`)
- `--connect-cdp-url`: attach to an existing Chrome started with remote debugging (example: `http://127.0.0.1:9222`)

## Output format

Each run appends one JSON object line:

Success:

```json
{"timestamp_utc":"...","prompt":"...","status":"success","answer":"..."}
```

Failure:

```json
{"timestamp_utc":"...","prompt":"...","status":"error","error":"...","debug_artifacts":{"screenshot":"...","html":"...","meta":"..."}}
```

When sampling (`--samples > 1`), each line also includes:
- `run_id`
- `sample_index`
- `sample_total`
- `provider`
- `sampling`

## Notes

- If a captcha/anti-bot screen appears, solve it manually in the opened browser window. The agent keeps polling and continues when the prompt input appears.
- UI selectors on `chatgpt.com` can change over time. If extraction fails, update selectors in `agent/run.py`.
- If a cookies banner appears, the agent attempts to click `Accept all` automatically.
- The default runtime uses a fresh ephemeral context (incognito-like) each run.
- Use `--session-mode persistent` only if you explicitly want profile reuse.
- If ChatGPT stays in an "in progress" state without meaningful text, the agent fails fast with `stalled_generation` and saves debug artifacts.
- Sampling is sequential in one browser session, with per-sample jitter and periodic cooldown.
- Each sample starts from a fresh conversation ("New chat") to reduce context contamination.

## Sampling examples

ChatGPT, 5 samples:

```bash
uv run python -m agent.run \
  --prompt "what are the best luxury bags i can buy for my wife?" \
  --samples 5
```

Gemini, 8 samples with custom pacing:

```bash
uv run python -m agent.run_gemini \
  --prompt "what are the best luxury bags i can buy for my wife?" \
  --samples 8 \
  --jitter-min 1.5 \
  --jitter-max 4.0 \
  --cooldown-every 10 \
  --cooldown-seconds 25
```

## Attach to your existing Chrome process (advanced)

If you want the agent to control your already-running Chrome instance and create a fresh incognito-like context, Chrome must be started with a DevTools port:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

Then run:

```bash
uv run python -m agent.run \
  --prompt "what are the best luxury bags i can buy for my wife?" \
  --connect-cdp-url http://127.0.0.1:9222
```

In CDP mode, the agent opens a dedicated new tab/context and will not close your main Chrome process.

You can use the same flag with Gemini:

```bash
uv run python -m agent.run_gemini \
  --prompt "what are the best luxury bags i can buy for my wife?" \
  --connect-cdp-url http://127.0.0.1:9222
```
