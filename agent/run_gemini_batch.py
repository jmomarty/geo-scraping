from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from agent import run_gemini

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

DEFAULT_CSV_PATH = Path("data/prompts_to_execute.csv")
DEFAULT_OUTPUT_PREFIX = "gemini_answers_"
DEFAULT_MAX_RETRIES = 1
DEFAULT_BATCH_CDP_URL = "http://127.0.0.1:9222"
DEFAULT_PRE_SUBMIT_DELAY_MS = 150
REQUIRED_COLUMNS = ("persona", "category", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemini once per prompt row from a CSV file, retry failures, and append "
            "enriched records to a JSONL output."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help=f"CSV input path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSONL output path. Defaults to data/gemini_answers_YYYYMMDDTHHMMSSZ.jsonl",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Number of retries for failed prompts (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=run_gemini.DEFAULT_READY_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for prompt input to become available "
            f"(default: {run_gemini.DEFAULT_READY_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--response-timeout",
        type=int,
        default=run_gemini.DEFAULT_RESPONSE_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for response generation "
            f"(default: {run_gemini.DEFAULT_RESPONSE_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=run_gemini.DEFAULT_STALL_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait before failing a stalled in-progress generation "
            f"(default: {run_gemini.DEFAULT_STALL_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--browser-channel",
        choices=["chrome", "chromium", "auto"],
        default=run_gemini.DEFAULT_BROWSER_CHANNEL,
        help=(
            "Browser channel: chrome (stable), chromium, or auto fallback "
            f"(default: {run_gemini.DEFAULT_BROWSER_CHANNEL})"
        ),
    )
    parser.add_argument(
        "--session-dir",
        default=str(run_gemini.DEFAULT_SESSION_DIR),
        help=(
            "Persistent profile directory when --session-mode=persistent "
            f"(default: {run_gemini.DEFAULT_SESSION_DIR})"
        ),
    )
    parser.add_argument(
        "--session-mode",
        choices=["ephemeral", "persistent"],
        default=run_gemini.DEFAULT_SESSION_MODE,
        help=(
            "Session mode: ephemeral (incognito-like) or persistent profile "
            f"(default: {run_gemini.DEFAULT_SESSION_MODE})"
        ),
    )
    parser.add_argument(
        "--debug-dir",
        default=str(run_gemini.DEFAULT_DEBUG_DIR),
        help=f"Directory for debug artifacts on failures (default: {run_gemini.DEFAULT_DEBUG_DIR})",
    )
    parser.add_argument(
        "--connect-cdp-url",
        default=DEFAULT_BATCH_CDP_URL,
        help=(
            "Chrome DevTools URL for existing Chrome "
            f"(default: {DEFAULT_BATCH_CDP_URL})."
        ),
    )
    parser.add_argument(
        "--pre-submit-delay-ms",
        type=int,
        default=DEFAULT_PRE_SUBMIT_DELAY_MS,
        help=f"Delay before send after input is ready (default: {DEFAULT_PRE_SUBMIT_DELAY_MS})",
    )
    return parser.parse_args()


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / f"{DEFAULT_OUTPUT_PREFIX}{timestamp}.jsonl"


def append_jsonl(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prompt_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        headers = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing:
            raise ValueError(
                "CSV missing required columns: "
                + ", ".join(missing)
                + f". Found columns: {headers}"
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "persona": (row.get("persona") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                    "prompt": (row.get("prompt") or "").strip(),
                }
            )
    return rows


def make_batch_error_record(prompt: str, message: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "provider": "gemini",
        "status": "error",
        "error": message,
    }


def enrich_record(
    record: dict[str, Any],
    batch_id: str,
    source_csv: str,
    row_index: int,
    persona: str,
    category: str,
    attempt: int,
) -> dict[str, Any]:
    enriched = dict(record)
    enriched["batch_id"] = batch_id
    enriched["source_csv"] = source_csv
    enriched["row_index"] = row_index
    enriched["persona"] = persona
    enriched["category"] = category
    enriched["attempt"] = attempt
    return enriched


def run_single_prompt_in_page(
    page: "Page",
    prompt: str,
    ready_timeout: int,
    response_timeout: int,
    stall_timeout: int,
    selected_channel: str,
    session_mode: str,
    debug_dir: Path,
    pre_submit_delay_ms: int,
) -> dict[str, Any]:
    run_id = f"gemini-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    record = run_gemini.build_sample_record(
        prompt=prompt,
        provider="gemini",
        run_id=run_id,
        sample_index=1,
        sample_total=1,
    )
    record["browser_channel"] = selected_channel
    record["session_mode"] = session_mode

    try:
        run_gemini.reset_conversation(page, ready_timeout=ready_timeout)
        prompt_input = run_gemini.wait_for_prompt_input(page, timeout_seconds=ready_timeout)
        if pre_submit_delay_ms > 0:
            page.wait_for_timeout(pre_submit_delay_ms)
        run_gemini.submit_prompt_with_fallback(page, prompt_input, prompt)
        answer = run_gemini.wait_for_final_answer(
            page,
            timeout_seconds=response_timeout,
            stall_timeout_seconds=stall_timeout,
        )
        record["status"] = "success"
        record["answer"] = answer
        return record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        record["debug_artifacts"] = run_gemini.save_debug_artifacts(
            page=page,
            debug_dir=debug_dir,
            error_message=str(exc),
        )
        return record


def main() -> int:
    args = parse_args()

    if args.max_retries < 0:
        print("--max-retries must be >= 0", file=sys.stderr)
        return 2
    if args.pre_submit_delay_ms < 0:
        print("--pre-submit-delay-ms must be >= 0", file=sys.stderr)
        return 2

    csv_path = Path(args.csv)
    output_path = Path(args.output) if args.output else default_output_path()
    session_dir = Path(args.session_dir)
    debug_dir = Path(args.debug_dir)
    batch_id = f"gemini-batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"

    try:
        rows = load_prompt_rows(csv_path)
    except Exception as exc:
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        return 2

    total_rows = len(rows)
    success_count = 0
    error_count = 0
    attempt_records = 0

    browser: Optional["Browser"] = None
    context: Optional["BrowserContext"] = None
    page: Optional["Page"] = None
    cleanup_browser = False
    cleanup_context = False
    force_new_page = False
    selected_channel = "unknown"

    try:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'playwright'. Install dependencies with "
                "`uv pip install -e .` and run `uv run playwright install chromium`."
            ) from exc

        with sync_playwright() as playwright:
            browser, context, selected_channel, cleanup_browser, cleanup_context, force_new_page = run_gemini.launch_browser_context(
                playwright=playwright,
                browser_channel=args.browser_channel,
                session_dir=session_dir,
                session_mode=args.session_mode,
                connect_cdp_url=args.connect_cdp_url,
            )
            page = context.new_page() if force_new_page else (context.pages[0] if context.pages else context.new_page())
            page.goto(run_gemini.GEMINI_URL, wait_until="domcontentloaded", timeout=60_000)
            run_gemini.handle_cookie_banner(page)

            for row_index, row in enumerate(rows, start=1):
                persona = row["persona"]
                category = row["category"]
                prompt = row["prompt"]

                if not prompt:
                    record = make_batch_error_record(prompt, "empty_prompt: CSV prompt value is empty")
                    enriched = enrich_record(
                        record=record,
                        batch_id=batch_id,
                        source_csv=str(csv_path),
                        row_index=row_index,
                        persona=persona,
                        category=category,
                        attempt=1,
                    )
                    append_jsonl(output_path, enriched)
                    attempt_records += 1
                    error_count += 1
                    continue

                row_succeeded = False

                for attempt in range(1, args.max_retries + 2):
                    raw_record = run_single_prompt_in_page(
                        page=page,
                        prompt=prompt,
                        ready_timeout=args.ready_timeout,
                        response_timeout=args.response_timeout,
                        stall_timeout=args.stall_timeout,
                        selected_channel=selected_channel,
                        session_mode=args.session_mode,
                        debug_dir=debug_dir,
                        pre_submit_delay_ms=args.pre_submit_delay_ms,
                    )

                    enriched = enrich_record(
                        record=raw_record,
                        batch_id=batch_id,
                        source_csv=str(csv_path),
                        row_index=row_index,
                        persona=persona,
                        category=category,
                        attempt=attempt,
                    )
                    append_jsonl(output_path, enriched)
                    attempt_records += 1

                    if raw_record.get("status") == "success":
                        row_succeeded = True
                        break

                if row_succeeded:
                    success_count += 1
                else:
                    error_count += 1

        print(
            f"Batch completed: total_rows={total_rows}, success_count={success_count}, "
            f"error_count={error_count}, attempt_records={attempt_records}, "
            f"output={output_path}, batch_id={batch_id}"
        )
        return 0 if error_count == 0 else 1

    except Exception as exc:
        print(f"Batch failed before completion: {exc}", file=sys.stderr)
        return 1
    finally:
        if page is not None and force_new_page:
            try:
                page.close()
            except Exception:
                pass
        if context is not None and cleanup_context:
            try:
                context.close()
            except Exception:
                pass
        if browser is not None and cleanup_browser:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
