from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast
from uuid import uuid4

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Locator, Page, Playwright

GEMINI_URL = "https://gemini.google.com/"
DEFAULT_OUTPUT_PATH = Path("data/gemini_answers.jsonl")
DEFAULT_DEBUG_DIR = Path("debug")
DEFAULT_SESSION_DIR = Path(".state/chrome-profile")
DEFAULT_BROWSER_CHANNEL = "chrome"
DEFAULT_SESSION_MODE = "ephemeral"
DEFAULT_CONNECT_CDP_URL = ""
DEFAULT_READY_TIMEOUT_SECONDS = 180
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 240
DEFAULT_STALL_TIMEOUT_SECONDS = 45
DEFAULT_PRE_SUBMIT_DELAY_MS = 1200
DEFAULT_SAMPLES = 1
DEFAULT_JITTER_MIN_SECONDS = 2.0
DEFAULT_JITTER_MAX_SECONDS = 6.0
DEFAULT_COOLDOWN_EVERY = 10
DEFAULT_COOLDOWN_SECONDS = 20.0
POLL_INTERVAL_MS = 1000
STABLE_TEXT_SECONDS = 3
SUBMIT_CONFIRM_TIMEOUT_SECONDS = 8

PROMPT_SELECTORS = [
    'textarea[aria-label*="Enter a prompt"]',
    'textarea[placeholder*="Enter a prompt"]',
    'div[contenteditable="true"][aria-label*="Enter a prompt"]',
    'div[contenteditable="true"][role="textbox"]',
    "textarea",
]

ASSISTANT_SELECTORS = [
    "model-response",
    "message-content",
    '[data-test-id*="response"]',
    'div[class*="response"]',
]

USER_SELECTORS = [
    "user-query",
    '[data-test-id*="user"]',
    'div[class*="query"]',
]

SUBMIT_BUTTON_SELECTORS = [
    "button[aria-label*='Send message']",
    "button[aria-label*='Submit']",
    "button[aria-label*='Send']",
    "button:has-text('Send message')",
    "button:has-text('Send')",
]

COOKIE_ACCEPT_SELECTORS = [
    "button:has-text('Accept all')",
    "button:has-text('Accept All')",
    "button:has-text('Tout accepter')",
]

ANTI_BOT_SELECTORS = [
    "text=Verify you are human",
    "text=Checking your browser",
    "text=Press & Hold",
    "iframe[title*='challenge']",
    "text=Cloudflare",
]

ERROR_SELECTORS = [
    "text=Something went wrong",
    "text=An error occurred",
    "text=Network error",
    "text=Try again",
    "text=Unable to complete",
    "text=temporarily unavailable",
]

STOP_GENERATING_SELECTORS = [
    "button:has-text('Stop response')",
    "button:has-text('Stop generating')",
    "button[aria-label*='Stop']",
    "button[aria-label*='Stop response']",
]

NEW_CHAT_SELECTORS = [
    "button:has-text('New chat')",
    "a:has-text('New chat')",
    "[aria-label*='New chat']",
    "button:has-text('Start a new chat')",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open gemini.google.com in a browser, submit a prompt, "
            "extract the answer, and append it to JSONL output."
        )
    )
    parser.add_argument("--prompt", required=True, help="Prompt text to send to Gemini")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"JSONL output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        help="Seconds to wait for prompt input to become available",
    )
    parser.add_argument(
        "--response-timeout",
        type=int,
        default=DEFAULT_RESPONSE_TIMEOUT_SECONDS,
        help="Seconds to wait for response generation",
    )
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=DEFAULT_STALL_TIMEOUT_SECONDS,
        help="Seconds to wait before failing a stalled in-progress generation",
    )
    parser.add_argument(
        "--browser-channel",
        choices=["chrome", "chromium", "auto"],
        default=DEFAULT_BROWSER_CHANNEL,
        help="Browser channel: chrome (stable), chromium, or auto fallback",
    )
    parser.add_argument(
        "--session-dir",
        default=str(DEFAULT_SESSION_DIR),
        help=f"Persistent profile directory when --session-mode=persistent (default: {DEFAULT_SESSION_DIR})",
    )
    parser.add_argument(
        "--session-mode",
        choices=["ephemeral", "persistent"],
        default=DEFAULT_SESSION_MODE,
        help="Session mode: ephemeral (incognito-like) or persistent profile",
    )
    parser.add_argument(
        "--debug-dir",
        default=str(DEFAULT_DEBUG_DIR),
        help=f"Directory for debug artifacts on failures (default: {DEFAULT_DEBUG_DIR})",
    )
    parser.add_argument(
        "--connect-cdp-url",
        default=DEFAULT_CONNECT_CDP_URL,
        help=(
            "Optional Chrome DevTools URL to attach to an existing Chrome "
            "(example: http://127.0.0.1:9222). When set, launches a fresh "
            "incognito-like context in that browser."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of times to sample the same prompt (default: 1)",
    )
    parser.add_argument(
        "--jitter-min",
        type=float,
        default=DEFAULT_JITTER_MIN_SECONDS,
        help="Minimum random delay between samples in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--jitter-max",
        type=float,
        default=DEFAULT_JITTER_MAX_SECONDS,
        help="Maximum random delay between samples in seconds (default: 6.0)",
    )
    parser.add_argument(
        "--cooldown-every",
        type=int,
        default=DEFAULT_COOLDOWN_EVERY,
        help="Apply long cooldown every N samples (default: 10, 0 disables)",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Long cooldown duration in seconds (default: 20.0)",
    )
    return parser.parse_args()


def append_jsonl(output_path: Path, record: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_sampling_args(
    samples: int,
    jitter_min: float,
    jitter_max: float,
    cooldown_every: int,
    cooldown_seconds: float,
) -> None:
    if samples < 1:
        raise ValueError("--samples must be >= 1")
    if jitter_min < 0 or jitter_max < 0:
        raise ValueError("--jitter-min and --jitter-max must be >= 0")
    if jitter_max < jitter_min:
        raise ValueError("--jitter-max must be >= --jitter-min")
    if cooldown_every < 0:
        raise ValueError("--cooldown-every must be >= 0")
    if cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be >= 0")


def build_sample_record(
    prompt: str,
    provider: str,
    run_id: str,
    sample_index: int,
    sample_total: int,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "provider": provider,
        "run_id": run_id,
        "sample_index": sample_index,
        "sample_total": sample_total,
        "sampling": sample_total > 1,
    }


def save_debug_artifacts(page: Optional["Page"], debug_dir: Path, error_message: str) -> dict:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = debug_dir / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict = {}
    current_url = ""

    if page is not None:
        try:
            current_url = page.url
        except Exception as exc:
            artifacts["url_error"] = str(exc)

        try:
            screenshot_path = run_dir / "final.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            artifacts["screenshot"] = str(screenshot_path)
        except Exception as exc:
            artifacts["screenshot_error"] = str(exc)

        try:
            html_path = run_dir / "page.html"
            html_path.write_text(page.content(), encoding="utf-8")
            artifacts["html"] = str(html_path)
        except Exception as exc:
            artifacts["html_error"] = str(exc)

    meta_path = run_dir / "meta.json"
    meta_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": error_message,
        "url": current_url,
    }

    try:
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["meta"] = str(meta_path)
    except Exception as exc:
        artifacts["meta_error"] = str(exc)

    return artifacts


def find_first_visible_locator(page: "Page", selectors: list[str]) -> Optional["Locator"]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    for selector in selectors:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=1200)
            return locator
        except PlaywrightTimeoutError:
            continue
    return None


def is_anti_bot_interstitial(page: Page) -> bool:
    for selector in ANTI_BOT_SELECTORS:
        try:
            if page.locator(selector).first.is_visible(timeout=100):
                return True
        except Exception:
            continue
    return False


def handle_cookie_banner(page: "Page") -> bool:
    for selector in COOKIE_ACCEPT_SELECTORS:
        button = page.locator(selector).first
        try:
            if not button.is_visible(timeout=120):
                continue
            button.click(timeout=1500)
            page.wait_for_timeout(300)
            return True
        except Exception:
            continue
    return False


def reset_conversation(page: "Page", ready_timeout: int) -> None:
    if not click_first_visible(page, NEW_CHAT_SELECTORS):
        page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60_000)
    handle_cookie_banner(page)
    wait_for_prompt_input(page, timeout_seconds=ready_timeout)


def wait_for_prompt_input(page: "Page", timeout_seconds: int) -> "Locator":
    deadline = time.monotonic() + timeout_seconds
    informed_manual_step = False

    while time.monotonic() < deadline:
        handle_cookie_banner(page)
        prompt_input = find_first_visible_locator(page, PROMPT_SELECTORS)
        if prompt_input is not None:
            return prompt_input

        if is_anti_bot_interstitial(page):
            if not informed_manual_step:
                print(
                    "Anti-bot or verification screen detected. "
                    "Solve it manually in the browser window; the agent will resume automatically.",
                    file=sys.stderr,
                )
                informed_manual_step = True

        page.wait_for_timeout(POLL_INTERVAL_MS)

    raise TimeoutError(
        "Timed out waiting for Gemini prompt input. "
        "If a verification or sign-in screen is shown, complete it and retry."
    )


def normalize_compact_text(value: str) -> str:
    return " ".join(value.split())


def get_prompt_input_text(prompt_input: "Locator") -> str:
    try:
        tag_name = prompt_input.evaluate("el => el.tagName.toLowerCase()")
        if tag_name == "textarea":
            return cast(str, prompt_input.input_value())
        return cast(str, prompt_input.evaluate("el => (el.innerText || el.textContent || '')"))
    except Exception:
        return ""


def set_contenteditable_text(prompt_input: "Locator", prompt_text: str) -> None:
    prompt_input.evaluate(
        """
        (el, text) => {
          el.focus();
          if ('value' in el) {
            el.value = text;
          } else {
            el.textContent = text;
          }
          el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'insertText', data: text }));
          el.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true, inputType: 'insertText', data: text }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        """,
        prompt_text,
    )


def submit_prompt(prompt_input: "Locator", prompt_text: str) -> None:
    tag_name = prompt_input.evaluate("el => el.tagName.toLowerCase()")
    prompt_input.click()

    if tag_name == "textarea":
        prompt_input.fill(prompt_text)
        if not normalize_compact_text(get_prompt_input_text(prompt_input)):
            prompt_input.type(prompt_text)
        prompt_input.press("Enter")
        return

    try:
        prompt_input.fill(prompt_text)
    except Exception:
        pass

    observed_text = normalize_compact_text(get_prompt_input_text(prompt_input))
    expected_text = normalize_compact_text(prompt_text)
    if observed_text != expected_text:
        try:
            prompt_input.click()
            try:
                prompt_input.press("Meta+A")
            except Exception:
                prompt_input.press("Control+A")
            prompt_input.press("Backspace")
            prompt_input.type(prompt_text)
        except Exception:
            set_contenteditable_text(prompt_input, prompt_text)

    observed_text = normalize_compact_text(get_prompt_input_text(prompt_input))
    if observed_text != expected_text:
        set_contenteditable_text(prompt_input, prompt_text)

    prompt_input.press("Enter")


def count_user_messages(page: "Page") -> int:
    for selector in USER_SELECTORS:
        try:
            count = page.locator(selector).count()
        except Exception:
            continue
        if count > 0:
            return count
    return 0


def click_first_visible(page: "Page", selectors: list[str]) -> bool:
    for selector in selectors:
        button = page.locator(selector).first
        try:
            button.wait_for(state="visible", timeout=1000)
            button.click()
            return True
        except Exception:
            continue
    return False


def ui_has_error_banner(page: "Page") -> Optional[str]:
    for selector in ERROR_SELECTORS:
        try:
            if page.locator(selector).first.is_visible(timeout=100):
                return selector.replace("text=", "")
        except Exception:
            continue
    return None


def has_assistant_turn(page: "Page") -> bool:
    for selector in ASSISTANT_SELECTORS:
        try:
            if page.locator(selector).count() > 0:
                return True
        except Exception:
            continue
    return False


def is_meaningful_assistant_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def wait_for_submit_confirmation(page: "Page", baseline_user_count: int) -> bool:
    deadline = time.monotonic() + SUBMIT_CONFIRM_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if count_user_messages(page) > baseline_user_count:
            return True
        if is_generating(page):
            return True
        page.wait_for_timeout(300)
    return False


def submit_prompt_with_fallback(page: "Page", prompt_input: "Locator", prompt_text: str) -> None:
    baseline_user_count = count_user_messages(page)

    submit_prompt(prompt_input, prompt_text)
    if wait_for_submit_confirmation(page, baseline_user_count):
        return

    prompt_input.click()
    prompt_input.press("Meta+Enter")
    if wait_for_submit_confirmation(page, baseline_user_count):
        return

    if click_first_visible(page, SUBMIT_BUTTON_SELECTORS):
        if wait_for_submit_confirmation(page, baseline_user_count):
            return

    raise RuntimeError(
        "Prompt submission could not be confirmed. "
        "Input was filled, but no send action was detected."
    )


def extract_latest_assistant_text(page: "Page") -> Optional[str]:
    for selector in ASSISTANT_SELECTORS:
        locator = page.locator(selector)
        count = locator.count()
        if count < 1:
            continue

        for idx in range(count - 1, -1, -1):
            candidate = locator.nth(idx)
            try:
                text = candidate.inner_text(timeout=1200).strip()
            except Exception:
                continue
            if text:
                return text

    return None


def is_generating(page: "Page") -> bool:
    for selector in STOP_GENERATING_SELECTORS:
        try:
            if page.locator(selector).first.is_visible(timeout=100):
                return True
        except Exception:
            continue
    return False


def launch_browser_context(
    playwright: "Playwright",
    browser_channel: str,
    session_dir: Path,
    session_mode: str,
    connect_cdp_url: str,
) -> tuple[Optional["Browser"], "BrowserContext", str, bool, bool, bool]:
    # Returns:
    # browser, context, selected_channel, cleanup_browser, cleanup_context, force_new_page
    if connect_cdp_url:
        try:
            browser = playwright.chromium.connect_over_cdp(connect_cdp_url)
            # Prefer the existing default context exposed by CDP.
            if browser.contexts:
                return browser, browser.contexts[0], "cdp", False, False, True
            # Fallback when no default context is visible.
            context = browser.new_context()
            return browser, context, "cdp", False, True, True
        except Exception as exc:
            raise RuntimeError(
                "Failed to connect to existing Chrome via CDP. "
                "Start Chrome with --remote-debugging-port=9222 and use "
                "--connect-cdp-url http://127.0.0.1:9222"
            ) from exc

    def launch_browser(selected_channel: str) -> tuple["Browser", str]:
        if selected_channel == "chrome":
            return playwright.chromium.launch(channel="chrome", headless=False), "chrome"
        return playwright.chromium.launch(headless=False), "chromium"

    def launch_persistent(selected_channel: str) -> tuple["BrowserContext", str]:
        launch_options = {
            "user_data_dir": str(session_dir),
            "headless": False,
        }
        if selected_channel == "chrome":
            context = playwright.chromium.launch_persistent_context(
                channel="chrome",
                **launch_options,
            )
            return context, "chrome"
        context = playwright.chromium.launch_persistent_context(**launch_options)
        return context, "chromium"

    def launch_auto() -> tuple[Optional["Browser"], "BrowserContext", str, bool, bool, bool]:
        if session_mode == "ephemeral":
            try:
                browser, selected = launch_browser("chrome")
                return browser, browser.new_context(), selected, True, True, False
            except Exception as exc:
                print(
                    f"Failed to launch Chrome stable ({exc}). Falling back to Playwright Chromium.",
                    file=sys.stderr,
                )
                browser, selected = launch_browser("chromium")
                return browser, browser.new_context(), selected, True, True, False

        try:
            context, selected = launch_persistent("chrome")
            return None, context, selected, False, True, False
        except Exception as exc:
            print(
                f"Failed to launch Chrome stable ({exc}). Falling back to Playwright Chromium.",
                file=sys.stderr,
            )
            context, selected = launch_persistent("chromium")
            return None, context, selected, False, True, False

    session_dir.mkdir(parents=True, exist_ok=True)
    if browser_channel == "auto":
        return launch_auto()

    if session_mode == "ephemeral":
        browser, selected = launch_browser(browser_channel)
        return browser, browser.new_context(), selected, True, True, False

    context, selected = launch_persistent(browser_channel)
    return None, context, selected, False, True, False


def wait_for_final_answer(
    page: "Page",
    timeout_seconds: int,
    stall_timeout_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    stall_deadline = time.monotonic() + stall_timeout_seconds
    last_text: Optional[str] = None
    stable_since: Optional[float] = None
    saw_meaningful_text = False

    while time.monotonic() < deadline:
        ui_error = ui_has_error_banner(page)
        if ui_error:
            raise RuntimeError(f"gemini_ui_error: {ui_error}")

        current_text = extract_latest_assistant_text(page)
        now = time.monotonic()

        if current_text:
            if current_text != last_text:
                last_text = current_text
                stable_since = now
            elif stable_since is None:
                stable_since = now
            if is_meaningful_assistant_text(current_text):
                saw_meaningful_text = True

        generating = is_generating(page)
        if not saw_meaningful_text and now >= stall_deadline and (generating or has_assistant_turn(page)):
            raise TimeoutError(
                "stalled_generation: Gemini remained in-progress without meaningful "
                f"assistant text for at least {stall_timeout_seconds}s."
            )

        if saw_meaningful_text and not generating and stable_since and (now - stable_since >= STABLE_TEXT_SECONDS):
            return (last_text or "").strip()

        page.wait_for_timeout(POLL_INTERVAL_MS)

    if last_text and is_meaningful_assistant_text(last_text):
        return last_text.strip()

    raise TimeoutError(
        "Timed out waiting for assistant response. "
        "No extractable assistant text was found before timeout."
    )


def run(
    prompt: str,
    output_path: Path,
    ready_timeout: int,
    response_timeout: int,
    stall_timeout: int,
    browser_channel: str,
    session_mode: str,
    session_dir: Path,
    debug_dir: Path,
    connect_cdp_url: str,
    samples: int,
    jitter_min: float,
    jitter_max: float,
    cooldown_every: int,
    cooldown_seconds: float,
) -> dict:
    validate_sampling_args(
        samples=samples,
        jitter_min=jitter_min,
        jitter_max=jitter_max,
        cooldown_every=cooldown_every,
        cooldown_seconds=cooldown_seconds,
    )

    run_id = f"gemini-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    summary = {
        "provider": "gemini",
        "run_id": run_id,
        "sample_total": samples,
        "success_count": 0,
        "error_count": 0,
    }

    browser: Optional["Browser"] = None
    context: Optional["BrowserContext"] = None
    page: Optional["Page"] = None
    cleanup_browser = False
    cleanup_context = False
    force_new_page = False

    try:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'playwright'. Install dependencies with "
                "`uv pip install -e .` and run `uv run playwright install chromium`."
            ) from exc

        with sync_playwright() as playwright:
            browser, context, selected_channel, cleanup_browser, cleanup_context, force_new_page = launch_browser_context(
                cast("Playwright", playwright),
                browser_channel=browser_channel,
                session_dir=session_dir,
                session_mode=session_mode,
                connect_cdp_url=connect_cdp_url,
            )
            page = context.new_page() if force_new_page else (context.pages[0] if context.pages else context.new_page())

            page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60_000)
            handle_cookie_banner(page)
            for sample_index in range(1, samples + 1):
                record = build_sample_record(
                    prompt=prompt,
                    provider="gemini",
                    run_id=run_id,
                    sample_index=sample_index,
                    sample_total=samples,
                )
                record["browser_channel"] = selected_channel
                record["session_mode"] = session_mode

                try:
                    reset_conversation(page, ready_timeout=ready_timeout)
                    prompt_input = wait_for_prompt_input(page, timeout_seconds=ready_timeout)
                    page.wait_for_timeout(DEFAULT_PRE_SUBMIT_DELAY_MS)
                    submit_prompt_with_fallback(page, prompt_input, prompt)
                    answer = wait_for_final_answer(
                        page,
                        timeout_seconds=response_timeout,
                        stall_timeout_seconds=stall_timeout,
                    )
                    record["status"] = "success"
                    record["answer"] = answer
                    summary["success_count"] += 1
                except Exception as sample_exc:
                    record["status"] = "error"
                    record["error"] = str(sample_exc)
                    record["debug_artifacts"] = save_debug_artifacts(
                        page=page,
                        debug_dir=debug_dir,
                        error_message=str(sample_exc),
                    )
                    summary["error_count"] += 1
                finally:
                    append_jsonl(output_path, record)

                if sample_index >= samples:
                    continue

                if cooldown_every > 0 and sample_index % cooldown_every == 0:
                    page.wait_for_timeout(int(cooldown_seconds * 1000))
                    continue

                delay_seconds = random.uniform(jitter_min, jitter_max)
                page.wait_for_timeout(int(delay_seconds * 1000))

            return summary
    except Exception as exc:
        # Fatal setup failure before or during the batch: emit one failed record per sample.
        for sample_index in range(1, samples + 1):
            record = build_sample_record(
                prompt=prompt,
                provider="gemini",
                run_id=run_id,
                sample_index=sample_index,
                sample_total=samples,
            )
            record["status"] = "error"
            record["error"] = str(exc)
            record["debug_artifacts"] = save_debug_artifacts(
                page=page,
                debug_dir=debug_dir,
                error_message=str(exc),
            )
            append_jsonl(output_path, record)
            summary["error_count"] += 1
        return summary
    finally:
        if page is not None and force_new_page:
            try:
                page.close()
            except Exception:
                pass
        if context is not None and cleanup_context:
            context.close()
        if browser is not None and cleanup_browser:
            browser.close()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    session_dir = Path(args.session_dir)
    debug_dir = Path(args.debug_dir)

    result = run(
        prompt=args.prompt,
        output_path=output_path,
        ready_timeout=args.ready_timeout,
        response_timeout=args.response_timeout,
        stall_timeout=args.stall_timeout,
        browser_channel=args.browser_channel,
        session_mode=args.session_mode,
        session_dir=session_dir,
        debug_dir=debug_dir,
        connect_cdp_url=args.connect_cdp_url,
        samples=args.samples,
        jitter_min=args.jitter_min,
        jitter_max=args.jitter_max,
        cooldown_every=args.cooldown_every,
        cooldown_seconds=args.cooldown_seconds,
    )

    success_count = int(result.get("success_count", 0))
    error_count = int(result.get("error_count", 0))
    sample_total = int(result.get("sample_total", 0))
    run_id = result.get("run_id", "unknown-run")

    if error_count == 0:
        print(
            f"Stored {success_count}/{sample_total} successful sample(s) in {output_path} "
            f"(run_id={run_id})"
        )
        return 0

    print(
        f"Sampling completed with {success_count} success and {error_count} error "
        f"record(s) in {output_path} (run_id={run_id})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
