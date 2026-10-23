"""Open the deployed demo in a real browser so the container stays awake.

Run by .github/workflows/keep-warm.yml. Fetching the URL with curl is not
enough - that returns the hosting shell, not the app. Streamlit only starts
the app when a client opens the websocket, which needs a browser.
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "").strip()
# The app's own heading. Waiting for it proves the websocket connected and
# the script ran, rather than just that some HTML came back.
READY = "Research Agent"
BOOT_BUDGET_MS = 180_000
WARM_SECONDS = 20


def main() -> int:
    if not URL:
        print("::error::APP_URL is empty")
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        start = time.monotonic()
        try:
            page.goto(URL, timeout=BOOT_BUDGET_MS, wait_until="domcontentloaded")
            page.wait_for_selector(f"text={READY}", timeout=BOOT_BUDGET_MS)
        except Exception as exc:
            print(f"::error::app did not render within {BOOT_BUDGET_MS // 1000}s: {exc}")
            page.screenshot(path="failed.png", full_page=True)
            browser.close()
            return 1

        elapsed = time.monotonic() - start
        # Hold the session open briefly. Closing the instant the first text
        # paints can land while the app is still finishing its run.
        page.wait_for_timeout(3_000)
        browser.close()

    print(f"rendered in {elapsed:.1f}s")
    if elapsed > WARM_SECONDS:
        print(f"::warning::cold start ({elapsed:.1f}s) - it had gone to sleep, "
              "so the schedule is not keeping up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
