"""Open the deployed demo in a real browser so the container stays awake.

Run by .github/workflows/keep-warm.yml.

Two things make this trickier than a curl:

1. Fetching the URL returns the hosting shell from share.streamlit.io. The
   app container only starts once a client opens the websocket, which needs
   a browser.
2. That shell renders the app inside an iframe (`/~/+/`), so the app's own
   markup never appears in the top-level document. The readiness check has
   to look inside every frame, not just the page.
"""

import os
import sys
import time

from playwright.sync_api import Page, sync_playwright

URL = os.environ.get("APP_URL", "").strip()
# The app's own heading. Finding it proves the websocket connected and the
# Streamlit script ran, rather than just that some HTML came back.
READY = "Research Agent"
BOOT_BUDGET = 180.0
WARM_SECONDS = 20.0
POLL = 2.0


def rendered(page: Page) -> bool:
    """True once any frame shows the app's heading."""
    for frame in page.frames:
        try:
            if frame.get_by_text(READY, exact=False).count() > 0:
                return True
        except Exception:
            # A frame can be detached mid-check while the shell settles.
            continue
    return False


def main() -> int:
    if not URL:
        print("::error::APP_URL is empty")
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        start = time.monotonic()

        try:
            page.goto(URL, timeout=int(BOOT_BUDGET * 1000), wait_until="domcontentloaded")
        except Exception as exc:
            print(f"::error::could not open {URL}: {exc}")
            browser.close()
            return 1

        while time.monotonic() - start < BOOT_BUDGET:
            if rendered(page):
                break
            page.wait_for_timeout(int(POLL * 1000))
        else:
            elapsed = time.monotonic() - start
            print(f"::error::app did not render within {elapsed:.0f}s")
            page.screenshot(path="failed.png", full_page=True)
            browser.close()
            return 1

        elapsed = time.monotonic() - start
        # Hold the session open briefly. Closing the instant the first text
        # paints can land while the app is still finishing its first run.
        page.wait_for_timeout(3_000)
        browser.close()

    print(f"rendered in {elapsed:.1f}s")
    if elapsed > WARM_SECONDS:
        print(f"::warning::cold start ({elapsed:.1f}s) - it had gone to sleep, "
              "so the schedule is not keeping up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
