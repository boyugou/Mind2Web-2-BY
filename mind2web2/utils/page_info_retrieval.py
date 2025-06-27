import asyncio
import hashlib
import re
import time
from logging import Logger
from pathlib import Path
from typing import Optional, Tuple, Union

from playwright.async_api import (
    async_playwright,
    BrowserContext,
    Page,
)
from pydantic import HttpUrl
import base64
from io import BytesIO

from mind2web2.llm_client.azure_openai_client import AsyncAzureOpenAIClient
from mind2web2.utils.logging_setup import create_logger

from PIL import Image   # pip install pillow



################################################################################
# ────────────────────────── MANUAL‑SCRAPE CONSTANTS ───────────────────────────
################################################################################

# 1) only block StoryGraph for now – you can add more domains later
BLACKLIST_PATTERNS = {
    r"^https?://(?:www\.)?app\.thestorygraph\.com/.*",
}
BLACKLIST_RE = re.compile("|".join(BLACKLIST_PATTERNS), re.I)

# 2) English message shown to downstream caller if the URL is on the list
MANUAL_TEXT = (
    "\u26A0\ufe0f This site uses advanced human‑verification. "
    "Please open it in a regular browser, complete any checks, and save the page as MHTML before continuing."
)


def is_blacklisted(url: str) -> bool:
    """Return True if *url* matches any pattern in `BLACKLIST_PATTERNS`."""
    return bool(BLACKLIST_RE.match(url))


def make_blank_png_b64() -> str:
    # Create 1×1 RGBA fully transparent pixel
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    # Convert to base64 and remove line breaks
    return base64.b64encode(buf.getvalue()).decode()


################################################################################
# ──────────────────────────────── CONSTANTS ──────────────────────────────────
################################################################################

BLANK_IMG_B64 = make_blank_png_b64()
ERROR_TEXT = "\u26A0\ufe0f This URL could not be loaded (navigation error)."


# ==== Stealth JS to evade automation detection ====
STEALTH_JS = r"""
window.navigator.chrome = {runtime: {}};
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
"""

default_extra_headers = {
    'sec-ch-ua': '"Chromium";v="113", "Google Chrome";v="113"',
    'accept-language': 'en-US,en;q=0.9',
}


def format_url(url: Union[str, Path]) -> str:
    text = str(url)
    if text.endswith("?utm_source=chatgpt.com"):
        text= text.replace("?utm_source=chatgpt.com","")
        print("REMOVED GPT SUFFIX")
    if not re.match(r'^(?:https?|ftp)://', text):
        return f'https://{text}'

    return text




# User-agent pool and default headers
default_user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
]

user_agent_strings = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0',
]

extra_headers_template = {
    'sec-ch-ua': '\"Chromium\";v=\"130\", \"Google Chrome\";v=\"130\", \"Not?A_Brand\";v=\"99\"',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'accept-Language': 'en-US,en;q=0.9'
}

import httpx
import random
from urllib.parse import urlparse

#
# async def is_pdf(url: str) -> bool:
#     if url.lower().endswith(".pdf"):
#         return True
#     url = format_url(url)
#
#     # 1) quick path-suffix heuristic
#     if urlparse(url).path.lower().endswith(".pdf"):
#         return True
#
#     try:
#         async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
#             # 2) try RANGE GET instead of HEAD (safer across CDNs)
#             extra_headers = extra_headers_template.copy()
#             extra_headers["user-agent"] = random.choice(user_agent_strings)
#             extra_headers["Range"] = "bytes=0-0"
#
#             r = await client.get(url, headers=extra_headers)
#             ctype = r.headers.get("content-type", "").split(";")[0].strip()
#             return ctype == "application/pdf"
#
#     except httpx.RequestError as e:
#         print(f"[is_pdf] network error: {type(e).__name__}: {e}")
#         print(f"[is_pdf] network error ({e}); fallback to False – URL: {url}")
#         return False

import asyncio
import httpx
import requests
import random
from urllib.parse import urlparse

def is_pdf_by_suffix(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf") or ("arxiv" in url and "pdf" in url)

def is_pdf_by_requests_head(url: str) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=8, headers={
            "User-Agent": random.choice(user_agent_strings)
        })
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        return content_type == "application/pdf"
    except Exception as e:
        print(f"[is_pdf_requests_head] error: {type(e).__name__}: {e}")
        return False  # continue fallback

async def is_pdf_by_httpx_get(url: str) -> bool:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
            r = await client.get(url, headers={
                "User-Agent": random.choice(user_agent_strings),
                "Range": "bytes=0-0"
            })
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            return ctype == "application/pdf"
    except httpx.RequestError as e:
        print(f"[is_pdf_httpx_get] network error: {type(e).__name__}: {e}")
        return False

async def is_pdf(url: str) -> bool:
    url = format_url(url)

    # 1. fast suffix check
    if is_pdf_by_suffix(url):
        print(f"{url} IS a PDF")
        return True

    # 2. requests HEAD (sync, reliable)
    if is_pdf_by_requests_head(url):
        print(f"{url} IS a PDF")
        return True

    # 3. httpx GET fallback
    if await is_pdf_by_httpx_get(url):
        print(f"{url} IS a PDF")
        return True

    # 4. Final fallback: Playwright or manual detection
    # Here, you can optionally integrate Playwright-based capture + type inspection
    print(f"{url} IS NOT a PDF")
    return False

async def _retry_async(
        func, retries: int = 3, delay: float = 1.0, backoff: float = 2.0, logger: Logger = None
):
    attempt = 1
    while True:
        try:
            return await func()
        except Exception as e:
            if attempt >= retries:
                if logger:
                    logger.error(f"Final attempt {attempt} failed: {e}")
                raise
            if logger:
                logger.warning(f"Attempt {attempt} failed: {e}, retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= backoff
            attempt += 1


class PageManager:
    """
    Manage active Page within a BrowserContext, handling new pages, closures, crashes, navigations.
    """

    def __init__(self, context: BrowserContext, logger: Logger):
        self.context = context
        self.logger = logger
        self.current: Optional[Page] = None
        self.closing = False
        self._handlers = []
        # Listen for new page events on context
        handler = lambda page: asyncio.create_task(self._on_new_page(page))
        context.on('page', handler)
        self._handlers.append((context, 'page', handler))
        for pg in context.pages:
            asyncio.create_task(self._on_new_page(pg))

    async def _on_new_page(self, page: Page):
        if self.closing:
            return
        self.logger.debug(f'New page opened: {page.url}')
        self.current = page
        self._attach_handlers(page)

    def _attach_handlers(self, page: Page):
        for event in ('close', 'crash', 'framenavigated'):
            if event == 'close':
                cb = lambda: asyncio.create_task(self._on_close(page))
            elif event == 'crash':
                cb = lambda: asyncio.create_task(self._on_crash(page))
            else:
                cb = lambda frame: asyncio.create_task(self._on_navigate(page, frame))
            page.on(event, cb)
            self._handlers.append((page, event, cb))

    async def _on_close(self, page: Page):
        if self.closing:
            return
        self.logger.warning(f'Page closed: {page.url}')
        pages = self.context.pages
        if pages:
            await self._on_new_page(pages[-1])
        else:
            try:
                new_pg = await self.context.new_page()
                await self._on_new_page(new_pg)
            except Exception as e:
                self.logger.error(f'Failed to reopen page after close: {e}')

    async def _on_crash(self, page: Page):
        if self.closing:
            return
        self.logger.error(f'Page crashed: {page.url}, refreshing...')
        try:
            await page.reload()
        except Exception as e:
            self.logger.error(f'Reload after crash failed: {e}')

    async def _on_navigate(self, page: Page, frame):
        if self.closing:
            return
        if frame == page.main_frame:
            self.logger.debug(f'Frame navigated: {page.url}')
            self.current = page

    async def get(self) -> Page:
        if self.closing:
            raise RuntimeError('Context is closing')
        if not self.current or self.current.is_closed():
            # self.logger.info('No active page, creating a new one')
            page = await self.context.new_page()
            await self._on_new_page(page)
        return self.current

    def dispose(self):
        """Stop listening and prevent new page opens."""
        self.closing = True
        for emitter, event, cb in self._handlers:
            try:
                emitter.off(event, cb)
            except Exception:
                pass
        self._handlers.clear()


class BlockingPopupError(RuntimeError):
    """Raised when a blocking popup / human-verification overlay is detected."""
    pass


async def capture_page_content_async(
        url: HttpUrl,
        logger: Logger,
        wait_until: str = "networkidle",
        headless: bool= False,  # ← Default becomes True
        user_data_dir: Union[str, Path] = None,
        grant_permissions: bool = True,
        detector_model: str ="gpt-4.1-mini"
) -> Tuple[Optional[str], Optional[str]]:
    # ----------- prepare persistent context dir -----------
    if user_data_dir is None:
        random_bytes = str(random.random()).encode("utf-8")
        hash_prefix = hashlib.sha256(random_bytes).hexdigest()[:6]
        user_data_dir = Path.cwd() / "tmp" / f"browser_context_{hash_prefix}"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    target = format_url(url)

    user_agent = random.choice(default_user_agents)
    headers = {**default_extra_headers, "user-agent": user_agent}

    screenshot_b64=make_blank_png_b64()
    page_text=ERROR_TEXT

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            channel="chrome",
            headless=headless,
            ignore_https_errors=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-site-isolation-trials",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--ignore-certificate-errors",
                "--safebrowsing-disable-auto-save",
                "--safebrowsing-disable-download-protection",
                '--password-store=basic',  # Use plain text basic storage instead of Keychain
                '--use-mock-keychain',  # Tell Chrome not to actually access Keychain
            ],
            extra_http_headers=headers,
            viewport={
                "width": random.randint(1050, 1150),
                "height": random.randint(700, 800),
            },
        )
        await context.add_init_script(STEALTH_JS)
        if grant_permissions:
            try:
                await context.grant_permissions(
                    [
                        "geolocation",
                        "notifications",
                        "camera",
                        "microphone",
                        "clipboard-read",
                        "clipboard-write",
                    ],
                    origin=target,
                )
            except Exception as e:
                logger.error(f'Failed to grant permissions: {e}')

        mgr = PageManager(context, logger)
        page = await mgr.get()
        await page.goto("https://www.google.com/", wait_until='load', timeout=15000)
        # page = await mgr.get()
        cdp = await context.new_cdp_session(page)
        await cdp.send("Page.enable")
        await cdp.send("DOM.enable")
        await cdp.send("Runtime.enable")

        start_ts = time.time()
        try:
            async def navigate():
                pg = await mgr.get()
                return await pg.goto(target, wait_until=wait_until, timeout=15000)

            try:
                await navigate()
            except Exception as e:
                logger.error(f"Navigation failed (Timeout is fine): {e}")

            # ---------- viewport screenshot ----------
            try:

                logger.info("Reset CDP to the latest page")
                page = await mgr.get()
                cdp = await context.new_cdp_session(page)
                vp_shot = await cdp.send(
                    "Page.captureScreenshot", {"format": "png"}
                )
                vp_b64 = vp_shot.get("data")
            except Exception as e:
                logger.error(f"Failed to capture screenshot: {e}")
                logger.error(f"Lets use blank information for: {target}")
                return BLANK_IMG_B64,ERROR_TEXT

            # ---------- blocking-popup detection ----------
            from pydantic import BaseModel

            class PopupEvalResult(BaseModel):
                thoughts: str
                blocking: bool
                reason: str

            llm = AsyncAzureOpenAIClient()
            prompt_headless = (
                "You are a scraping assistant. Given a screenshot of the current "
                "viewport, determine whether there is a major blocking popup overlay that would prevent accurate scraping of the "
                "main content. A minor floating element that does not block the "
                "main content is fine (i.e., blocking = False). If you find human "
                "verification, ignore and proceed, i.e. blocking=False."
            )

            prompt_headful = (
                "You are a scraping assistant. Given a screenshot of the current "
                "viewport, determine whether there is a major blocking popup overlay that would prevent accurate scraping of the "
                "main content. A minor floating element that does not block the "
                "main content is fine (i.e., blocking = False).\nA speciail case is human verification or totally get blocked by the website. If you find the current website is blocked by human detector, just proceed with blocking = False."
            )

            try:
                verification_result = await llm.response(
                    model=detector_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_headless if headless else prompt_headful},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{vp_b64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    response_format=PopupEvalResult,
                    temperature=0.0,
                )
                if verification_result.blocking and headless == True:
                    logger.warning(
                        f"Blocking detected on {target} during Headless Mode:\n {verification_result.thoughts} \n {verification_result.blocking} \n {verification_result.reason}"
                    )
                    logger.warning(
                        f"Raising a BlockingPopupError"
                    )
                    raise BlockingPopupError(verification_result.reason)
                elif verification_result.blocking and headless == False:
                    logger.warning(
                        f"Blocking detected on {target} during HeadFUL Mode:\n {verification_result.thoughts} \n {verification_result.blocking} \n {verification_result.reason}"
                    )
                    page = await mgr.get()
                    await page.pause()



            except BlockingPopupError:
                logger.error(f"Blocking detected on {target}:\n {verification_result.thoughts} \n {verification_result.blocking} \n {verification_result.reason}")
                return None,None,None
            except Exception as e:
                logger.error(f'Failed to call LM Blocker Detector, likely because the url is bad {url}: {e}')

            # ---------- scroll & full-page capture ----------
            page = await mgr.get()
            # await page.pause()
            for _ in range(3):
                await page.keyboard.press("End")
                await asyncio.sleep(random.uniform(1.0, 2.0))
            for _ in range(random.randint(5, 10)):
                await page.mouse.wheel(0, random.randint(-500, 500))
                await asyncio.sleep(random.uniform(0.5, 1.5))
            await page.keyboard.press("Home")
            await asyncio.sleep(random.uniform(1.0, 2.0))

            try:
                logger.debug(f"Start trying to collect info like Screenshots {target}")
                logger.info("Reset CDP to the latest page")
                page = await mgr.get()
                cdp = await context.new_cdp_session(page)
                metrics = await cdp.send("Page.getLayoutMetrics")
                css_vp = metrics["cssVisualViewport"]
                css_content = metrics["cssContentSize"]
                width = round(css_vp["clientWidth"])
                height = round(min(css_content["height"], 6000))
                scale = round(metrics.get("visualViewport", {}).get("scale", 1))
                logger.info(f"Set device override to {width}, {height}, {scale}")
                await cdp.send(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "mobile": False,
                        "width": width,
                        "height": height,
                        "deviceScaleFactor": scale,
                    },
                )
                await asyncio.sleep(random.uniform(2.1, 3.2))
                logger.info(f"Try to Capture Screenshot at {target}")
                shot = await cdp.send(
                    "Page.captureScreenshot",
                    {"format": "png", "captureBeyondViewport": True},
                )
                screenshot_b64 = shot.get("data")
                logger.info(f"Try to Capture TEXT at {target}")
                text_res = await cdp.send(
                    "Runtime.evaluate",
                    {
                        "expression": "document.documentElement.innerText",
                        "returnByValue": True,
                    },
                )
                page_text = text_res.get("result", {}).get("value")

                elapsed = time.time() - start_ts
                logger.debug(f"Completed capture for {target} in {elapsed:.2f}s")
                return screenshot_b64, page_text
            except Exception as e:
                logger.error(f"Error capturing content: {e}")
                return screenshot_b64, page_text
        finally:
            mgr.dispose()
            try:
                await cdp.detach()
            except Exception:
                pass
            try:
                await context.close()
            except Exception:
                pass



if __name__ == '__main__':
    logger, _ = create_logger(__name__, r"tmp")
    test_url = 'https://www.akc.org/dog-breeds/west-highland-white-terrier/'
    shot_b64, text = asyncio.run(capture_page_content_async(test_url, logger, headless=False))
    print(text)
    if shot_b64:
        Path('screenshot.png').write_bytes(base64.b64decode(shot_b64))
        logger.info('Saved screenshot.png')
    if text:
        Path('page.txt').write_text(text, encoding='utf-8')
        logger.info('Saved page.txt')
#
#
# if __name__ == "__main__":
#     asyncio.run(is_pdf("https://www.nobelprize.org/prizes/physics/2004/wilczek/biographical/"))
