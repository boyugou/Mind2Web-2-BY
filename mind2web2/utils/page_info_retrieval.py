# Standard library imports
import asyncio
import base64
import hashlib
import random
import re
import ssl
import time
from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse, unquote

# Third-party imports
import certifi
import httpx
import requests
from PIL import Image
from playwright.async_api import (
    BrowserContext,
    Page,
    async_playwright,
)
from pydantic import HttpUrl
from mind2web2.utils.logging_setup import create_logger


# ================================ Constants ================================

def make_blank_png_b64() -> str:
    # Create 1×1 RGBA fully transparent pixel
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    # Convert to base64 and remove line breaks
    return base64.b64encode(buf.getvalue()).decode()


# Error handling constants
BLANK_IMG_B64 = make_blank_png_b64()
ERROR_TEXT = "\u26A0\ufe0f This URL could not be loaded (navigation error)."


def format_url(url: Union[str, Path]) -> str:
    text = str(url)
    if text.endswith("?utm_source=chatgpt.com"):
        text = text.replace("?utm_source=chatgpt.com", "")
        print("REMOVED GPT SUFFIX")
    if not re.match(r'^(?:https?|ftp)://', text):
        return f'https://{text}'
    return text



# User-agent pools
DEFAULT_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
]

USER_AGENT_STRINGS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0',
]

# ================================ PDF Detection Functions ================================


#TODO: There still can be fake PDFs. For those cases, we should just make a blank PDF for them.
def is_pdf_by_suffix(url: str) -> bool:
    """Check if URL likely points to PDF based on path/query patterns."""
    parsed = urlparse(url.lower())
    path = unquote(parsed.path)

    # Direct .pdf extension
    if path.endswith('.pdf'):
        return True
    
    # Common PDF URL patterns
    pdf_patterns = [
        'arxiv.org/pdf/',
        '/download/pdf',
        '/fulltext.pdf',
        '/article/pdf',
        '/content/pdf',
        'type=pdf',
        'format=pdf',
        'download=pdf',
        '.pdf?',
        '/pdf/',
        'pdfviewer',
    ]
    
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in pdf_patterns)

def is_pdf_by_requests_head(url: str, timeout: int = 10) -> bool:
    """Check PDF via HEAD request with proper error handling."""
    try:
        headers = {
            "User-Agent": random.choice(USER_AGENT_STRINGS),
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        # Create SSL context that's more permissive
        session = requests.Session()
        session.verify = certifi.where()
        
        resp = session.head(
            url, 
            allow_redirects=True, 
            timeout=timeout, 
            headers=headers,
            verify=False  # Less strict SSL verification
        )
        
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        
        # Check various PDF content types
        pdf_types = [
            "application/pdf",
            "application/x-pdf",
            "application/acrobat",
            "applications/vnd.pdf",
            "text/pdf",
            "text/x-pdf"
        ]
        
        return any(pdf_type in content_type for pdf_type in pdf_types)
        
    except requests.exceptions.SSLError as e:
        print(f"[is_pdf_requests_head] SSL error for {url}: {type(e).__name__}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"[is_pdf_requests_head] Connection error for {url}: {type(e).__name__}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"[is_pdf_requests_head] Timeout error for {url}: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"[is_pdf_requests_head] Unexpected error for {url}: {type(e).__name__}: {e}")
        return False

async def is_pdf_by_httpx_get_range(url: str, timeout: int = 10) -> bool:
    """Check PDF via partial GET request to read file header."""
    try:
        # Configure httpx with custom SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            verify=False,
            limits=httpx.Limits(max_redirects=10)
        ) as client:
            
            headers = {
                "User-Agent": random.choice(USER_AGENT_STRINGS),
                "Range": "bytes=0-1023",  # Get first 1KB to check magic number
                "Accept": "*/*",
            }
            
            r = await client.get(url, headers=headers)
            
            # First check Content-Type
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if "pdf" in ctype:
                return True
            
            # If we got content, check PDF magic number
            if r.content:
                # PDF files start with %PDF-
                return r.content.startswith(b'%PDF-')
                
    except httpx.TimeoutException:
        print(f"[is_pdf_httpx_get_range] Timeout for {url}")
        return False
    except httpx.ConnectError:
        print(f"[is_pdf_httpx_get_range] Connection error for {url}")
        return False
    except Exception as e:
        print(f"[is_pdf_httpx_get_range] Error for {url}: {type(e).__name__}: {e}")
        return False

async def is_pdf_by_full_get(url: str, timeout: int = 15) -> bool:
    """Last resort: download beginning of file to check magic number."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            verify=False
        ) as client:
            
            headers = {
                "User-Agent": random.choice(USER_AGENT_STRINGS),
                "Accept": "*/*",
            }
            
            # Stream the response to avoid downloading large files
            async with client.stream('GET', url, headers=headers) as response:
                # Read first 5 bytes to check PDF magic number
                chunk = await response.aread(5)
                if chunk and chunk.startswith(b'%PDF-'):
                    return True
                    
                # Also check Content-Type from response
                ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
                return "pdf" in ctype
                
    except Exception as e:
        print(f"[is_pdf_by_full_get] Error for {url}: {type(e).__name__}: {e}")
        return False

async def is_pdf(url: str, logger: Logger = None) -> bool:
    """
    Robustly detect if a URL points to a PDF file using multiple strategies.
    
    Args:
        url: The URL to check
        logger: Optional logger instance
        
    Returns:
        bool: True if URL points to a PDF, False otherwise
    """
    url = format_url(url)
    
    if logger:
        logger.debug(f"Checking if URL is PDF: {url}")
    
    # 1. Fast URL pattern check
    if is_pdf_by_suffix(url):
        if logger:
            logger.info(f"URL pattern indicates PDF: {url}")
        else:
            print(f"{url} IS a PDF (by URL pattern)")
        return True
    
    # 2. Try HEAD request first (fastest network check)
    if is_pdf_by_requests_head(url):
        if logger:
            logger.info(f"HEAD request confirms PDF: {url}")
        else:
            print(f"{url} IS a PDF (by HEAD request)")
        return True
    
    # 3. Try partial GET with magic number check
    if await is_pdf_by_httpx_get_range(url):
        if logger:
            logger.info(f"Partial GET confirms PDF: {url}")
        else:
            print(f"{url} IS a PDF (by partial GET)")
        return True
    
    # 4. Last resort: stream beginning of file
    if await is_pdf_by_full_get(url):
        if logger:
            logger.info(f"Full GET confirms PDF: {url}")
        else:
            print(f"{url} IS a PDF (by full GET)")
        return True
    
    # Not a PDF
    if logger:
        logger.debug(f"URL is not a PDF: {url}")
    else:
        print(f"{url} IS NOT a PDF")
    return False


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
        wait_until: str = "load",
        headless: bool= True,
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

    user_agent = random.choice(DEFAULT_USER_AGENTS)
    headers = {"user-agent": user_agent}

    screenshot_b64=make_blank_png_b64()
    page_text=ERROR_TEXT


    # Set location. Set Language to English
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            # channel="chrome",
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
                '--password-store=basic',
                '--use-mock-keychain',
            ],
            extra_http_headers=headers,
            viewport={
                "width": random.randint(1050, 1150),
                "height": random.randint(700, 800),
            },
        )
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
        # page = await mgr.get()
        # cdp = await context.new_cdp_session(page)
        # await cdp.send("Page.enable")
        # await cdp.send("DOM.enable")
        # await cdp.send("Runtime.enable")

        start_ts = time.time()
        try:
            async def navigate():
                pg = await mgr.get()
                return await pg.goto(target, wait_until=wait_until, timeout=15000)

            try:
                await navigate()
            except Exception as e:
                logger.info(f"Navigation failed (Timeout is fine): {e}")

            # ---------- scroll & full-page capture ----------
            page = await mgr.get()
            for _ in range(3):
                await page.keyboard.press("End")
                await asyncio.sleep(random.uniform(1.0, 2.0))
            for _ in range(random.randint(5, 10)):
                await page.mouse.wheel(0, random.randint(-500, 500))
                await asyncio.sleep(random.uniform(0.5, 1.5))
            await page.keyboard.press("Home")
            await asyncio.sleep(random.uniform(1.0, 2.0))

            try:
                page = await mgr.get()
                cdp = await context.new_cdp_session(page)
                metrics = await cdp.send("Page.getLayoutMetrics")
                css_vp = metrics["cssVisualViewport"]
                css_content = metrics["cssContentSize"]
                width = round(css_vp["clientWidth"])
                height = round(min(css_content["height"], 6000))
                scale = round(metrics.get("visualViewport", {}).get("scale", 1))
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
                shot = await cdp.send(
                    "Page.captureScreenshot",
                    {"format": "png", "captureBeyondViewport": True},
                )
                screenshot_b64 = shot.get("data")
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
