# pdf_parser.py  ---------------------------------------------------------
"""
Lightweight PDF parser:
    * extract()   - Pass URL / local path / bytes, asynchronously returns (imgs, text)
    * If download or parsing fails, always returns (None, None)
    * imgs: screenshot of each page (JPEG, base64), up to 50 pages
    * text: all plain text, up to 100 pages
Dependencies:
    pip install aiohttp pymupdf pillow
"""

import aiohttp
import asyncio
import base64
from io import BytesIO
from typing import List, Tuple, Union, Optional

import fitz                # PyMuPDF
from PIL import Image

# ------------------ Constants ------------------
PDF_MAGIC = b"%PDF-"        # PDF file header
UA_CHROME = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

class PDFParser:
    """
    Download and parse PDF. Returns (None, None) on failure.
    """

    # Default limits
    MAX_PAGES: int        = 100
    MAX_IMAGE_PAGES: int  = 50
    RENDER_DPI: int       = 144
    JPEG_QUALITY: int     = 70

    # ------------------ Public API ------------------
    async def extract(
        self,
        source: Union[str, bytes, BytesIO],
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Parameters
        ----------
        source : str | bytes | BytesIO
            URL / local file path / PDF byte stream

        Returns
        -------
        imgs : list[str] | None
        text : str | None
        """
        # 1) Obtain PDF bytes
        if isinstance(source, (bytes, BytesIO)):
            data = source.getvalue() if isinstance(source, BytesIO) else source
        elif isinstance(source, str) and source.lower().startswith(("http://", "https://")):
            data = await self._fetch_pdf_bytes(source)
        else:  # Local file
            data = await asyncio.to_thread(lambda p: open(p, "rb").read(), str(source))

        # 2) Magic number check
        if not data.lstrip().startswith(PDF_MAGIC):
            return None, None

        # 3) Parsing (CPU-intensive, synchronous), run in thread
        return await asyncio.to_thread(self._extract_from_bytes, data)

    # ------------------ Internal Implementation ------------------
    async def _fetch_pdf_bytes(self, url: str) -> bytes:
        """
        Fetch PDF with browser User-Agent; if necessary, switch to export.arxiv.org as backup.
        """
        headers = {
            "User-Agent": UA_CHROME,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        }

        async def _download(u: str) -> bytes:
            async with aiohttp.ClientSession(headers=headers) as s:
                async with s.get(u, allow_redirects=True, timeout=30) as r:
                    r.raise_for_status()
                    return await r.read()

        data = await _download(url)
        # print(data)

        # If returned HTML, try backup domain for arxiv
        if not data.lstrip().startswith(PDF_MAGIC) and "arxiv.org" in url:
            backup = url.replace("://arxiv.org", "://export.arxiv.org")
            try:
                data = await _download(backup)
            except Exception as e:
                print(f"failed to download from {url} with backup export arxiv: {e}")

        # print(data)

        return data

    def _extract_from_bytes(
        self, data: bytes
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Actual parsing logic. Returns (None, None) on failure.
        """
        # Double-check magic number (in case called directly by other modules)
        if not data.lstrip().startswith(PDF_MAGIC):
            return None, None

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except (fitz.FileDataError, RuntimeError):
            return None, None

        imgs: List[str] = []
        texts: List[str] = []
        zoom = self.RENDER_DPI / 72

        max_pages = min(self.MAX_PAGES, doc.page_count)
        max_img_pages = min(self.MAX_IMAGE_PAGES, doc.page_count)

        for i in range(max_pages):
            page = doc.load_page(i)
            texts.append(page.get_text("text"))

            if i < max_img_pages:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                buf = BytesIO()
                img.save(buf, "JPEG", quality=self.JPEG_QUALITY,
                         optimize=True, progressive=True)
                imgs.append(base64.b64encode(buf.getvalue()).decode())
        # print(texts)
        return imgs, "\n".join(texts)


# ------------------ Local Quick Test ------------------
if __name__ == "__main__":
    async def _demo() -> None:
        parser = PDFParser()
        #
        # # ✅ Normal PDF
        # ok_imgs, ok_txt = await parser.extract(
        #     "https://arxiv.org/pdf/2505.07880.pdf"
        # )
        # print("Normal PDF:", "Success" if ok_txt else "Failed")

        # ❌ Fake PDF
        bad_imgs, bad_txt = await parser.extract(
            "https://arxiv.org/pdf/2408.XXXXXv1.pdf"
        )
        print("Fake PDF:", "Success" if bad_txt else "Failed")

    asyncio.run(_demo())
