"""Render the same three-slide horizontal README preview pattern as SAR/RSS.

The images are screenshots of the actual local tutorial, not redesigned panels.
Slide order matches SAR: cover, a theory/results page, a practical outcome page.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SLIDES = (0, 3, 5)  # Cover · exact support horizon · observed policy training
SLIDE_SIZE = (1920, 1080)
THUMB_SIZE = (640, 360)
TARGET = ROOT / "assets" / "tour_preview.png"


def main():
    source = (ROOT / "docs" / "index.html").resolve().as_uri()
    with TemporaryDirectory() as temp:
        paths = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(
                viewport={"width": SLIDE_SIZE[0], "height": SLIDE_SIZE[1]},
                device_scale_factor=1,
                reduced_motion="reduce",
                color_scheme="light",
            )
            page.goto(source, wait_until="load")
            page.wait_for_function("document.querySelectorAll('section.slide').length === 7")
            try:
                page.evaluate("() => document.fonts.ready")
            except Exception:
                pass
            for i, slide in enumerate(SLIDES):
                page.evaluate("(index) => deck.go(index)", slide)
                page.wait_for_function(
                    "(index) => document.querySelectorAll('section.slide')[index]?.classList.contains('active')",
                    arg=slide,
                )
                page.wait_for_timeout(650)
                path = Path(temp) / f"slide_{slide + 1}.png"
                page.screenshot(path=str(path), full_page=False, animations="disabled")
                paths.append(path)
            browser.close()

        strip = Image.new("RGB", (THUMB_SIZE[0] * len(paths), THUMB_SIZE[1]), "#1C2644")
        for i, path in enumerate(paths):
            with Image.open(path) as screenshot:
                thumb = screenshot.convert("RGB").resize(THUMB_SIZE, Image.Resampling.LANCZOS)
                strip.paste(thumb, (i * THUMB_SIZE[0], 0))
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        strip.save(TARGET, optimize=True)
        print(f"Saved {TARGET.relative_to(ROOT)} ({strip.width} × {strip.height}), slides 1 / 4 / 6")


if __name__ == "__main__":
    main()
