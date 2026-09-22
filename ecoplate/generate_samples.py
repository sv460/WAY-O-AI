"""Draw simple rear-of-car pictures with number plates so the demo runs without a camera."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "samples"
FONT = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"

DEMO_PLATES = [
    ("DL3CA1234", (170, 80, 40)),
    ("MH02AB5678", (30, 30, 30)),
    ("KA05CD4321", (40, 80, 200)),
    ("TN07EF8765", (90, 50, 50)),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _draw_car(plate: str, color: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (720, 420), (210, 205, 198))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 300, 720, 420), fill=(70, 70, 70))
    d.line((0, 360, 720, 360), fill=(230, 230, 230), width=4)
    d.rounded_rectangle((90, 90, 630, 310), radius=18, fill=color, outline=(20, 20, 20), width=3)
    d.rounded_rectangle((140, 110, 580, 190), radius=8, fill=(170, 210, 230))
    d.rectangle((110, 230, 190, 270), fill=(255, 200, 40))
    d.rectangle((530, 230, 610, 270), fill=(255, 200, 40))
    d.rounded_rectangle((210, 228, 510, 302), radius=6, fill=(255, 255, 255), outline=(10, 10, 10), width=3)
    font = _font(36)
    bbox = d.textbbox((0, 0), plate, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((360 - tw / 2, 265 - th / 2), plate, fill=(10, 10, 10), font=font)
    return img


def make_samples() -> list[Path]:
    SAMPLE_DIR.mkdir(exist_ok=True)
    paths = []
    for plate, color in DEMO_PLATES:
        path = SAMPLE_DIR / f"{plate}.jpg"
        _draw_car(plate, color).save(path, quality=95)
        paths.append(path)
    return paths


if __name__ == "__main__":
    made = make_samples()
    print("Wrote", len(made), "sample images")
