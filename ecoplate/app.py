#!/usr/bin/env python3
"""
WAY-O-AI software prototype
  python3 app.py --demo
  python3 app.py --image samples/DL3CA1234.jpg
  python3 app.py --folder samples
  python3 app.py --plate MH02AB5678 --ppm 1100
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from detector import annotate, detect_plates
from emissions import build_alert, lookup_vehicle


ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
LOG_FILE = ROOT / "data" / "detection_log.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_log(row: dict) -> None:
    OUTPUT.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
    frame = pd.DataFrame([row])
    header = not LOG_FILE.exists()
    frame.to_csv(LOG_FILE, mode="a", header=header, index=False)


def print_report(record, alert: dict, source: str, air_ppm: int | None) -> None:
    print("\n" + "=" * 60)
    print("WAY-O-AI  |  plate + pollution report")
    print("=" * 60)
    print(f"Source          : {source}")
    print(f"Plate           : {record.plate}")
    print(f"Vehicle         : {record.display_name}")
    print(f"Fuel / standard : {record.fuel} / {record.emission_standard}")
    print(f"City            : {record.city}")
    print(f"CO2 per km      : {record.co2_g_per_km} g")
    print(f"CO2 per day     : {record.co2_kg_per_day} kg  (demo estimate)")
    print(f"Pollution band  : {alert['band']}")
    print(f"PUC valid       : {'Yes' if record.puc_valid else 'No'}")
    if air_ppm is not None:
        print(f"Air sensor ppm  : {air_ppm}  (MQ-135 style demo value)")
    print(f"Alert level     : {alert['level']}")
    for reason in alert["reasons"]:
        print(f"  - {reason}")
    print(alert["message"])
    print("=" * 60)


def process_plate_text(plate: str, source: str, air_ppm: int | None) -> dict:
    record = lookup_vehicle(plate)
    alert = build_alert(record, air_ppm=air_ppm)
    print_report(record, alert, source, air_ppm)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "plate": record.plate,
        "known": record.known,
        "vehicle": record.display_name,
        "fuel": record.fuel,
        "co2_g_per_km": record.co2_g_per_km,
        "co2_kg_per_day": record.co2_kg_per_day,
        "puc_valid": record.puc_valid,
        "alert_level": alert["level"],
        "air_ppm": air_ppm if air_ppm is not None else "",
    }
    save_log(row)
    return {"record": record, "alert": alert}


def process_image(path: Path, air_ppm: int | None) -> None:
    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read image: {path}")
        return
    hits = detect_plates(image)
    if not hits:
        print(f"No plate text found in {path.name}. Try a clearer close-up of the plate.")
        return
    for i, hit in enumerate(hits, start=1):
        result = process_plate_text(hit.text, source=str(path.name), air_ppm=air_ppm)
        extra = [
            f"{hit.text}  {result['alert']['level']}",
            f"{result['record'].display_name}",
            f"CO2 {result['record'].co2_g_per_km} g/km  PUC {'Yes' if result['record'].puc_valid else 'No'}",
        ]
        labeled = annotate(image, [hit], extra)
        out = OUTPUT / f"{path.stem}_result_{i}.jpg"
        OUTPUT.mkdir(exist_ok=True)
        cv2.imwrite(str(out), labeled)
        print(f"Saved annotated photo -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WAY-O-AI number-plate + pollution prototype")
    parser.add_argument("--image", type=Path, help="Photo of a car / plate")
    parser.add_argument("--folder", type=Path, help="Folder of photos")
    parser.add_argument("--plate", type=str, help="Type a plate number directly (skip camera)")
    parser.add_argument("--demo", action="store_true", help="Generate sample plates and run them")
    parser.add_argument("--ppm", type=int, default=None, help="Optional MQ-135 style air reading")
    args = parser.parse_args()

    if args.demo:
        from generate_samples import make_samples

        samples = make_samples()
        print(f"Created {len(samples)} demo photos in samples/")
        for photo in samples:
            process_image(photo, args.ppm)
        return

    if args.plate:
        process_plate_text(args.plate, source="typed", air_ppm=args.ppm)
        return

    if args.image:
        process_image(args.image, args.ppm)
        return

    if args.folder:
        files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            print(f"No images in {args.folder}")
            return
        for photo in files:
            process_image(photo, args.ppm)
        return

    parser.print_help()
    print("\nQuick start:  python3 app.py --demo")


if __name__ == "__main__":
    main()
