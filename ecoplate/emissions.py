"""
Match a number plate to the local WAY-O-AI registry and decide the alert.
Owner names and phone numbers in the sample file are FAKE classroom data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).parent / "data" / "vehicles.csv"

DEFAULT_CO2_PER_KM = 145
DEFAULT_CO2_PER_DAY = 16


@dataclass
class VehicleRecord:
    plate: str
    owner_name: str
    phone: str
    fuel: str
    make: str
    model: str
    year: int | str
    emission_standard: str
    co2_g_per_km: float
    co2_kg_per_day: float
    puc_valid: bool
    city: str
    known: bool

    @property
    def display_name(self) -> str:
        if self.known:
            return f"{self.make} {self.model} ({self.year})"
        return "Unknown vehicle"


def normalize_plate(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _load_registry() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing registry: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df["plate_norm"] = df["plate"].astype(str).str.replace(" ", "", regex=False).str.upper()
    df["puc_flag"] = df["puc_valid"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
    return df


def lookup_vehicle(plate_text: str) -> VehicleRecord:
    plate = normalize_plate(plate_text)
    registry = _load_registry()
    match = registry[registry["plate_norm"] == plate]
    if match.empty:
        return VehicleRecord(
            plate=plate or "UNREADABLE",
            owner_name="Unknown",
            phone="—",
            fuel="Unknown",
            make="Unknown",
            model="Unknown",
            year="—",
            emission_standard="Unknown",
            co2_g_per_km=DEFAULT_CO2_PER_KM,
            co2_kg_per_day=DEFAULT_CO2_PER_DAY,
            puc_valid=False,
            city="—",
            known=False,
        )
    row = match.iloc[0]
    return VehicleRecord(
        plate=str(row["plate"]),
        owner_name=str(row["owner_name"]),
        phone=str(row["phone"]),
        fuel=str(row["fuel"]),
        make=str(row["make"]),
        model=str(row["model"]),
        year=int(row["year"]),
        emission_standard=str(row["emission_standard"]),
        co2_g_per_km=float(row["co2_g_per_km"]),
        co2_kg_per_day=float(row["co2_kg_per_day"]),
        puc_valid=bool(row["puc_flag"]),
        city=str(row["city"]),
        known=True,
    )


def pollution_band(record: VehicleRecord) -> str:
    if record.fuel.lower() == "electric":
        return "Zero tailpipe"
    if record.co2_g_per_km <= 100:
        return "Low"
    if record.co2_g_per_km <= 150:
        return "Moderate"
    if record.co2_g_per_km <= 180:
        return "High"
    return "Very high"


def build_alert(record: VehicleRecord, air_ppm: int | None = None) -> dict:
    """
    Classroom alert logic from the pitch deck:
    - missing / expired PUC -> warning
    - very high CO2 or bad air reading -> warning
    This only prints a simulated message. It does not send real SMS.
    """
    reasons = []
    level = "OK"
    if not record.known:
        reasons.append("Plate not in local registry — using average car values.")
        level = "WATCH"
    if record.known and not record.puc_valid:
        reasons.append("PUC certificate is not valid for this year.")
        level = "WARNING"
    if record.co2_kg_per_day >= 22 and record.fuel.lower() != "electric":
        reasons.append("Estimated daily CO2 is high for this vehicle class.")
        if level == "OK":
            level = "WATCH"
    if air_ppm is not None and air_ppm >= 1000:
        reasons.append(f"Roadside air sensor reading is high ({air_ppm} ppm).")
        level = "WARNING"
    if not reasons:
        reasons.append("PUC valid and emission estimate is within the demo limit.")

    if level == "WARNING":
        message = (
            f"SIMULATED ALERT: Vehicle {record.plate} needs a PUC check / cleaner trip. "
            "In a real city system this would notify the owner. This prototype only logs it."
        )
    elif level == "WATCH":
        message = f"WATCH: {record.plate} was logged. Add it to the registry or review emissions."
    else:
        message = f"CLEAR: {record.plate} is in the registry with a valid PUC in this demo."

    return {
        "level": level,
        "reasons": reasons,
        "message": message,
        "band": pollution_band(record),
    }
