# WAY-O-AI — Python prototype

pitch deck for **WAY-O-AI: An Innovation for Traffic & Pollution Management**.

This folder is the **computer-vision + database** part of the idea:

1. Reads a car number plate from a photo.
2. Looks the plate up in a local sample registry.
3. Shows estimated CO2 and whether the PUC is valid.
4. Prints a **simulated** warning (no real SMS, no real police system).
5. Saves a log and an annotated picture.

The names and phone numbers in `data/vehicles.csv` are **fake classroom data**.

---

 WAY-O-AI has three layers:

| Layer | In the pitch | What you still do |
| --- | --- | --- |
| Hardware | Webcam, NodeMCU, MQ-135 gas sensor, cardboard city model | Keep using that for the live demo |
| Vision | OpenCV vehicle box + plate text | This Python app |
| Data + alerts | Registry with plate, PUC, daily CO2; SMS / chalan idea | Local CSV + printed / logged alert |
| Future | Raspberry Pi, YOLO, solar | Optional upgrade after the prototype works |

---

## Setup

You need Python 3.10+, Tesseract OCR, and a few packages.

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

cd ecoplate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows install Tesseract from the official installer, then `pip install -r requirements.txt`.

---

## Run it

```bash
# Built-in demo (draws sample cars and reads their plates)
python3 app.py --demo

# Your own photo
python3 app.py --image path/to/car.jpg

# Whole folder of photos
python3 app.py --folder samples

# Type a plate from the registry (no camera)
python3 app.py --plate MH02AB5678

# Include a fake roadside air reading (like MQ-135 ppm)
python3 app.py --plate TN07EF8765 --ppm 1100
```

Results:

- Terminal report (vehicle, CO2, PUC, alert level)
- Annotated image in `output/`
- Row added to `data/detection_log.csv`

Try these demo plates (they are already in the CSV):

- `DL3CA1234` — PUC valid, moderate CO2
- `MH02AB5678` — PUC **not** valid → WARNING
- `KA05CD4321` — electric, zero tailpipe
- `TN07EF8765` — high daily CO2 + PUC not valid → WARNING

---

## How the software works

```
photo → OpenCV finds a plate-shaped rectangle
      → Tesseract reads letters/digits
      → vehicles.csv lookup
      → CO2 band + PUC check
      → log + on-screen alert
```

Pollution numbers are **estimates for the project**, not a lab measurement of that exact car. Daily kg values match the style of the table on slide 5 of your deck.

---

## Hooking up the Arduino later

Keep hardware and Python separate at first.

1. Arduino / NodeMCU reads MQ-135 and prints one number per line, for example `PPM:1024`.
2. On the laptop, read that serial port and pass it in:

```bash
python3 app.py --image samples/TN07EF8765.jpg --ppm 1024
```


---

##  (project checklist)

1. Run `python3 app.py --demo` and save screenshots for the report.
2. Photograph your cardboard cars so the plate is large and straight. Add those photos to `samples/`.
3. Put every demo plate into `data/vehicles.csv` (plate, fake owner, fuel, CO2/day, PUC Yes/No).
4. Show one **valid PUC** car and one **expired PUC** car in the viva.
5. If you have the MQ-135, show the live ppm on the laptop next to the Python alert.
6. In the report, write the limits honestly: OCR fails on blurry plates; this is not a government database; alerts are simulated.
7. Only if extra time: try YOLOv8 plate detection on a Raspberry Pi (your “future prospects” slide).

---
