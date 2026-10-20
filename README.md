# AttendanceLogger Pro — Face check-in with no double log (Windows app)

CSC 126 Midterm Project. Classical OpenCV only (Haar + LBPH, no DNN), Tkinter desktop UI, builds to `.exe`.

## Problem
Manual attendance is slow and cheatable. One laptop + webcam enrolls faces once, then auto-logs `name,date,time,confidence` to `attendance.csv` — one row per person per day, unknowns flagged orange.

## Rubric map (5 topics, see `face_core.py`)
| Topic | Code |
|---|---|
| a. Image Operations | `resize`, `cvtColor`, `GaussianBlur`, `equalizeHist` |
| b. Image Segmentation | MOG2 mask → `threshold` → `dilate` → `findContours` motion gate |
| c. Haar Cascade | `CascadeClassifier` + `detectMultiScale` faces |
| d. Analysis + Transform | `boundingRect`, face crop 150×150, `putText`/`rectangle` overlays |
| e. Motion + Tracking | centroid tracker (stable IDs, one row per person per day) |

Output: annotated frame (green=KNOWN, orange=UNKNOWN + `#id` + distance), motion-mask panel, 4 stat cards, Treeview table mirroring CSV, event log.

## Setup (wheels only, no C++ tools)
```bash
cd AttendanceLoggerPro
python -m pip install --only-binary :all: -r requirements.txt
```

## Run / Build
```bash
python AttendanceApp.py   # run from source
build_exe.bat             # -> dist\AttendanceLoggerPro.exe
```

Flow: 1) type name → Capture faces (30) → Train model. 2) Start session → walk in → row appears in table + CSV. 3) Same person returns within 60 s → NOT logged again. 4) Export CSV as… for submission.

## Files
- `AttendanceApp.py` — Tkinter UI (enroll, session, table, log)
- `face_core.py` — detection/recognition/tracking/CSV core
- `requirements.txt`, `build_exe.bat`, `DOCUMENTATION.md`
- `data/faces/<name>/*.jpg`, `data/model.yml`, `attendance.csv` (auto-created)
- `samples/` demo inputs, `screenshots/` report figures, `alerts/` snapshots

## 15-min demo
1. Enroll 1 classmate live (30 faces) + Train (4 min). 2. Start session, show KNOWN green + mask (3 min). 3. Show UNKNOWN orange on stranger (2 min). 4. Walk out/in → no double row (3 min). 5. Open `attendance.csv` + walk code a→e (3 min).
