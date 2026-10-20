"""AttendanceLogger Pro - Windows desktop app (Tkinter, builds to .exe).

Run:   python AttendanceApp.py
Build: build_exe.bat  ->  dist\\AttendanceLoggerPro.exe

Flow: enroll name -> capture faces -> train LBPH -> start session -> auto-log CSV once per person per day.
Core vision lives in face_core.py; this file is UI only.
"""
import os
import csv
import queue
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

import face_core as fc

# ---------- Theme (indigo, distinct from SecureVision) ----------
BG, PANEL, CARD = "#101827", "#1c2942", "#263550"
ACCENT, GREEN, ORANGE, RED = "#818cf8", "#34d399", "#fb923c", "#f87171"
TEXT, DIM = "#e5e9f0", "#9aa7bd"
FONT, FONT_BIG = ("Segoe UI", 10), ("Segoe UI", 11, "bold")
NUM_FONT = ("Segoe UI", 24, "bold")
VIEW_W = 440


class StatCard(tk.Frame):
    def __init__(self, parent, caption, color=TEXT):
        super().__init__(parent, bg=CARD, padx=12, pady=8)
        self.num = tk.Label(self, text="0", font=NUM_FONT, bg=CARD, fg=color)
        self.num.pack(anchor="w")
        tk.Label(self, text=caption, font=("Segoe UI", 9), bg=CARD, fg=DIM).pack(anchor="w")

    def set(self, v):
        self.num.config(text=str(v))


class App:
    def __init__(self, root):
        self.root = root
        root.title("AttendanceLogger Pro")
        root.configure(bg=BG)
        root.minsize(1200, 800)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cascade = fc.load_cascade()
        self.rec, self.names = fc.load_model()  # None until trained
        self.st = fc.new_state()
        self.cap, self.running = None, False
        self.fps, self._t = 0.0, time.time()
        self.present = set()  # names logged this session (no double count)
        self._view_w = VIEW_W  # video panel width; follows window size
        self._rz_after = None
        self.cur_mask = None
        self._q = queue.Queue(maxsize=2)  # worker -> UI; drop-oldest, never blocks
        self._stop_evt = threading.Event()
        self._worker = None
        self._sess = 0  # session id; lets the UI ignore msgs from an old worker
        self._cleanup_t = None  # background stop() cleanup thread
        self._thr = 75.0       # plain-float mirror of slider (worker never touches Tk)
        self._train_thread = None
        self._train_result = None
        os.makedirs(fc.FACES_DIR, exist_ok=True)

        self._header()
        self._sidebar()
        self._content()
        self._statusbar()
        self.placeholder()
        self.refresh_model_label()
        self.tick_clock()
        root.bind("s", lambda e: self.snapshot())
        root.bind("q", lambda e: self.on_close())
        root.bind("<Delete>", lambda e: self.delete_selected())

    # ============ layout ============
    def _header(self):
        h = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        h.pack(fill="x")
        tk.Label(h, text="✔ AttendanceLogger Pro", font=("Segoe UI", 14, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left")
        self.dot = tk.Canvas(h, width=14, height=14, bg=PANEL, highlightthickness=0)
        self.dot.pack(side="right")
        self._d = self.dot.create_oval(2, 2, 12, 12, fill="#475569", outline="")
        self.live_txt = tk.Label(h, text="IDLE", font=FONT_BIG, bg=PANEL, fg=DIM)
        self.live_txt.pack(side="right", padx=(0, 8))
        self.clock = tk.Label(h, text="", font=FONT, bg=PANEL, fg=DIM)
        self.clock.pack(side="right", padx=(0, 18))

    def _sidebar(self):
        # Scrollable column so all buttons fit on short screens.
        wrap = tk.Frame(self.root, bg=PANEL)
        wrap.pack(side="left", fill="y")
        canvas = tk.Canvas(wrap, bg=PANEL, highlightthickness=0, width=360)
        bar = tk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="y", expand=True)
        s = tk.Frame(canvas, bg=PANEL, padx=12, pady=12)
        canvas.create_window((0, 0), window=s, anchor="nw")
        s.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _wheel(e):  # mouse wheel scrolls sidebar when hovered
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self._side_canvas = canvas
        tk.Label(s, text="1  ENROLL", font=("Segoe UI", 9, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(s, text="Full name", bg=PANEL, fg=DIM, font=FONT).pack(anchor="w", pady=(4, 0))
        self.e_name = tk.Entry(s, font=FONT, width=24)
        self.e_name.pack(fill="x")
        r = tk.Frame(s, bg=PANEL)
        r.pack(fill="x", pady=4)
        tk.Label(r, text="Samples", bg=PANEL, fg=DIM, font=FONT).pack(side="left")
        self.e_n = tk.Spinbox(r, from_=15, to=60, width=5, font=FONT)
        self.e_n.delete(0, "end")
        self.e_n.insert(0, "30")
        self.e_n.pack(side="right")
        self.btn(s, "📷  Capture faces", None, self.enroll)
        self.prog = tk.Label(s, text="", bg=PANEL, fg=DIM, font=("Segoe UI", 9))
        self.prog.pack()
        self.btn(s, "🧠  Train model", GREEN, self.train)
        self.model_lbl = tk.Label(s, text="Model: none", bg=PANEL, fg=DIM, font=("Segoe UI", 9))
        self.model_lbl.pack(pady=(0, 6))

        tk.Label(s, text="2  SESSION", font=("Segoe UI", 9, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w", pady=(8, 0))
        # Camera row: label + dropdown
        cam_row = tk.Frame(s, bg=PANEL)
        cam_row.pack(fill="x", pady=(4, 0))
        tk.Label(cam_row, text="Camera", bg=PANEL, fg=DIM, font=FONT, width=8).pack(side="left")
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.cam_var, width=40, font=FONT, state="readonly")
        self.cam_combo.pack(side="left", padx=(4, 0), fill="x", expand=True)
        # Refresh button on its own row
        btn_row = tk.Frame(s, bg=PANEL)
        btn_row.pack(fill="x", pady=(2, 4))
        self.btn(btn_row, "⟳ Refresh", None, self.refresh_cameras).pack(side="left")
        self.refresh_cameras()
        self.btn(s, "▶  Start session", GREEN, self.start)
        self.btn(s, "■  Stop", RED, self.stop)
        self.btn(s, "📂  Test video…", None, self.open_video)
        tk.Label(s, text="Accept threshold (lower=stricter)", bg=PANEL, fg=DIM, font=FONT).pack(anchor="w", pady=(6, 0))
        self.s_thr = tk.Scale(s, from_=50, to=100, orient="horizontal", bg=PANEL,
                              fg=TEXT, highlightthickness=0, length=210,
                              command=lambda v: setattr(self, "_thr", float(v)))
        self.s_thr.set(75)
        self.s_thr.pack()
        self.v_save = tk.BooleanVar(value=True)
        tk.Checkbutton(s, text="Auto-log to CSV", variable=self.v_save, bg=PANEL,
                       fg=TEXT, selectcolor=CARD, activebackground=PANEL, font=FONT).pack(anchor="w", pady=4)

        tk.Label(s, text="3  ACTIONS", font=("Segoe UI", 9, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w", pady=(8, 0))
        self.btn(s, "📸  Snapshot  (S)", None, self.snapshot)
        self.btn(s, "💾  Export CSV as…", None, self.export)
        self.btn(s, "🗑  Delete selected row", None, self.delete_selected)
        self.btn(s, "🧹  Clear attendance", None, self.clear_attendance)
        self.btn(s, "👤  Delete enrolled person…", None, self.delete_person)
        self.btn(s, "📁  Open data folder", None, self.open_data)

    def _content(self):
        c = tk.Frame(self.root, bg=BG, padx=12, pady=10)
        c.pack(side="left", fill="both", expand=True)

        # Stat cards
        strip = tk.Frame(c, bg=BG)
        strip.pack(fill="x", pady=(0, 8))
        self.k_present = StatCard(strip, "Present (unique)", GREEN)
        self.k_faces = StatCard(strip, "Faces in frame", ACCENT)
        self.k_unknown = StatCard(strip, "Unknown seen", ORANGE)
        self.k_motion = StatCard(strip, "Motion", DIM)
        for w in (self.k_present, self.k_faces, self.k_unknown, self.k_motion):
            w.pack(side="left", expand=True, fill="x", padx=4)

        # Video panels - responsive grid
        v = tk.Frame(c, bg=BG)
        v.pack(fill="both", expand=True, pady=(0, 8))
        v.columnconfigure(0, weight=1)
        v.columnconfigure(1, weight=1)
        v.rowconfigure(1, weight=1)

        tk.Label(v, text="PROCESSED", bg=BG, fg=DIM,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(v, text="MOTION MASK", bg=BG, fg=DIM,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Video containers that scale
        self.live = tk.Label(v, bg="black", bd=2, relief="sunken")
        self.live.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.mask = tk.Label(v, bg="black", bd=2, relief="sunken")
        self.mask.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        v.bind("<Configure>", self._on_v_resize)

        # Attendance section - scrollable if needed
        tk.Label(c, text="ATTENDANCE", bg=BG, fg=DIM,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))

        # Table with horizontal scroll
        table_frame = tk.Frame(c, bg=BG)
        table_frame.pack(fill="both", expand=True, pady=(0, 6))
        cols = ("name", "date", "time", "conf")
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings", height=6)
        for col, w in zip(cols, (180, 110, 110, 90)):
            self.table.heading(col, text=col.upper())
            self.table.column(col, width=w, minwidth=80, stretch=False)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Log with vertical scroll
        log_frame = tk.Frame(c, bg=BG)
        log_frame.pack(fill="x", pady=(6, 0))
        self.log = tk.Listbox(log_frame, height=4, bg=PANEL, fg=TEXT,
                              font=("Consolas", 9), activestyle="none")
        log_sb = tk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")

        self.load_csv_to_table()
        self.log_insert("Ready. Enroll faces, train, then start session.")

    def _statusbar(self):
        self.status = tk.Label(self.root, text="Idle", bg=PANEL, fg=DIM, font=FONT, anchor="w", padx=12)
        self.status.pack(side="bottom", fill="x")

    # ============ small helpers ============
    def btn(self, parent, text, color, cmd):
        b = tk.Button(parent, text=text, command=cmd, font=FONT, relief="flat", padx=10, pady=6,
                      bg=(CARD if color is None else color),
                      fg=("white" if color in (GREEN, RED) else TEXT),
                      activebackground=CARD, cursor="hand2")
        b.pack(fill="x", pady=3)
        return b

    def log_insert(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}")
        if self.log.size() > 200:
            self.log.delete(0)
        self.log.see("end")

    def set_live(self, on):
        self.dot.itemconfig(self._d, fill=GREEN if on else "#475569")
        self.live_txt.config(text="● LIVE" if on else "IDLE", fg=GREEN if on else DIM)

    def tick_clock(self):
        self.clock.config(text=f"{datetime.now():%H:%M:%S}   {self.fps:0.0f} fps")
        self.root.after(500, self.tick_clock)

    def to_photo(self, bgr, w=None):
        w = w or self._view_w
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, ww = rgb.shape[:2]
        nw, nh = w, int(h * w / ww)
        return ImageTk.PhotoImage(Image.fromarray(rgb).resize((nw, nh)))

    def _on_v_resize(self, event):
        # Grow/shrink both video panels to share the row (kills dead space).
        # event.widget is the video frame container
        w = max(280, (event.width - 10) // 2)
        if abs(w - self._view_w) < 20:
            return
        self._view_w = w
        if self._rz_after is not None:  # throttle: redraw once drag pauses
            self.root.after_cancel(self._rz_after)
        self._rz_after = self.root.after(150, self._rerender)

    def _rerender(self):
        self._rz_after = None
        if getattr(self, "cur", None) is None:
            self.placeholder()
            return
        self.ph = self.to_photo(self.cur)
        self.live.config(image=self.ph)
        if self.cur_mask is not None:
            self.ph2 = self.to_photo(self.cur_mask)
            self.mask.config(image=self.ph2)

    def placeholder(self):
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(img, "Enroll + Train, then Start", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (154, 167, 189), 2)
        self.ph = self.to_photo(img)
        self.live.config(image=self.ph)
        m = np.zeros((480, 640), np.uint8)
        self.ph2 = self.to_photo(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        self.mask.config(image=self.ph2)

    def refresh_model_label(self):
        self.model_lbl.config(text=f"Model: {len(self.names)} person(s)" if self.rec else "Model: none — train first")

    def load_csv_to_table(self):
        for i in self.table.get_children():
            self.table.delete(i)
        if os.path.isfile(fc.CSV_FILE):
            import csv as _csv
            try:
                with open(fc.CSV_FILE) as fh:
                    r = list(_csv.DictReader(fh))
            except Exception as e:
                self.log_insert("CSV unreadable, starting fresh: " + str(e))
                r = []
            shown = 0
            for row in r[-500:]:  # table shows recent; present-set still covers all
                try:
                    vals = (row["name"], row["date"], row["time"], row["confidence"])
                except (KeyError, TypeError):
                    continue  # skip hand-edited/broken rows, never crash
                if not vals[0] or not vals[1]:
                    continue
                self.table.insert("", "end", values=vals)
                shown += 1
            for row in r:
                try:
                    self.present.add(row["name"] + " " + row["date"])
                except (KeyError, TypeError):
                    continue
            self.k_present.set(self._today_present_count())
            if shown < len(r):
                self.log_insert(f"Loaded {shown} recent rows ({len(r)} total in CSV).")

    def refresh_cameras(self):
        # Enumerate available cameras and populate dropdown with friendly names."
        cams = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    label = f"Camera {i} ({'Built-in' if i == 0 else 'External'}) — {w}x{h}"
                    cams.append((label, str(i)))
                cap.release()
            else:
                cap.release()
        if not cams:
            cams = [("No cameras found", "")]
        self.cam_combo["values"] = [c[0] for c in cams]
        self._cam_map = {c[0]: c[1] for c in cams}
        if cams:
            self.cam_var.set(cams[0][0])

    def open_camera(self, label):
        # Open laptop cam by label from dropdown. Returns an opened cap delivering real frames, else None.
        idx = self._cam_map.get(label, "")
        try:
            idx = int(str(idx).strip())
        except ValueError:
            return None
        for be in (getattr(cv2, "CAP_DSHOW", None),
                   getattr(cv2, "CAP_MSMF", None), None):
            try:
                cap = cv2.VideoCapture(idx, be) if be is not None else cv2.VideoCapture(idx)
            except Exception:
                continue
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    try:  # ask for SD frames: HD decode per frame is what kills fps
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency
                    except Exception:
                        pass
                    return cap
                cap.release()
        return None

    def enroll(self):
        name = self.e_name.get().strip().replace(" ", "_")
        if not name:
            messagebox.showwarning("Enroll", "Type a name first (e.g. Juan_Dela_Cruz).")
            return
        try:
            want = int(self.e_n.get())
        except ValueError:
            want = 30
        self.stop()  # free camera if a session was running
        cap = self.open_camera(self.cam_var.get())
        if cap is None:
            messagebox.showerror("Camera", "Cannot open camera (tried all drivers).\n"
                                 "Close Zoom/browser, allow camera privacy, try index 0 or phone URL.")
            return
        got, missed = [], 0
        self.prog.config(text="Look at camera… 0/%d" % want)
        self.status.config(text="Enrolling: look at the camera, turn head slightly.")
        self.root.update()
        while len(got) < want and missed < 200:
            ok, frame = cap.read()
            if not ok:
                missed += 1
                continue
            small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            boxes = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            # --- live preview in the app so you can frame your face ---
            view = small.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(view, (x, y), (x + w, y + h), (52, 211, 153), 2)
            cv2.putText(view, f"SAMPLES {len(got)}/{want}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (129, 140, 248), 2)
            self.cur = view.copy()
            self.cur_mask = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.ph = self.to_photo(view)
            self.live.config(image=self.ph)
            self.ph2 = self.to_photo(self.cur_mask)
            self.mask.config(image=self.ph2)
            if len(boxes):
                f = fc.prep_face(gray, max(boxes, key=lambda b: b[2] * b[3]))
                if f is not None:
                    got.append(f)
                    self.prog.config(text=f"Look at camera… {len(got)}/{want} (turn head slightly)")
                    self.root.update()
            else:
                missed += 1
                self.prog.config(text=f"No face seen — move into frame ({len(got)}/{want})")
                self.root.update()
            cv2.waitKey(30)
        cap.release()
        if len(got) < 10:
            messagebox.showerror("Enroll", f"Only {len(got)} faces found. Improve light / move closer.")
            self.prog.config(text="")
            return
        folder, total = fc.save_dataset(name, got)
        self.prog.config(text=f"Saved {len(got)} to {folder}")
        self.log_insert(f"Enrolled {name}: +{len(got)} samples (total {total}). Now press Train.")
        messagebox.showinfo("Enrolled", f"Saved {len(got)} faces for {name}.\nPress Train model.")

    def _today(self):
        return datetime.now().strftime("%Y-%m-%d")

    def _today_present_count(self):
        # Unique names logged TODAY (not all-time, so the card stays truthful)."
        t = self._today()
        return len({k.rsplit(" ", 1)[0] for k in self.present if k.endswith(" " + t)})

    def train(self):
        if getattr(self, "_train_thread", None) is not None and self._train_thread.is_alive():
            self.log_insert("Training already running…")
            return
        self._train_result = None
        self._train_thread = threading.Thread(target=self._train_job, daemon=True)
        self._train_thread.start()
        self.status.config(text="Training model (UI stays usable)…")
        self.log_insert("Training started in background…")
        self.root.after(500, self._poll_train)

    def _train_job(self):
        try:
            names, n = fc.train_model()
            self._train_result = ("ok", names, n)
        except Exception as e:
            self._train_result = ("err", str(e))

    def _poll_train(self):
        t = getattr(self, "_train_thread", None)
        if t is not None and t.is_alive():
            self.root.after(500, self._poll_train)
            return
        res = self._train_result or ("err", "Training failed.")
        if res[0] == "ok":
            _, names, n = res
            self.rec, self.names = fc.load_model()
            self.refresh_model_label()
            self.status.config(text=f"Model ready: {len(names)} person(s).")
            self.log_insert(f"Model trained: {names} ({n} samples).")
            messagebox.showinfo("Trained", f"Persons: {', '.join(names)}\nSamples: {n}")
        else:
            self.status.config(text="Training failed.")
            messagebox.showerror("Train", res[1])

    # ============ session ============
    def start(self):
        if self.rec is None:
            messagebox.showwarning("No model", "Enroll + Train first.")
            return
        self.stop()
        self._await_cleanup()  # previous camera must be fully released first
        cap = self.open_camera(self.cam_var.get())
        if cap is None:
            messagebox.showerror("Camera", "Cannot open camera (tried all drivers).\n"
                                 "Close Zoom/browser, allow camera privacy, try index 0 or phone URL.")
            return
        self.st = fc.new_state()
        while not self._q.empty():  # drop stale frames from previous run
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._stop_evt.clear()
        self.cap = cap  # worker owns reads; main thread never touches it
        self.running = True
        self._sess += 1
        self._worker = threading.Thread(target=self._worker_loop, args=(cap, self._sess), daemon=True)
        self._worker.start()
        self.set_live(True)
        self.status.config(text="Session running…")
        self.log_insert("Session started.")
        self.loop()

    def open_video(self):
        if self.rec is None:
            messagebox.showwarning("No model", "Enroll + Train first (video still needs a model).")
            return
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if not p:
            return
        self.stop()
        self._await_cleanup()
        self.st = fc.new_state()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._stop_evt.clear()
        cap = cv2.VideoCapture(p)
        self.cap = cap
        self.running = True
        self._sess += 1
        self._worker = threading.Thread(target=self._worker_loop, args=(cap, self._sess), daemon=True)
        self._worker.start()
        self.set_live(True)
        self.status.config(text="Testing " + os.path.basename(p))
        self.loop()

    def stop(self):
        # Instant on the UI thread: flags, feedback and log FIRST.
        # Joining/releasing the camera happens on a background thread so a
        # slow driver can never freeze the window ("Stop" must not hang).
        was = self.running
        self.running = False
        self._stop_evt.set()  # worker exits its read loop and releases the camera
        w, self._worker = self._worker, None
        cap, self.cap = self.cap, None
        self.fps = 0.0           # header shows 0 fps -> clearly idle
        self.set_live(False)
        self.status.config(text="Session stopped.")
        self._drain()
        if was:
            self.log_insert("Session stopped.")
        if w is not None or cap is not None:
            self._cleanup_t = threading.Thread(target=self._cleanup, args=(w, cap),
                                               daemon=True)
            self._cleanup_t.start()

    def _cleanup(self, w, cap):
        try:
            if w is not None and w.is_alive() and w is not threading.current_thread():
                w.join(timeout=2)  # normal case: worker exits within one frame
            if cap is not None:
                try:
                    cap.release()  # no-op if worker already released it; also
                except Exception:  # unblocks a worker stuck inside cap.read()
                    pass
            if w is not None and w.is_alive() and w is not threading.current_thread():
                w.join(timeout=1)
            self._drain()  # worker may have pushed one last frame while exiting
        except Exception:
            pass

    def _drain(self):
        # Queue is thread-safe; drop any frames still in flight (never painted:
        # loop() stops painting as soon as running=False).
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _await_cleanup(self):
        # Called before opening a new camera: the old one must be free.
        t = self._cleanup_t
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3)

    def snapshot(self):
        if hasattr(self, "cur"):
            f = f"alerts/snap_{int(time.time())}.jpg"
            os.makedirs("alerts", exist_ok=True)
            cv2.imwrite(f, self.cur)
            self.log_insert("Snapshot: " + os.path.basename(f))

    def export(self):
        if not os.path.isfile(fc.CSV_FILE):
            messagebox.showinfo("Export", "No attendance.csv yet.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            shutil.copy(fc.CSV_FILE, p)
            self.log_insert("Exported to " + os.path.basename(p))

    def open_data(self):
        try:
            os.startfile(".")
        except Exception as e:
            messagebox.showinfo("Data", str(e))

    # ============ delete data (table + CSV + enrolled faces) ============
    def _save_table_to_csv(self):
        # Rewrite attendance.csv from current table rows."
        rows = [self.table.item(i, "values") for i in self.table.get_children()]
        with open(fc.CSV_FILE, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "date", "time", "confidence"])
            w.writerows(rows)
        if not rows and os.path.isfile(fc.CSV_FILE):
            pass  # keep empty file with header so Export still works

    def _rebuild_present(self):
        # Resync Present set from table (deleted names become loggable again)."
        self.present = set()
        for i in self.table.get_children():
            v = self.table.item(i, "values")
            self.present.add(v[0] + " " + v[1])
        self.k_present.set(self._today_present_count())

    def delete_selected(self):
        sel = self.table.selection()
        if not sel:
            messagebox.showinfo("Delete", "Select row(s) in the table first.")
            return
        if not messagebox.askyesno("Delete", f"Delete {len(sel)} row(s)? CSV updates too."):
            return
        for i in sel:
            self.table.delete(i)
        self._save_table_to_csv()
        self._rebuild_present()
        self.log_insert(f"Deleted {len(sel)} attendance row(s).")

    def clear_attendance(self):
        if not self.table.get_children():
            return
        if not messagebox.askyesno("Clear", "Delete ALL attendance rows? CSV is cleared too."):
            return
        for i in self.table.get_children():
            self.table.delete(i)
        self._save_table_to_csv()
        self._rebuild_present()
        self.log_insert("Attendance cleared.")

    def delete_person(self):
        # Delete data/faces/<name>, then retrain (or drop model if empty)."
        import shutil as _sh
        if not os.path.isdir(fc.FACES_DIR):
            return
        names = sorted([d for d in os.listdir(fc.FACES_DIR)
                        if os.path.isdir(os.path.join(fc.FACES_DIR, d))])
        if not names:
            messagebox.showinfo("Delete person", "No enrolled faces found.")
            return
        win = tk.Toplevel(self.root)
        win.title("Delete enrolled person")
        win.configure(bg=PANEL)
        tk.Label(win, text="Select person to delete:", bg=PANEL, fg=TEXT, font=FONT).pack(padx=14, pady=(12, 4))
        box = tk.Listbox(win, height=min(8, len(names)), font=FONT)
        for n in names:
            box.insert("end", n)
        box.pack(padx=14, fill="x")
        box.selection_set(0)
        tk.Button(win, text="Delete", bg=RED, fg="white", relief="flat", padx=12, pady=6,
                  command=lambda: self._do_delete_person(win, box.get(box.curselection() or 0))).pack(pady=12)

    def _do_delete_person(self, win, name):
        import shutil as _sh
        win.destroy()
        if not messagebox.askyesno("Delete person", f"Delete all faces of {name} and retrain?"):
            return
        _sh.rmtree(os.path.join(fc.FACES_DIR, name), ignore_errors=True)
        rest = [d for d in os.listdir(fc.FACES_DIR)
                if os.path.isdir(os.path.join(fc.FACES_DIR, d))]
        if not rest:  # nobody left -> drop model so session forces re-enroll
            for f in (fc.MODEL_FILE, fc.LABELS_FILE):
                if os.path.isfile(f):
                    os.remove(f)
            self.rec, self.names = None, []
            self.refresh_model_label()
            self.log_insert(f"Deleted {name}. No persons left, model removed.")
        else:
            try:
                names, n = fc.train_model()
                self.rec, self.names = fc.load_model()
                self.refresh_model_label()
                self.log_insert(f"Deleted {name}. Retrained: {names} ({n} samples).")
            except RuntimeError as e:
                messagebox.showerror("Retrain", str(e))
                self.rec, self.names = fc.load_model()
                self.refresh_model_label()
        messagebox.showinfo("Deleted", f"Removed enrolled data for {name}.")

    def on_close(self):
        self.stop()
        self.root.destroy()

    def _push(self, msg):
        # Worker -> UI handoff; drops oldest so a slow UI never backlogs.
        try:
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(msg)
        except queue.Full:
            pass

    def _worker_loop(self, cap, sess):
        # Background thread: camera/video reads + ALL vision. Never touches Tk.
        n, cache = 0, None
        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._push(("eof", sess))
                    break
                # Detect on every 4th frame; reuse last result between (fps boost).
                n += 1
                if n % 4 == 0 or cache is None:
                    try:
                        out, m, infos, motion = fc.process_frame(
                            frame, self.cascade, self.rec, self.names, self.st,
                            thresh_val=self._thr)
                    except Exception as e:
                        self._push(("error", str(e), sess))
                        break
                    cache = (out, m, infos, motion)
                self._push(("frame",) + cache + (sess,))
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if self.cap is cap:
                self.cap = None

    def loop(self):
        # UI tick (main thread only): drain queue, paint latest, schedule next.
        if not self.running:
            # Stop is final: drop frames still in flight so the "Session stopped."
            # status / 0 fps are not overwritten by one last queued frame.
            self._drain()
            return
        latest, eof, err = None, False, None
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "frame":
                if len(msg) > 5 and msg[5] != self._sess:
                    continue  # stray frame from a previous session's worker
                latest = msg  # keep newest, drop older (no lag buildup)
            elif msg[0] == "eof" and (len(msg) < 2 or msg[1] == self._sess):
                eof = True  # (a stale eof from an old session must not kill this one)
            elif msg[0] == "error" and (len(msg) < 3 or msg[2] == self._sess):
                err = msg[1]
        if err is not None:
            self.log_insert("Vision error: " + err)
            self.stop()
            return
        if eof:
            self.log_insert("Video ended.")
            self.stop()
            return
        if latest is not None:
            _, out, m, infos, motion, _ = latest
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-3)
            self._t = now
            self.cur = out.copy()
            self.cur_mask = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            self.ph = self.to_photo(out)
            self.live.config(image=self.ph)
            self.ph2 = self.to_photo(self.cur_mask)
            self.mask.config(image=self.ph2)
            nf = len(infos)
            unk = sum(1 for i in infos if not i["known"])
            self.k_faces.set(nf)
            self.k_unknown.set(unk)
            self.k_motion.set("YES" if motion else "no")
            logged = False
            if self.v_save.get():
                today = datetime.now().strftime("%Y-%m-%d")
                for i in infos:
                    # One row per person per day: already present -> never logged again.
                    if i["known"] and (i["name"] + " " + today) not in self.present:
                        fc.log_row(fc.CSV_FILE, i["name"], i["conf"])
                        key = i["name"] + " " + today
                        self.present.add(key)
                        self.table.insert("", "end", values=(i["name"], datetime.now().strftime("%Y-%m-%d"),
                                                             datetime.now().strftime("%H:%M:%S"), f"{i['conf']:.1f}"))
                        self.table.see(self.table.get_children()[-1])
                        self.log_insert(f"✔ {i['name']} logged (conf {i['conf']:.0f})")
                        logged = True
            self.k_present.set(self._today_present_count())
            self.status.config(text=f"{'MOTION ' if motion else ''}Faces {nf}  Present {len(self.present)}"
                               + ("  • logged" if logged else ""))
        if self.running or not self._q.empty():
            self.root.after(50, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    App(root).root.mainloop()
