"""
Hand Teleop GUI — customtkinter interface.

Usage:
    python app.py
"""

import os
import sys
import glob
import subprocess
import threading
import shutil
import re
import tempfile

import customtkinter as ctk
from tkinter import filedialog, messagebox

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR   = os.path.join(BASE_DIR, "configs")
VIDEO_DIR     = os.path.join(BASE_DIR, "video")
ORIGINALS_DIR = os.path.join(VIDEO_DIR, "originals")
RETARGETED_DIR= os.path.join(VIDEO_DIR, "retargeted")
CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR      = os.path.join(BASE_DIR, "data")

for d in (ORIGINALS_DIR, RETARGETED_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def get_hands():
    ymls = sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.yml")))
    return [os.path.splitext(os.path.basename(y))[0] for y in ymls]


def get_originals():
    return [os.path.basename(p)
            for p in sorted(glob.glob(os.path.join(ORIGINALS_DIR, "*.mp4")))]


def get_retargeted():
    return [os.path.basename(p)
            for p in sorted(glob.glob(os.path.join(RETARGETED_DIR, "*.mp4")))]


def checkpoint_exists(hand_name):
    return os.path.isfile(os.path.join(CKPT_DIR, f"mlp_ss_{hand_name}_best.pt"))


class ProgressDialog(ctk.CTkToplevel):
    def __init__(self, parent, title: str, on_cancel):
        super().__init__(parent)
        self.title(title)
        self.geometry("380x150")
        self.resizable(False, False)
        self.grab_set()
        self._on_cancel = on_cancel
        self._closed    = False

        self._label = ctk.CTkLabel(self, text="Ініціалізація...",
                                   font=ctk.CTkFont(size=13))
        self._label.pack(pady=(20, 8))

        self._bar = ctk.CTkProgressBar(self, width=320)
        self._bar.set(0)
        self._bar.pack(pady=(0, 12))

        ctk.CTkButton(self, text="Скасувати", width=120,
                      fg_color="#c0392b", hover_color="#922b21",
                      command=self._cancel).pack()

    def set_total(self, total: int):
        self._label.configure(text=f"Кадр 0 / {total}")

    def update(self, current: int, total: int):
        if self._closed:
            return
        progress = current / total if total else 0
        self._bar.set(progress)
        pct = int(progress * 100)
        self._label.configure(text=f"Кадр {current} / {total}  ({pct}%)")

    def close(self):
        if not self._closed:
            self._closed = True
            self.destroy()

    def _cancel(self):
        self._on_cancel()
        self.close()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hand Teleop System")
        self.geometry("960x780")
        self.resizable(False, False)

        self._process           = None
        self._selected_video    = None
        self._input_video_path  = None
        self._btn_refs          = {"orig": {}, "ret": {}}

        self._build_ui()

    # ─── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sb = ctk.CTkFrame(self, width=250, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)

        ctk.CTkLabel(sb, text="Hand Teleop",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 2))
        ctk.CTkLabel(sb, text="Visual Teleoperation System",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(0, 18))

        # Hand
        ctk.CTkLabel(sb, text="Роботична кисть", anchor="w").pack(fill="x", padx=16)
        hands = get_hands()
        self._hand_var = ctk.StringVar(value=hands[0] if hands else "")
        self._hand_menu = ctk.CTkOptionMenu(
            sb, variable=self._hand_var, values=hands,
            command=self._on_hand_changed)
        self._hand_menu.pack(fill="x", padx=16, pady=(4, 14))

        # Mode
        ctk.CTkLabel(sb, text="Режим", anchor="w").pack(fill="x", padx=16)
        self._mode = ctk.StringVar(value="camera")
        ctk.CTkRadioButton(sb, text="Камера (реальний час)",
                           variable=self._mode, value="camera",
                           command=self._on_mode_changed).pack(anchor="w", padx=24, pady=2)
        ctk.CTkRadioButton(sb, text="Відео з галереї",
                           variable=self._mode, value="video",
                           command=self._on_mode_changed).pack(anchor="w", padx=24, pady=2)

        # Selected video hint (shown only in video mode)
        self._vpick_frame = ctk.CTkFrame(sb, fg_color="transparent")
        ctk.CTkLabel(self._vpick_frame, text="Вхідне відео", anchor="w").pack(fill="x")
        self._vpick_label = ctk.CTkLabel(
            self._vpick_frame, text="не обрано — оберіть з галереї",
            font=ctk.CTkFont(size=10), text_color="gray",
            wraplength=210, anchor="w")
        self._vpick_label.pack(fill="x")

        # Camera position (video mode only)
        self._cam_distance = 0.7
        self._cam_yaw      = 45.0
        self._cam_pitch    = -30.0

        self._cam_frame = ctk.CTkFrame(sb, fg_color="transparent")
        ctk.CTkLabel(self._cam_frame, text="Позиція камери",
                     anchor="w").pack(fill="x", padx=16, pady=(14, 2))
        self._cam_info = ctk.CTkLabel(
            self._cam_frame,
            text=self._cam_text(),
            font=ctk.CTkFont(size=10), text_color="gray", anchor="w")
        self._cam_info.pack(fill="x", padx=16)
        ctk.CTkButton(self._cam_frame, text="🎥  Налаштувати ракурс",
                      height=30, command=self._open_camera_preview
                      ).pack(fill="x", padx=16, pady=(6, 0))

        # Smoothness
        ctk.CTkFrame(sb, height=1, fg_color="gray30").pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(sb, text="Плавність", anchor="w").pack(fill="x", padx=16)

        self._cutoff_var = ctk.DoubleVar(value=0.3)
        self._beta_var   = ctk.DoubleVar(value=0.02)

        for label, var, lo, hi in [
            ("Згладжування", self._cutoff_var, 0.05, 2.0),
            ("Реакція",      self._beta_var,   0.0,  0.3),
        ]:
            row = ctk.CTkFrame(sb, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=1)
            ctk.CTkLabel(row, text=label, width=100, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(row, text=f"{var.get():.2f}", width=42)
            lbl.pack(side="right")
            ctk.CTkSlider(
                row, from_=lo, to=hi, variable=var,
                command=lambda v, l=lbl: l.configure(text=f"{float(v):.2f}")
            ).pack(side="left", fill="x", expand=True, padx=4)

        # Buttons
        ctk.CTkFrame(sb, height=1, fg_color="gray30").pack(fill="x", padx=16, pady=12)

        self._start_btn = ctk.CTkButton(sb, text="▶  Запустити",
                                        height=36, command=self._start)
        self._start_btn.pack(fill="x", padx=16, pady=3)

        self._stop_btn = ctk.CTkButton(
            sb, text="⏹  Зупинити", height=36,
            fg_color="#c0392b", hover_color="#922b21",
            command=self._stop, state="disabled")
        self._stop_btn.pack(fill="x", padx=16, pady=3)

        ctk.CTkButton(sb, text="＋  Додати нову руку",
                      height=34, fg_color="gray30", hover_color="gray40",
                      command=self._add_hand).pack(fill="x", padx=16, pady=(10, 4))

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(self, corner_radius=0, fg_color="gray14")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=1)
        right.grid_rowconfigure(4, weight=0)
        right.grid_columnconfigure(0, weight=1)

        # Originals section header
        orig_hdr = ctk.CTkFrame(right, fg_color="transparent")
        orig_hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        ctk.CTkLabel(orig_hdr, text="Галерея відео",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(orig_hdr, text="↺ Оновити", width=90, height=28,
                      fg_color="gray30", hover_color="gray40",
                      command=self._refresh_lists).pack(side="right", padx=(6, 0))
        ctk.CTkButton(orig_hdr, text="＋ Додати", width=90, height=28,
                      command=self._add_to_gallery).pack(side="right")

        self._orig_list = ctk.CTkScrollableFrame(right, height=140)
        self._orig_list.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 6))

        # Retargeted section
        ctk.CTkLabel(right, text="Ретаргетовані відео",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=16, pady=(4, 4))

        self._ret_list = ctk.CTkScrollableFrame(right, height=140)
        self._ret_list.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 6))

        ctk.CTkLabel(right, text="2× клік — відкрити   |   ПКМ — видалити",
                     font=ctk.CTkFont(size=10), text_color="gray40").grid(
            row=4, column=0, sticky="w", padx=16, pady=(0, 4))

        # Status bar
        self._status_var = ctk.StringVar(value="Готово")
        ctk.CTkLabel(right, textvariable=self._status_var,
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=5, column=0, sticky="w", padx=16, pady=(0, 8))

        self._selected      = {"orig": None, "ret": None}
        self._last_selected = None  # ("orig"|"ret", name)
        self._on_mode_changed()
        self._refresh_lists()
        if self._hand_var.get():
            self._on_hand_changed(self._hand_var.get())

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        if self._mode.get() == "video":
            self._vpick_frame.pack(fill="x", padx=16, pady=(10, 0))
            self._cam_frame.pack(fill="x")
        else:
            self._vpick_frame.pack_forget()
            self._cam_frame.pack_forget()
            self._input_video_path = None
            self._vpick_label.configure(text="не обрано — оберіть з галереї")

    def _on_hand_changed(self, name):
        if checkpoint_exists(name):
            self._status_var.set(f"✓  {name} готовий до запуску")
        else:
            self._status_var.set(f"⚠  Checkpoint для '{name}' не знайдено")

    def _cam_text(self):
        return (f"dist={self._cam_distance:.2f}  "
                f"yaw={self._cam_yaw:.1f}°  "
                f"pitch={self._cam_pitch:.1f}°")

    def _open_camera_preview(self):
        hand = self._hand_var.get()
        if not hand:
            messagebox.showerror("Помилка", "Спочатку оберіть руку")
            return
        config = os.path.join(CONFIGS_DIR, f"{hand}.yml")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.close()

        cmd = [sys.executable,
               os.path.join(BASE_DIR, "camera_preview.py"),
               "--config",       config,
               "--cam-distance", str(self._cam_distance),
               "--cam-yaw",      str(self._cam_yaw),
               "--cam-pitch",    str(self._cam_pitch),
               "--out",          tmp.name]

        self._status_var.set("Виставте ракурс у PyBullet, потім натисніть S")
        proc = subprocess.Popen(cmd, cwd=BASE_DIR)

        def wait():
            proc.wait()
            self.after(0, lambda: self._apply_camera_params(tmp.name))

        threading.Thread(target=wait, daemon=True).start()

    def _apply_camera_params(self, tmp_path: str):
        try:
            content = open(tmp_path).read().strip()
            os.unlink(tmp_path)
            if content:
                parts = content.split()
                self._cam_distance = float(parts[0])
                self._cam_yaw      = float(parts[1])
                self._cam_pitch    = float(parts[2])
                self._cam_info.configure(text=self._cam_text())
                self._status_var.set("Ракурс збережено ✓")
            else:
                self._status_var.set("Ракурс не збережено")
        except Exception:
            self._status_var.set("Ракурс не збережено")

    def _add_to_gallery(self):
        paths = filedialog.askopenfilenames(
            title="Оберіть відео для галереї",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        for path in paths:
            dst = os.path.join(ORIGINALS_DIR, os.path.basename(path))
            if not os.path.exists(dst):
                shutil.copy2(path, dst)
        if paths:
            self._refresh_lists()
            self._status_var.set(f"Додано {len(paths)} відео до галереї")

    def _refresh_lists(self):
        self._populate_list(self._orig_list, get_originals(), "orig")
        self._populate_list(self._ret_list,  get_retargeted(), "ret")

    def _populate_list(self, frame, items, key):
        for w in frame.winfo_children():
            w.destroy()
        if not items:
            ctk.CTkLabel(frame, text="Порожньо", text_color="gray").pack(pady=10)
            return
        self._btn_refs[key] = {}
        for name in items:
            is_sel  = self._selected.get(key) == name
            bg      = "#1f538d" if is_sel else "#2b2b2b"
            row = ctk.CTkFrame(frame, fg_color=bg, corner_radius=6, height=32)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)
            lbl = ctk.CTkLabel(row, text=name, anchor="w",
                               fg_color="transparent",
                               font=ctk.CTkFont(size=12))
            lbl.pack(fill="both", expand=True, padx=10)

            for widget in (row, lbl):
                widget.bind("<Button-1>",
                            lambda e, n=name, k=key: self._select(n, k))
                widget.bind("<Double-Button-1>",
                            lambda e, n=name, k=key: self._open_video(n, k))
                widget.bind("<Button-3>",
                            lambda e, n=name, k=key: self._ctx_menu(e, n, k))
                widget.bind("<Enter>",
                            lambda e, r=row, n=name, k=key: r.configure(
                                fg_color="#2563a8" if self._selected.get(k)==n else "#3a3a3a"))
                widget.bind("<Leave>",
                            lambda e, r=row, n=name, k=key: r.configure(
                                fg_color="#1f538d" if self._selected.get(k)==n else "#2b2b2b"))

            self._btn_refs[key][name] = row

    def _select(self, name, key):
        # deselect other list
        other = "ret" if key == "orig" else "orig"
        prev_other = self._selected.get(other)
        if prev_other and prev_other in self._btn_refs.get(other, {}):
            self._btn_refs[other][prev_other].configure(fg_color="#2b2b2b")
        self._selected[other] = None

        # deselect previous in same list
        prev = self._selected.get(key)
        if prev and prev in self._btn_refs.get(key, {}):
            self._btn_refs[key][prev].configure(fg_color="#2b2b2b")

        self._selected[key]  = name
        self._last_selected  = (key, name)

        if name in self._btn_refs.get(key, {}):
            self._btn_refs[key][name].configure(fg_color="#1f538d")

        if key == "orig" and self._mode.get() == "video":
            self._input_video_path = os.path.join(ORIGINALS_DIR, name)
            self._vpick_label.configure(text=name)

    def _open_video(self, name, key):
        folder = ORIGINALS_DIR if key == "orig" else RETARGETED_DIR
        os.startfile(os.path.join(folder, name))

    def _ctx_menu(self, event, name, key):
        from tkinter import Menu
        m = Menu(self, tearoff=0)
        m.add_command(label=f"Видалити  '{name}'",
                      command=lambda: self._delete_item(name, key))
        m.tk_popup(event.x_root, event.y_root)

    def _delete_item(self, name, key):
        folder = ORIGINALS_DIR if key == "orig" else RETARGETED_DIR
        if not messagebox.askyesno("Видалення", f"Видалити '{name}'?"):
            return
        os.remove(os.path.join(folder, name))
        if self._last_selected == (key, name):
            self._last_selected = None
        self._selected[key] = None
        self._refresh_lists()

    # ─── Start / Stop ─────────────────────────────────────────────────────────

    def _start(self):
        hand = self._hand_var.get()
        if not hand:
            messagebox.showerror("Помилка", "Оберіть руку")
            return
        if not checkpoint_exists(hand):
            messagebox.showerror("Помилка",
                f"Checkpoint для '{hand}' не знайдено.\n"
                "Спочатку навчіть мережу через 'Додати нову руку'.")
            return

        config  = os.path.join(CONFIGS_DIR, f"{hand}.yml")
        dist    = str(self._cam_distance)
        yaw     = str(self._cam_yaw)
        pitch   = str(self._cam_pitch)
        cutoff  = str(self._cutoff_var.get())
        beta    = str(self._beta_var.get())

        if self._mode.get() == "camera":
            cmd = [sys.executable, os.path.join(BASE_DIR, "main.py"),
                   "--config",       config,
                   "--cam-distance", dist,
                   "--cam-yaw",      yaw,
                   "--cam-pitch",    pitch,
                   "--min-cutoff",   cutoff,
                   "--beta",         beta]
        else:
            if not self._input_video_path:
                messagebox.showerror("Помилка",
                    "Оберіть відео з галереї (клік на відео в списку 'Галерея відео')")
                return
            base = os.path.splitext(os.path.basename(self._input_video_path))[0]
            out  = os.path.join(RETARGETED_DIR, f"{base}_{hand}.mp4")
            cmd = [sys.executable, os.path.join(BASE_DIR, "video_retarget.py"),
                   "--input",        self._input_video_path,
                   "--config",       config,
                   "--out",          out,
                   "--cam-distance", dist,
                   "--cam-yaw",      yaw,
                   "--cam-pitch",    pitch,
                   "--min-cutoff",   cutoff,
                   "--beta",         beta]

        if self._mode.get() == "video":
            cmd = [cmd[0], "-u"] + cmd[1:]
            self._process = subprocess.Popen(
                cmd, cwd=BASE_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1)
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._status_var.set("Обробка відео...")
            dlg = ProgressDialog(self, title="Обробка відео",
                                 on_cancel=self._stop)
            threading.Thread(target=self._watch_video_progress,
                             args=(dlg,), daemon=True).start()
        else:
            self._process = subprocess.Popen(cmd, cwd=BASE_DIR)
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._status_var.set("Запущено...")
            threading.Thread(target=self._wait_process, daemon=True).start()

    def _watch_video_progress(self, dlg: "ProgressDialog"):
        total = 0
        for line in self._process.stdout:
            line = line.strip()
            m_total = re.search(r"Processing (\d+) frames", line)
            if m_total:
                total = int(m_total.group(1))
                self.after(0, lambda t=total: dlg.set_total(t))
            m_frame = re.search(r"frame (\d+)/(\d+)", line)
            if m_frame:
                cur = int(m_frame.group(1))
                tot = int(m_frame.group(2))
                self.after(0, lambda c=cur, t=tot: dlg.update(c, t))
        self._process.wait()
        self.after(0, dlg.close)
        self.after(0, self._on_done)

    def _wait_process(self):
        if self._process:
            self._process.wait()
        self.after(0, self._on_done)

    def _on_done(self):
        self._process = None
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._status_var.set("Завершено")
        self._refresh_lists()

    def _stop(self):
        if self._process:
            self._process.terminate()

    # ─── Add hand ─────────────────────────────────────────────────────────────

    def _add_hand(self):
        yml = filedialog.askopenfilename(
            title="Оберіть конфіг руки (.yml)",
            filetypes=[("YAML", "*.yml *.yaml"), ("All", "*.*")])
        if not yml:
            return

        assets_src = filedialog.askdirectory(title="Оберіть папку assets руки")
        if not assets_src:
            return

        hand_name  = os.path.splitext(os.path.basename(yml))[0]
        dst_yml    = os.path.join(CONFIGS_DIR, os.path.basename(yml))
        dst_assets = os.path.join(BASE_DIR, "assets", os.path.basename(assets_src))

        if not os.path.exists(dst_yml):
            shutil.copy2(yml, dst_yml)
        if not os.path.exists(dst_assets):
            shutil.copytree(assets_src, dst_assets)

        hands = get_hands()
        self._hand_menu.configure(values=hands)
        self._hand_var.set(hand_name)

        if messagebox.askyesno("Навчання", f"Навчити мережу для '{hand_name}'?"):
            self._train(hand_name, dst_yml)

    def _train(self, hand_name, config_path):
        data_files = sorted(glob.glob(os.path.join(DATA_DIR, "kps_*.npz")))
        if not data_files:
            messagebox.showwarning(
                "Дані відсутні",
                f"Файли kps_*.npz не знайдено в {DATA_DIR}.\n"
                "Спочатку зберіть дані через collect_data.py")
            return

        cmd = ([sys.executable,
                os.path.join(BASE_DIR, "mlp_selfsupervised", "train.py"),
                "--config", config_path, "--data"] + data_files)

        self._process = subprocess.Popen(cmd, cwd=BASE_DIR)
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._status_var.set(f"Навчання '{hand_name}'...")
        threading.Thread(target=self._wait_process, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
