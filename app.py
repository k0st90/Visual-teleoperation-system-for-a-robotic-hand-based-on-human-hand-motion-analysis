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
from database import run_migrations
from database.repositories import camera_settings as cam_repo
from database.repositories import hands as hands_repo
from database.repositories import models as models_repo
from database.repositories import videos as videos_repo
from database.repositories import usage_sessions as sessions_repo
from database.repositories import training_epochs as epochs_repo

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR   = os.path.join(BASE_DIR, "configs")
VIDEO_DIR     = os.path.join(BASE_DIR, "video")
ORIGINALS_DIR = os.path.join(VIDEO_DIR, "originals")
RETARGETED_DIR= os.path.join(VIDEO_DIR, "retargeted")
CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR      = os.path.join(BASE_DIR, "data")

for d in (RETARGETED_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def get_hands():
    try:
        return [row["name"] for row in hands_repo.get_all()]
    except Exception:
        return []


def get_originals():
    try:
        return [row["filename"] for row in videos_repo.get_all_originals()]
    except Exception:
        return []


def get_retargeted():
    try:
        return [row["filename"] for row in videos_repo.get_all_retargeted()]
    except Exception:
        return []


def checkpoint_exists(hand_name):
    try:
        row = models_repo.get_latest(hand_name)
        return row is not None and os.path.isfile(row["checkpoint_path"])
    except Exception:
        return False


class ProgressDialog(ctk.CTkToplevel):
    def __init__(self, parent, title: str, on_cancel):
        super().__init__(parent)
        self.title(title)
        self.geometry("380x150")
        self.resizable(False, False)
        self.after(200, self.grab_set)
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
            self.after(150, self.destroy)

    def _cancel(self):
        self._on_cancel()
        self.close()


class TrainingProgressDialog(ctk.CTkToplevel):
    def __init__(self, parent, hand_name: str, on_cancel):
        super().__init__(parent)
        self.title(f"Навчання — {hand_name}")
        self.geometry("420x200")
        self.resizable(False, False)
        self.after(200, self.grab_set)
        self._on_cancel = on_cancel
        self._closed    = False

        self._label = ctk.CTkLabel(self, text="Ініціалізація...",
                                   font=ctk.CTkFont(size=13))
        self._label.pack(pady=(20, 6))

        self._bar = ctk.CTkProgressBar(self, width=360)
        self._bar.set(0)
        self._bar.pack(pady=(0, 6))

        self._loss_label = ctk.CTkLabel(self, text="",
                                        font=ctk.CTkFont(size=11),
                                        text_color="gray")
        self._loss_label.pack(pady=(0, 12))

        ctk.CTkButton(self, text="Скасувати", width=120,
                      fg_color="#c0392b", hover_color="#922b21",
                      command=self._cancel).pack()

    def update(self, epoch: int, total: int, train_loss: float, val_loss: float):
        if self._closed:
            return
        self._bar.set(epoch / total)
        self._label.configure(text=f"Епоха {epoch} / {total}  ({int(epoch/total*100)}%)")
        self._loss_label.configure(text=f"train={train_loss:.5f}   val={val_loss:.5f}")

    def close(self):
        if not self._closed:
            self._closed = True
            self.after(150, self.destroy)

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

        try:
            run_migrations()
        except Exception as e:
            print(f"DB init warning: {e}")

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

        ctk.CTkButton(sb, text="📈  Графіки навчання",
                      height=34, fg_color="gray30", hover_color="gray40",
                      command=self._show_training_charts).pack(fill="x", padx=16, pady=(2, 4))

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
        if not name:
            return
        try:
            ok, err = hands_repo.validate_paths(name)
            if not ok:
                self._status_var.set(f"⚠  Зламаний шлях: {err}")
                self._show_broken_path_dialog(name)
                return
        except Exception:
            pass
        if checkpoint_exists(name):
            self._status_var.set(f"✓  {name} готовий до запуску")
        else:
            self._status_var.set(f"⚠  Checkpoint для '{name}' не знайдено")
        try:
            row = cam_repo.get(name)
            if row:
                self._cam_distance = row["cam_distance"]
                self._cam_yaw      = row["cam_yaw"]
                self._cam_pitch    = row["cam_pitch"]
                self._cam_info.configure(text=self._cam_text())
        except Exception:
            pass

    def _show_training_charts(self):
        import tempfile, webbrowser
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            messagebox.showerror("Помилка", "Встанови plotly: pip install plotly")
            return

        hand = self._hand_var.get()
        if not hand:
            messagebox.showerror("Помилка", "Спочатку оберіть руку")
            return

        runs = models_repo.get_all_for_hand(hand)
        if not runs:
            messagebox.showinfo("Немає даних", f"Для '{hand}' немає навчених моделей")
            return

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Train Loss vs Val Loss",
                "Links Vec Loss",
                "Joint Pos Loss",
                "Learning Rate",
                "Epoch Time (s)"),
            vertical_spacing=0.06)

        colors = ["#4da6ff", "#ff6b6b", "#51cf66", "#ffd43b",
                  "#cc5de8", "#ff922b", "#20c997", "#f06595"]

        for i, run in enumerate(runs):
            epochs = epochs_repo.get_for_run(run["run_id"])
            if not epochs:
                continue

            color     = colors[i % len(colors)]
            run_label = run["run_id"]
            xs        = [e["epoch"]           for e in epochs]
            train_l   = [e["train_loss"]       for e in epochs]
            val_l     = [e["val_loss"]         for e in epochs]
            links_l   = [e["links_vec_loss"]   for e in epochs]
            jpos_l    = [e["joint_pos_loss"]   for e in epochs]
            lr_l      = [e["lr"]               for e in epochs]
            time_l    = [e["epoch_time_sec"]   for e in epochs]
            best_xs   = [e["epoch"]    for e in epochs if e["is_best"]]
            best_vals = [e["val_loss"] for e in epochs if e["is_best"]]

            hover = [
                f"<b>Epoch {e['epoch']}</b><br>"
                f"train: {e['train_loss']:.5f}<br>"
                f"val: {e['val_loss']:.5f}<br>"
                f"links: {e['links_vec_loss']:.5f}<br>"
                f"jpos: {e['joint_pos_loss']:.5f}<br>"
                f"lr: {e['lr']:.2e}<br>"
                f"time: {e['epoch_time_sec']:.1f}s<br>"
                f"best: {'✓' if e['is_best'] else '—'}"
                for e in epochs]

            kw_hover = dict(hovertemplate="%{text}<extra></extra>", text=hover)

            fig.add_trace(go.Scatter(x=xs, y=train_l,
                name=f"{run_label} train",
                line=dict(color=color, width=1.5, dash="dot"),
                **kw_hover), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=val_l,
                name=f"{run_label} val",
                line=dict(color=color, width=2),
                **kw_hover), row=1, col=1)
            fig.add_trace(go.Scatter(x=best_xs, y=best_vals,
                name="best", mode="markers",
                marker=dict(color="white", size=9, symbol="star",
                            line=dict(color=color, width=1.5)),
                hoverinfo="skip", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=links_l,
                name=f"{run_label} links",
                line=dict(color=color, width=2),
                showlegend=False, **kw_hover), row=2, col=1)
            fig.add_trace(go.Scatter(x=xs, y=jpos_l,
                name=f"{run_label} jpos",
                line=dict(color=color, width=2),
                showlegend=False, **kw_hover), row=3, col=1)
            fig.add_trace(go.Scatter(x=xs, y=lr_l,
                name=f"{run_label} lr",
                line=dict(color=color, width=2),
                showlegend=False, **kw_hover), row=4, col=1)
            fig.add_trace(go.Scatter(x=xs, y=time_l,
                name=f"{run_label} time",
                line=dict(color=color, width=2),
                showlegend=False, **kw_hover), row=5, col=1)

        fig.update_layout(
            title=dict(text=f"Навчання — {hand}", x=0.5, xanchor="center",
                       font=dict(size=18)),
            template="plotly_dark",
            height=1100,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.04,
                        xanchor="center", x=0.5),
            margin=dict(t=80, b=120))
        fig.update_xaxes(title_text="Епоха", row=5, col=1)

        html = fig.to_html(full_html=True, include_plotlyjs=True)
        dark_html = html.replace(
            "<body>",
            "<body style='background-color:#1a1a2e;margin:0;padding:16px;box-sizing:border-box;'>"
        ).replace(
            "<div>",
            "<div style='width:100%;'>", 1
        )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                          mode="w", encoding="utf-8")
        tmp.write(dark_html)
        tmp.close()
        webbrowser.open(f"file:///{tmp.name}")

    def _cam_text(self):
        return (f"dist={self._cam_distance:.2f}  "
                f"yaw={self._cam_yaw:.1f}°  "
                f"pitch={self._cam_pitch:.1f}°")

    def _open_camera_preview(self):
        hand = self._hand_var.get()
        if not hand:
            messagebox.showerror("Помилка", "Спочатку оберіть руку")
            return
        row = hands_repo.get_by_name(hand)
        if not row:
            messagebox.showerror("Помилка", "Руку не знайдено в БД")
            return
        config      = row["yml_path"]
        assets_path = row["assets_path"]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.close()

        cmd = [sys.executable,
               os.path.join(BASE_DIR, "camera_preview.py"),
               "--config",       config,
               "--assets-path",  assets_path,
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
                try:
                    cam_repo.save(
                        self._hand_var.get(),
                        self._cam_distance,
                        self._cam_yaw,
                        self._cam_pitch)
                    print(f"[db] camera saved for {self._hand_var.get()}")
                except Exception as e:
                    print(f"[db] camera save error: {e}")
            else:
                self._status_var.set("Ракурс не збережено")
        except Exception:
            self._status_var.set("Ракурс не збережено")

    @staticmethod
    def _video_duration(path: str) -> float | None:
        try:
            cap = cv2.VideoCapture(path)
            fps    = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return round(frames / fps, 2) if fps > 0 else None
        except Exception:
            return None

    def _add_to_gallery(self):
        paths = filedialog.askopenfilenames(
            title="Оберіть відео для галереї",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        for path in paths:
            abs_path = os.path.abspath(path)
            try:
                videos_repo.add_original(
                    filename=os.path.basename(abs_path),
                    full_path=abs_path,
                    duration_sec=self._video_duration(abs_path))
            except Exception as e:
                print(f"[db] add original error: {e}")
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
            try:
                row = videos_repo.get_original_by_filename(name)
                self._input_video_path = row["full_path"] if row else None
            except Exception:
                self._input_video_path = None
            self._vpick_label.configure(text=name)

    def _open_video(self, name, key):
        try:
            if key == "orig":
                row = videos_repo.get_original_by_filename(name)
            else:
                rows = videos_repo.get_all_retargeted()
                row = next((r for r in rows if r["filename"] == name), None)
            if row and os.path.exists(row["full_path"]):
                os.startfile(row["full_path"])
                return
        except Exception:
            pass
        folder = ORIGINALS_DIR if key == "orig" else RETARGETED_DIR
        os.startfile(os.path.join(folder, name))

    def _ctx_menu(self, event, name, key):
        from tkinter import Menu
        m = Menu(self, tearoff=0)
        m.add_command(label=f"Видалити  '{name}'",
                      command=lambda: self._delete_item(name, key))
        m.tk_popup(event.x_root, event.y_root)

    def _delete_item(self, name, key):
        if not messagebox.askyesno("Видалення", f"Видалити '{name}'?"):
            return
        try:
            if key == "orig":
                videos_repo.delete_original(name)
            else:
                path = os.path.join(RETARGETED_DIR, name)
                if os.path.exists(path):
                    os.remove(path)
                videos_repo.delete_retargeted(name)
        except Exception as e:
            print(f"[db] delete error: {e}")
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

        row         = hands_repo.get_by_name(hand)
        config      = row["yml_path"]      if row else os.path.join(CONFIGS_DIR, f"{hand}.yml")
        assets_path = row["assets_path"]   if row else None

        model_row = models_repo.get_latest(hand)
        ckpt = model_row["checkpoint_path"] if model_row else None
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
            if ckpt:
                cmd += ["--checkpoint", ckpt]
            if assets_path:
                cmd += ["--assets-path", assets_path]
        else:
            if not self._input_video_path:
                messagebox.showerror("Помилка",
                    "Оберіть відео з галереї (клік на відео в списку 'Галерея відео')")
                return
            import time as _time
            base = os.path.splitext(os.path.basename(self._input_video_path))[0]
            ts   = _time.strftime("%Y%m%d_%H%M%S")
            out  = os.path.join(RETARGETED_DIR, f"{base}_{hand}_{ts}.mp4")
            self._last_retarget_out = out
            cmd = [sys.executable, os.path.join(BASE_DIR, "video_retarget.py"),
                   "--input",        self._input_video_path,
                   "--config",       config,
                   "--out",          out,
                   "--cam-distance", dist,
                   "--cam-yaw",      yaw,
                   "--cam-pitch",    pitch,
                   "--min-cutoff",   cutoff,
                   "--beta",         beta]
            if ckpt:
                cmd += ["--checkpoint", ckpt]
            if assets_path:
                cmd += ["--assets-path", assets_path]

        if self._mode.get() == "video":
            cmd = [cmd[0], "-u"] + cmd[1:]
            self._process = subprocess.Popen(
                cmd, cwd=BASE_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1)
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._status_var.set("Обробка відео...")
            self._current_session_id = None
            dlg = ProgressDialog(self, title="Обробка відео",
                                 on_cancel=self._stop)
            threading.Thread(target=self._watch_video_progress,
                             args=(dlg,), daemon=True).start()
        else:
            self._process = subprocess.Popen(cmd, cwd=BASE_DIR)
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._status_var.set("Запущено...")
            try:
                self._current_session_id = sessions_repo.start(
                    hand, self._cutoff_var.get(), self._beta_var.get())
            except Exception:
                self._current_session_id = None
            threading.Thread(target=self._wait_process, daemon=True).start()

    def _watch_video_progress(self, dlg: "ProgressDialog"):
        total = 0
        last_frame = 0
        for line in self._process.stdout:
            line = line.strip()
            m_total = re.search(r"Processing (\d+) frames", line)
            if m_total:
                total = int(m_total.group(1))
                self.after(0, lambda t=total: dlg.set_total(t))
            m_frame = re.search(r"frame (\d+)/(\d+)", line)
            if m_frame:
                last_frame = int(m_frame.group(1))
                tot = int(m_frame.group(2))
                self.after(0, lambda c=last_frame, t=tot: dlg.update(c, t))
        self._process.wait()
        self._last_frames = last_frame or total
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
        try:
            sid = getattr(self, "_current_session_id", None)
            if sid:
                sessions_repo.finish(sid)
                self._current_session_id = None
        except Exception:
            pass

        if getattr(self, "_last_retarget_out", None):
            out = self._last_retarget_out
            try:
                model_row = models_repo.get_latest(self._hand_var.get())
                videos_repo.add_retargeted(
                    filename=os.path.basename(out),
                    full_path=out,
                    original_filename=os.path.basename(
                        getattr(self, "_input_video_path", "") or ""),
                    hand_name=self._hand_var.get(),
                    min_cutoff=self._cutoff_var.get(),
                    beta=self._beta_var.get(),
                    cam_distance=self._cam_distance,
                    cam_yaw=self._cam_yaw,
                    cam_pitch=self._cam_pitch,
                    model_id=model_row["id"] if model_row else None)
            except Exception:
                pass
            self._last_retarget_out = None

        self._refresh_lists()

    def _stop(self):
        if self._process:
            self._process.terminate()

    # ─── Add hand ─────────────────────────────────────────────────────────────

    def _show_broken_path_dialog(self, name: str):
        answer = messagebox.askquestion(
            "Зламаний шлях",
            f"Файли для '{name}' не знайдено.\n\nОновити шлях чи видалити руку?",
            type=messagebox.YESNOCANCEL,
            icon=messagebox.WARNING,
            detail="Так — оновити шлях\nНі — видалити руку\nСкасувати — нічого не робити"
        )
        if answer == "yes":
            self._update_hand_paths(name)
        elif answer == "no":
            if messagebox.askyesno("Підтвердження", f"Видалити '{name}' з БД?"):
                hands_repo.delete(name)
                hands = get_hands()
                self._hand_menu.configure(values=hands)
                self._hand_var.set(hands[0] if hands else "")

    def _update_hand_paths(self, name: str):
        yml = filedialog.askopenfilename(
            title=f"Новий конфіг для {name}",
            filetypes=[("YAML", "*.yml *.yaml"), ("All", "*.*")])
        if not yml:
            return
        assets = filedialog.askdirectory(title=f"Нова папка assets для {name}")
        if not assets:
            return
        hands_repo.update_paths(name, os.path.abspath(yml), os.path.abspath(assets))
        self._status_var.set(f"✓  Шлях оновлено для '{name}'")
        self._on_hand_changed(name)

    def _add_hand(self):
        yml = filedialog.askopenfilename(
            title="Оберіть конфіг руки (.yml)",
            filetypes=[("YAML", "*.yml *.yaml"), ("All", "*.*")])
        if not yml:
            return

        assets_src = filedialog.askdirectory(title="Оберіть папку assets руки")
        if not assets_src:
            return

        hand_name   = os.path.splitext(os.path.basename(yml))[0]
        yml_abs     = os.path.abspath(yml)
        assets_abs  = os.path.abspath(assets_src)

        try:
            hands_repo.get_or_create(hand_name, yml_abs, assets_abs)
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалось зареєструвати руку: {e}")
            return

        hands = get_hands()
        self._hand_menu.configure(values=hands)
        self._hand_var.set(hand_name)

        row = hands_repo.get_by_name(hand_name)
        self._train(hand_name, row["yml_path"])

    def _train(self, hand_name, config_path):
        data_files = sorted(glob.glob(os.path.join(DATA_DIR, "kps_*.npz")))
        if not data_files:
            messagebox.showwarning(
                "Дані відсутні",
                f"Файли kps_*.npz не знайдено в {DATA_DIR}.\n"
                "Спочатку зберіть дані через collect_data.py")
            return

        import time as _time
        run_id = _time.strftime("%Y%m%d_%H%M%S")
        self._current_run_id = run_id
        hand_row   = hands_repo.get_by_name(hand_name)
        assets_path = hand_row["assets_path"] if hand_row else None

        cmd = [sys.executable, "-u",
               os.path.join(BASE_DIR, "mlp_selfsupervised", "train.py"),
               "--config", config_path,
               "--run-id", run_id,
               "--data"] + data_files
        if assets_path:
            cmd += ["--assets-path", assets_path]

        self._process = subprocess.Popen(
            cmd, cwd=BASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1)

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._status_var.set(f"Навчання '{hand_name}'...")

        dlg = TrainingProgressDialog(self, hand_name, on_cancel=self._stop)
        threading.Thread(target=self._watch_training,
                         args=(dlg,), daemon=True).start()

    def _watch_training(self, dlg: "TrainingProgressDialog"):
        hand      = self._hand_var.get()
        run_id    = getattr(self, "_current_run_id", None)
        ckpt_path = None

        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            print(f"[train] {line}")
            m = re.search(
                r"Epoch\s+(\d+)/(\d+)\s+train=([\d.]+)\s+val=([\d.]+)\s+"
                r"links=([\d.]+)\s+jpos=([\d.]+)\s+lr=([\d.e+-]+)\s+"
                r"time=([\d.]+)\s+best=(\w+)", line)
            if m:
                epoch      = int(m.group(1))
                total      = int(m.group(2))
                train_loss = float(m.group(3))
                val_loss   = float(m.group(4))
                links      = float(m.group(5))
                jpos       = float(m.group(6))
                lr         = float(m.group(7))
                etime      = float(m.group(8))
                is_best    = m.group(9) == "yes"
                self.after(0, lambda e=epoch, t=total, tr=train_loss, v=val_loss:
                           dlg.update(e, t, tr, v))
                if run_id:
                    try:
                        epochs_repo.save(
                            run_id=run_id, epoch=epoch,
                            train_loss=train_loss, val_loss=val_loss,
                            links_vec_loss=links, joint_pos_loss=jpos,
                            lr=lr, epoch_time_sec=etime, is_best=is_best)
                    except Exception as e:
                        print(f"[db] epoch save error: {e}")
            m2 = re.search(r"Best checkpoint: (.+)", line)
            if m2:
                ckpt_path = m2.group(1).strip()

        self._process.wait()

        if ckpt_path and run_id and hand:
            try:
                abs_ckpt = os.path.join(BASE_DIR, ckpt_path)
                models_repo.save(hand, run_id, abs_ckpt)
            except Exception as e:
                print(f"DB save model warning: {e}")

        self.after(0, dlg.close)
        self.after(0, self._on_done)


if __name__ == "__main__":
    app = App()
    app.mainloop()
