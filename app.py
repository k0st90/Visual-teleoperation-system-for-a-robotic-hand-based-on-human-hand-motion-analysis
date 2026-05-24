"""
Hand Teleop GUI — customtkinter interface.

Usage:
    python app.py
"""

import os
import sys
import glob
import shutil
import subprocess
import threading
import re
import tempfile
import time
import webbrowser

import cv2

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
VIDEO_DIR     = os.path.join(BASE_DIR, "video")
RETARGETED_DIR= os.path.join(VIDEO_DIR, "retargeted")
CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR      = os.path.join(BASE_DIR, "data")

_RT_CAM_DISTANCE = 0.6
_RT_CAM_YAW      = 0.0
_RT_CAM_PITCH    = -30.0

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
        return [{"filename": row["filename"], "full_path": row["full_path"]}
                for row in videos_repo.get_all_originals()]
    except Exception:
        return []


def get_retargeted():
    try:
        return [{"filename": row["filename"], "full_path": row["full_path"]}
                for row in videos_repo.get_all_retargeted()]
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


class EpochsDialog(ctk.CTkToplevel):
    def __init__(self, parent, hand_name: str):
        super().__init__(parent)
        self.title("Параметри навчання")
        self.geometry("340x160")
        self.resizable(False, False)
        self.after(200, self.grab_set)
        self.result = None

        ctk.CTkLabel(self, text=f"Кількість епох для '{hand_name}':",
                     font=ctk.CTkFont(size=13)).pack(pady=(20, 6), padx=20, anchor="w")

        self._entry = ctk.CTkEntry(self, placeholder_text="наприклад: 100")
        self._entry.pack(fill="x", padx=20)
        self._entry.insert(0, "100")
        self._entry.select_range(0, "end")
        self._entry.focus()

        self._err = ctk.CTkLabel(self, text="", text_color="#e74c3c",
                                 font=ctk.CTkFont(size=11))
        self._err.pack(pady=(4, 0), padx=20, anchor="w")

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(6, 16))
        ctk.CTkButton(btns, text="Скасувати", fg_color="gray30",
                      hover_color="gray40", command=self.destroy).pack(side="left")
        ctk.CTkButton(btns, text="Почати навчання",
                      command=self._ok).pack(side="right")

        self._entry.bind("<Return>", lambda e: self._ok())

    def _ok(self):
        val = self._entry.get().strip()
        try:
            n = int(val)
            if n < 1:
                raise ValueError
        except ValueError:
            self._err.configure(text="Введіть ціле число більше 0")
            self._entry.focus()
            return
        self.result = n
        self.destroy()


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
        self.geometry("960x920")
        self.resizable(False, True)

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
        hand_hdr = ctk.CTkFrame(sb, fg_color="transparent")
        hand_hdr.pack(fill="x", padx=16)
        ctk.CTkLabel(hand_hdr, text="Роботична кисть", anchor="w").pack(side="left")
        ctk.CTkButton(hand_hdr, text="＋ Нова", width=72, height=24,
                      command=self._add_hand).pack(side="right")
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
        self._cam_yaw      = 0.0
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

        # ── Група 1: Запуск ───────────────────────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color="gray30").pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(sb, text="КЕРУВАННЯ", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="gray50", anchor="w").pack(fill="x", padx=18, pady=(0, 4))

        self._start_btn = ctk.CTkButton(sb, text="▶  Запустити",
                                        height=36, command=self._start)
        self._start_btn.pack(fill="x", padx=16, pady=3)

        self._stop_btn = ctk.CTkButton(
            sb, text="⏹  Зупинити", height=36,
            fg_color="#c0392b", hover_color="#922b21",
            command=self._stop, state="disabled")
        self._stop_btn.pack(fill="x", padx=16, pady=3)

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(self, corner_radius=0, fg_color="gray14")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # row=0  Галерея відео header ──────────────────────────────────────────
        orig_hdr = ctk.CTkFrame(right, fg_color="transparent")
        orig_hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        ctk.CTkLabel(orig_hdr, text="Галерея відео",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(orig_hdr, text="↺ Оновити", width=90, height=28,
                      fg_color="gray30", hover_color="gray40",
                      command=self._refresh_lists).pack(side="right", padx=(6, 0))
        ctk.CTkButton(orig_hdr, text="＋ Додати", width=90, height=28,
                      command=self._add_to_gallery).pack(side="right")

        # row=3  Orig list (expandable) ────────────────────────────────────────
        self._orig_list = ctk.CTkScrollableFrame(right, height=120)
        self._orig_list.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))

        # row=4  Retargeted header ─────────────────────────────────────────────
        ctk.CTkLabel(right, text="Ретаргетовані відео",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=16, pady=(4, 4))

        # row=5  Ret list (expandable) ─────────────────────────────────────────
        self._ret_list = ctk.CTkScrollableFrame(right, height=120)
        self._ret_list.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 4))

        # row=6  Hint ──────────────────────────────────────────────────────────
        ctk.CTkLabel(right, text="2× клік — відкрити   |   ПКМ — видалити",
                     font=ctk.CTkFont(size=10), text_color="gray40").grid(
            row=4, column=0, sticky="w", padx=16, pady=(0, 2))

        # row=7  Separator ─────────────────────────────────────────────────────
        ctk.CTkFrame(right, height=1, fg_color="gray30").grid(
            row=5, column=0, sticky="ew", padx=16, pady=(6, 0))

        # row=8  ІНСТРУМЕНТИ ───────────────────────────────────────────────────
        tools_hdr = ctk.CTkFrame(right, fg_color="transparent")
        tools_hdr.grid(row=6, column=0, sticky="ew", padx=16, pady=(6, 6))
        ctk.CTkLabel(tools_hdr, text="ІНСТРУМЕНТИ",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="gray50").pack(side="left")
        ctk.CTkButton(tools_hdr, text="🗑  Видалити руку",
                      width=150, height=28,
                      fg_color="#5a1a1a", hover_color="#7a2a2a",
                      command=self._delete_hand).pack(side="right", padx=(6, 0))
        ctk.CTkButton(tools_hdr, text="💾  Checkpoint",
                      width=130, height=28,
                      fg_color="gray30", hover_color="gray40",
                      command=self._export_checkpoint).pack(side="right", padx=(6, 0))
        ctk.CTkButton(tools_hdr, text="📈  Графіки",
                      width=110, height=28,
                      fg_color="gray30", hover_color="gray40",
                      command=self._show_training_charts).pack(side="right", padx=(6, 0))

        # row=9  Status bar ────────────────────────────────────────────────────
        self._status_var = ctk.StringVar(value="Готово")
        ctk.CTkLabel(right, textvariable=self._status_var,
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=7, column=0, sticky="w", padx=16, pady=(0, 8))

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

        latest_run_id = runs[-1]["run_id"] if runs else None

        for i, run in enumerate(runs):
            epochs = epochs_repo.get_for_run(run["run_id"])
            if not epochs:
                continue

            color     = colors[i % len(colors)]
            is_latest = run["run_id"] == latest_run_id
            run_label = f"{run['run_id']} (поточне)" if is_latest else run["run_id"]
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

            lg_t = f"{run_label}_train"
            lg_v = f"{run_label}_val"

            fig.add_trace(go.Scatter(x=xs, y=train_l,
                name=f"{run_label} train",
                legendgroup=lg_t,
                line=dict(color=color, width=1.5, dash="dot"),
                **kw_hover), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=val_l,
                name=f"{run_label} val",
                legendgroup=lg_v,
                line=dict(color=color, width=2),
                **kw_hover), row=1, col=1)
            fig.add_trace(go.Scatter(x=best_xs, y=best_vals,
                name="best", mode="markers",
                legendgroup=lg_v,
                marker=dict(color="white", size=9, symbol="star",
                            line=dict(color=color, width=1.5)),
                hoverinfo="skip", showlegend=False), row=1, col=1)

            # підграфіки — дублюються в обох групах; hover тільки в train-копії
            for y_data, row_n in [(links_l, 2), (jpos_l, 3), (lr_l, 4), (time_l, 5)]:
                fig.add_trace(go.Scatter(x=xs, y=y_data,
                    legendgroup=lg_t, showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate="%{y:.5g}<extra></extra>"), row=row_n, col=1)
                fig.add_trace(go.Scatter(x=xs, y=y_data,
                    legendgroup=lg_v, showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate="%{y:.5g}<extra></extra>"), row=row_n, col=1)

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

    def _export_checkpoint(self):
        hand = self._hand_var.get()
        if not hand:
            messagebox.showerror("Помилка", "Спочатку оберіть руку")
            return
        model_row = models_repo.get_latest(hand)
        if not model_row or not os.path.isfile(model_row["checkpoint_path"]):
            messagebox.showerror("Помилка", f"Checkpoint для '{hand}' не знайдено")
            return
        src = model_row["checkpoint_path"]
        dst = filedialog.asksaveasfilename(
            title="Зберегти checkpoint як...",
            initialfile=os.path.basename(src),
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")])
        if not dst:
            return
        shutil.copy2(src, dst)
        self._status_var.set(f"Checkpoint збережено: {os.path.basename(dst)}")

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
        for item in items:
            path     = item["full_path"]
            label    = item["filename"]
            is_sel   = self._selected.get(key) == path
            bg       = "#1f538d" if is_sel else "#2b2b2b"
            row = ctk.CTkFrame(frame, fg_color=bg, corner_radius=6, height=32)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)
            lbl = ctk.CTkLabel(row, text=label, anchor="w",
                               fg_color="transparent",
                               font=ctk.CTkFont(size=12))
            lbl.pack(fill="both", expand=True, padx=10)

            for widget in (row, lbl):
                widget.bind("<Button-1>",
                            lambda e, p=path, k=key: self._select(p, k))
                widget.bind("<Double-Button-1>",
                            lambda e, p=path, k=key: self._open_video(p, k))
                widget.bind("<Button-3>",
                            lambda e, p=path, k=key: self._ctx_menu(e, p, k))
                widget.bind("<Enter>",
                            lambda e, r=row, p=path, k=key: r.configure(
                                fg_color="#2563a8" if self._selected.get(k)==p else "#3a3a3a"))
                widget.bind("<Leave>",
                            lambda e, r=row, p=path, k=key: r.configure(
                                fg_color="#1f538d" if self._selected.get(k)==p else "#2b2b2b"))

            self._btn_refs[key][path] = row

    def _select(self, path, key):
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

        self._selected[key]  = path
        self._last_selected  = (key, path)

        if path in self._btn_refs.get(key, {}):
            self._btn_refs[key][path].configure(fg_color="#1f538d")

        if key == "orig" and self._mode.get() == "video":
            self._input_video_path = path
            self._vpick_label.configure(text=os.path.basename(path))

    def _open_video(self, path, key):
        if os.path.exists(path):
            os.startfile(path)
            return
        if key == "ret":
            fallback = os.path.join(RETARGETED_DIR, os.path.basename(path))
            if os.path.exists(fallback):
                os.startfile(fallback)
                return
        if messagebox.askyesno(
            "Файл не знайдено",
            f"Файл не знайдено за шляхом:\n{path}\n\nВидалити запис з галереї?",
            icon="warning"
        ):
            self._delete_item(path, key)

    def _ctx_menu(self, event, path, key):
        from tkinter import Menu
        m = Menu(self, tearoff=0)
        if key == "ret":
            m.add_command(label=f"Вивантажити  '{os.path.basename(path)}'",
                          command=lambda: self._export_video(path))
            m.add_separator()
        m.add_command(label=f"Видалити  '{os.path.basename(path)}'",
                      command=lambda: self._delete_item(path, key))
        m.tk_popup(event.x_root, event.y_root)

    def _export_video(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Помилка", f"Файл не знайдено:\n{path}")
            return
        dst_dir = filedialog.askdirectory(title="Оберіть папку для збереження")
        if not dst_dir:
            return
        dst = os.path.join(dst_dir, os.path.basename(path))
        shutil.copy2(path, dst)
        self._status_var.set(f"Збережено: {os.path.basename(dst)}")

    def _delete_item(self, path, key):
        if not messagebox.askyesno("Видалення", f"Видалити '{os.path.basename(path)}'?"):
            return
        try:
            if key == "orig":
                videos_repo.delete_original(path)
            else:
                if os.path.exists(path):
                    os.remove(path)
                videos_repo.delete_retargeted(path)
        except Exception as e:
            print(f"[db] delete error: {e}")
        if self._last_selected == (key, path):
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
        config      = row["yml_path"] if row else None
        if not config:
            messagebox.showerror("Помилка", f"Руку '{hand}' не знайдено в БД")
            return
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
                   "--cam-distance", str(_RT_CAM_DISTANCE),
                   "--cam-yaw",      str(_RT_CAM_YAW),
                   "--cam-pitch",    str(_RT_CAM_PITCH),
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
            if not os.path.exists(self._input_video_path):
                if messagebox.askyesno(
                    "Файл не знайдено",
                    f"Файл не знайдено за шляхом:\n{self._input_video_path}\n\n"
                    "Видалити запис з галереї?",
                    icon="warning"
                ):
                    self._delete_item(self._input_video_path, "orig")
                    self._input_video_path = None
                    self._vpick_label.configure(text="не обрано — оберіть з галереї")
                return
            base = os.path.splitext(os.path.basename(self._input_video_path))[0]
            ts   = time.strftime("%Y%m%d_%H%M%S")
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
                    original_filename=getattr(self, "_input_video_path", None),
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

    def _delete_hand(self):
        hand = self._hand_var.get()
        if not hand:
            messagebox.showerror("Помилка", "Спочатку оберіть руку")
            return

        ckpt_paths = [
            r["checkpoint_path"]
            for r in models_repo.get_all_for_hand(hand)
            if r["checkpoint_path"] and os.path.isfile(r["checkpoint_path"])
        ]

        msg = f"Видалити '{hand}' з бази даних?\n\nБуде видалено: конфігурацію, всі моделі та епохи навчання."
        if ckpt_paths:
            msg += f"\n\nЗнайдено {len(ckpt_paths)} checkpoint-файл(ів) на диску."
        answer = messagebox.askyesnocancel(
            "Видалення руки", msg,
            icon="warning",
            detail="Так — видалити з БД\nНі — скасувати" if not ckpt_paths
                   else "Так — видалити з БД та файли\nНі — видалити тільки з БД\nСкасувати — нічого не робити")
        if answer is None:
            return

        delete_files = answer and bool(ckpt_paths)

        try:
            hands_repo.delete(hand)
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалось видалити: {e}")
            return

        if delete_files:
            failed = []
            for path in ckpt_paths:
                try:
                    os.remove(path)
                except Exception:
                    failed.append(path)
            if failed:
                messagebox.showwarning(
                    "Увага", f"Не вдалось видалити файли:\n" + "\n".join(failed))

        hands = get_hands()
        self._hand_menu.configure(values=hands)
        self._hand_var.set(hands[0] if hands else "")
        suffix = f" + {len(ckpt_paths) - len(failed if delete_files else [])} файл(ів)" if delete_files else ""
        self._status_var.set(f"'{hand}' видалено{suffix}")

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
            from retargeting import load_retargeting_config
            load_retargeting_config(yml_abs, assets_abs)
        except FileNotFoundError as e:
            messagebox.showerror(
                "Файл не знайдено",
                f"{e}\n\nПеревірте що обрані YML і папка assets відповідають одній руці.")
            return
        except Exception as e:
            messagebox.showerror("Помилка конфігу", str(e))
            return

        data_files = sorted(glob.glob(os.path.join(DATA_DIR, "kps_*.npz")))
        if not data_files:
            messagebox.showwarning(
                "Дані відсутні",
                f"Файли kps_*.npz не знайдено в {DATA_DIR}.\n"
                "Спочатку зберіть дані через collect_data.py")
            return

        dlg_epochs = EpochsDialog(self, hand_name)
        self.wait_window(dlg_epochs)
        if dlg_epochs.result is None:
            return
        n_epochs = dlg_epochs.result

        try:
            hands_repo.get_or_create(hand_name, yml_abs, assets_abs)
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалось зареєструвати руку: {e}")
            return

        hands = get_hands()
        self._hand_menu.configure(values=hands)
        self._hand_var.set(hand_name)

        row = hands_repo.get_by_name(hand_name)
        self._train(hand_name, row["yml_path"], n_epochs)

    def _train(self, hand_name, config_path, n_epochs: int = 100):
        data_files = sorted(glob.glob(os.path.join(DATA_DIR, "kps_*.npz")))

        run_id = time.strftime("%Y%m%d_%H%M%S")
        self._current_run_id = run_id
        hand_row   = hands_repo.get_by_name(hand_name)
        assets_path = hand_row["assets_path"] if hand_row else None

        cmd = [sys.executable, "-u",
               os.path.join(BASE_DIR, "mlp_selfsupervised", "train.py"),
               "--config", config_path,
               "--run-id", run_id,
               "--epochs", str(n_epochs),
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
