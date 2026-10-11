"""
Storyteller Studio Pro - Kinetic Typography Edition
A PySide6 application for creating YouTube storytelling videos
with Neural TTS narration and beautiful text visualization.
"""

import sys
import os
import io
import time
import wave
import math
import shutil
import tempfile
import subprocess
import asyncio
import importlib
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Auto-Install edge-tts if missing
# ---------------------------------------------------------------------------
try:
    import edge_tts
except ImportError:
    print("edge-tts not found. Attempting to automatic installation via pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
        import edge_tts
        print("edge-tts installed successfully!")
    except Exception as e:
        print(f"Failed to automatically install edge-tts: {e}")
        print("Please install it manually by running: pip install edge-tts")
        sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    sd = None # Playback is optional, but recommended

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QObject, QRectF, QPointF
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, 
    QRadialGradient, QImage, QFontMetrics, QAction, QPalette
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QLabel, QSlider, QComboBox, QSpinBox,
    QGroupBox, QFileDialog, QProgressBar, QMessageBox, QSplitter,
    QCheckBox, QDoubleSpinBox, QSizePolicy, QStatusBar, QDialog, 
    QFormLayout, QLineEdit
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    char_start: int
    char_end: int


@dataclass
class VideoSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    output_dir: str = str(Path.home() / "Videos" / "StorytellerStudio")
    title: str = "My Story"
    subtitle: str = ""
    font_family: str = "Georgia"
    font_size: int = 42
    bg_hue_start: int = 220
    accent_color: Tuple[int, int, int] = (255, 215, 100)
    show_progress_bar: bool = True
    title_duration: float = 3.0


# ---------------------------------------------------------------------------
# TTS generation worker (Edge-TTS)
# ---------------------------------------------------------------------------

class EdgeTTSGenerator(QThread):
    """Generates high-quality TTS using edge-tts and captures exact word timings."""
    progress = Signal(str)
    finished_ok = Signal(str, list)   # (wav_file, word_timings)
    error = Signal(str)

    def __init__(self, text: str, output_dir: str, voice: str, rate: str = "+0%"):
        super().__init__()
        self.text = text
        self.output_dir = output_dir
        self.voice = voice
        self.rate = rate

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            mp3_file = os.path.join(self.output_dir, "narration.mp3")
            wav_file = os.path.join(self.output_dir, "narration.wav")
            
            self.progress.emit("Connecting to Edge TTS...")
            communicate = edge_tts.Communicate(self.text, self.voice, rate=self.rate)
            
            word_timings = []
            text_ptr = 0
            
            self.progress.emit("Synthesizing speech...")
            
            with open(mp3_file, "wb") as f:
                async def amain():
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            offset_sec = chunk["offset"] / 10_000_000  # 100ns to sec
                            duration_sec = chunk["duration"] / 10_000_000
                            word_text = chunk["text"]
                            
                            try:
                                idx = self.text.index(word_text, text_ptr)
                                char_start = idx
                                char_end = idx + len(word_text)
                                text_ptr = char_end
                            except ValueError:
                                char_start = text_ptr
                                char_end = text_ptr + len(word_text)
                                text_ptr = char_end
                            
                            wt = WordTiming(
                                word=word_text,
                                start=offset_sec,
                                end=offset_sec + duration_sec,
                                char_start=char_start,
                                char_end=char_end
                            )
                            word_timings.append(wt)
                
                asyncio.run(amain())

            self.progress.emit("Converting MP3 to WAV...")
            ffmpeg = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
            if not ffmpeg:
                raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
            
            cmd = [ffmpeg, '-y', '-i', mp3_file, '-ar', '44100', '-ac', '1', wav_file]
            subprocess.run(cmd, capture_output=True)
            
            self.progress.emit("TTS generation complete.")
            self.finished_ok.emit(wav_file, [w.__dict__ for w in word_timings])
            
        except Exception as e:
            self.error.emit(f"TTS Error: {e}")


# ---------------------------------------------------------------------------
# Audio player
# ---------------------------------------------------------------------------

class AudioPlayer(QThread):
    """Plays audio and emits precise elapsed time for the visualizer."""
    progress = Signal(float)  # elapsed seconds
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, audio_file: str):
        super().__init__()
        self.audio_file = audio_file
        self.stop_requested = False

    def run(self):
        if sd is None:
            self.error.emit("sounddevice not installed. Cannot play audio.")
            return
        try:
            with wave.open(self.audio_file, 'rb') as wf:
                rate = wf.getframerate()
                frames = wf.getnframes()
                data = wf.readframes(frames)
            
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(samples) / rate
            
            start_time = time.perf_counter()
            sd.play(samples, rate)

            while not self.stop_requested:
                elapsed = time.perf_counter() - start_time
                if elapsed >= duration:
                    break
                self.progress.emit(elapsed)
                time.sleep(0.03)

            sd.stop()
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(f"Playback error: {e}")

    def stop(self):
        self.stop_requested = True


# ---------------------------------------------------------------------------
# Rendering Logic (Kinetic Typography)
# ---------------------------------------------------------------------------

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def render_frame(
    width: int, height: int, 
    time_sec: float, settings: VideoSettings, 
    full_text: str, word_timings: List[dict]
) -> QImage:
    """Pure function to render a single frame to a QImage."""
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(Qt.black)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)

    # --- Atmospheric Background ---
    hue_shift = (time_sec * 5) % 360
    h1 = (settings.bg_hue_start + hue_shift) % 360
    h2 = (h1 + 40) % 360
    r1, g1, b1 = hsl_to_rgb(h1, 0.45, 0.08)
    r2, g2, b2 = hsl_to_rgb(h2, 0.55, 0.02)

    grad = QLinearGradient(0, 0, 0, height)
    grad.setColorAt(0.0, QColor(r1, g1, b1))
    grad.setColorAt(1.0, QColor(r2, g2, b2))
    painter.fillRect(0, 0, width, height, QBrush(grad))

    # --- Floating Particles (Fireflies) ---
    ar, ag, ab = settings.accent_color
    painter.setPen(Qt.NoPen)
    for i in range(25):
        # Pseudo-random but deterministic particle movement
        seed = i * 12.34
        p_x = (math.sin(time_sec * 0.2 + seed) * 0.5 + 0.5) * width
        p_y = ((time_sec * 15 * (1 + (i % 3)*0.5) + seed * 100) % height)
        p_size = 2 + (i % 4)
        p_alpha = int(80 + 60 * math.sin(time_sec * 1.5 + seed))
        painter.setBrush(QBrush(QColor(ar, ag, ab, max(0, p_alpha))))
        painter.drawEllipse(QPointF(p_x, height - p_y), p_size, p_size)

    # --- Soft Pulsing Glow ---
    pulse = 0.5 + 0.5 * math.sin(time_sec * 1.2)
    glow_radius = int(min(width, height) * (0.3 + 0.05 * pulse))
    cx, cy = width // 2, int(height * 0.5)
    glow = QRadialGradient(cx, cy, glow_radius)
    glow.setColorAt(0.0, QColor(ar, ag, ab, int(20 + 15 * pulse)))
    glow.setColorAt(1.0, QColor(ar, ag, ab, 0))
    painter.setBrush(QBrush(glow))
    painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)

    # --- Title Overlay ---
    if settings.title and time_sec < settings.title_duration + 1.0:
        fade = 1.0
        if time_sec > settings.title_duration:
            fade = max(0.0, 1.0 - (time_sec - settings.title_duration))
        
        alpha = int(255 * fade)
        title_font = QFont(settings.font_family, max(28, int(width * 0.045)))
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255, alpha))
        fm = QFontMetrics(title_font)
        tw = fm.horizontalAdvance(settings.title)
        painter.drawText(QPointF((width - tw) / 2, height * 0.30), settings.title)

    # --- Story Text (Kinetic Typography) ---
    if full_text and word_timings:
        _draw_story_text(painter, width, height, time_sec, settings, full_text, word_timings)

    # --- Progress Bar ---
    if settings.show_progress_bar and word_timings:
        _draw_progress(painter, width, height, time_sec, settings, word_timings)

    painter.end()
    return img


def _draw_story_text(painter, w, h, t, s, full_text, word_timings):
    # 1. Find current word index
    cur_idx = -1
    for i, wt in enumerate(word_timings):
        if wt['start'] <= t < wt['end']:
            cur_idx = i
            break
    
    # Determine target character position for paragraph extraction
    target_char = 0
    if cur_idx >= 0:
        target_char = word_timings[cur_idx]['char_start']
    else:
        # Fallback: if in silence, keep showing the last spoken paragraph
        last_spoken_idx = -1
        for i, wt in enumerate(word_timings):
            if wt['start'] <= t:
                last_spoken_idx = i
        if last_spoken_idx != -1:
            target_char = word_timings[last_spoken_idx]['char_start']

    # 2. Find paragraph bounds
    p_start = full_text.rfind('\n\n', 0, target_char)
    p_start = 0 if p_start == -1 else p_start + 2
    p_end = full_text.find('\n\n', target_char)
    if p_end == -1: p_end = len(full_text)
    
    paragraph_text = full_text[p_start:p_end].strip()
    if not paragraph_text: return

    # Get word timings ONLY for this paragraph
    para_words = [wt for wt in word_timings if p_start <= wt['char_start'] < p_end]

    # 3. Setup Typography
    font_size = max(20, int(w * s.font_size / 1920))
    font = QFont(s.font_family, font_size)
    fm = QFontMetrics(font)
    
    max_width = int(w * 0.75)
    line_height = int(fm.height() * 1.6)
    space_w = fm.horizontalAdvance(' ')

    # 4. Word Wrap
    lines = []
    current_line = []
    current_width = 0

    for wt in para_words:
        word_str = wt['word']
        ww = fm.horizontalAdvance(word_str)
        
        if current_width + ww > max_width and current_line:
            lines.append(current_line)
            current_line = []
            current_width = 0
            
        current_line.append(wt)
        current_width += ww + space_w
        
    if current_line: lines.append(current_line)

    # 5. Layout and Draw
    total_h = len(lines) * line_height
    start_y = int(h * 0.5 - total_h / 2 + fm.ascent())

    # Subtle "breathing" zoom effect
    scale = 1.0 + 0.005 * math.sin(t * 2.0)
    painter.save()
    painter.translate(w/2, h/2)
    painter.scale(scale, scale)
    painter.translate(-w/2, -h/2)

    for li, line in enumerate(lines):
        line_w = sum(fm.horizontalAdvance(wt['word']) for wt in line) + space_w * (len(line) - 1)
        x = int((w - line_w) / 2)
        y = start_y + li * line_height

        for wt in line:
            word_str = wt['word']
            ww = fm.horizontalAdvance(word_str)
            is_past = t >= wt['end']
            is_active = wt['start'] <= t < wt['end']

            if is_active:
                # Draw Glowing Pill Background
                pulse = 0.7 + 0.3 * math.sin(t * 8)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(ar, ag, ab, int(50 * pulse))))
                rect = QRectF(x - 8, y - fm.ascent() - 6, ww + 16, fm.height() + 12)
                painter.drawRoundedRect(rect, 8, 8)

                # Active Text
                painter.setPen(QColor(ar, ag, ab))
                bold_font = QFont(font)
                bold_font.setBold(True)
                painter.setFont(bold_font)
                painter.drawText(QPointF(x, y), word_str)
                painter.setFont(font)
            elif is_past:
                # Spoken Text
                painter.setPen(QColor(240, 240, 240, 255))
                painter.drawText(QPointF(x, y), word_str)
            else:
                # Unspoken Text (Dimmed)
                painter.setPen(QColor(180, 180, 180, 100))
                painter.drawText(QPointF(x, y), word_str)

            x += ww + space_w

    painter.restore()


def _draw_progress(painter, w, h, t, s, word_timings):
    total = word_timings[-1]['end']
    if total <= 0: return
    progress = min(1.0, t / total)

    bar_w = int(w * 0.8)
    bar_h = 3
    bar_x = (w - bar_w) // 2
    bar_y = h - int(h * 0.04)

    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
    painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), bar_h / 2, bar_h / 2)

    ar, ag, ab = s.accent_color
    grad = QLinearGradient(bar_x, 0, bar_x + bar_w * progress, 0)
    grad.setColorAt(0.0, QColor(ar, ag, ab, 200))
    grad.setColorAt(1.0, QColor(min(255, ar + 40), min(255, ag + 40), min(255, ab + 40), 255))
    painter.setBrush(QBrush(grad))
    painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w * progress, bar_h), bar_h / 2, bar_h / 2)


# ---------------------------------------------------------------------------
# GUI Components
# ---------------------------------------------------------------------------

class VisualizerCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.settings = VideoSettings()
        self.full_text = ""
        self.word_timings = []
        self.current_time = 0.0
        
        # Timer to keep animation running even when audio isn't playing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33) # ~30fps
        
        self.last_tick_time = time.perf_counter()

    def _tick(self):
        now = time.perf_counter()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        # Only advance time if we aren't being updated by AudioPlayer
        # (Simple hack: just update visuals, time is set externally)
        self.update()

    def set_data(self, text, timings):
        self.full_text = text
        self.word_timings = timings
        self.current_time = 0.0
        self.update()

    def set_time(self, t: float):
        self.current_time = t
        self.update()

    def paintEvent(self, event):
        img = render_frame(
            self.width(), self.height(), self.current_time, self.settings, 
            self.full_text, self.word_timings
        )
        painter = QPainter(self)
        painter.drawImage(0, 0, img)


class VideoExporter(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    finished_ok = Signal(str)
    error = Signal(str)

    def __init__(self, settings, full_text, word_timings, audio_file, output_path):
        super().__init__()
        self.settings = settings
        self.full_text = full_text
        self.word_timings = word_timings
        self.audio_file = audio_file
        self.output_path = output_path

    def run(self):
        try:
            import cv2
            self.status.emit("Analyzing audio duration...")
            
            with wave.open(self.audio_file, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()

            total_frames = int(duration * self.settings.fps) + 1
            self.status.emit(f"Rendering {total_frames} frames...")

            tmp_dir = tempfile.mkdtemp(prefix="storyteller_")
            tmp_video = os.path.join(tmp_dir, "video_no_audio.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp_video, fourcc, self.settings.fps, 
                                     (self.settings.width, self.settings.height))

            for i in range(total_frames):
                t = i / self.settings.fps
                
                img = render_frame(
                    self.settings.width, self.settings.height, t, self.settings,
                    self.full_text, self.word_timings
                )
                
                # Convert QImage to OpenCV BGR
                ptr = img.constBits()
                ptr.setsize(img.sizeInBytes())
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape(self.settings.height, self.settings.width, 4)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                writer.write(bgr)

                self.progress.emit(i + 1, total_frames)

            writer.release()
            self.status.emit("Muxing audio with ffmpeg...")

            ffmpeg = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
            if ffmpeg:
                cmd = [
                    ffmpeg, '-y',
                    '-i', tmp_video,
                    '-i', self.audio_file,
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-shortest',
                    self.output_path
                ]
                subprocess.run(cmd, capture_output=True)
            else:
                shutil.copy(tmp_video, self.output_path)

            shutil.rmtree(tmp_dir)
            self.finished_ok.emit(self.output_path)

        except Exception as e:
            self.error.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Settings")
        self.settings = settings
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)
        self.title_edit = QLineEdit(self.settings.title)
        layout.addRow("Title:", self.title_edit)
        self.subtitle_edit = QLineEdit(self.settings.subtitle)
        layout.addRow("Subtitle:", self.subtitle_edit)
        self.font_edit = QLineEdit(self.settings.font_family)
        layout.addRow("Font Family:", self.font_edit)
        self.font_size_spin = QSpinBox(); self.font_size_spin.setRange(20, 80); self.font_size_spin.setValue(self.settings.font_size)
        layout.addRow("Font Size:", self.font_size_spin)
        self.w_spin = QSpinBox(); self.w_spin.setRange(640, 3840); self.w_spin.setValue(self.settings.width)
        layout.addRow("Width:", self.w_spin)
        self.h_spin = QSpinBox(); self.h_spin.setRange(360, 2160); self.h_spin.setValue(self.settings.height)
        layout.addRow("Height:", self.h_spin)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(15, 60); self.fps_spin.setValue(self.settings.fps)
        layout.addRow("FPS:", self.fps_spin)
        self.hue_spin = QSpinBox(); self.hue_spin.setRange(0, 360); self.hue_spin.setValue(self.settings.bg_hue_start)
        layout.addRow("Bg Hue:", self.hue_spin)
        
        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btns.addWidget(ok_btn); btns.addWidget(cancel_btn)
        layout.addRow(btns)
        ok_btn.clicked.connect(self._save_and_accept)
        cancel_btn.clicked.connect(self.reject)

    def _save_and_accept(self):
        s = self.settings
        s.title = self.title_edit.text()
        s.subtitle = self.subtitle_edit.text()
        s.font_family = self.font_edit.text() or "Georgia"
        s.font_size = self.font_size_spin.value()
        s.width = self.w_spin.value()
        s.height = self.h_spin.value()
        s.fps = self.fps_spin.value()
        s.bg_hue_start = self.hue_spin.value()
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Storyteller Studio Pro - Kinetic Edition")
        self.resize(1400, 850)
        self.settings = VideoSettings()
        
        self.audio_file = None
        self.word_timings = []
        self.tts_worker = None
        self.player = None
        self.exporter = None
        self.output_dir = tempfile.mkdtemp(prefix="storyteller_data_")

        self._build_ui()
        self._fetch_voices()
        
        sample = """The Lighthouse at World's Edge

In a forgotten village where the sea met the sky, there stood a lighthouse older than memory itself. Its light, they said, did not warn ships of rocks, but beckoned travelers from beyond the horizon.

One stormy evening, a small boat appeared from the mist."""
        self.editor.setPlainText(sample)
        self.canvas.set_data(sample, [])

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Left Panel
        left = QWidget()
        l_layout = QVBoxLayout(left)
        
        l_layout.addWidget(QLabel("Story Editor:"))
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Georgia", 12))
        self.editor.textChanged.connect(lambda: self.canvas.set_data(self.editor.toPlainText(), []))
        l_layout.addWidget(self.editor, 1)

        tts_group = QGroupBox("TTS Settings")
        tts_layout = QVBoxLayout(tts_group)
        
        self.voice_combo = QComboBox()
        tts_layout.addWidget(QLabel("Voice:"))
        tts_layout.addWidget(self.voice_combo)
        
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Rate:"))
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(-50, 100)
        self.rate_slider.setValue(0)
        self.rate_label = QLabel("+0%")
        self.rate_slider.valueChanged.connect(lambda v: self.rate_label.setText(f"{'+' if v>=0 else ''}{v}%"))
        rate_layout.addWidget(self.rate_slider)
        rate_layout.addWidget(self.rate_label)
        tts_layout.addLayout(rate_layout)

        l_layout.addWidget(tts_group)

        self.generate_btn = QPushButton("Generate Neural TTS")
        self.generate_btn.setStyleSheet("background-color: #4a90d9; color: white; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self.generate_tts)
        l_layout.addWidget(self.generate_btn)

        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play Preview")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_audio)
        btn_layout.addWidget(self.play_btn)
        
        self.export_btn = QPushButton("Export MP4")
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("background-color: #d99415; color: white; font-weight: bold; padding: 10px;")
        self.export_btn.clicked.connect(self.export_video)
        btn_layout.addWidget(self.export_btn)
        l_layout.addLayout(btn_layout)

        self.settings_btn = QPushButton("Video Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        l_layout.addWidget(self.settings_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        l_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        l_layout.addWidget(self.status_label)

        splitter.addWidget(left)

        # Right Panel
        self.canvas = VisualizerCanvas()
        self.canvas.settings = self.settings
        splitter.addWidget(self.canvas)
        splitter.setSizes([450, 950])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _fetch_voices(self):
        self.voice_combo.clear()
        self.status_bar.showMessage("Fetching TTS voices...")
        QApplication.processEvents()
        try:
            voices = asyncio.run(edge_tts.list_voices())
            for v in voices:
                if v["Locale"].startswith("en"):
                    name = f"{v['ShortName']} - {v['Gender']}"
                    self.voice_combo.addItem(name, v["ShortName"])
            self.voice_combo.setCurrentText("en-US-GuyNeural - Male")
        except Exception:
            self.voice_combo.addItem("Default", "en-US-GuyNeural")
        self.status_bar.showMessage("Ready")

    def generate_tts(self):
        text = self.editor.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty", "Please write a story first.")
            return

        self.generate_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Generating neural TTS audio...")

        voice = self.voice_combo.currentData()
        rate_val = self.rate_slider.value()
        rate_str = f"{'+' if rate_val>=0 else ''}{rate_val}%"

        self.tts_worker = EdgeTTSGenerator(text, self.output_dir, voice, rate_str)
        self.tts_worker.progress.connect(self.status_label.setText)
        self.tts_worker.error.connect(self._on_tts_error)
        self.tts_worker.finished_ok.connect(self._on_tts_done)
        self.tts_worker.start()

    def _on_tts_done(self, wav_file, word_timings):
        self.audio_file = wav_file
        self.word_timings = word_timings
        
        self.canvas.set_data(self.editor.toPlainText(), word_timings)
        
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        duration = word_timings[-1]['end'] if word_timings else 0
        self.status_label.setText(f"Audio ready. Duration: {duration:.1f}s")

    def _on_tts_error(self, msg):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "TTS Error", msg)
        self.status_label.setText("Error.")

    def play_audio(self):
        if not self.audio_file: return
        self.player = AudioPlayer(self.audio_file)
        self.player.progress.connect(self.canvas.set_time)
        self.player.error.connect(lambda m: QMessageBox.warning(self, "Audio", m))
        self.player.start()

    def export_video(self):
        if not self.audio_file: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Video", "story.mp4", "MP4 (*.mp4)")
        if not path: return

        self.export_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText("Exporting video...")

        self.exporter = VideoExporter(
            self.settings, self.editor.toPlainText(), 
            self.word_timings, self.audio_file, path
        )
        self.exporter.progress.connect(lambda c, t: self.progress_bar.setValue(int(c/t*100)))
        self.exporter.status.connect(self.status_label.setText)
        self.exporter.error.connect(self._on_export_error)
        self.exporter.finished_ok.connect(self._on_export_done)
        self.exporter.start()

    def _on_export_done(self, path):
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        QMessageBox.information(self, "Success", f"Video exported to:\n{path}")

    def _on_export_error(self, msg):
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Export Error", msg)

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.Accepted:
            self.canvas.settings = self.settings
            self.canvas.update()

    def closeEvent(self, e):
        if self.player: self.player.stop()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Storyteller Studio Pro")
    
    # Dark Theme
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(30, 30, 35))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.Base, QColor(40, 40, 45))
    palette.setColor(QPalette.Text, QColor(230, 230, 230))
    palette.setColor(QPalette.Button, QColor(50, 50, 55))
    palette.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    palette.setColor(QPalette.Highlight, QColor(74, 144, 217))
    app.setPalette(palette)
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec())