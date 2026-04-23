"""
mol3d_gui.py — Interactive 3D Molecule Viewer
==============================================
PyQt6 GUI with full menu bar, appearance controls, and atom colour editor.

Controls:
  Left-drag    Rotate molecule
  Right-drag   Pan
  Scroll       Zoom

Menus:
  File    Open / Save SVG / Quit
  Edit    Appearance (ball size, bond width) / Atom Colours / Reset Colors
  View    Preset orientations / Reset View / Background color
  Help    About

Dependencies:
    pip install PyQt6 numpy svgwrite
"""

import sys, os, math, tempfile
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QSlider, QStatusBar, QFrame, QSizePolicy,
    QGroupBox, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QColorDialog, QPushButton, QGridLayout,
    QScrollArea, QToolBar, QMenu,
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, QByteArray, QPoint, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QAction, QColor, QPalette, QFont, QCursor, QIcon, QPixmap

# ── renderer / parsers ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from mol3d_avogadro import (
    parse_xyz, parse_gaussian, parse_gaussian_log, infer_bonds,
    render_avogadro, Molecule, CPK_BASE, CPK_DARK, VDW_RADII,
    lighten, darken, hex_to_rgb, rgb_to_hex, auto_dark,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ELEM_FULL_NAME = {
    "H":"Hydrogen","C":"Carbon","N":"Nitrogen","O":"Oxygen",
    "F":"Fluorine","S":"Sulfur","P":"Phosphorus","Cl":"Chlorine",
    "Br":"Bromine","I":"Iodine","B":"Boron","Si":"Silicon",
}

DARK_BG  = "#0f0f1a"
PANEL_BG = "#0d0d18"
CARD_BG  = "#13131f"
BORDER   = "#2a2a44"
FG       = "#ccd6f6"
FG_DIM   = "#8899bb"
ACCENT   = "#4488cc"

STYLESHEET = f"""
QMainWindow, QDialog {{ background:{DARK_BG}; }}
QWidget {{ color:{FG}; }}
QMenuBar {{
    background:{CARD_BG}; color:{FG};
    border-bottom:1px solid {BORDER}; padding:2px 4px;
}}
QMenuBar::item:selected {{ background:{BORDER}; border-radius:3px; }}
QMenu {{
    background:{CARD_BG}; color:{FG};
    border:1px solid {BORDER}; border-radius:6px;
    padding:4px;
}}
QMenu::item {{ padding:5px 24px 5px 10px; border-radius:4px; }}
QMenu::item:selected {{ background:{ACCENT}; color:#fff; }}
QMenu::separator {{ height:1px; background:{BORDER}; margin:3px 6px; }}
QToolBar {{
    background:{CARD_BG}; border-bottom:1px solid {BORDER};
    spacing:4px; padding:3px 8px;
}}
QToolBar QToolButton {{
    background:transparent; color:{FG};
    border:none; border-radius:4px; padding:4px 8px;
}}
QToolBar QToolButton:hover {{ background:{BORDER}; }}
QPushButton {{
    background:#1e1e32; color:{FG};
    border:1px solid {BORDER}; border-radius:5px;
    padding:5px 14px; font-size:12px;
}}
QPushButton:hover  {{ background:#2a2a48; border-color:{ACCENT}; }}
QPushButton:pressed{{ background:{ACCENT}; }}
QPushButton#accent {{
    background:#1e4080; border-color:{ACCENT}; color:#fff;
}}
QPushButton#accent:hover {{ background:#2255aa; }}
QPushButton#color_btn {{
    border-radius:4px; min-width:36px; min-height:24px;
    padding:2px; border:2px solid {BORDER};
}}
QPushButton#color_btn:hover {{ border-color:{ACCENT}; }}
QLabel {{ color:{FG}; }}
QSlider::groove:horizontal {{
    height:4px; background:{BORDER}; border-radius:2px;
}}
QSlider::handle:horizontal {{
    background:{ACCENT}; border-radius:6px;
    width:14px; height:14px; margin:-5px 0;
}}
QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
QDoubleSpinBox, QSpinBox {{
    background:#1a1a28; border:1px solid {BORDER};
    border-radius:4px; padding:3px 6px; color:{FG};
}}
QGroupBox {{
    color:#88aadd; border:1px solid {BORDER};
    border-radius:6px; margin-top:10px; font-size:11px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
QStatusBar {{ background:{PANEL_BG}; color:{FG_DIM}; font-size:11px; }}
QScrollArea {{ background:transparent; border:none; }}
QScrollArea > QWidget > QWidget {{ background:transparent; }}
QDialogButtonBox QPushButton {{ min-width:80px; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SWATCH BUTTON
# ─────────────────────────────────────────────────────────────────────────────

def color_pixmap(hex_color: str, size: int = 22) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(QColor(hex_color))
    return px

class ColorButton(QPushButton):
    """Push-button that shows a solid colour swatch and opens a colour dialog."""
    colorChanged = pyqtSignal(str)   # emits new hex string

    def __init__(self, initial_color: str = "#ffffff", parent=None):
        super().__init__(parent)
        self.setObjectName("color_btn")
        self.setFixedSize(36, 26)
        self._color = initial_color
        self._update_swatch()
        self.clicked.connect(self._pick)

    def color(self) -> str:
        return self._color

    def set_color(self, hex_color: str):
        self._color = hex_color
        self._update_swatch()

    def _update_swatch(self):
        self.setStyleSheet(
            f"QPushButton#color_btn {{"
            f"  background:{self._color};"
            f"  border:2px solid {BORDER}; border-radius:4px;"
            f"}}"
            f"QPushButton#color_btn:hover {{ border-color:{ACCENT}; }}"
        )

    def _pick(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if col.isValid():
            self._color = col.name()
            self._update_swatch()
            self.colorChanged.emit(self._color)


# ─────────────────────────────────────────────────────────────────────────────
# APPEARANCE DIALOG  (ball size + bond width)
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceDialog(QDialog):
    def __init__(self, atom_scale: float, bond_width: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Appearance")
        self.setFixedWidth(320)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        form = QFormLayout()
        form.setSpacing(10)

        self._atom_spin = QDoubleSpinBox()
        self._atom_spin.setRange(0.3, 3.0)
        self._atom_spin.setSingleStep(0.05)
        self._atom_spin.setDecimals(2)
        self._atom_spin.setValue(atom_scale)
        self._atom_spin.setSuffix("×")
        self._atom_spin.setToolTip("Multiplier on Van der Waals radius")
        form.addRow("Ball size:", self._atom_spin)

        self._bond_spin = QDoubleSpinBox()
        self._bond_spin.setRange(1.0, 30.0)
        self._bond_spin.setSingleStep(0.5)
        self._bond_spin.setDecimals(1)
        self._bond_spin.setValue(bond_width)
        self._bond_spin.setSuffix(" px")
        self._bond_spin.setToolTip("Bond cylinder half-width in pixels")
        form.addRow("Bond width:", self._bond_spin)

        layout.addLayout(form)

        # Preview slider for quick feel
        hint = QLabel("Tip: use the scale slider in the toolbar to adjust overall size.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{FG_DIM}; font-size:10px;")
        layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @property
    def atom_scale(self) -> float:
        return self._atom_spin.value()

    @property
    def bond_width(self) -> float:
        return self._bond_spin.value()


# ─────────────────────────────────────────────────────────────────────────────
# ATOM COLOUR EDITOR DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class AtomColorDialog(QDialog):
    """Shows one colour-picker row per element present in the molecule."""

    def __init__(self, elements: list, current_overrides: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atom Colours")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        lbl = QLabel("Click a swatch to change an element's colour.")
        lbl.setStyleSheet(f"color:{FG_DIM}; font-size:11px;")
        layout.addWidget(lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(10)
        grid.setContentsMargins(6, 6, 6, 6)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        self._buttons: dict[str, ColorButton] = {}

        for row, elem in enumerate(sorted(elements)):
            default = CPK_BASE.get(elem, "#cc44aa")
            current = current_overrides.get(elem, default)

            # Element symbol + name label
            sym_lbl = QLabel(f"<b>{elem}</b>")
            sym_lbl.setFixedWidth(28)
            sym_lbl.setStyleSheet("font-size:13px;")
            name_lbl = QLabel(ELEM_FULL_NAME.get(elem, elem))
            name_lbl.setStyleSheet(f"color:{FG_DIM}; font-size:11px;")

            btn = ColorButton(current)
            self._buttons[elem] = btn

            # Reset to CPK button
            reset_btn = QPushButton("CPK")
            reset_btn.setFixedWidth(40)
            reset_btn.setStyleSheet(f"font-size:10px; padding:2px 4px;")
            reset_btn.setToolTip(f"Reset {elem} to CPK default")
            reset_btn.clicked.connect(lambda _, e=elem, b=btn: b.set_color(CPK_BASE.get(e, "#cc44aa")))

            grid.addWidget(sym_lbl,   row, 0)
            grid.addWidget(name_lbl,  row, 1)
            grid.addWidget(btn,       row, 2)
            grid.addWidget(reset_btn, row, 3)

        grid.setColumnStretch(1, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_overrides(self) -> dict:
        return {elem: btn.color() for elem, btn in self._buttons.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CPK LEGEND PANEL  (sidebar)
# ─────────────────────────────────────────────────────────────────────────────

class LegendPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            LegendPanel {{
                background:{CARD_BG};
                border:1px solid {BORDER};
                border-radius:8px;
            }}
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(4)
        self._layout.setContentsMargins(10,10,10,10)

        title = QLabel("Elements")
        title.setStyleSheet(f"color:{ACCENT}; font-weight:bold; font-size:12px;")
        self._layout.addWidget(title)
        self._layout.addSpacing(4)

        self._rows: list = []

    def update_for(self, mol: Molecule, overrides: dict):
        for w in self._rows:
            w.setParent(None)
        self._rows.clear()

        seen = set()
        for atom in mol.atoms:
            e = atom.element
            if e in seen:
                continue
            seen.add(e)
            color = overrides.get(e, CPK_BASE.get(e, "#cc44aa"))
            name  = ELEM_FULL_NAME.get(e, e)

            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0)
            rl.setSpacing(7)

            swatch = QLabel()
            swatch.setFixedSize(14,14)
            swatch.setStyleSheet(
                f"background:{color}; border-radius:7px; border:1px solid #555;"
            )
            lbl = QLabel(f"{e}  {name}")
            lbl.setStyleSheet(f"color:{FG_DIM}; font-size:11px;")
            rl.addWidget(swatch)
            rl.addWidget(lbl)
            rl.addStretch()

            self._layout.addWidget(row)
            self._rows.append(row)

        self._layout.addStretch()


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class MoleculeCanvas(QSvgWidget):
    rotationChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.molecule: Molecule | None = None
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self._drag_start: QPoint | None = None
        self._drag_mode = "none"

        self.setMinimumSize(500, 450)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        # Render parameters — all public, set by MainWindow
        self.base_scale     = 110.0
        self.atom_scale     = 1.1
        self.bond_width_px  = 10.0
        self.background     = "#0a0a12"
        self.color_overrides: dict = {}

        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)

    # ── public ───────────────────────────────────────────────────────────────

    def load_molecule(self, mol: Molecule):
        self.molecule = mol
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def reset_view(self):
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def set_preset(self, rx, ry, rz):
        from mol3d_avogadro import rotation_matrix
        self._rot  = rotation_matrix(math.radians(rx), math.radians(ry), math.radians(rz))
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def request_render(self, delay_ms: int = 0):
        if not self._render_timer.isActive():
            self._render_timer.start(delay_ms)

    def get_svg_bytes(self) -> bytes:
        return self._render_to_bytes()

    # ── internal ─────────────────────────────────────────────────────────────

    def _render_to_bytes(self, w=None, h=None) -> bytes:
        if self.molecule is None:
            return b""
        w = w or max(self.width(),  500)
        h = h or max(self.height(), 450)

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            tmp = f.name
        try:
            render_avogadro(
                self.molecule,
                rot_matrix_override=self._rot,
                pan_x=self._pan[0], pan_y=self._pan[1],
                canvas_w=w, canvas_h=h,
                scale=self.base_scale * self._zoom,
                atom_scale=self.atom_scale,
                bond_width_px=self.bond_width_px,
                background=self.background,
                color_overrides=self.color_overrides or None,
                output_path=tmp,
            )
            with open(tmp,"rb") as f:
                return f.read()
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    def _do_render(self):
        data = self._render_to_bytes()
        if data:
            self.load(QByteArray(data))

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_mode  = "rotate"
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start = event.position().toPoint()
            self._drag_mode  = "pan"
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

    def mouseMoveEvent(self, event):
        if self._drag_start is None or self.molecule is None:
            return
        cur = event.position().toPoint()
        dx  = cur.x() - self._drag_start.x()
        dy  = cur.y() - self._drag_start.y()
        self._drag_start = cur

        if self._drag_mode == "rotate":
            sens = 0.008
            ax, ay = dy*sens, dx*sens
            cx,sx  = math.cos(ax), math.sin(ax)
            cy,sy  = math.cos(ay), math.sin(ay)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            self._rot = Rx @ Ry @ self._rot
        elif self._drag_mode == "pan":
            self._pan[0] += dx
            self._pan[1] += dy

        self.rotationChanged.emit()
        self.request_render(delay_ms=16)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        self._drag_mode  = "none"
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.request_render(delay_ms=0)

    def wheelEvent(self, event):
        if self.molecule is None:
            return
        factor = 1.12 if event.angleDelta().y() > 0 else (1/1.12)
        self._zoom = max(0.15, min(6.0, self._zoom * factor))
        self.rotationChanged.emit()
        self.request_render(delay_ms=16)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.request_render(delay_ms=80)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mol3d — Molecule Viewer")
        self.resize(1080, 720)
        self.setStyleSheet(STYLESHEET)
        self._color_overrides: dict = {}   # elem -> hex
        self._build_menubar()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._show_placeholder()

    # ── menu bar ─────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()

        # ── File ──
        file_menu = mb.addMenu("&File")

        act_open = QAction("&Open…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_file)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_save_svg = QAction("Save &SVG…", self)
        act_save_svg.setShortcut("Ctrl+S")
        act_save_svg.triggered.connect(self._save_svg)
        file_menu.addAction(act_save_svg)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ── Edit ──
        edit_menu = mb.addMenu("&Edit")

        act_appearance = QAction("&Appearance…", self)
        act_appearance.setShortcut("Ctrl+P")
        act_appearance.triggered.connect(self._edit_appearance)
        edit_menu.addAction(act_appearance)

        act_colors = QAction("&Atom Colours…", self)
        act_colors.setShortcut("Ctrl+Shift+C")
        act_colors.triggered.connect(self._edit_atom_colors)
        edit_menu.addAction(act_colors)

        act_reset_colors = QAction("Reset Colours to &CPK", self)
        act_reset_colors.triggered.connect(self._reset_colors)
        edit_menu.addAction(act_reset_colors)

        # ── View ──
        view_menu = mb.addMenu("&View")

        act_reset_view = QAction("&Reset View", self)
        act_reset_view.setShortcut("Ctrl+R")
        act_reset_view.triggered.connect(lambda: self._canvas.reset_view())
        view_menu.addAction(act_reset_view)

        view_menu.addSeparator()

        presets_menu = view_menu.addMenu("&Preset Orientation")
        for label, (rx,ry,rz) in [
            ("Top",         ( 5,  0,  0)),
            ("Bottom",      (175, 0,  0)),
            ("Front",       ( 0,  0,  0)),
            ("Back",        ( 0,180,  0)),
            ("Left",        ( 0,-90,  0)),
            ("Right",       ( 0, 90,  0)),
            ("Perspective", (55, 20, 15)),
        ]:
            a = QAction(label, self)
            a.triggered.connect(lambda _, r=(rx,ry,rz): self._canvas.set_preset(*r))
            presets_menu.addAction(a)

        view_menu.addSeparator()

        act_bg = QAction("&Background Colour…", self)
        act_bg.triggered.connect(self._pick_background)
        view_menu.addAction(act_bg)

        # ── Help ──
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About mol3d", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ── toolbar  (only the most frequent actions) ─────────────────────────────

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16,16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        def tb_action(label, slot, shortcut=None, tooltip=None):
            a = QAction(label, self)
            if shortcut: a.setShortcut(shortcut)
            if tooltip:  a.setToolTip(tooltip)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        tb_action("⊕ Open",      self._open_file, "Ctrl+O", "Open molecule file")
        tb_action("⊞ Save SVG",  self._save_svg,  "Ctrl+S", "Export current view as SVG")
        tb.addSeparator()
        tb_action("⟳ Reset",     lambda: self._canvas.reset_view(), "Ctrl+R")
        tb.addSeparator()

        # Zoom readout
        self._zoom_lbl = QLabel("100%")
        self._zoom_lbl.setFixedWidth(46)
        self._zoom_lbl.setStyleSheet(f"color:{ACCENT}; font-size:11px; padding:0 4px;")
        tb.addWidget(self._zoom_lbl)

        tb.addSeparator()

        tb.addWidget(QLabel("Scale:"))
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(40, 220)
        self._scale_slider.setValue(110)
        self._scale_slider.setFixedWidth(120)
        self._scale_slider.setToolTip("Base render scale (Å → pixels)")
        self._scale_slider.valueChanged.connect(self._on_scale_change)
        tb.addWidget(self._scale_slider)

    # ── central layout ────────────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(172)
        sidebar.setStyleSheet(f"background:{PANEL_BG}; border-right:1px solid {BORDER};")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10,14,10,14)
        sl.setSpacing(10)

        # Molecule info
        info = QGroupBox("Molecule")
        il   = QVBoxLayout(info)
        self._lbl_name  = QLabel("—")
        self._lbl_name.setWordWrap(True)
        self._lbl_name.setStyleSheet("color:#e0e0ff; font-weight:bold; font-size:11px;")
        self._lbl_atoms = QLabel("Atoms: —")
        self._lbl_bonds = QLabel("Bonds: —")
        self._lbl_src   = QLabel("Source: —")
        self._lbl_src.setStyleSheet(f"color:{FG_DIM}; font-size:10px;")
        for w in (self._lbl_name, self._lbl_atoms, self._lbl_bonds, self._lbl_src):
            il.addWidget(w)
        sl.addWidget(info)

        # Legend
        self._legend = LegendPanel()
        sl.addWidget(self._legend)
        sl.addStretch()

        # Hint
        hint = QLabel("Drag  rotate\nRight-drag  pan\nScroll  zoom")
        hint.setStyleSheet(f"color:#334455; font-size:10px; line-height:160%;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(hint)

        root.addWidget(sidebar)

        # Canvas
        self._canvas = MoleculeCanvas()
        self._canvas.rotationChanged.connect(self._on_rotation_changed)
        root.addWidget(self._canvas, 1)

    # ── status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "No file loaded — use File → Open to load an .xyz, .gjf, .com or .log file."
        )

    # ── placeholder ───────────────────────────────────────────────────────────

    def _show_placeholder(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="450" viewBox="0 0 600 450">' 
            '<rect width="600" height="450" fill="#0a0a12"/>' 
            '<text x="300" y="210" text-anchor="middle" font-family="Courier New" font-size="15" fill="#2a3a55">' 
            'Open a molecule file to begin' 
            '</text>' 
            '</svg>'
        ).encode("ascii")
        self._canvas.load(QByteArray(svg))
    # ── file open / save ──────────────────────────────────────────────────────

    def _load_mol_from_path(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        ext  = os.path.splitext(path)[1].lower()
        base = os.path.basename(path)

        if ext == ".xyz":
            mol = parse_xyz(text, name=os.path.splitext(base)[0])
            src = "XYZ"
        elif ext in (".gjf", ".com"):
            mol = parse_gaussian(text)
            src = "Gaussian input"
        elif ext in (".log", ".out"):
            mol = parse_gaussian_log(text)
            src = "Gaussian log (last geometry)"
        else:
            # Try XYZ first, then Gaussian input
            try:
                mol = parse_xyz(text, name=base)
                src = "XYZ (auto)"
            except Exception:
                mol = parse_gaussian(text)
                src = "Gaussian input (auto)"

        infer_bonds(mol)
        return mol, src

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "All supported (*.xyz *.gjf *.com *.log *.out);;"
            "XYZ (*.xyz);;Gaussian input (*.gjf *.com);;"
            "Gaussian log (*.log *.out);;All files (*)"
        )
        if not path:
            return
        try:
            mol, src = self._load_mol_from_path(path)
            self._color_overrides = {}
            self._canvas.color_overrides = {}
            self._canvas.base_scale = float(self._scale_slider.value())
            self._canvas.load_molecule(mol)
            self._legend.update_for(mol, {})
            self._lbl_name.setText(mol.name)
            self._lbl_atoms.setText(f"Atoms: {len(mol.atoms)}")
            self._lbl_bonds.setText(f"Bonds: {len(mol.bonds)}")
            self._lbl_src.setText(f"Source: {src}")
            self._status.showMessage(
                f"{path}  |  {len(mol.atoms)} atoms, {len(mol.bonds)} bonds  [{src}]"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error loading file",
                                 f"{type(e).__name__}: {e}")

    def _save_svg(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load a molecule first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", "molecule.svg", "SVG files (*.svg)"
        )
        if not path:
            return
        try:
            data = self._canvas.get_svg_bytes()
            with open(path, "wb") as f:
                f.write(data)
            self._status.showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error saving", str(e))

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _edit_appearance(self):
        dlg = AppearanceDialog(
            self._canvas.atom_scale,
            self._canvas.bond_width_px,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._canvas.atom_scale    = dlg.atom_scale
            self._canvas.bond_width_px = dlg.bond_width
            self._canvas.request_render()

    def _edit_atom_colors(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load a molecule first.")
            return
        elements = sorted({a.element for a in mol.atoms})
        dlg = AtomColorDialog(elements, self._color_overrides, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._color_overrides = dlg.get_overrides()
            self._canvas.color_overrides = self._color_overrides
            self._legend.update_for(mol, self._color_overrides)
            self._canvas.request_render()

    def _reset_colors(self):
        self._color_overrides = {}
        self._canvas.color_overrides = {}
        if self._canvas.molecule:
            self._legend.update_for(self._canvas.molecule, {})
        self._canvas.request_render()
        self._status.showMessage("Atom colours reset to CPK defaults.")

    # ── View actions ──────────────────────────────────────────────────────────

    def _pick_background(self):
        col = QColorDialog.getColor(
            QColor(self._canvas.background), self, "Pick Background Colour"
        )
        if col.isValid():
            self._canvas.background = col.name()
            self._canvas.request_render()

    # ── Help ──────────────────────────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(self, "About mol3d",
            "<b>mol3d</b> — 3D Molecule Viewer<br><br>"
            "Avogadro-style ball-and-stick rendering.<br>"
            "Parsers: XYZ · Gaussian input (.gjf/.com) · Gaussian log (.log/.out)<br><br>"
            "Controls:<br>"
            "  &nbsp; Left-drag &nbsp;&nbsp; Rotate<br>"
            "  &nbsp; Right-drag &nbsp; Pan<br>"
            "  &nbsp; Scroll &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Zoom<br><br>"
            "Dependencies: PyQt6 · NumPy · svgwrite"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_scale_change(self, value):
        self._canvas.base_scale = float(value)
        self._canvas.request_render(delay_ms=60)

    def _on_rotation_changed(self):
        self._zoom_lbl.setText(f"{int(self._canvas._zoom*100)}%")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("mol3d")
    app.setFont(QFont("Segoe UI", 10))

    win = MainWindow()
    win.show()

    # Optional: open file from command line
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        try:
            mol, src = win._load_mol_from_path(sys.argv[1])
            win._canvas.load_molecule(mol)
            win._legend.update_for(mol, {})
            win._lbl_name.setText(mol.name)
            win._lbl_atoms.setText(f"Atoms: {len(mol.atoms)}")
            win._lbl_bonds.setText(f"Bonds: {len(mol.bonds)}")
            win._lbl_src.setText(f"Source: {src}")
        except Exception as e:
            print(f"Warning: could not open {sys.argv[1]}: {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()