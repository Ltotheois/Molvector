"""
molvector_gui.py — Interactive 3D Molecule Viewer
==============================================
PyQt6 GUI with full menu bar, appearance controls, and atom colour editor.

Controls:
  Left-drag    Rotate molecule
  Right-drag   Pan
  Scroll       Zoom

Menus:
  File    Open / Save SVG / Quit
  Edit    Appearance (ball size, bond width) / Atom Colours / Reset Colors / Info
  View    Preset orientations / Reset View / Background color
  Help    About

Dependencies:
    pip install PyQt6 numpy svgwrite
"""

import sys, os, math, json, tempfile, platform
from typing import List, Tuple, Optional, Dict
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QSlider, QStatusBar, QFrame, QSizePolicy,
    QGroupBox, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QColorDialog, QPushButton, QGridLayout,
    QScrollArea, QToolBar, QMenu, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QComboBox, QPlainTextEdit, QLineEdit,
    QButtonGroup, QRadioButton, QKeySequenceEdit,
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import Qt, QByteArray, QPoint, QPointF, pyqtSignal, QTimer, QSize, QRect, QRectF, QUrl, QMimeData
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPalette, QFont, QCursor, QIcon, QPixmap, QImage, QPainter, QPdfWriter, QPageSize, QKeySequence
from PyQt6.QtGui import QDrag

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── renderer / parsers ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from molvector_render import (
    parse_xyz, parse_gaussian, parse_gaussian_log, parse_pdb, infer_bonds,
    render_molecule, Molecule, CPK_BASE, CPK_DARK, VDW_RADII,
    lighten, darken, hex_to_rgb, rgb_to_hex, auto_dark,
    chemical_formula, molecular_mass, VibrationalMode, ExcitedState,
    save_xyz, save_gaussian_input, save_pdb, project_molecule, Atom, Bond,
    optimize_geometry, HAS_OPENBABEL,
)

def load_colored_icon(svg_path: str, color: str, size: int = 22) -> QIcon:
    """Load an SVG file, replace #000000 fills with color, return QIcon."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    svg_content = svg_content.replace('#000000', color)
    renderer = QSvgRenderer(QByteArray(svg_content.encode("utf-8")))
    default_size = renderer.defaultSize()
    if default_size.isValid() and default_size.width() > 0 and default_size.height() > 0:
        aspect = default_size.width() / default_size.height()
        if aspect >= 1.0:
            w, h = size, int(size / aspect)
        else:
            w, h = int(size * aspect), size
    else:
        w = h = size
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    x = (size - w) // 2
    y = (size - h) // 2
    renderer.render(painter, QRectF(x, y, w, h))
    painter.end()
    return QIcon(pix)

def set_icons(app, win):
    if platform.system() == "Darwin":
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "molvector_macos.png")
    else:
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icons", "icon.svg")
    
    if not os.path.isfile(icon_path):
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.svg")
    
    if not os.path.isfile(icon_path):
        return
    
    icon = QIcon(icon_path)
    app.setWindowIcon(icon)
    win.setWindowIcon(icon)

def get_safe_filename(name: str) -> str:
    """C60+ -> C60p, removes special characters."""
    s = name.replace("+", "p").replace("-", "m")
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '', s)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ELEM_FULL_NAME = {
    "H":"Hydrogen", "He":"Helium",  "Li":"Lithium", "Be":"Beryllium","B":"Boron",
    "C":"Carbon",   "N":"Nitrogen", "O":"Oxygen",   "F":"Fluorine",  "Ne":"Neon",
    "Na":"Sodium",  "Mg":"Magnesium","Al":"Aluminium","Si":"Silicon", "P":"Phosphorus",
    "S":"Sulfur",   "Cl":"Chlorine", "Ar":"Argon",   "K":"Potassium", "Ca":"Calcium",
    "Sc":"Scandium","Ti":"Titanium", "V":"Vanadium", "Cr":"Chromium", "Mn":"Manganese",
    "Fe":"Iron",     "Co":"Cobalt",   "Ni":"Nickel",   "Cu":"Copper",   "Zn":"Zinc",
    "Ga":"Gallium",  "Ge":"Germanium","As":"Arsenic",  "Se":"Selenium", "Br":"Bromine",
    "Kr":"Krypton",  "Rb":"Rubidium", "Sr":"Strontium","Y":"Yttrium",   "Zr":"Zirconium",
    "Nb":"Niobium",  "Mo":"Molybdenum","Tc":"Technetium","Ru":"Ruthenium","Rh":"Rhodium",
    "Pd":"Palladium","Ag":"Silver",   "Cd":"Cadmium",  "In":"Indium",   "Sn":"Tin",
    "Sb":"Antimony", "Te":"Tellurium","I":"Iodine",   "Xe":"Xenon",    "Cs":"Caesium",
    "Ba":"Barium",   "La":"Lanthanum", "Ce":"Cerium",   "Pr":"Praseodymium","Nd":"Neodymium",
    "Pm":"Promethium","Sm":"Samarium", "Eu":"Europium", "Gd":"Gadolinium","Tb":"Terbium",
    "Dy":"Dysprosium","Ho":"Holmium",  "Er":"Erbium",   "Tm":"Thulium",  "Yb":"Ytterbium",
    "Lu":"Lutetium", "Hf":"Hafnium",  "Ta":"Tantalum", "W":"Tungsten",  "Re":"Rhenium",
    "Os":"Osmium",   "Ir":"Iridium",  "Pt":"Platinum", "Au":"Gold",     "Hg":"Mercury",
    "Tl":"Thallium", "Pb":"Lead",     "Bi":"Bismuth",  "Po":"Polonium", "At":"Astatine",
    "Rn":"Radon",    "Fr":"Francium", "Ra":"Radium",
    "Ac":"Actinium","Th":"Thorium",  "Pa":"Protactinium","U":"Uranium",
    "Np":"Neptunium","Pu":"Plutonium","Am":"Americium","Cm":"Curium",
    "Bk":"Berkelium","Cf":"Californium","Es":"Einsteinium","Fm":"Fermium",
    "Md":"Mendelevium","No":"Nobelium","Lr":"Lawrencium",
    "Rf":"Rutherfordium","Db":"Dubnium","Sg":"Seaborgium","Bh":"Bohrium",
    "Hs":"Hassium","Mt":"Meitnerium","Ds":"Darmstadtium","Rg":"Roentgenium",
    "Cn":"Copernicium","Nh":"Nihonium","Fl":"Flerovium","Mc":"Moscovium",
    "Lv":"Livermorium","Ts":"Tennessine","Og":"Oganesson",
}

# Theme Colors
THEMES = {
    "dark": {
        "DARK_BG":  "#0f0f1a",
        "PANEL_BG": "#0d0d18",
        "CARD_BG":  "#13131f",
        "BORDER":   "#2a2a44",
        "FG":       "#ccd6f6",
        "FG_DIM":   "#99aacc",
        "ACCENT":   "#4488cc",
        "CANVAS":   "#0a0a12"
    },
    "light": {
        "DARK_BG":  "#f0f2f5",
        "PANEL_BG": "#e4e6eb",
        "CARD_BG":  "#ffffff",
        "BORDER":   "#ced4da",
        "FG":       "#1c1e21",
        "FG_DIM":   "#65676b",
        "ACCENT":   "#007bff",
        "CANVAS":   "#ffffff"
    }
}

def get_stylesheet(theme_name: str) -> str:
    t = THEMES[theme_name]
    return f"""
    QMainWindow, QDialog {{ background:{t['DARK_BG']}; }}
    QWidget {{ color:{t['FG']}; }}
    QMenuBar {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border-bottom:1px solid {t['BORDER']}; padding:2px 4px;
    }}
    QMenuBar::item:selected {{ background:{t['BORDER']}; border-radius:3px; }}
    QMenu {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border:1px solid {t['BORDER']}; border-radius:6px;
        padding:4px;
    }}
    QMenu::item {{ padding:5px 24px 5px 10px; border-radius:4px; }}
    QMenu::item:selected {{ background:{t['ACCENT']}; color:#fff; }}
    QMenu::separator {{ height:1px; background:{t['BORDER']}; margin:3px 6px; }}
    QToolBar {{
        background:{t['CARD_BG']}; border-bottom:1px solid {t['BORDER']};
        spacing:4px; padding:3px 8px;
    }}
    QToolBar QToolButton {{
        background:transparent; color:{t['FG']};
        border:1px solid transparent; border-radius:4px; padding:4px 8px;
    }}
    QToolBar QToolButton:hover {{ background:{t['BORDER']}; }}
    QToolBar QToolButton:checked {{
        background:{t['ACCENT']}; color:white;
        border:1px solid {t['ACCENT']};
    }}
    QPushButton {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border:1px solid {t['BORDER']}; border-radius:5px;
        padding:5px 14px; font-size:12px;
    }}
    QPushButton:hover  {{ background:{t['BORDER']}; border-color:{t['ACCENT']}; }}
    QToolTip {{
        background: {t['CARD_BG']}; color: {t['FG']};
        border: 1px solid {t['ACCENT']}; border-radius: 4px;
        padding: 4px;
    }}
    QPushButton:pressed{{ background:{t['ACCENT']}; }}
    QPushButton#accent {{
        background:#1e4080; border-color:{t['ACCENT']}; color:#fff;
    }}
    QPushButton#accent:hover {{ background:#2255aa; }}
    QPushButton#color_btn {{
        border-radius:4px; min-width:36px; min-height:24px;
        padding:2px; border:2px solid {t['BORDER']};
    }}
    QPushButton#color_btn:hover {{ border-color:{t['ACCENT']}; }}
    QLabel {{ color:{t['FG']}; }}
    QSlider::groove:horizontal {{
        height:4px; background:{t['BORDER']}; border-radius:2px;
    }}
    QSlider::handle:horizontal {{
        background:{t['ACCENT']}; border-radius:6px;
        width:14px; height:14px; margin:-5px 0;
    }}
    QSlider::sub-page:horizontal {{ background:{t['ACCENT']}; border-radius:2px; }}
    QDoubleSpinBox, QSpinBox, QComboBox {{
        background:{t['DARK_BG']}; border:1px solid {t['BORDER']};
        border-radius:4px; padding:3px 6px; color:{t['FG']};
    }}
    QComboBox::drop-down {{
        border:none; width:20px;
    }}
    QComboBox QAbstractItemView {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border:1px solid {t['BORDER']}; selection-background-color:{t['ACCENT']};
        selection-color:#fff; outline:none;
    }}
    QCheckBox {{
        spacing:6px; color:{t['FG']};
    }}
    QCheckBox::indicator {{
        width:14px; height:14px; border:1px solid {t['BORDER']};
        border-radius:3px; background:{t['DARK_BG']};
    }}
    QCheckBox::indicator:checked {{
        background:{t['ACCENT']}; border-color:{t['ACCENT']};
    }}
    QGroupBox {{
        color:{t['ACCENT']}; border:1px solid {t['BORDER']};
        border-radius:6px; margin-top:10px; font-size:11px;
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
    QStatusBar {{ background:{t['PANEL_BG']}; color:{t['FG_DIM']}; font-size:11px; }}
    QScrollArea {{ background:transparent; border:none; }}
    QScrollArea > QWidget > QWidget {{ background:transparent; }}
    QDialogButtonBox QPushButton {{ min-width:80px; }}
    QTabWidget::pane {{ border: 1px solid {t['BORDER']}; background: transparent; }}
    QTabBar::tab {{
        background: {t['DARK_BG']}; border: 1px solid {t['BORDER']};
        padding: 5px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{ background: {t['CARD_BG']}; border-bottom-color: {t['CARD_BG']}; }}
    QTableWidget {{ background: {t['DARK_BG']}; gridline-color: {t['BORDER']}; border: 1px solid {t['BORDER']}; }}
    QHeaderView::section {{ background: {t['CARD_BG']}; color: {t['FG']}; border: 1px solid {t['BORDER']}; }}
    
    QLabel#dim {{
        color:{t['FG_DIM']}; font-size:11px;
    }}
    QLabel#zoom_label {{
        color:{t['ACCENT']}; font-size:11px; padding:0 4px;
    }}
    QWidget#sidebar {{
        background:{t['PANEL_BG']}; border-right:1px solid {t['BORDER']};
    }}
    QLabel#mol_name {{
        color:{t['FG']}; font-weight:bold; font-size:13px; letter-spacing:1px;
    }}
    QLabel#hint {{
        color:{t['FG_DIM']}; font-size:10px; line-height:160%;
    }}
    QToolBar {{
        background:{t['PANEL_BG']}; border-bottom:1px solid {t['BORDER']};
        spacing: 4px; padding: 2px;
    }}
    QToolButton {{
        border-radius: 4px; padding: 4px; color: {t['FG']};
        border: 1px solid transparent;
    }}
    QToolButton:hover {{ background: {t['BORDER']}; }}
    QToolButton:checked {{
        background: {t['ACCENT']}; color: white;
        border: 1px solid {t['ACCENT']};
    }}
    QToolButton:checked {{
        background: {t['ACCENT']}; color: white;
        border: 1px solid {t['ACCENT']};
    }}
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

    def set_color(self, hex_color: str, emit: bool = True):
        orig = self._color
        self._color = hex_color
        self._update_swatch()
        if emit and orig != hex_color:
            self.colorChanged.emit(self._color)

    def _update_swatch(self):
        # The border/size is handled by the main stylesheet via #color_btn
        self.setStyleSheet(f"background:{self._color};")

    def _pick(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if col.isValid():
            self._color = col.name()
            self._update_swatch()
            self.colorChanged.emit(self._color)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS DIALOG  (Theme, ball size, bond width, background)
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceDialog(QDialog):
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "molvector_config.json")

    def __init__(self, atom_scale, bond_width, bond_style, color_overrides,
                 live_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Appearance")
        self.setMinimumWidth(400)
        self._live_callback = live_callback
        self._orig = (atom_scale, bond_width, bond_style,
                      dict(color_overrides))

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        # Ball size
        self._ball_slider = QSlider(Qt.Orientation.Horizontal)
        self._ball_slider.setRange(20, 150)
        self._ball_slider.setValue(int(atom_scale * 100))
        self._ball_slider.setFixedWidth(160)
        self._ball_lbl = QLabel(f"{atom_scale:.2f}")
        self._ball_slider.valueChanged.connect(self._on_change)
        ball_row = QHBoxLayout()
        ball_row.addWidget(self._ball_slider)
        ball_row.addWidget(self._ball_lbl)
        form.addRow("Ball Size:", ball_row)

        # Bond width
        self._bondw_slider = QSlider(Qt.Orientation.Horizontal)
        self._bondw_slider.setRange(2, 30)
        self._bondw_slider.setValue(int(bond_width))
        self._bondw_slider.setFixedWidth(160)
        self._bondw_lbl = QLabel(f"{bond_width:.0f}")
        self._bondw_slider.valueChanged.connect(self._on_change)
        bondw_row = QHBoxLayout()
        bondw_row.addWidget(self._bondw_slider)
        bondw_row.addWidget(self._bondw_lbl)
        form.addRow("Bond Width:", bondw_row)

        # Bond style
        style_group = QButtonGroup(self)
        style_row = QHBoxLayout()
        for s in ("gradient", "grey", "splitted"):
            rb = QRadioButton(s.capitalize())
            if s == bond_style:
                rb.setChecked(True)
            style_group.addButton(rb)
            style_row.addWidget(rb)
        self._style_btns = style_group
        style_row.addStretch()
        form.addRow("Bond Style:", style_row)

        # Atom colours
        self._color_overrides = color_overrides
        self._edit_colors_btn = QPushButton("Edit Atom Colours…")
        self._edit_colors_btn.clicked.connect(self._edit_atom_colors)
        form.addRow("Atom Colours:", self._edit_colors_btn)

        layout.addLayout(form)

        # Bottom buttons
        layout.addSpacing(6)
        btn_row = QHBoxLayout()
        btn_make_config = QPushButton("Make Default")
        btn_make_config.clicked.connect(self._save_config)
        btn_restore = QPushButton("Restore Defaults")
        btn_restore.clicked.connect(self._restore_defaults)
        btn_row.addWidget(btn_make_config)
        btn_row.addWidget(btn_restore)
        btn_row.addStretch()
        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self.accept)
        dialog_btns.rejected.connect(self.reject)
        btn_row.addWidget(dialog_btns)
        layout.addLayout(btn_row)

    def _on_change(self):
        if self._live_callback:
            self._live_callback(
                self.ball_scale, self.bond_width, self.bond_style,
                self._color_overrides,
            )

    @property
    def ball_scale(self) -> float:
        return self._ball_slider.value() / 100.0

    @property
    def bond_width(self) -> float:
        return float(self._bondw_slider.value())

    @property
    def bond_style(self) -> str:
        for rb in self._style_btns.buttons():
            if rb.isChecked():
                return rb.text().lower()
        return "gradient"

    def _edit_atom_colors(self):
        mol = self.parent()._canvas.molecule if self.parent() else None
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        elements = sorted({a.element for a in mol.atoms})
        dlg = AtomColorDialog(elements, self._color_overrides, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._color_overrides = dlg.get_overrides()
            self._on_change()

    def _save_config(self):
        config = {
            "atom_scale": self.ball_scale,
            "bond_width_px": self.bond_width,
            "bond_style": self.bond_style,
            "color_overrides": self._color_overrides,
        }
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Config Saved",
                f"Settings saved to:\n{self.CONFIG_FILE}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config:\n{e}")

    def _restore_defaults(self):
        self._ball_slider.setValue(70)
        self._bondw_slider.setValue(10)
        for rb in self._style_btns.buttons():
            if rb.text().lower() == "grey":
                rb.setChecked(True)
                break
        self._color_overrides = {}
        self._on_change()

    @staticmethod
    def load_config():
        path = AppearanceDialog.CONFIG_FILE
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return None


class SettingsDialog(QDialog):
    def __init__(self, theme, bg_color, live_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(350)
        self._live_callback = live_callback

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        form = QFormLayout()
        form.setSpacing(10)

        # Theme
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.setCurrentText(theme.capitalize())
        self._theme_combo.currentTextChanged.connect(self._on_change)
        form.addRow("Theme:", self._theme_combo)

        # Background Color
        self._bg_btn = ColorButton(bg_color)
        self._bg_btn.colorChanged.connect(self._on_change)
        form.addRow("Background:", self._bg_btn)

        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_change(self):
        if self._live_callback:
            self._live_callback(
                self._theme_combo.currentText().lower(),
                self._bg_btn.color(),
            )

    @property
    def theme(self) -> str: return self._theme_combo.currentText().lower()
    @property
    def bg_color(self) -> str: return self._bg_btn.color()


# ─────────────────────────────────────────────────────────────────────────────
# ATOM COLOUR EDITOR DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class AtomColorDialog(QDialog):
    """Shows one colour-picker row per element present in the molecule."""

    def __init__(self, elements: list, current_overrides: dict, live_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atom Colours")
        self.setMinimumWidth(360)
        self._live_callback = live_callback

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        lbl = QLabel("Click a swatch to change an element's colour.")
        lbl.setObjectName("dim")
        lbl.setStyleSheet("font-size:11px;")
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
            name_lbl.setObjectName("dim")

            btn = ColorButton(current)
            btn.colorChanged.connect(self._trigger_live)
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

    def _trigger_live(self, *_):
        if self._live_callback:
            self._live_callback(self.get_overrides())

    def get_overrides(self) -> dict:
        return {elem: btn.color() for elem, btn in self._buttons.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRUM DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumDialog(QDialog):
    def __init__(self, x_data, y_data, x_label, y_label, title, metadata="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(700, 500)
        self._x = np.array(x_data)
        self._y = np.array(y_data)
        self._meta = metadata
        self._xl, self._yl = x_label, y_label
        self._default_file = "spectrum.txt"

        layout = QVBoxLayout(self)
        
        if not HAS_MPL:
            layout.addWidget(QLabel("Matplotlib not found. Cannot display plot."))
        else:
            # Detect current theme from parent
            theme_name = "dark"
            p = parent
            while p:
                if hasattr(p, "_current_theme"):
                    theme_name = p._current_theme
                    break
                p = p.parent()
            
            t = THEMES[theme_name]
            
            self.fig = Figure(figsize=(6, 4), facecolor=t['CARD_BG'])
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(t['DARK_BG'])
            
            # Plot stems/peaks
            self.ax.vlines(self._x, 0, self._y, color=t['ACCENT'], linewidth=2)
            self.ax.scatter(self._x, self._y, color=t['ACCENT'], s=20)
            
            self.ax.set_xlabel(x_label, color=t['FG'])
            self.ax.set_ylabel(y_label, color=t['FG'])
            self.ax.tick_params(colors=t['FG_DIM'])
            for spine in self.ax.spines.values():
                spine.set_edgecolor(t['BORDER'])
            
            self.fig.tight_layout()
            layout.addWidget(self.canvas)

            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            layout.addWidget(self.toolbar)

        btns = QHBoxLayout()
        btn_export = QPushButton("Export Data (.txt)")
        btn_export.clicked.connect(self._export)
        btns.addStretch()
        btns.addWidget(btn_export)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    def set_default_filename(self, name: str):
        self._default_file = name

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Spectrum", self._default_file, "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                if self._meta:
                    for line in self._meta.splitlines():
                        f.write(f"# {line}\n")
                f.write(f"# {self._xl}\t{self._yl}\n")
                for xi, yi in zip(self._x, self._y):
                    f.write(f"{xi:.6f}\t{yi:.6f}\n")
            QMessageBox.information(self, "Export Successful", f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not export: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATIONS DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class CalculationsDialog(QDialog):
    modeSelected = pyqtSignal(object)  # VibrationalMode
    stateSelected = pyqtSignal(object) # ExcitedState
    viewSpectrum = pyqtSignal(str)     # "ir" or "uvvis"
    animationToggled = pyqtSignal(bool)
    vectorsToggled = pyqtSignal(bool)

    def __init__(self, mol: Molecule, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Calculations — {mol.name}")
        self.setMinimumSize(650, 500)
        self.mol = mol

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Frequencies Tab ──
        if mol.vibrational_modes:
            freq_page = QWidget()
            fpl = QVBoxLayout(freq_page)
            
            self._freq_table = QTableWidget(len(mol.vibrational_modes), 3)
            self._freq_table.setHorizontalHeaderLabels(["Mode", "Freq (cm⁻¹)", "Intensity (km/mol)"])
            self._freq_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._freq_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self._freq_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            for i, m in enumerate(mol.vibrational_modes):
                self._freq_table.setItem(i, 0, QTableWidgetItem(str(m.index)))
                self._freq_table.setItem(i, 1, QTableWidgetItem(f"{m.frequency:.2f}"))
                self._freq_table.setItem(i, 2, QTableWidgetItem(f"{m.intensity:.2f}"))
            
            self._freq_table.itemSelectionChanged.connect(lambda t=self._freq_table: self._on_freq_sel(t))
            fpl.addWidget(self._freq_table)
            
            btn_ir = QPushButton("View IR Spectrum…")
            btn_ir.clicked.connect(lambda: self.viewSpectrum.emit("ir"))
            fpl.addWidget(btn_ir)
            
            # Animation control
            self._anim_check = QCheckBox("Show animation")
            self._anim_check.setObjectName("dim")
            self._anim_check.toggled.connect(self.animationToggled.emit)
            fpl.addWidget(self._anim_check)

            # Vector control
            self._vector_check = QCheckBox("Show displacement vectors")
            self._vector_check.setChecked(False)
            self._vector_check.setObjectName("dim")
            self._vector_check.toggled.connect(self.vectorsToggled.emit)
            fpl.addWidget(self._vector_check)

            btn_export_freq = QPushButton("Export Data (.txt)")
            btn_export_freq.clicked.connect(lambda: self._export_table(self._freq_table, "frequencies"))
            fpl.addWidget(btn_export_freq)
            
            self.tabs.addTab(freq_page, "Frequencies")

        # ── TDDFT Tab ──
        if mol.excited_states:
            td_page = QWidget()
            tpl = QVBoxLayout(td_page)
            
            self._state_table = QTableWidget(len(mol.excited_states), 4)
            self._state_table.setHorizontalHeaderLabels(["State", "Energy (eV)", "Wavelength (nm)", "f"])
            self._state_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._state_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self._state_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._state_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            for i, s in enumerate(mol.excited_states):
                self._state_table.setItem(i, 0, QTableWidgetItem(f"S{s.index} ({s.symmetry})"))
                self._state_table.setItem(i, 1, QTableWidgetItem(f"{s.energy_ev:.4f}"))
                self._state_table.setItem(i, 2, QTableWidgetItem(f"{s.wavelength_nm:.2f}"))
                self._state_table.setItem(i, 3, QTableWidgetItem(f"{s.oscillator_strength:.4f}"))
            
            self._state_table.itemSelectionChanged.connect(lambda t=self._state_table: self._on_state_sel(t))
            tpl.addWidget(self._state_table)
            
            btn_uv = QPushButton("View UV-Vis Spectrum…")
            btn_uv.clicked.connect(lambda: self.viewSpectrum.emit("uvvis"))
            tpl.addWidget(btn_uv)

            btn_export_state = QPushButton("Export Data (.txt)")
            btn_export_state.clicked.connect(lambda: self._export_table(self._state_table, "excited_states"))
            tpl.addWidget(btn_export_state)
            
            self.tabs.addTab(td_page, "Excited States (TDDFT)")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _on_freq_sel(self, table):
        rows = table.selectedItems()
        if not rows: return
        idx = int(rows[0].text()) - 1
        self.modeSelected.emit(self.mol.vibrational_modes[idx])

    def _on_state_sel(self, table):
        rows = table.selectedItems()
        if not rows: return
        # Extract "S1" -> 1
        row_text = rows[0].text()
        idx = int(row_text.split()[0][1:]) - 1
        self.stateSelected.emit(self.mol.excited_states[idx])

    def _export_table(self, table, kind):
        safe = get_safe_filename(self.mol.name)
        default_file = f"{safe}_{kind}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Export Table", default_file, "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Molecule: {self.mol.name}\n")
                f.write(f"# Calculation: {kind}\n")
                headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
                f.write("# " + "\t".join(headers) + "\n")
                for row in range(table.rowCount()):
                    vals = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        vals.append(item.text() if item else "")
                    f.write("\t".join(vals) + "\n")
            QMessageBox.information(self, "Export Successful", f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not export: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# G16 INPUT GENERATOR DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class G16InputDialog(QDialog):
    def __init__(self, mol, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate G16 Input")
        self.setMinimumSize(680, 540)
        self.setStyleSheet("""
            QComboBox:disabled, QSpinBox:disabled {
                background: #e8e8e8;
                color: #aaa;
                border: 1px solid #ccc;
            }
        """)
        self.mol = mol
        self._generating = False
        self._last_generated = ""
        self._all_fields = []

        layout = QVBoxLayout(self)

        # ── Route card fields (each with custom override) ──
        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def combo_row(items):
            row = QHBoxLayout()
            combo = QComboBox()
            combo.addItems(items + ["Custom"])
            row.addWidget(combo)
            custom = QLineEdit()
            custom.setPlaceholderText("Enter custom value…")
            custom.setFixedWidth(180)
            custom.setVisible(False)
            row.addWidget(custom)
            row.addStretch()
            return combo, custom, row

        self._job_combo, self._job_custom, j_row = combo_row(
            ["opt", "freq", "opt freq", "sp", "scan"])
        form.addRow("Job type:", j_row)

        self._method_combo, self._method_custom, m_row = combo_row(
            ["b3lyp", "wb97xd", "m062x", "mp2", "hf", "ccsd(t)"])
        form.addRow("Method:", m_row)

        self._basis_combo, self._basis_custom, b_row = combo_row(
            ["6-31g(d)", "6-311+g(d,p)", "aug-cc-pvdz", "def2svp", "def2tzvp"])
        form.addRow("Basis set:", b_row)

        # Toggle custom field visibility and reconnect signals
        for combo, custom in [(self._job_combo, self._job_custom),
                              (self._method_combo, self._method_custom),
                              (self._basis_combo, self._basis_custom)]:
            combo.currentTextChanged.connect(
                lambda t, c=custom: c.setVisible(t == "Custom"))
            combo.currentTextChanged.connect(self._on_field_changed)
            custom.textChanged.connect(self._on_field_changed)

        btn_style = """
            QSpinBox::up-button, QSpinBox::down-button { width: 24px; }
            QSpinBox:disabled { background: #e8e8e8; color: #aaa; border: 1px solid #ccc; }
        """

        chg_layout = QHBoxLayout()
        self._charge_spin = QSpinBox()
        self._charge_spin.setRange(-10, 10)
        self._charge_spin.setValue(mol.charge)
        self._charge_spin.setMinimumWidth(90)
        self._charge_spin.setStyleSheet(btn_style)
        self._charge_spin.valueChanged.connect(self._on_field_changed)
        chg_layout.addWidget(self._charge_spin)
        chg_layout.addSpacing(4)
        chg_layout.addWidget(QLabel("Mult:"))
        self._mult_spin = QSpinBox()
        self._mult_spin.setRange(1, 10)
        self._mult_spin.setValue(1)
        self._mult_spin.setMinimumWidth(90)
        self._mult_spin.setStyleSheet(btn_style)
        self._mult_spin.valueChanged.connect(self._on_field_changed)
        chg_layout.addWidget(self._mult_spin)
        chg_layout.addStretch()
        form.addRow("Charge / Mult:", chg_layout)

        proc_row = QHBoxLayout()
        self._nproc_spin = QSpinBox()
        self._nproc_spin.setRange(1, 64)
        self._nproc_spin.setValue(4)
        self._nproc_spin.setMinimumWidth(90)
        self._nproc_spin.setStyleSheet(btn_style)
        self._nproc_spin.valueChanged.connect(self._on_field_changed)
        proc_row.addWidget(self._nproc_spin)
        proc_row.addStretch()
        form.addRow("Processors:", proc_row)

        mem_row = QHBoxLayout()
        self._mem_combo = QComboBox()
        self._mem_combo.addItems(["2GB", "4GB", "8GB", "16GB", "32GB", "64GB"])
        self._mem_combo.setCurrentText("8GB")
        self._mem_combo.currentTextChanged.connect(self._on_field_changed)
        mem_row.addWidget(self._mem_combo)
        mem_row.addStretch()
        form.addRow("Memory:", mem_row)

        layout.addLayout(form)

        # ── Preview (editable) ──
        layout.addWidget(QLabel("Preview (editable):"))
        self._preview = QPlainTextEdit()
        self._preview.setMinimumHeight(180)
        self._preview.setStyleSheet("font-family: 'Courier New', monospace;")
        self._preview.textChanged.connect(self._on_preview_edited)
        layout.addWidget(self._preview)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save…")
        btn_save.clicked.connect(self._save)
        btn_layout.addWidget(btn_save)
        self._btn_sync = QPushButton("Sync from fields")
        self._btn_sync.clicked.connect(self._sync_from_fields)
        btn_layout.addWidget(self._btn_sync)
        btn_layout.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        # Collect all form fields for enable/disable
        self._all_fields = [
            self._job_combo, self._method_combo, self._basis_combo,
            self._charge_spin, self._mult_spin, self._nproc_spin,
            self._mem_combo,
        ]

        self._sync_from_fields()

    def _set_fields_enabled(self, enabled: bool):
        for w in self._all_fields:
            w.setEnabled(enabled)
        self._btn_sync.setEnabled(not enabled)

    def _on_preview_edited(self):
        if self._generating:
            return
        if self._preview.toPlainText() == self._last_generated:
            return
        self._set_fields_enabled(False)

    def _on_field_changed(self):
        self._sync_from_fields()

    def _field_val(self, combo, custom) -> str:
        if combo.currentText() == "Custom":
            txt = custom.text().strip()
            return txt if txt else "CUSTOM"
        return combo.currentText()

    def _build_route(self) -> str:
        job = self._field_val(self._job_combo, self._job_custom)
        method = self._field_val(self._method_combo, self._method_custom)
        basis = self._field_val(self._basis_combo, self._basis_custom)
        return f"# {job} {method}/{basis}"

    def _build_text(self) -> str:
        nproc = self._nproc_spin.value()
        mem = self._mem_combo.currentText()
        route = self._build_route()
        charge = self._charge_spin.value()
        mult = self._mult_spin.value()

        lines = [
            f"%nprocshared={nproc}",
            f"%mem={mem}",
            route,
            "",
            self.mol.name,
            "",
            f"{charge} {mult}",
        ]
        for a in self.mol.atoms:
            lines.append(f"{a.element:3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
        lines.append("")
        return "\n".join(lines)

    def _sync_from_fields(self):
        self._generating = True
        self._last_generated = self._build_text()
        self._preview.setPlainText(self._last_generated)
        self._generating = False
        self._set_fields_enabled(True)
        self._btn_sync.setEnabled(False)

    def _save(self):
        safe = self.mol.name.replace(" ", "_").replace("/", "_") or "input"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gaussian Input", f"{safe}.gjf",
            "Gaussian input (*.gjf *.com);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._build_text())
            QMessageBox.information(self, "Saved", f"Gaussian input saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CPK LEGEND PANEL  (sidebar)
# ─────────────────────────────────────────────────────────────────────────────

class LegendPanel(QGroupBox):
    elementColorChanged = pyqtSignal(str, str)  # (element_symbol, hex_color)

    def __init__(self, parent=None):
        super().__init__("Elements", parent)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(4)
        self._layout.setContentsMargins(10,14,10,10)

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

            swatch = QPushButton()
            swatch.setFixedSize(14, 14)
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(f"Click to change {name} colour")
            swatch.setStyleSheet(
                f"background:{color}; border-radius:7px; border:1px solid #555;"
            )
            swatch.clicked.connect(lambda _, e=e, c=color: self._pick_color(e, c))

            lbl = QLabel(f"{e} - {name}")
            lbl.setObjectName("dim")
            rl.addWidget(swatch)
            rl.addWidget(lbl)
            rl.addStretch()

            self._layout.addWidget(row)
            self._rows.append(row)

    def _pick_color(self, elem: str, current_hex: str):
        dlg = QColorDialog(QColor(current_hex), self)
        dlg.setWindowTitle(f"Pick Colour for {ELEM_FULL_NAME.get(elem, elem)}")
        if dlg.exec():
            new_hex = dlg.selectedColor().name()
            self.elementColorChanged.emit(elem, new_hex)

        self._layout.addStretch()


# ─────────────────────────────────────────────────────────────────────────────
# PERIODIC TABLE DIALOG
# ─────────────────────────────────────────────────────────────────────────────

_PERIODIC_TABLE_LAYOUT = [
    # (symbol, Z, row, col)
    ("H",1,0,0), ("He",2,0,17),
    ("Li",3,1,0), ("Be",4,1,1), ("B",5,1,12), ("C",6,1,13), ("N",7,1,14), ("O",8,1,15), ("F",9,1,16), ("Ne",10,1,17),
    ("Na",11,2,0), ("Mg",12,2,1), ("Al",13,2,12), ("Si",14,2,13), ("P",15,2,14), ("S",16,2,15), ("Cl",17,2,16), ("Ar",18,2,17),
    ("K",19,3,0), ("Ca",20,3,1), ("Sc",21,3,2), ("Ti",22,3,3), ("V",23,3,4), ("Cr",24,3,5), ("Mn",25,3,6), ("Fe",26,3,7), ("Co",27,3,8), ("Ni",28,3,9), ("Cu",29,3,10), ("Zn",30,3,11), ("Ga",31,3,12), ("Ge",32,3,13), ("As",33,3,14), ("Se",34,3,15), ("Br",35,3,16), ("Kr",36,3,17),
    ("Rb",37,4,0), ("Sr",38,4,1), ("Y",39,4,2), ("Zr",40,4,3), ("Nb",41,4,4), ("Mo",42,4,5), ("Tc",43,4,6), ("Ru",44,4,7), ("Rh",45,4,8), ("Pd",46,4,9), ("Ag",47,4,10), ("Cd",48,4,11), ("In",49,4,12), ("Sn",50,4,13), ("Sb",51,4,14), ("Te",52,4,15), ("I",53,4,16), ("Xe",54,4,17),
    ("Cs",55,5,0), ("Ba",56,5,1), ("La",57,5,2), ("Hf",72,5,3), ("Ta",73,5,4), ("W",74,5,5), ("Re",75,5,6), ("Os",76,5,7), ("Ir",77,5,8), ("Pt",78,5,9), ("Au",79,5,10), ("Hg",80,5,11), ("Tl",81,5,12), ("Pb",82,5,13), ("Bi",83,5,14), ("Po",84,5,15), ("At",85,5,16), ("Rn",86,5,17),
    ("Fr",87,6,0), ("Ra",88,6,1), ("Ac",89,6,2), ("Rf",104,6,3), ("Db",105,6,4), ("Sg",106,6,5), ("Bh",107,6,6), ("Hs",108,6,7), ("Mt",109,6,8), ("Ds",110,6,9), ("Rg",111,6,10), ("Cn",112,6,11), ("Nh",113,6,12), ("Fl",114,6,13), ("Mc",115,6,14), ("Lv",116,6,15), ("Ts",117,6,16), ("Og",118,6,17),
    # F-block: lanthanides
    ("Ce",58,7,2), ("Pr",59,7,3), ("Nd",60,7,4), ("Pm",61,7,5), ("Sm",62,7,6), ("Eu",63,7,7), ("Gd",64,7,8), ("Tb",65,7,9), ("Dy",66,7,10), ("Ho",67,7,11), ("Er",68,7,12), ("Tm",69,7,13), ("Yb",70,7,14), ("Lu",71,7,15),
    # F-block: actinides
    ("Th",90,8,2), ("Pa",91,8,3), ("U",92,8,4), ("Np",93,8,5), ("Pu",94,8,6), ("Am",95,8,7), ("Cm",96,8,8), ("Bk",97,8,9), ("Cf",98,8,10), ("Es",99,8,11), ("Fm",100,8,12), ("Md",101,8,13), ("No",102,8,14), ("Lr",103,8,15),
]

_ELEM_CATEGORY_COLORS = {
    "nonmetal":       "#4CAF50",
    "noble_gas":      "#9C27B0",
    "alkali_metal":   "#FF5722",
    "alkaline_earth": "#FF9800",
    "metalloid":      "#00BCD4",
    "halogen":        "#1E88E5",
    "transition":     "#42A5F5",
    "post_transition":"#78909C",
    "lanthanide":     "#E91E63",
    "actinide":       "#D32F2F",
    "unknown":        "#757575",
}

def _element_category(sym: str, Z: int) -> str:
    if Z == 1: return "nonmetal"
    if Z == 2: return "noble_gas"
    if 3 <= Z <= 4: return "alkali_metal" if Z == 3 else "alkaline_earth"
    if 5 <= Z <= 10:
        if Z == 5: return "metalloid"
        return "nonmetal" if Z in (6,7,8) else "halogen" if Z in (9,17,35,53,85,117) else "noble_gas"
    if 11 <= Z <= 18:
        if Z == 11: return "alkali_metal"
        if Z == 12: return "alkaline_earth"
        if Z in (13,): return "post_transition"
        if Z in (14,): return "metalloid"
        if Z in (15,16): return "nonmetal"
        if Z == 17: return "halogen"
        if Z == 18: return "noble_gas"
    if 19 <= Z <= 36:
        if Z in (19,37,55,87): return "alkali_metal"
        if Z in (20,38,56,88): return "alkaline_earth"
        if 21 <= Z <= 30: return "transition"
        if 31 <= Z <= 36:
            if Z == 31: return "post_transition"
            if Z in (32,): return "metalloid"
            if Z in (33,34): return "metalloid" if Z == 33 else "nonmetal"
            return "halogen" if Z == 35 else "noble_gas"
    if 37 <= Z <= 54:
        if Z in (37,55,87): return "alkali_metal"
        if Z in (38,56,88): return "alkaline_earth"
        if Z == 39 or (40 <= Z <= 48):
            return "transition"
        if 49 <= Z <= 54:
            if Z == 49: return "post_transition"
            if Z == 50: return "post_transition"
            if Z == 51: return "metalloid"
            if Z == 52: return "metalloid"
            return "halogen" if Z == 53 else "noble_gas"
    if 55 <= Z <= 86:
        if Z in (55,87): return "alkali_metal"
        if Z in (56,88): return "alkaline_earth"
        if 57 <= Z <= 71: return "lanthanide"
        if 72 <= Z <= 80: return "transition"
        if 81 <= Z <= 86:
            if Z == 81: return "post_transition"
            if Z in (82,83): return "post_transition"
            if Z in (84,): return "post_transition"
            return "halogen" if Z == 85 else "noble_gas"
    if 89 <= Z <= 103: return "actinide"
    if 104 <= Z <= 111: return "transition"
    if 112 <= Z <= 118:
        if Z in (113,114,115,116): return "post_transition"
        if Z == 117: return "halogen"
        if Z == 118: return "noble_gas"
    return "unknown"


class PeriodicTableDialog(QDialog):
    elementSelected = pyqtSignal(str)

    def __init__(self, parent=None, current_element: str = "C"):
        super().__init__(parent)
        self.setWindowTitle("Select Element — Periodic Table")
        self.setModal(True)
        self.setMinimumSize(1020, 640)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(3)

        btn_size = 52
        btn_font = QFont("Segoe UI", 10, QFont.Weight.Bold)

        by_Z = {}
        for sym, Z, r, c in _PERIODIC_TABLE_LAYOUT:
            by_Z[Z] = (sym, r, c)

        self._all_buttons: Dict[str, QPushButton] = {}

        for sym, Z, r, c in _PERIODIC_TABLE_LAYOUT:
            btn = QPushButton(sym)
            btn.setFixedSize(btn_size, btn_size)
            btn.setFont(btn_font)
            tooltip = f"{sym} ({Z}) — {ELEM_FULL_NAME.get(sym, 'Unknown')}"
            btn.setToolTip(tooltip)

            cat = _element_category(sym, Z)
            bg = _ELEM_CATEGORY_COLORS.get(cat, "#757575")
            is_dark = sum(int(bg[i:i+2],16) for i in (1,3,5)) < 400
            text_color = "#ffffff" if is_dark else "#000000"
            btn.setStyleSheet(
                f"QPushButton {{ background:{bg}; color:{text_color}; "
                f"border:1px solid rgba(0,0,0,0.2); border-radius:4px; }}"
                f"QPushButton:hover {{ border:2px solid white; }}"
            )

            is_current = (sym == current_element)
            if is_current:
                btn.setStyleSheet(
                    f"QPushButton {{ background:{bg}; color:{text_color}; "
                    f"border:2px solid #FFD700; border-radius:4px; }}"
                )

            def _make_handler(s):
                return lambda checked, sym=s: self._select_element(sym)

            btn.clicked.connect(_make_handler(sym))
            grid.addWidget(btn, r, c)
            self._all_buttons[sym] = btn

        # Headers for characteristic rows
        label_font = QFont("Segoe UI", 8)
        lbl_la = QLabel("Lanthanides")
        lbl_la.setFont(label_font)
        lbl_la.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(lbl_la, 7, 0, 1, 2)

        lbl_ac = QLabel("Actinides")
        lbl_ac.setFont(label_font)
        lbl_ac.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(lbl_ac, 8, 0, 1, 2)

        layout.addLayout(grid)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Cancel")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _select_element(self, sym: str):
        self.elementSelected.emit(sym)
        self.accept()





# ── Shortcut Configuration Dialog ───────────────────────────────────────────

class ShortcutDialog(QDialog):
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "molvector_config.json")

    def __init__(self, shortcut_actions: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Shortcuts")
        self.setMinimumSize(500, 400)
        self._actions = dict(shortcut_actions)

        layout = QVBoxLayout(self)

        # Table
        self._table = QTableWidget(len(self._actions), 2)
        self._table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self._editors = []
        for row, (aid, action) in enumerate(self._actions.items()):
            name_item = QTableWidgetItem(action.text().replace("&", ""))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            ks = QKeySequenceEdit(action.shortcut())
            ks.setClearButtonEnabled(True)
            self._table.setCellWidget(row, 1, ks)
            self._editors.append(ks)

        layout.addWidget(self._table)

        # Buttons
        btn_row = QHBoxLayout()
        btn_restore = QPushButton("Restore Defaults")
        btn_restore.clicked.connect(self._restore_defaults)
        btn_row.addWidget(btn_restore)

        btn_make_default = QPushButton("Make Default")
        btn_make_default.clicked.connect(self._make_default)
        btn_row.addWidget(btn_make_default)

        btn_row.addStretch()

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

    def _restore_defaults(self):
        for row, (aid, action) in enumerate(self._actions.items()):
            default = MainWindow.DEFAULT_SHORTCUTS.get(aid, "")
            self._editors[row].setKeySequence(QKeySequence(default))

    def _make_default(self):
        cfg = self._collect_shortcuts()
        config = {}
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
        except Exception:
            pass
        config["shortcuts"] = cfg
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Saved", f"Shortcuts saved to:\n{self.CONFIG_FILE}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save shortcuts:\n{e}")

    def get_shortcuts(self) -> dict:
        return self._collect_shortcuts()

    def _collect_shortcuts(self) -> dict:
        result = {}
        for row, (aid, action) in enumerate(self._actions.items()):
            ks = self._editors[row].keySequence()
            if not ks.isEmpty():
                result[aid] = ks.toString()
            else:
                result[aid] = ""
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Quick SVG Export Dialog ────────────────────────────────────────────────────

class QuickExportDialog(QDialog):
    """A dialog with a draggable SVG target for drag-and-drop into external editors."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._tmp_path = None
        self.setWindowTitle("Quick SVG Export")
        self.setFixedSize(320, 220)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl = QLabel("Drag the icon into Inkscape or another SVG editor:")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._drag_widget = _DragWidget(self._get_svg_path, self)
        self._drag_widget.setFixedSize(120, 80)
        layout.addWidget(self._drag_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        note = QLabel("Note: in Inkscape, you may have to drag the molecule to see gradients.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        note.setStyleSheet("font-size:10px;")
        layout.addWidget(note)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)

    def _get_svg_path(self) -> str:
        if self._tmp_path is None:
            svg_data = self._canvas.get_svg_bytes(export_mode=True)
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                f.write(svg_data)
                self._tmp_path = f.name
        return self._tmp_path

    def closeEvent(self, event):
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
            self._tmp_path = None
        super().closeEvent(event)


class _DragWidget(QPushButton):
    """A draggable pushbutton that initiates a file drag on mouse move."""

    def __init__(self, get_svg_path_cb, parent=None):
        super().__init__("Drag SVG", parent)
        self._get_svg_path = get_svg_path_cb
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setToolTip("Drag this into an SVG editor")
        self.setStyleSheet("""
            QPushButton {
                border: 2px dashed #888;
                border-radius: 8px;
                font-size: 14px;
                background: #f0f0f0;
                padding: 8px;
            }
            QPushButton:hover {
                border-color: #4a9eff;
                background: #e0edff;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.position().toPoint() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return

        svg_path = self._get_svg_path()

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(svg_path)])

        with open(svg_path, "rb") as f:
            mime.setData("image/svg+xml", f.read())

        drag = QDrag(self)
        drag.setMimeData(mime)

        pixmap = self.grab().scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        drag.setPixmap(pixmap)
        drag.setHotSpot(self.rect().center())

        drag.exec(Qt.DropAction.CopyAction)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CANVAS
# ─────────────────────────────────────────────────────────────────────────────

def _ideal_directions(n: int, angle_rad: float):
    """Return n unit vectors in ideal VSEPR-like geometry (tetrahedral, trigonal, bent)."""
    if n == 4:
        a = 1.0 / math.sqrt(3.0)
        return [np.array([a,a,a]), np.array([a,-a,-a]),
                np.array([-a,a,-a]), np.array([-a,-a,a])]
    elif n == 3:
        s3 = math.sqrt(3.0)/2.0
        return [np.array([1.0,0.0,0.0]), np.array([-0.5,s3,0.0]), np.array([-0.5,-s3,0.0])]
    elif n == 2:
        half = angle_rad / 2.0
        return [np.array([math.sin(half),0.0,math.cos(half)]),
                np.array([-math.sin(half),0.0,math.cos(half)])]
    dirs = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        phi = math.acos(2.0 * i / n - 1.0)
        dirs.append(np.array([math.sin(phi)*math.cos(theta),
                              math.sin(phi)*math.sin(theta), math.cos(phi)]))
    return dirs


class MoleculeCanvas(QSvgWidget):
    rotationChanged = pyqtSignal()
    fileDropped = pyqtSignal(str)
    moleculeChanged = pyqtSignal() # Emitted when atoms/bonds change
    requestHistorySave = pyqtSignal() # Emitted BEFORE a change

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
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._show_vectors = False
        self.build_mode = False
        self.selection_mode = False
        self.build_element = "C"
        self._bonding_from: int | None = None
        self._mouse_pos: QPoint | None = None

        # Selection state
        self.selected_atoms: set = set()
        self._sel_drag_start: QPoint | None = None
        self._sel_rect: QRectF | None = None

        # Atom dragging (Alt+click in build/select mode)
        self._drag_atom_idx: int | None = None

        # Render parameters — all public, set by MainWindow
        self.base_scale     = 110.0
        self.atom_scale     = 0.7
        self.bond_width_px  = 10.0
        self.background     = "#ffffff"
        self.bond_style     = "gradient"
        self.color_overrides: dict = {}
        self.animation_phase: float = 0.0
        self.animation_amplitude: float = 0.0

        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)

        self.arrows: Optional[List[Tuple[int, float, float, float, str]]] = None
        self.active_vectors: Optional[np.ndarray] = None
        self.animation_phase: float = 0.0
        self.animation_amplitude: float = 0.0
        
        # Build options
        self.build_mode: bool = False
        self.build_element: str = "C"
        self.auto_adjust_h: bool = False

    # ── drag and drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self.fileDropped.emit(path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    # ── public ───────────────────────────────────────────────────────────────

    def load_molecule(self, mol: Molecule):
        self.molecule = mol
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.selected_atoms.clear()
        # Auto-scale: fit the molecule to 80% of the smaller canvas dimension
        positions = np.array([[a.x, a.y, a.z] for a in mol.atoms])
        if len(positions) > 0:
            centroid = positions.mean(axis=0)
            centered = positions - centroid
            max_extent = np.max(np.linalg.norm(centered, axis=1)) + 1.5  # +1.5 Å for atom radius margin
            if max_extent > 0:
                w = max(self.width(),  500)
                h = max(self.height(), 450)
                self.base_scale = 0.40 * min(w, h) / max_extent
        self.request_render()

    def reset_view(self):
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def set_preset(self, rx, ry, rz):
        from molvector_render import rotation_matrix
        self._rot  = rotation_matrix(math.radians(rx), math.radians(ry), math.radians(rz))
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def edit_background_color(self):
        col = QColorDialog.getColor(QColor(self.background), self, "Background Colour")
        if col.isValid():
            self.background = col.name()
            self.request_render()

    def request_render(self, delay_ms: int = 0):
        if not self._render_timer.isActive():
            self._render_timer.start(delay_ms)

    def get_svg_bytes(self, export_mode: bool = False) -> bytes:
        return self._render_to_bytes(export_mode=export_mode)

    # ── internal ─────────────────────────────────────────────────────────────

    def _render_to_bytes(self, w=None, h=None, export_mode: bool = False) -> bytes:
        if self.molecule is None:
            return b""
        w = w or max(self.width(),  500)
        h = h or max(self.height(), 450)

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            tmp = f.name
        try:
            render_molecule(
                self.molecule,
                rot_matrix_override=self._rot,
                pan_x=self._pan[0], pan_y=self._pan[1],
                canvas_w=w, canvas_h=h,
                scale=self.base_scale * self._zoom,
                atom_scale=self.atom_scale,
                bond_width_px=self.bond_width_px,
                bond_style=self.bond_style,
                background=self.background,
                color_overrides=self.color_overrides or None,
                active_vectors=self.active_vectors,
                animation_phase=self.animation_phase,
                animation_amplitude=self.animation_amplitude,
                output_path=tmp,
                export_mode=export_mode,
                selected_indices=self.selected_atoms if not export_mode else None,
                vectors=self.arrows if self._show_vectors else None,
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
        # Overlay for bonding line
        if self.build_mode and self._bonding_from is not None and self._mouse_pos is not None:
            self.update() # Force paintEvent for overlay

    def paintEvent(self, event):
        super().paintEvent(event)
        draw_sel = self.selection_mode and self._sel_rect is not None and self._sel_rect.isValid()
        draw_bond = self.build_mode and self._bonding_from is not None and self._mouse_pos is not None and self.molecule
        if not draw_sel and not draw_bond:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if draw_sel:
            p.fillRect(self._sel_rect, QColor(0, 120, 255, 30))
            p.setPen(QColor(0, 140, 255, 200))
            p.drawRect(self._sel_rect.toRect())

        if draw_bond:
            p.setPen(QColor("#007bff"))
            atoms, _ = project_molecule(
                self.molecule, self._rot, self._pan[0], self._pan[1],
                self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
            )
            ax, ay, az, ar = atoms[self._bonding_from]
            p.drawLine(int(ax), int(ay), self._mouse_pos.x(), self._mouse_pos.y())

        p.end()

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _get_hit_atom(self, pos: QPoint) -> int | None:
        if not self.molecule: return None
        atoms, _ = project_molecule(
            self.molecule, self._rot, self._pan[0], self._pan[1],
            self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
        )
        for i, (ax, ay, az, ar) in enumerate(atoms):
            dist = math.hypot(pos.x() - ax, pos.y() - ay)
            if dist < max(ar, 10):
                return i
        return None

    def _get_hit_bond(self, pos: QPoint) -> int | None:
        if not self.molecule: return None
        _, bonds = project_molecule(
            self.molecule, self._rot, self._pan[0], self._pan[1],
            self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
        )
        threshold = 8.0
        for ax, ay, bx, by, z_avg, idx in bonds:
            # Distance from point to segment
            L2 = (bx-ax)**2 + (by-ay)**2
            if L2 < 1e-6: continue
            t = max(0, min(1, ((pos.x()-ax)*(bx-ax) + (pos.y()-ay)*(by-ay)) / L2))
            proj_x = ax + t * (bx-ax)
            proj_y = ay + t * (by-ay)
            dist = math.hypot(pos.x() - proj_x, pos.y() - proj_y)
            if dist < threshold:
                return idx
        return None

    def _unproject(self, pos: QPoint) -> np.ndarray:
        # returns 3D position in molecule local frame (relative to centroid)
        cx = self.width()/2 + self._pan[0]
        cy = self.height()/2 + self._pan[1]
        scale = self.base_scale * self._zoom
        
        # Assume z_local = 0
        rp_x = (pos.x() - cx) / scale
        rp_y = (cy - pos.y()) / scale
        rp = np.array([rp_x, rp_y, 0.0])
        # local = rot^-1 @ rp
        local = np.linalg.inv(self._rot) @ rp
        return local

    def mousePressEvent(self, event):
        self._update_cursor(event.position().toPoint(), event.modifiers())
        # Alt/Option+click drags an atom in build or selection mode
        mod = event.modifiers()
        if (self.build_mode or self.selection_mode) and event.button() == Qt.MouseButton.LeftButton and (mod & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)):
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None and self.molecule:
                self.requestHistorySave.emit()
                self._drag_atom_idx = idx
                self._drag_start = event.position().toPoint()
                self._drag_mode = "atom"
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                return

        if self.selection_mode and event.button() == Qt.MouseButton.LeftButton:
            mod = event.modifiers()
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None:
                if mod & Qt.KeyboardModifier.ShiftModifier:
                    if idx in self.selected_atoms:
                        self.selected_atoms.discard(idx)
                    else:
                        self.selected_atoms.add(idx)
                else:
                    self.selected_atoms = {idx} if idx not in self.selected_atoms else set()
                self.request_render()
                return
            self.selected_atoms.clear()
            self._sel_drag_start = event.position().toPoint()
            self._sel_rect = QRectF()
            self.request_render()
            return

        if self.build_mode and event.button() == Qt.MouseButton.LeftButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None:
                self._bonding_from = idx
                self._mouse_pos = event.position().toPoint()
                return
            
            b_idx = self._get_hit_bond(event.position().toPoint())
            if b_idx is not None:
                self.requestHistorySave.emit()
                self.molecule.bonds[b_idx].order = (self.molecule.bonds[b_idx].order % 3) + 1
                
                if self.auto_adjust_h:
                    at1, at2 = self.molecule.atoms[self.molecule.bonds[b_idx].i], self.molecule.atoms[self.molecule.bonds[b_idx].j]
                    self._apply_auto_h(at1)
                    self._apply_auto_h(at2)
                    self._relax_h()
                
                self.moleculeChanged.emit()
                self.request_render()
                return

            # Add new atom if nothing hit
            if not self.molecule:
                self.molecule = Molecule("New Molecule", atoms=[])
            
            self.requestHistorySave.emit()
            loc = self._unproject(event.position().toPoint())
            # Convert local to absolute (if we have a centroid)
            if self.molecule.atoms:
                pos_arr = np.array([[a.x, a.y, a.z] for a in self.molecule.atoms])
                centroid = pos_arr.mean(axis=0)
                abs_pos = loc + centroid
            else:
                abs_pos = loc
            
            new_atom = Atom(self.build_element, abs_pos[0], abs_pos[1], abs_pos[2])
            self.molecule.atoms.append(new_atom)
            new_idx = len(self.molecule.atoms) - 1
            
            if self.auto_adjust_h:
                self._apply_auto_h(self.molecule.atoms[new_idx])
                self._relax_h()
                
            self.moleculeChanged.emit()
            self.request_render()
            return

        if self.build_mode and event.button() == Qt.MouseButton.RightButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None:
                self.requestHistorySave.emit()
                # 1. Identify connected H atoms to remove
                h_to_remove = []
                neighbors_to_adjust = []
                
                for b in self.molecule.bonds:
                    if b.i == idx:
                        nb_idx = b.j
                    elif b.j == idx:
                        nb_idx = b.i
                    else:
                        continue
                    
                    nb_atom = self.molecule.atoms[nb_idx]
                    if nb_atom.element == "H":
                        # Only remove if it has no other bonds
                        other_bonds = 0
                        for b2 in self.molecule.bonds:
                            if b2.i == nb_idx or b2.j == nb_idx: other_bonds += 1
                        if other_bonds == 1:
                            h_to_remove.append(nb_idx)
                    else:
                        neighbors_to_adjust.append(nb_atom)

                # 2. Delete the atoms (highest index first to avoid shifts)
                all_to_del = sorted(list(set(h_to_remove + [idx])), reverse=True)
                for d_idx in all_to_del:
                    self.molecule.atoms.pop(d_idx)
                    new_bonds = []
                    for b in self.molecule.bonds:
                        if b.i == d_idx or b.j == d_idx: continue
                        ni = b.i - 1 if b.i > d_idx else b.i
                        nj = b.j - 1 if b.j > d_idx else b.j
                        new_bonds.append(Bond(ni, nj, b.order))
                    self.molecule.bonds = new_bonds

                # 3. Adjust neighbors if auto_adjust_h is on
                if self.auto_adjust_h:
                    for nb_atom in neighbors_to_adjust:
                        self._apply_auto_h(nb_atom)
                    self._relax_h()

                self.moleculeChanged.emit()
                self.request_render()
                return
            
            b_idx = self._get_hit_bond(event.position().toPoint())
            if b_idx is not None:
                self.requestHistorySave.emit()
                # Remove bond
                self.molecule.bonds.pop(b_idx)
                self.moleculeChanged.emit()
                self.request_render()
                return

        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_mode  = "rotate"
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start = event.position().toPoint()
            self._drag_mode  = "pan"
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

    def _update_cursor(self, pos: QPoint, mods):
        has_atom = self._get_hit_atom(pos) is not None
        alt_held = bool(mods & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier))

        if self.build_mode:
            if alt_held and has_atom:
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif self.selection_mode:
            if alt_held and has_atom:
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            elif has_atom:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.position().toPoint()
        self._update_cursor(event.position().toPoint(), event.modifiers())
        if self.build_mode and self._bonding_from is not None:
            self.request_render(delay_ms=16)
            return

        if self.selection_mode and self._sel_drag_start is not None:
            cur = event.position().toPoint()
            self._sel_rect = QRectF(QPointF(self._sel_drag_start), QPointF(cur)).normalized()
            self.update()
            return

        if self._drag_mode == "atom" and self._drag_atom_idx is not None and self.molecule:
            cur = event.position().toPoint()
            dx = cur.x() - self._drag_start.x()
            dy = cur.y() - self._drag_start.y()
            self._drag_start = cur
            scale = self.base_scale * self._zoom
            inv_rot = np.linalg.inv(self._rot)
            delta_3d = inv_rot @ np.array([dx / scale, -dy / scale, 0.0])
            # Move clicked atom
            atom = self.molecule.atoms[self._drag_atom_idx]
            atom.x += delta_3d[0]
            atom.y += delta_3d[1]
            atom.z += delta_3d[2]
            # Move all other selected atoms together
            if self._drag_atom_idx in self.selected_atoms:
                for i in self.selected_atoms:
                    if i == self._drag_atom_idx:
                        continue
                    a = self.molecule.atoms[i]
                    a.x += delta_3d[0]
                    a.y += delta_3d[1]
                    a.z += delta_3d[2]
            self.request_render(delay_ms=16)
            return

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
        self._update_cursor(event.position().toPoint(), event.modifiers())
        if self.selection_mode and self._sel_drag_start is not None:
            final_pt = event.position().toPoint()
            rect = QRectF(QPointF(self._sel_drag_start), QPointF(final_pt)).normalized()
            if rect.width() > 5 and rect.height() > 5 and self.molecule:
                atoms, _ = project_molecule(
                    self.molecule, self._rot, self._pan[0], self._pan[1],
                    self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
                )
                for i, (ax, ay, az, ar) in enumerate(atoms):
                    if rect.contains(ax, ay):
                        self.selected_atoms.add(i)
                self.request_render()
            self._sel_drag_start = None
            self._sel_rect = None
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.update()
            return

        if self.build_mode and self._bonding_from is not None:
            end_idx = self._get_hit_atom(event.position().toPoint())
            if end_idx is not None and end_idx == self._bonding_from:
                self.requestHistorySave.emit()
                self.molecule.atoms[self._bonding_from].element = self.build_element
                if self.auto_adjust_h:
                    self._apply_auto_h(self.molecule.atoms[self._bonding_from])
                    self._relax_h()
                self.moleculeChanged.emit()
            elif end_idx is not None and end_idx != self._bonding_from:
                self.requestHistorySave.emit()
                # Bond existing
                self.molecule.bonds.append(Bond(self._bonding_from, end_idx))
                
                if self.auto_adjust_h:
                    at1, at2 = self.molecule.atoms[self._bonding_from], self.molecule.atoms[end_idx]
                    self._apply_auto_h(at1)
                    self._apply_auto_h(at2)
                    self._relax_h()
                
                self.moleculeChanged.emit()
            elif end_idx is None:
                self.requestHistorySave.emit()
                # Add new atom and bond
                loc = self._unproject(event.position().toPoint())
                pos_arr = np.array([[a.x, a.y, a.z] for a in self.molecule.atoms])
                centroid = pos_arr.mean(axis=0)
                abs_pos = loc + centroid
                new_idx = len(self.molecule.atoms)
                self.molecule.atoms.append(Atom(self.build_element, abs_pos[0], abs_pos[1], abs_pos[2]))
                self.molecule.bonds.append(Bond(self._bonding_from, new_idx))
                
                if self.auto_adjust_h:
                    at1, at2 = self.molecule.atoms[self._bonding_from], self.molecule.atoms[new_idx]
                    self._apply_auto_h(at1)
                    self._apply_auto_h(at2)
                    self._relax_h()
                    
                self.moleculeChanged.emit()
            
            self._bonding_from = None
            self._mouse_pos = None
            self.request_render()
            return

        was_atom_drag = self._drag_mode == "atom"
        self._drag_atom_idx = None
        self._drag_start = None
        self._drag_mode  = "none"
        if was_atom_drag:
            if self.build_mode or self.selection_mode:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.request_render(delay_ms=0)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.SelectAll):
            if self.molecule:
                self.selected_atoms = set(range(len(self.molecule.atoms)))
                self.request_render()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self.selected_atoms:
                self.selected_atoms.clear()
                self.request_render()
            return
        mod = event.modifiers()
        if event.key() == Qt.Key.Key_A and (mod & Qt.KeyboardModifier.ShiftModifier) and (mod & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)):
            if self.selected_atoms:
                self.selected_atoms.clear()
                self.request_render()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._mouse_pos and self._get_hit_atom(self._mouse_pos) is not None:
            self._update_cursor(self._mouse_pos, event.modifiers())
        super().keyReleaseEvent(event)

    def _apply_auto_h(self, atom: Atom):
        valencies = {
            "H": 1, "C": 4, "N": 3, "O": 2, "F": 1,
            "P": 3, "S": 2, "Cl": 1, "Br": 1, "I": 1,
            "Si": 4, "B": 3
        }
        # Ideal X-H bond lengths per atom
        bond_lengths = {"H":0.74,"C":1.09,"N":1.01,"O":0.96,"F":0.92,
                        "P":1.42,"S":1.34,"Cl":1.27,"Br":1.41,"I":1.61,
                        "Si":1.48,"B":1.19}
        try:
            atom_idx = -1
            for i, a in enumerate(self.molecule.atoms):
                if a is atom:
                    atom_idx = i
                    break
            if atom_idx == -1: return
        except Exception: return

        v = valencies.get(atom.element, 0)
        if v <= 0: return

        # Collect existing neighbor indices and bond orders
        neighbors = []
        neighbor_orders = {}
        for b in self.molecule.bonds:
            if b.i == atom_idx:
                neighbors.append(b.j)
                neighbor_orders[b.j] = b.order
            elif b.j == atom_idx:
                neighbors.append(b.i)
                neighbor_orders[b.i] = b.order

        # Separate current H (only bonded to this atom) from heavy neighbors
        h_indices = []
        heavy_indices = []
        heavy_order_sum = 0
        for nb_idx in set(neighbors):
            if self.molecule.atoms[nb_idx].element == "H":
                other_bonds = sum(1 for b in self.molecule.bonds if b.i == nb_idx or b.j == nb_idx)
                if other_bonds == 1:
                    h_indices.append(nb_idx)
                else:
                    heavy_indices.append(nb_idx)
                    heavy_order_sum += neighbor_orders.get(nb_idx, 1)
            else:
                heavy_indices.append(nb_idx)
                heavy_order_sum += neighbor_orders.get(nb_idx, 1)

        needed = max(0, v - heavy_order_sum)

        # Remove excess H atoms
        if len(h_indices) > needed:
            to_del = sorted(h_indices[needed:], reverse=True)
            for idx in to_del:
                self.molecule.atoms.pop(idx)
                new_bonds = []
                for b in self.molecule.bonds:
                    if b.i == idx or b.j == idx: continue
                    ni = b.i - 1 if b.i > idx else b.i
                    nj = b.j - 1 if b.j > idx else b.j
                    new_bonds.append(Bond(ni, nj, b.order))
                self.molecule.bonds = new_bonds
                if atom_idx > idx: atom_idx -= 1
            return

        if len(h_indices) >= needed:
            return

        # Need to add (needed - len(h_indices)) H atoms
        n_add = needed - len(h_indices)

        # Build existing bond direction vectors (from atom to heavy neighbors)
        atom_pos = np.array([atom.x, atom.y, atom.z])
        existing_vecs = []
        for nb_idx in heavy_indices:
            nb = self.molecule.atoms[nb_idx]
            d = np.array([nb.x - atom.x, nb.y - atom.y, nb.z - atom.z])
            nd = np.linalg.norm(d)
            if nd > 1e-4:
                existing_vecs.append(d / nd)
            else:
                existing_vecs.append(np.array([1.0, 0.0, 0.0]))

        bl = bond_lengths.get(atom.element, 1.09)
        steric = len(heavy_indices) + needed  # total bonded atoms (including H to add)
        if atom.element == "C":
            angle_by_s = {4: 109.47, 3: 120.0, 2: 180.0}
        elif atom.element == "O":
            angle_by_s = {4: 109.47, 3: 120.0, 2: 104.5, 1: 180.0}
        elif atom.element == "N":
            angle_by_s = {4: 109.47, 3: 120.0, 2: 104.5, 1: 180.0}
        else:
            angle_by_s = {4: 109.47, 3: 120.0, 2: 104.5, 1: 180.0}
        angle_deg = angle_by_s.get(steric, 109.47)
        angle_rad = math.radians(angle_deg)

        # Generate ideal geometry directions for the full steric number
        ideal_dirs = _ideal_directions(steric, angle_rad)

        # Match existing bond directions to closest ideal directions
        used = set()
        h_dirs = []
        for vec in existing_vecs:
            best = -1
            best_dot = -2
            for j, d in enumerate(ideal_dirs):
                if j in used: continue
                dot = np.dot(vec, d)
                if dot > best_dot:
                    best_dot = dot
                    best = j
            used.add(best)

        for j in range(len(ideal_dirs)):
            if j not in used:
                h_dirs.append(ideal_dirs[j])

        # Add H atoms in the available ideal directions
        for i in range(min(n_add, len(h_dirs))):
            h_idx = len(self.molecule.atoms)
            off = h_dirs[i] * bl
            self.molecule.atoms.append(Atom("H", atom.x + off[0], atom.y + off[1], atom.z + off[2]))
            self.molecule.bonds.append(Bond(atom_idx, h_idx))
        # If more H needed than available ideal slots (edge case), place at uniform offsets
        for i in range(n_add - min(n_add, len(h_dirs))):
            h_idx = len(self.molecule.atoms)
            off = np.array([1.0, 0.0, 0.0])
            self.molecule.atoms.append(Atom("H", atom.x + off[0], atom.y + off[1], atom.z + off[2]))
            self.molecule.bonds.append(Bond(atom_idx, h_idx))

    def _relax_h(self):
        """Geometrically re-place all H atoms (no FF solver needed)."""
        if not self.molecule:
            return
        done = set()
        for b in self.molecule.bonds:
            for parent_idx in (b.i, b.j):
                if parent_idx in done:
                    continue
                if self.molecule.atoms[parent_idx].element == "H":
                    continue
                has_h = False
                for b2 in self.molecule.bonds:
                    other = b2.j if b2.i == parent_idx else (b2.i if b2.j == parent_idx else -1)
                    if other >= 0 and self.molecule.atoms[other].element == "H":
                        has_h = True
                        break
                if has_h:
                    done.add(parent_idx)
                    self._apply_auto_h(self.molecule.atoms[parent_idx])

    def _get_non_h_indices(self) -> List[int]:
        if not self.molecule: return []
        return [i for i, a in enumerate(self.molecule.atoms) if a.element != "H"]

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

    DEFAULT_SHORTCUTS = {
        "open": "Ctrl+O",
        "save_as": "Ctrl+S",
        "export_svg": "Ctrl+Shift+S",
        "export_view": "Ctrl+E",
        "quick_svg_export": "Ctrl+Shift+X",
        "quit": "Ctrl+Q",
        "info": "Ctrl+I",
        "settings": "Ctrl+P",
        "shortcuts": "",
        "reset_view": "R",
        "build_mode": "B",
        "selection_mode": "S",
        "clean_molecule": "Ctrl+L",
        "undo": "Ctrl+Z",
        "redo": "Ctrl+Shift+Z",
        "calc_results": "Ctrl+M",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Molvector — Molecule Viewer")
        self.resize(1080, 720)
        self._current_theme = "light"
        self._apply_theme("light")
        self._current_source = "None"
        self._current_path   = ""
        self._color_overrides: dict = {}   # elem -> hex
        
        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(40)  # ~25 FPS
        self._anim_timer.timeout.connect(self._on_anim_step)
        self._anim_phase = 0.0
        
        # FF Parameters
        self._ff_max_steps = 500
        self._ff_tol = 0.01

        # Undo/Redo history
        self._history = [] # list of deepcopied Molecule
        self._redo_stack = []
        self._max_history = 50

        # Toolbar icon size for mode buttons
        self._mode_icon_size = QSize(22, 22)

        self._shortcut_actions: dict = {}

        self._build_central()
        self._build_menubar()
        self._build_toolbar()
        self._setup_builder_toolbar()
        self._build_statusbar()
        self._load_appearance_config()
        self._apply_shortcut_config()
        self._show_placeholder()
        self.setAcceptDrops(True)

    # ── menu bar ─────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()

        # ── File ──
        file_menu = mb.addMenu("&File")

        act_open = QAction("&Open…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_file)
        file_menu.addAction(act_open)
        self._shortcut_actions["open"] = act_open

        act_save_as = QAction("Save &As…", self)
        act_save_as.setShortcut("Ctrl+S")
        act_save_as.triggered.connect(self._save_as)
        file_menu.addAction(act_save_as)
        self._shortcut_actions["save_as"] = act_save_as

        file_menu.addSeparator()

        act_save_svg = QAction("Export as &SVG…", self)
        act_save_svg.setShortcut("Ctrl+Shift+S")
        act_save_svg.triggered.connect(self._save_svg)
        file_menu.addAction(act_save_svg)
        self._shortcut_actions["export_svg"] = act_save_svg

        act_export = QAction("&Export View…", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._export_view)
        file_menu.addAction(act_export)
        self._shortcut_actions["export_view"] = act_export

        act_quick_svg = QAction("Quick SVG E&xport…", self)
        act_quick_svg.setShortcut("Ctrl+Shift+X")
        act_quick_svg.triggered.connect(self._quick_svg_export)
        file_menu.addAction(act_quick_svg)
        self._shortcut_actions["quick_svg_export"] = act_quick_svg

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        self._shortcut_actions["quit"] = act_quit

        # ── Edit ──
        edit_menu = mb.addMenu("&Edit")

        act_appearance = QAction("&Appearance…", self)
        act_appearance.triggered.connect(self._edit_appearance)
        edit_menu.addAction(act_appearance)

        act_info = QAction("&Info…", self)
        act_info.setShortcut("Ctrl+I")
        act_info.triggered.connect(self._show_molecule_info)
        edit_menu.addAction(act_info)
        self._shortcut_actions["info"] = act_info

        edit_menu.addSeparator()

        act_settings = QAction("&Settings…", self)
        act_settings.setShortcut("Ctrl+P")
        act_settings.triggered.connect(self._edit_settings)
        edit_menu.addAction(act_settings)
        self._shortcut_actions["settings"] = act_settings

        act_shortcuts = QAction("&Shortcuts…", self)
        act_shortcuts.triggered.connect(self._edit_shortcuts)
        edit_menu.addAction(act_shortcuts)
        self._shortcut_actions["shortcuts"] = act_shortcuts

        # ── View ──
        view_menu = mb.addMenu("&View")
        act_reset_view = QAction("&Reset View", self)
        act_reset_view.setShortcut("R")
        act_reset_view.triggered.connect(lambda: self._canvas.reset_view())
        view_menu.addAction(act_reset_view)
        self._shortcut_actions["reset_view"] = act_reset_view
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
        act_bg.triggered.connect(self._canvas.edit_background_color)
        view_menu.addAction(act_bg)

        # ── Calculations ──
        self._menu_calc = mb.addMenu("&Calculations")
        act_g16 = QAction("Generate G16 Input…", self)
        act_g16.triggered.connect(self._generate_g16_input)
        self._menu_calc.addAction(act_g16)
        self._menu_calc.setEnabled(True)

        # ── Build ──
        self._menu_build = mb.addMenu("&Build")
        
        act_toggle = QAction("Build Mode", self)
        act_toggle.setCheckable(True)
        act_toggle.setShortcut("B")
        act_toggle.triggered.connect(self._toggle_build_mode)
        self._menu_build.addAction(act_toggle)
        self._act_build_toggle = act_toggle # reference for toolbar sync
        self._shortcut_actions["build_mode"] = act_toggle

        act_select_toggle = QAction("Selection Mode", self)
        act_select_toggle.setCheckable(True)
        act_select_toggle.setShortcut("S")
        act_select_toggle.triggered.connect(self._toggle_selection_mode)
        self._menu_build.addAction(act_select_toggle)
        self._act_select_toggle = act_select_toggle
        self._shortcut_actions["selection_mode"] = act_select_toggle

        self._menu_build.addSeparator()
        
        act_clean_m = QAction("Clean Molecule", self)
        act_clean_m.setShortcut("Ctrl+L")
        act_clean_m.triggered.connect(self._clean_molecule)
        self._menu_build.addAction(act_clean_m)
        self._shortcut_actions["clean_molecule"] = act_clean_m

        act_undo = QAction("Undo", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self._undo)
        self._menu_build.addAction(act_undo)
        self._shortcut_actions["undo"] = act_undo

        act_redo = QAction("Redo", self)
        act_redo.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        act_redo.triggered.connect(self._redo)
        self._menu_build.addAction(act_redo)

        act_ff = QAction("Optimize Settings…", self)
        act_ff.triggered.connect(self._edit_ff_settings)
        self._menu_build.addAction(act_ff)

        self._menu_build.addSeparator()

        act_clear_m = QAction("Clear All", self)
        act_clear_m.triggered.connect(self._clear_molecule)
        self._menu_build.addAction(act_clear_m)

        # ── Help ──
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About Molvector", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _setup_builder_toolbar(self):
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        self._build_toolbar_obj = QToolBar("Builder")
        self._build_toolbar_obj.setObjectName("builderToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._build_toolbar_obj)
        self._build_toolbar_obj.setIconSize(QSize(22, 22))
        self._build_toolbar_obj.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._build_toolbar_obj.setVisible(True)

        icon_color = '#ccd6f6' if self._current_theme == 'dark' else '#000000'

        # Selection tool
        select_path = os.path.join(assets_dir, "icon_select.svg")
        self._act_select_btn = QAction(load_colored_icon(select_path, icon_color), "Selection", self)
        self._act_select_btn.setCheckable(True)
        self._act_select_btn.setToolTip("Selection tool — click or drag to select atoms")
        self._act_select_btn.triggered.connect(self._toggle_selection_mode)
        self._build_toolbar_obj.addAction(self._act_select_btn)

        # Build tool
        draw_path = os.path.join(assets_dir, "icon_draw.svg")
        self._act_build_btn = QAction(load_colored_icon(draw_path, icon_color), "Build Mode", self)
        self._act_build_btn.setCheckable(True)
        self._act_build_btn.setToolTip("Build mode — add / bond atoms")
        self._act_build_btn.triggered.connect(self._toggle_build_mode)
        self._build_toolbar_obj.addAction(self._act_build_btn)

        self._build_toolbar_obj.addSeparator()
        self._build_toolbar_obj.addWidget(QLabel(" Element: "))
        self._elem_combo = QComboBox()
        self._elem_combo.setMinimumContentsLength(4)
        self._elem_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._common_elements = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        self._elem_combo.addItems(self._common_elements + ["Others…"])
        self._elem_combo.setCurrentText("C")
        self._elem_combo.currentTextChanged.connect(self._on_build_elem_change)
        self._build_toolbar_obj.addWidget(self._elem_combo)

        self._build_toolbar_obj.addSeparator()
        self._auto_h_check = QCheckBox("Auto adjust H")
        self._auto_h_check.setChecked(False)
        self._auto_h_check.toggled.connect(self._on_auto_h_toggle)
        self._build_toolbar_obj.addWidget(self._auto_h_check)

        act_clear = QAction("Clear All", self)
        act_clear.triggered.connect(self._clear_molecule)
        self._build_toolbar_obj.addAction(act_clear)

        self._build_toolbar_obj.addSeparator()
        act_clean = QAction("Clean Molecule", self)
        act_clean.setToolTip("Rapidly optimize geometry (Force Field)")
        act_clean.triggered.connect(self._clean_molecule)
        self._build_toolbar_obj.addAction(act_clean)

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

        tb_action("Open",      self._open_file, "Ctrl+O", "Open molecule file")
        tb_action("Save SVG",  self._save_svg,  None, "Export current view as SVG")
        tb.addSeparator()
        tb_action("Reset",     lambda: self._canvas.reset_view())
        tb.addSeparator()

        # Zoom readout
        self._zoom_lbl = QLabel("100%")
        self._zoom_lbl.setFixedWidth(46)
        self._zoom_lbl.setObjectName("zoom_label")
        # Style this in get_stylesheet instead of here
        tb.addWidget(self._zoom_lbl)

        # Zoom slider
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(15, 600)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(120)
        self._zoom_slider.setToolTip("Zoom level (%)")
        self._zoom_slider.valueChanged.connect(self._on_zoom_change)
        tb.addWidget(self._zoom_slider)

    # ── central layout ────────────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(172)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10,14,10,14)
        sl.setSpacing(10)

        # Molecule info
        info = QGroupBox("Molecule")
        il   = QVBoxLayout(info)
        self._lbl_name  = QLabel("—")
        self._lbl_name.setWordWrap(True)
        self._lbl_name.setObjectName("mol_name")
        self._lbl_charge = QLabel("Charge: —")
        self._lbl_atoms = QLabel("Atoms: —")
        self._lbl_bonds = QLabel("Bonds: —")
        self._lbl_mass  = QLabel("Mass: —")
        self._lbl_src   = QLabel("Source: —")
        self._lbl_src.setObjectName("dim")
        self._lbl_src.setStyleSheet("font-size:10px;")
        for w in (self._lbl_name, self._lbl_atoms, self._lbl_bonds, self._lbl_mass, self._lbl_charge, self._lbl_src):
            il.addWidget(w)
        sl.addWidget(info)

        # Calculations info
        self._calc_group = QGroupBox("Calculations")
        cl = QVBoxLayout(self._calc_group)
        self._lbl_vib = QLabel("Vibrations: —")
        self._lbl_td  = QLabel("States: —")
        for w in (self._lbl_vib, self._lbl_td):
            cl.addWidget(w)
        self._btn_view_calc = QPushButton("View Results…")
        self._btn_view_calc.clicked.connect(self._show_calculations_dialog)
        cl.addWidget(self._btn_view_calc)
        sl.addWidget(self._calc_group)
        self._calc_group.hide()

        # Legend
        self._legend = LegendPanel()
        self._legend.elementColorChanged.connect(self._on_legend_color_changed)
        sl.addWidget(self._legend)
        sl.addStretch()

        # Hint
        hint = QLabel("Drag  rotate\nRight-drag  pan\nScroll  zoom")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(hint)

        root.addWidget(sidebar)

        # Canvas
        self._canvas = MoleculeCanvas()
        self._canvas.rotationChanged.connect(self._on_rotation_changed)
        self._canvas.fileDropped.connect(self._load_and_display)
        self._canvas.moleculeChanged.connect(self._on_structure_changed)
        self._canvas.requestHistorySave.connect(self._save_history)
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
        bg = THEMES[self._current_theme]["CANVAS"]
        fg = THEMES[self._current_theme]["FG_DIM"]
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="450" viewBox="0 0 600 450">' 
            f'<rect width="600" height="450" fill="{bg}"/>' 
            f'<text x="300" y="210" text-anchor="middle" font-family="Courier New" font-size="15" fill="{fg}">' 
            'Open a molecule file to begin' 
            '</text>' 
            '</svg>'
        ).encode("ascii")
        self._canvas.load(QByteArray(svg))

    # ── drag and drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self._load_and_display(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ── file open / save ──────────────────────────────────────────────────────

    def _load_mol_from_path(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]

        mol = None
        src = ""
        errors = []

        # Try all parsers regardless of file extension
        parsers = [
            ("Gaussian log", lambda t: parse_gaussian_log(t)),
            ("XYZ", lambda t: parse_xyz(t, name=name)),
            ("PDB", lambda t: parse_pdb(t, name=name)),
            ("Gaussian input", lambda t: parse_gaussian(t))
        ]

        for source_name, parser in parsers:
            try:
                mol = parser(text)
                if mol and mol.atoms:
                    src = source_name
                    break
            except Exception as e:
                errors.append(f"{source_name}: {e}")

        if mol is None or not mol.atoms:
            err_msg = "\n".join(errors)
            raise ValueError(f"Could not parse file format.\nAttempts:\n{err_msg}")

        infer_bonds(mol)
        return mol, src

    def _load_and_display(self, path: str):
        """Parse a file and update all UI elements. Used by drag-and-drop and Open."""
        try:
            mol, src = self._load_mol_from_path(path)
            self._color_overrides = {}
            self._canvas.color_overrides = {}
            self._canvas.load_molecule(mol)
            self._legend.update_for(mol, {})
            self._current_source = src
            self._current_path = path
            self._update_info_panel(mol)
            self._update_calculations_menu(mol)
        except Exception as e:
            QMessageBox.critical(self, "Error loading file",
                                 f"{type(e).__name__}: {e}")

    def _update_calculations_menu(self, mol: Molecule):
        # Remove any actions beyond the first (Generate G16 Input…)
        while len(self._menu_calc.actions()) > 1:
            self._menu_calc.removeAction(self._menu_calc.actions()[-1])

        if mol and (mol.vibrational_modes or mol.excited_states):
            self._menu_calc.addSeparator()
            act_results = QAction("Calculation Results…", self)
            overrides = self._load_shortcut_overrides()
            ks = overrides.get("calc_results", "Ctrl+M")
            act_results.setShortcut(ks)
            act_results.triggered.connect(self._show_calculations_dialog)
            self._menu_calc.addAction(act_results)
            self._shortcut_actions["calc_results"] = act_results

    def _generate_g16_input(self):
        mol = self._canvas.molecule
        if not mol or not mol.atoms:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        dlg = G16InputDialog(mol, parent=self)
        dlg.exec()

    def _show_calculations_dialog(self):
        mol = self._canvas.molecule
        if not mol: return
        dlg = CalculationsDialog(mol, parent=self)
        dlg.modeSelected.connect(self._show_vibration)
        dlg.stateSelected.connect(self._show_excited_state)
        dlg.viewSpectrum.connect(self._on_view_spectrum)
        dlg.animationToggled.connect(self._on_anim_toggle)
        dlg.vectorsToggled.connect(self._on_vectors_toggle)
        dlg.finished.connect(self._on_calculations_closed)
        dlg.show()

    def _on_anim_toggle(self, enabled: bool):
        if enabled:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
            self._canvas.animation_amplitude = 0.0
            self._canvas.request_render()

    def _on_vectors_toggle(self, enabled: bool):
        self._canvas._show_vectors = enabled
        self._canvas.request_render()

    def _on_anim_step(self):
        self._anim_phase += 0.25
        if self._anim_phase > 2*math.pi:
            self._anim_phase -= 2*math.pi
        self._canvas.animation_phase = self._anim_phase
        self._canvas.animation_amplitude = 0.6
        self._canvas.request_render()

    def _on_calculations_closed(self):
        self._anim_timer.stop()
        self._anim_phase = 0.0
        self._canvas.animation_phase = 0.0
        self._canvas.animation_amplitude = 0.0
        self._canvas.active_vectors = None
        self._canvas.arrows = None
        self._canvas.request_render()
        self._status.clearMessage()

    def _on_view_spectrum(self, kind):
        if kind == "ir": self._view_ir_spectrum()
        else: self._view_uvvis_spectrum()

    def _view_ir_spectrum(self):
        mol = self._canvas.molecule
        if not mol: return
        x = [m.frequency for m in mol.vibrational_modes]
        y = [m.intensity for m in mol.vibrational_modes]
        meta = f"Molecule: {mol.name}\nCalculation: Vibrational Frequencies (IR)"
        safe_name = get_safe_filename(mol.name)
        default_file = f"{safe_name}_ir_spectrum.txt"
        dlg = SpectrumDialog(x, y, "Frequency / cm-1", "Intensity (arb. units)", "IR Spectrum", meta, self)
        dlg.set_default_filename(default_file)
        dlg.exec()

    def _view_uvvis_spectrum(self):
        mol = self._canvas.molecule
        if not mol: return
        x = [s.wavelength_nm for s in mol.excited_states]
        y = [s.oscillator_strength for s in mol.excited_states]
        meta = f"Molecule: {mol.name}\nCalculation: Excited States (TDDFT)"
        safe_name = get_safe_filename(mol.name)
        default_file = f"{safe_name}_uvvis_spectrum.txt"
        dlg = SpectrumDialog(x, y, "Wavelength / nm", "Oscillator Strength / f", "UV-Vis Spectrum", meta, self)
        dlg.set_default_filename(default_file)
        dlg.exec()

    def _reset_state(self):
        self._canvas.arrows = None
        self._canvas.active_vectors = None
        self._canvas.request_render()
        if self._canvas.molecule:
            self._update_info_panel(self._canvas.molecule)

    def _show_vibration(self, mode: VibrationalMode):
        # Scale displacements for visibility
        self._canvas.active_vectors = mode.displacements
        scale = 1.2
        arrows = []
        for i, d in enumerate(mode.displacements):
            arrows.append((i, d[0]*scale, d[1]*scale, d[2]*scale, "#00ff00"))
        
        self._canvas.arrows = arrows
        self._canvas.request_render()
        self._status.showMessage(f"Visualizing mode {mode.index}: {mode.frequency:.1f} cm⁻¹")

    def _show_excited_state(self, state: ExcitedState):
        self._canvas.arrows = None
        self._canvas.active_vectors = None
        self._canvas.request_render()
        
        msg = f"Excited State {state.index}: {state.symmetry} | {state.energy_ev:.3f} eV | f={state.oscillator_strength:.4f}"
        self._status.showMessage(msg)
        
        formula = chemical_formula(self._canvas.molecule)
        # Keep name consistent, don't add (S1) here if user wants consistency
        # Or just use plain text
        self._lbl_name.setText(formula)

    def _on_legend_color_changed(self, elem: str, hex_color: str):
        self._color_overrides[elem] = hex_color
        self._canvas.color_overrides[elem] = hex_color
        self._canvas.request_render()
        if self._canvas.molecule:
            self._legend.update_for(self._canvas.molecule, self._color_overrides)

    def _toggle_theme(self):
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self._apply_theme(new_theme)
        self._canvas.background = THEMES[new_theme]["CANVAS"]
        self._canvas.request_render()

    def _apply_theme(self, theme_name: str):
        self._current_theme = theme_name
        self.setStyleSheet(get_stylesheet(theme_name))
        
        if hasattr(self, "_act_theme"):
            self._act_theme.setText("Switch to Dark Mode" if theme_name == "light" else "Switch to Light Mode")
            
        # Update toolbar icon colors
        self._update_tool_icons()

        self.setProperty("theme", theme_name)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_tool_icons(self):
        if not hasattr(self, '_act_select_btn') or not hasattr(self, '_act_build_btn'):
            return
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        color = '#ccd6f6' if self._current_theme == 'dark' else '#000000'
        self._act_select_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_select.svg"), color))
        self._act_build_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_draw.svg"), color))

    def _update_info_panel(self, mol):
        """Populate sidebar labels and status bar for a loaded molecule."""
        formula = chemical_formula(mol)
        mass    = molecular_mass(mol)
        charge  = mol.charge
        src     = self._current_source
        path    = self._current_path

        # Build name: plain text C60+ etc.
        if charge == 0:
            display_name = formula
        elif charge > 0:
            display_name = f"{formula}{charge}+" if charge > 1 else f"{formula}+"
        else:
            display_name = f"{formula}{abs(charge)}-" if charge < -1 else f"{formula}-"

        charge_str = "neutral" if charge == 0 else (f"+{charge}" if charge > 0 else str(charge))

        self._lbl_name.setText(display_name)
        self._lbl_charge.setText(f"Charge: {charge_str}")
        self._lbl_atoms.setText(f"Atoms: {len(mol.atoms)}")
        self._lbl_bonds.setText(f"Bonds: {len(mol.bonds)}")
        self._lbl_mass.setText(f"Mass: {mass:.3f} uma")
        self._lbl_src.setText(f"Source: {src}")

        # Update Calculations Group
        has_vib = bool(mol.vibrational_modes)
        has_td  = bool(mol.excited_states)
        has_any = has_vib or has_td
        if has_any:
            self._calc_group.show()
            self._lbl_vib.setVisible(has_vib)
            self._lbl_td.setVisible(has_td)
            self._lbl_vib.setText(f"Vibrations: {len(mol.vibrational_modes)} modes")
            self._lbl_td.setText(f"TD-DFT: {len(mol.excited_states)} states")
        else:
            self._calc_group.hide()

        # Update main window title
        self.setWindowTitle(f"Molvector — {display_name}")

        status_path = f"{path}  |  " if path else ""
        self._status.showMessage(
            f"{status_path}{display_name}  |  {len(mol.atoms)} atoms, "
            f"{len(mol.bonds)} bonds  |  {mass:.3f} uma  [{src}]"
        )
        self._legend.update_for(mol, self._color_overrides)

    def _show_molecule_info(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return

        from collections import Counter
        formula = chemical_formula(mol)
        mass = molecular_mass(mol)
        charge = mol.charge
        charge_str = "neutral" if charge == 0 else (f"+{charge}" if charge > 0 else str(charge))
        counts = Counter(a.element for a in mol.atoms)
        elem_parts = [f"{e}\u2009{counts[e]}" for e in sorted(counts.keys())]
        elem_str = "  " + ", ".join(elem_parts) if mol.atoms else "—"
        src = self._current_source or "—"
        path = self._current_path or "—"

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Info \u2014 {formula}")
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)

        for label, value in [
            ("Formula:", formula),
            ("Charge:", charge_str),
            ("Mass:", f"{mass:.3f} uma"),
            ("Atoms:", str(len(mol.atoms))),
            ("Bonds:", str(len(mol.bonds))),
            ("Source:", src),
            ("Path:", path),
            ("Elements:", elem_str),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold;")
            val = QLabel(value)
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            val.setWordWrap(True)
            row.addWidget(lbl)
            row.addWidget(val, 1)
            layout.addLayout(row)

        layout.addSpacing(10)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn.rejected.connect(dlg.reject)
        layout.addWidget(btn)
        dlg.exec()

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "All supported (*.xyz *.gjf *.com *.log *.out *.pdb *.txt);;"
            "XYZ (*.xyz);;Gaussian input (*.gjf *.com);;"
            "Gaussian log (*.log *.out);;PDB (*.pdb);;Text (*.txt);;All files (*)"
        )
        if path:
            self._load_and_display(path)

    def _save_svg(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        safe_name = get_safe_filename(self._canvas.molecule.name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", f"{safe_name}_view.svg", "SVG files (*.svg)"
        )
        if not path:
            return
        try:
            data = self._canvas.get_svg_bytes(export_mode=True)
            with open(path, "wb") as f:
                f.write(data)
            self._status.showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error saving", str(e))

    def _quick_svg_export(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        dlg = QuickExportDialog(self._canvas, self)
        dlg.show()

    def _edit_shortcuts(self):
        dlg = ShortcutDialog(self._shortcut_actions, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            overrides = dlg.get_shortcuts()
            self._apply_shortcuts(overrides)
            self._save_shortcut_overrides(overrides)

    def _apply_shortcut_config(self):
        overrides = self._load_shortcut_overrides()
        if overrides:
            self._apply_shortcuts(overrides)

    def _apply_shortcuts(self, overrides: dict):
        for aid, seq_str in overrides.items():
            if aid in self._shortcut_actions and seq_str:
                self._shortcut_actions[aid].setShortcut(seq_str)

    def _load_shortcut_overrides(self) -> dict:
        cfg_path = os.path.join(os.path.dirname(__file__), "molvector_config.json")
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return cfg.get("shortcuts", {})
        except Exception:
            pass
        return {}

    def _save_shortcut_overrides(self, overrides: dict):
        cfg_path = os.path.join(os.path.dirname(__file__), "molvector_config.json")
        config = {}
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
        except Exception:
            pass
        config["shortcuts"] = overrides
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass

    def _export_view(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return

        safe_name = get_safe_filename(self._canvas.molecule.name)
        filters = "PDF files (*.pdf);;PNG files (*.png);;JPEG files (*.jpg *.jpeg);;SVG files (*.svg)"
        path, sel_filter = QFileDialog.getSaveFileName(
            self, "Export View", f"{safe_name}_view.pdf", filters
        )
        if not path:
            return

        try:
            # 1. Get SVG data from canvas
            svg_data = self._canvas.get_svg_bytes(export_mode=True)
            renderer = QSvgRenderer(QByteArray(svg_data))
            
            # 2. Determine size (use high res for raster)
            # We'll use a 2x or 3x scale for images to make them crisp
            view_size = renderer.defaultSize()
            if view_size.isEmpty():
                view_size = QSize(1200, 900)
            
            ext = os.path.splitext(path)[1].lower()
            
            if ext == ".pdf":
                pdf = QPdfWriter(path)
                pdf.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                # Center the molecule on the page
                painter = QPainter(pdf)
                try:
                    # Scale molecule to fit page comfortably
                    target_rect = painter.viewport()
                    # Maintain aspect ratio
                    svg_ratio = view_size.width() / view_size.height()
                    page_ratio = target_rect.width() / target_rect.height()
                    
                    if svg_ratio > page_ratio:
                        w = target_rect.width()
                        h = int(w / svg_ratio)
                    else:
                        h = target_rect.height()
                        w = int(h * svg_ratio)
                    
                    # Center it
                    x = (target_rect.width() - w) // 2
                    y = (target_rect.height() - h) // 2
                    renderer.render(painter, QRectF(float(x), float(y), float(w), float(h)))
                finally:
                    painter.end()
            
            elif ext in (".png", ".jpg", ".jpeg"):
                # Scale up for high quality
                scale_factor = 2.0
                img_size = view_size * scale_factor
                img = QImage(img_size, QImage.Format.Format_ARGB32)
                img.fill(Qt.GlobalColor.transparent if ext == ".png" else Qt.GlobalColor.white)
                
                painter = QPainter(img)
                try:
                    renderer.render(painter)
                finally:
                    painter.end()
                
                img.save(path)
            
            elif ext == ".svg":
                with open(path, "wb") as f:
                    f.write(svg_data)
            
            self._status.showMessage(f"Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _save_as(self):
        mol = self._canvas.molecule
        if not mol:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
            
        safe_name = get_safe_filename(mol.name)
        filters = "XYZ (*.xyz);;PDB (*.pdb);;Gaussian input (*.gjf *.com);;All files (*)"
        path, sel_filter = QFileDialog.getSaveFileName(
            self, "Save Molecule As", f"{safe_name}.xyz", filters
        )
        if not path:
            return
            
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".xyz":
                text = save_xyz(mol)
            elif ext == ".pdb":
                text = save_pdb(mol)
            elif ext in (".gjf", ".com"):
                text = save_gaussian_input(mol)
            else:
                # Default to XYZ if extension is unknown
                text = save_xyz(mol)
                
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._status.showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ── Tool mode actions ────────────────────────────────────────────────────

    def _toggle_selection_mode(self, enabled: bool):
        self._act_select_btn.setChecked(enabled)
        self._act_select_toggle.setChecked(enabled)
        self._canvas.selection_mode = enabled
        if enabled:
            self._act_build_toggle.setChecked(False)
            self._act_build_btn.setChecked(False)
            self._canvas.build_mode = False
            self._status.showMessage("Selection Mode: Click atoms to select, drag to rectangle-select")
        else:
            self._canvas.selected_atoms.clear()
            self._canvas.request_render()
            self._status.showMessage("Selection Mode Off")
        self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)

    def _toggle_build_mode(self, enabled: bool):
        if enabled:
            self._act_select_btn.setChecked(False)
            self._act_select_toggle.setChecked(False)
            self._canvas.selection_mode = False
        self._act_build_toggle.setChecked(enabled)
        self._act_build_btn.setChecked(enabled)
        self._canvas.build_mode = enabled
        self._status.showMessage("Build Mode Active: Click to add, Drag to bond, Click bond to change order" if enabled else "Build Mode Off")
        self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)

    def _on_build_elem_change(self, elem: str):
        if elem == "Others…":
            prev = self._canvas.build_element
            dlg = PeriodicTableDialog(self, prev)
            dlg.elementSelected.connect(self._on_pick_from_periodic_table)
            if not dlg.exec():
                self._elem_combo.setCurrentText(prev)
            return
        self._canvas.build_element = elem

    def _on_pick_from_periodic_table(self, sym: str):
        idx = self._elem_combo.findText(sym)
        if idx < 0:
            others_idx = self._elem_combo.findText("Others…")
            self._elem_combo.insertItem(others_idx, sym)
        self._elem_combo.setCurrentText(sym)

    def _on_auto_h_toggle(self, checked: bool):
        self._canvas.auto_adjust_h = checked

    def _clear_molecule(self):
        if QMessageBox.question(self, "Clear", "Clear the entire molecule?") == QMessageBox.StandardButton.Yes:
            self._save_history()
            self._canvas.molecule = Molecule("New Molecule", atoms=[])
            self._canvas.request_render()
            self._update_info_panel(self._canvas.molecule)

    def _clean_molecule(self):
        if not self._canvas.molecule or not self._canvas.molecule.atoms:
            return

        if not HAS_OPENBABEL:
            QMessageBox.warning(
                self, "OpenBabel Not Found",
                "Geometry optimization requires OpenBabel.\n\n"
                "Install it with:  pip install openbabel-wheel\n\n"
                "If already installed, set the BABEL_DATADIR environment "
                "variable to the folder containing UFF.prm."
            )
            return

        self._save_history()
        steps_taken = optimize_geometry(
            self._canvas.molecule, 
            max_steps=self._ff_max_steps, 
            tol=self._ff_tol,
        )
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._status.showMessage(f"Geometry optimized ({steps_taken} iterations).", 3000)

    def _on_structure_changed(self):
        # UI updates only; history should be saved BEFORE the change occurs
        # to capture the 'before' state correctly.
        self._update_info_panel(self._canvas.molecule)

    def _save_history(self):
        import copy
        if self._canvas.molecule:
            # We save a snapshot of the CURRENT state before it gets modified.
            snap = copy.deepcopy(self._canvas.molecule)
            self._history.append(snap)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            # Clear redo stack on NEW action
            self._redo_stack.clear()

    def _undo(self):
        if not self._history:
            self._status.showMessage("Nothing to undo.")
            return
        
        import copy
        # Save CURRENT state to redo stack before going back
        if self._canvas.molecule:
            self._redo_stack.append(copy.deepcopy(self._canvas.molecule))

        # Restore the most recent snapshot
        prev = self._history.pop()
        self._canvas.molecule = prev
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._status.showMessage("Undo successful.")

    def _redo(self):
        if not self._redo_stack:
            self._status.showMessage("Nothing to redo.")
            return
        
        import copy
        # Save CURRENT state to history before going forward
        if self._canvas.molecule:
            self._history.append(copy.deepcopy(self._canvas.molecule))
            if len(self._history) > self._max_history:
                self._history.pop(0)

        # Restore from redo stack
        next_state = self._redo_stack.pop()
        self._canvas.molecule = next_state
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._status.showMessage("Redo successful.")

    def _edit_ff_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Optimization Parameters")
        l = QVBoxLayout(dlg)
        
        from PyQt6.QtWidgets import QFormLayout, QSpinBox, QDialogButtonBox
        form = QFormLayout()
        
        s_steps = QSpinBox()
        s_steps.setRange(10, 100000)
        s_steps.setValue(self._ff_max_steps)
        form.addRow("Max Steps:", s_steps)
        
        l.addLayout(form)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        l.addWidget(btns)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._ff_max_steps = s_steps.value()
            self._clean_molecule()

    # ── View actions ─────────────────────────────────────────────────────────

    def _set_bond_style(self, style: str):
        self._canvas.bond_style = style
        self._canvas.request_render()

    def _set_ball_size(self, scale: float):
        self._canvas.atom_scale = scale
        self._canvas.request_render()

    def _set_bond_width(self, w: float):
        self._canvas.bond_width_px = w
        self._canvas.request_render()

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _load_appearance_config(self):
        cfg = AppearanceDialog.load_config()
        if cfg is None:
            return
        self._canvas.atom_scale = cfg.get("atom_scale", self._canvas.atom_scale)
        self._canvas.bond_width_px = cfg.get("bond_width_px", self._canvas.bond_width_px)
        self._canvas.bond_style = cfg.get("bond_style", self._canvas.bond_style)
        self._color_overrides = cfg.get("color_overrides", {})
        self._canvas.color_overrides = self._color_overrides
        self._canvas.request_render()

    def _edit_appearance(self):
        orig = (self._canvas.atom_scale, self._canvas.bond_width_px,
                self._canvas.bond_style, dict(self._color_overrides))

        def _live_update(ball, bw, style, colors):
            self._canvas.atom_scale = ball
            self._canvas.bond_width_px = bw
            self._canvas.bond_style = style
            self._color_overrides = colors
            self._canvas.color_overrides = colors
            if self._canvas.molecule:
                self._legend.update_for(self._canvas.molecule, colors)
            self._canvas.request_render()

        dlg = AppearanceDialog(*orig, live_callback=_live_update, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._canvas.atom_scale = dlg.ball_scale
            self._canvas.bond_width_px = dlg.bond_width
            self._canvas.bond_style = dlg.bond_style
            self._color_overrides = dlg._color_overrides
            self._canvas.color_overrides = dlg._color_overrides
            if self._canvas.molecule:
                self._legend.update_for(self._canvas.molecule, dlg._color_overrides)
            self._canvas.request_render()
        else:
            (self._canvas.atom_scale, self._canvas.bond_width_px,
             self._canvas.bond_style, self._color_overrides) = orig
            self._canvas.color_overrides = self._color_overrides
            if self._canvas.molecule:
                self._legend.update_for(self._canvas.molecule, self._color_overrides)
            self._canvas.request_render()

    def _edit_settings(self):
        orig_theme = self._current_theme
        orig_bg    = self._canvas.background

        def _live_update(theme, bg):
            if theme != self._current_theme:
                self._apply_theme(theme)
            self._canvas.background = bg
            self._canvas.request_render()

        dlg = SettingsDialog(
            orig_theme,
            orig_bg,
            live_callback=_live_update,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._apply_theme(dlg.theme)
            self._canvas.background = dlg.bg_color
            self._canvas.request_render()
        else:
            self._apply_theme(orig_theme)
            self._canvas.background = orig_bg
            self._canvas.request_render()

    def _edit_atom_colors(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        elements = sorted({a.element for a in mol.atoms})
        orig_overrides = dict(self._color_overrides)

        def _live_update(overrides):
            self._color_overrides = overrides
            self._canvas.color_overrides = overrides
            self._legend.update_for(mol, overrides)
            self._canvas.request_render()

        dlg = AtomColorDialog(elements, orig_overrides, live_callback=_live_update, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._color_overrides = dlg.get_overrides()
            self._canvas.color_overrides = self._color_overrides
            self._legend.update_for(mol, self._color_overrides)
            self._canvas.request_render()
        else:
            self._color_overrides = orig_overrides
            self._canvas.color_overrides = orig_overrides
            self._legend.update_for(mol, orig_overrides)
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
        QMessageBox.about(self, "About Molvector",
            "<b>Molvector</b> — 3D Molecule Viewer<br><br>"
            "Ball-and-stick rendering.<br>"
            "Parsers: XYZ · Gaussian input (.gjf/.com) · Gaussian log (.log/.out)<br><br>"
            "Controls:<br>"
            "  &nbsp; Left-drag &nbsp;&nbsp; Rotate<br>"
            "  &nbsp; Right-drag &nbsp; Pan<br>"
            "  &nbsp; Scroll &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Zoom<br><br>"
            "Dependencies: PyQt6 · NumPy · svgwrite"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_zoom_change(self, value):
        self._canvas._zoom = value / 100.0
        self._zoom_lbl.setText(f"{value}%")
        self._canvas.request_render(delay_ms=60)

    def _on_rotation_changed(self):
        pct = int(self._canvas._zoom * 100)
        self._zoom_lbl.setText(f"{pct}%")
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(pct)
        self._zoom_slider.blockSignals(False)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Molvector")
    app.setFont(QFont("Segoe UI", 10))

    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Molvector")
        except Exception:
            pass


    win = MainWindow()
    set_icons(app, win)
    win.show()

    # Optional: open file from command line
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win._load_and_display(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()