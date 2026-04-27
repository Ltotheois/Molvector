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
    QScrollArea, QToolBar, QMenu, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QComboBox,
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import Qt, QByteArray, QPoint, pyqtSignal, QTimer, QSize, QRect, QRectF
from PyQt6.QtGui import QAction, QColor, QPalette, QFont, QCursor, QIcon, QPixmap, QImage, QPainter, QPdfWriter, QPageSize, QKeySequence

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── renderer / parsers ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from molvector_avogadro import (
    parse_xyz, parse_gaussian, parse_gaussian_log, parse_pdb, infer_bonds,
    render_avogadro, Molecule, CPK_BASE, CPK_DARK, VDW_RADII,
    lighten, darken, hex_to_rgb, rgb_to_hex, auto_dark,
    chemical_formula, molecular_mass, VibrationalMode, ExcitedState,
    save_xyz, save_gaussian_input, save_pdb, project_molecule, Atom, Bond,
    optimize_geometry,
)

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
    "Rn":"Radon",
}

# Theme Colors
THEMES = {
    "dark": {
        "DARK_BG":  "#0f0f1a",
        "PANEL_BG": "#0d0d18",
        "CARD_BG":  "#13131f",
        "BORDER":   "#2a2a44",
        "FG":       "#ccd6f6",
        "FG_DIM":   "#8899bb",
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
        border:none; border-radius:4px; padding:4px 8px;
    }}
    QToolBar QToolButton:hover {{ background:{t['BORDER']}; }}
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
    QDoubleSpinBox, QSpinBox {{
        background:{t['DARK_BG']}; border:1px solid {t['BORDER']};
        border-radius:4px; padding:3px 6px; color:{t['FG']};
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
        color:{t['FG_DIM']};
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
    }}
    QToolButton:hover {{ background: {t['BORDER']}; }}
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

class SettingsDialog(QDialog):
    def __init__(self, theme, atom_scale, bond_width, bg_color, live_callback=None, parent=None):
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

        # Ball size
        self._atom_spin = QDoubleSpinBox()
        self._atom_spin.setRange(0.2, 3.0)
        self._atom_spin.setSingleStep(0.05)
        self._atom_spin.setValue(atom_scale)
        self._atom_spin.valueChanged.connect(self._on_change)
        form.addRow("Ball Size:", self._atom_spin)

        # Bond Width
        self._bond_spin = QDoubleSpinBox()
        self._bond_spin.setRange(1, 30)
        self._bond_spin.setValue(bond_width)
        self._bond_spin.valueChanged.connect(self._on_change)
        form.addRow("Bond Width:", self._bond_spin)

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
                self._atom_spin.value(),
                self._bond_spin.value(),
                self._bg_btn.color()
            )

    @property
    def theme(self) -> str: return self._theme_combo.currentText().lower()
    @property
    def atom_scale(self) -> float: return self._atom_spin.value()
    @property
    def bond_width(self) -> float: return self._bond_spin.value()
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
            name_lbl.setStyleSheet("font-size:11px;")

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
            
            table = QTableWidget(len(mol.vibrational_modes), 3)
            table.setHorizontalHeaderLabels(["Mode", "Freq (cm⁻¹)", "Intensity (km/mol)"])
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            for i, m in enumerate(mol.vibrational_modes):
                table.setItem(i, 0, QTableWidgetItem(str(m.index)))
                table.setItem(i, 1, QTableWidgetItem(f"{m.frequency:.2f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{m.intensity:.2f}"))
            
            table.itemSelectionChanged.connect(lambda t=table: self._on_freq_sel(t))
            fpl.addWidget(table)
            
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
            
            self.tabs.addTab(freq_page, "Frequencies")

        # ── TDDFT Tab ──
        if mol.excited_states:
            td_page = QWidget()
            tpl = QVBoxLayout(td_page)
            
            table = QTableWidget(len(mol.excited_states), 4)
            table.setHorizontalHeaderLabels(["State", "Energy (eV)", "Wavelength (nm)", "f"])
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            for i, s in enumerate(mol.excited_states):
                table.setItem(i, 0, QTableWidgetItem(f"S{s.index} ({s.symmetry})"))
                table.setItem(i, 1, QTableWidgetItem(f"{s.energy_ev:.4f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{s.wavelength_nm:.2f}"))
                table.setItem(i, 3, QTableWidgetItem(f"{s.oscillator_strength:.4f}"))
            
            table.itemSelectionChanged.connect(lambda t=table: self._on_state_sel(t))
            tpl.addWidget(table)
            
            btn_uv = QPushButton("View UV-Vis Spectrum…")
            btn_uv.clicked.connect(lambda: self.viewSpectrum.emit("uvvis"))
            tpl.addWidget(btn_uv)
            
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
            lbl.setStyleSheet("font-size:11px;")
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
# INTERACTIVE CANVAS
# ─────────────────────────────────────────────────────────────────────────────

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
        self._show_vectors = False
        self.build_mode = False
        self.build_element = "C"
        self._bonding_from: int | None = None
        self._mouse_pos: QPoint | None = None

        # Render parameters — all public, set by MainWindow
        self.base_scale     = 110.0
        self.atom_scale     = 0.7
        self.bond_width_px  = 10.0
        self.background     = "#ffffff"
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
        from molvector_avogadro import rotation_matrix
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
                active_vectors=self.active_vectors,
                animation_phase=self.animation_phase,
                animation_amplitude=self.animation_amplitude,
                output_path=tmp,
                export_mode=export_mode,
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
        if self.build_mode and self._bonding_from is not None and self._mouse_pos is not None and self.molecule:
            # Draw temporary bonding line
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QColor("#007bff"))
            # Get start atom projected pos
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
            self.moleculeChanged.emit()
            self.request_render()
            return

        if self.build_mode and event.button() == Qt.MouseButton.RightButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None:
                self.requestHistorySave.emit()
                # Remove atom
                self.molecule.atoms.pop(idx)
                # Remove and re-index bonds
                new_bonds = []
                for b in self.molecule.bonds:
                    if b.i == idx or b.j == idx:
                        continue
                    ni = b.i - 1 if b.i > idx else b.i
                    nj = b.j - 1 if b.j > idx else b.j
                    new_bonds.append(Bond(ni, nj, b.order))
                self.molecule.bonds = new_bonds
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

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.position().toPoint()
        if self.build_mode and self._bonding_from is not None:
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
        if self.build_mode and self._bonding_from is not None:
            end_idx = self._get_hit_atom(event.position().toPoint())
            if end_idx is not None and end_idx != self._bonding_from:
                self.requestHistorySave.emit()
                # Bond existing
                self.molecule.bonds.append(Bond(self._bonding_from, end_idx))
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
                self.moleculeChanged.emit()
            
            self._bonding_from = None
            self._mouse_pos = None
            self.request_render()
            return

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
        self._ff_steps = 150
        self._ff_k_bond = 8.0
        self._ff_k_rep = 1.5

        # Undo/Redo history
        self._history = [] # list of deepcopied Molecule
        self._redo_stack = []
        self._max_history = 50

        self._build_central()
        self._build_menubar()
        self._build_toolbar()
        self._setup_builder_toolbar()
        self._build_statusbar()
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

        act_save_as = QAction("Save &As…", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._save_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_save_svg = QAction("Export as &SVG…", self)
        act_save_svg.setShortcut("Ctrl+S")
        act_save_svg.triggered.connect(self._save_svg)
        file_menu.addAction(act_save_svg)

        act_export = QAction("&Export View…", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._export_view)
        file_menu.addAction(act_export)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ── Edit ──
        edit_menu = mb.addMenu("&Edit")
        act_settings = QAction("&Settings…", self)
        act_settings.setShortcut("Ctrl+P")
        act_settings.triggered.connect(self._edit_settings)
        edit_menu.addAction(act_settings)

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
        act_bg.triggered.connect(self._canvas.edit_background_color)
        view_menu.addAction(act_bg)

        # ── Calculations ──
        self._menu_calc = mb.addMenu("&Calculations")
        self._menu_calc.setEnabled(False)

        # ── Build ──
        self._menu_build = mb.addMenu("&Build")
        
        act_toggle = QAction("Build Mode", self)
        act_toggle.setCheckable(True)
        act_toggle.setShortcut("Ctrl+B")
        act_toggle.triggered.connect(self._toggle_build_mode)
        self._menu_build.addAction(act_toggle)
        self._act_build_toggle = act_toggle # reference for toolbar sync

        self._menu_build.addSeparator()
        
        act_clean_m = QAction("Clean Molecule", self)
        act_clean_m.setShortcut("Ctrl+L")
        act_clean_m.triggered.connect(self._clean_molecule)
        self._menu_build.addAction(act_clean_m)

        act_undo = QAction("Undo", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self._undo)
        self._menu_build.addAction(act_undo)

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
        # ── Build ──
        self._build_toolbar_obj = QToolBar("Builder")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._build_toolbar_obj)
        self._build_toolbar_obj.setVisible(True)

        self._act_build_btn = QAction("Build Mode", self)
        self._act_build_btn.setCheckable(True)
        self._act_build_btn.triggered.connect(self._toggle_build_mode)
        self._build_toolbar_obj.addAction(self._act_build_btn)
        
        self._build_toolbar_obj.addSeparator()
        self._build_toolbar_obj.addWidget(QLabel(" Element: "))
        self._elem_combo = QComboBox()
        self._elem_combo.addItems(["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"])
        self._elem_combo.setCurrentText("C")
        self._elem_combo.currentTextChanged.connect(self._on_build_elem_change)
        self._build_toolbar_obj.addWidget(self._elem_combo)

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
        tb_action("Save SVG",  self._save_svg,  "Ctrl+S", "Export current view as SVG")
        tb.addSeparator()
        tb_action("Reset",     lambda: self._canvas.reset_view(), "Ctrl+R")
        tb.addSeparator()

        # Zoom readout
        self._zoom_lbl = QLabel("100%")
        self._zoom_lbl.setFixedWidth(46)
        self._zoom_lbl.setObjectName("zoom_label")
        # Style this in get_stylesheet instead of here
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
        elif ext == ".pdb":
            mol = parse_pdb(text, name=os.path.splitext(base)[0])
            src = "PDB"
        else:
            # Try XYZ first, then Gaussian input
            try:
                mol = parse_xyz(text, name=base)
                src = "XYZ (auto)"
            except Exception:
                try:
                    mol = parse_pdb(text, name=base)
                    src = "PDB (auto)"
                except Exception:
                    mol = parse_gaussian(text)
                    src = "Gaussian input (auto)"

        infer_bonds(mol)
        return mol, src

    def _load_and_display(self, path: str):
        """Parse a file and update all UI elements. Used by drag-and-drop and Open."""
        try:
            mol, src = self._load_mol_from_path(path)
            self._color_overrides = {}
            self._canvas.color_overrides = {}
            self._canvas.base_scale = float(self._scale_slider.value())
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
        self._menu_calc.clear()
        self._menu_calc.setEnabled(True)

        act_results = QAction("Calculation Results…", self)
        act_results.setShortcut("Ctrl+M")
        act_results.triggered.connect(self._show_calculations_dialog)
        self._menu_calc.addAction(act_results)
        
        if not mol.vibrational_modes and not mol.excited_states:
            self._menu_calc.setEnabled(False)

    def _show_calculations_dialog(self):
        mol = self._canvas.molecule
        if not mol: return
        dlg = CalculationsDialog(mol, parent=self)
        dlg.modeSelected.connect(self._show_vibration)
        dlg.stateSelected.connect(self._show_excited_state)
        dlg.viewSpectrum.connect(self._on_view_spectrum)
        dlg.animationToggled.connect(self._on_anim_toggle)
        dlg.vectorsToggled.connect(self._on_vectors_toggle)
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
        dlg = SpectrumDialog(x, y, "Frequency (cm-1)", "Intensity (km/mol)", "IR Spectrum", meta, self)
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
        dlg = SpectrumDialog(x, y, "Wavelength (nm)", "Oscillator Strength (f)", "UV-Vis Spectrum", meta, self)
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
            
        self.setProperty("theme", theme_name)
        self.style().unpolish(self)
        self.style().polish(self)

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
        if has_vib or has_td:
            self._calc_group.show()
            self._lbl_vib.setText(f"Vibrations: {len(mol.vibrational_modes)} modes")
            self._lbl_vib.setVisible(has_vib)
            self._lbl_td.setText(f"TD-DFT: {len(mol.excited_states)} states")
            self._lbl_td.setVisible(has_td)
        else:
            self._calc_group.hide()

        # Update main window title
        self.setWindowTitle(f"Molvector — {display_name}")

        status_path = f"{path}  |  " if path else ""
        self._status.showMessage(
            f"{status_path}{display_name}  |  {len(mol.atoms)} atoms, "
            f"{len(mol.bonds)} bonds  |  {mass:.3f} uma  [{src}]"
        )

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "All supported (*.xyz *.gjf *.com *.log *.out *.pdb);;"
            "XYZ (*.xyz);;Gaussian input (*.gjf *.com);;"
            "Gaussian log (*.log *.out);;PDB (*.pdb);;All files (*)"
        )
        if path:
            self._load_and_display(path)

    def _save_svg(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load a molecule first.")
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

    def _export_view(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load a molecule first.")
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
            QMessageBox.information(self, "No molecule", "Load a molecule first.")
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

    # ── Builder actions ───────────────────────────────────────────────────────

    def _toggle_build_mode(self, enabled: bool):
        # Sync menu and toolbar buttons
        self._act_build_toggle.setChecked(enabled)
        self._act_build_btn.setChecked(enabled)
        
        self._canvas.build_mode = enabled
        self._status.showMessage("Build Mode Active: Click to add, Drag to bond, Click bond to change order" if enabled else "Build Mode Off")
        if enabled:
            self._canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self._canvas.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def _on_build_elem_change(self, elem: str):
        self._canvas.build_element = elem

    def _clear_molecule(self):
        if QMessageBox.question(self, "Clear", "Clear the entire molecule?") == QMessageBox.StandardButton.Yes:
            self._save_history()
            self._canvas.molecule = Molecule("New Molecule", atoms=[])
            self._canvas.request_render()
            self._update_info_panel(self._canvas.molecule)

    def _clean_molecule(self):
        if not self._canvas.molecule or not self._canvas.molecule.atoms:
            return
        
        # Apply the simple spring-repulsion force field with user params
        self._save_history()
        optimize_geometry(
            self._canvas.molecule, 
            steps=self._ff_steps, 
            k_bond=self._ff_k_bond, 
            k_rep=self._ff_k_rep
        )
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._status.showMessage(f"Geometry optimized ({self._ff_steps} steps).", 3000)

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
        
        from PyQt6.QtWidgets import QFormLayout, QDoubleSpinBox, QSpinBox, QDialogButtonBox
        form = QFormLayout()
        
        s_steps = QSpinBox()
        s_steps.setRange(10, 2000)
        s_steps.setValue(self._ff_steps)
        form.addRow("Steps:", s_steps)
        
        s_kb = QDoubleSpinBox()
        s_kb.setRange(0.1, 100.0)
        s_kb.setValue(self._ff_k_bond)
        form.addRow("Bond Stiffness (k):", s_kb)
        
        s_kr = QDoubleSpinBox()
        s_kr.setRange(0.0, 50.0)
        s_kr.setValue(self._ff_k_rep)
        form.addRow("Repulsion (stiffness):", s_kr)
        
        l.addLayout(form)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        l.addWidget(btns)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._ff_steps = s_steps.value()
            self._ff_k_bond = s_kb.value()
            self._ff_k_rep = s_kr.value()
            self._clean_molecule()

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _edit_settings(self):
        orig_theme = self._current_theme
        orig_scale = self._canvas.atom_scale
        orig_width = self._canvas.bond_width_px
        orig_bg    = self._canvas.background

        def _live_update(theme, scale, width, bg):
            if theme != self._current_theme:
                self._apply_theme(theme)
            self._canvas.atom_scale = scale
            self._canvas.bond_width_px = width
            self._canvas.background = bg
            self._canvas.request_render()

        dlg = SettingsDialog(
            orig_theme,
            orig_scale,
            orig_width,
            orig_bg,
            live_callback=_live_update,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._apply_theme(dlg.theme)
            self._canvas.atom_scale = dlg.atom_scale
            self._canvas.bond_width_px = dlg.bond_width
            self._canvas.background = dlg.bg_color
            self._canvas.request_render()
        else:
            # Restore
            self._apply_theme(orig_theme)
            self._canvas.atom_scale = orig_scale
            self._canvas.bond_width_px = orig_width
            self._canvas.background = orig_bg
            self._canvas.request_render()

    def _edit_atom_colors(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load a molecule first.")
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
    app.setApplicationName("Molvector")
    app.setFont(QFont("Segoe UI", 10))

    win = MainWindow()
    win.show()

    # Optional: open file from command line
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win._load_and_display(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()