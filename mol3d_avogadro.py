"""
mol3d_avogadro.py — Renderer + Parsers library
================================================
Avogadro-style ball-and-stick SVG renderer for molecules.

Parsers:
  parse_xyz(text, name)       — standard XYZ format
  parse_gaussian(text)        — Gaussian .gjf / .com input
  parse_gaussian_log(text)    — Gaussian .log / .out output (last geometry)
  infer_bonds(mol)            — distance-threshold bond detection

Renderer:
  render_avogadro(mol, ...)   — produces an SVG file
  Accepts color_overrides={"C": "#ff0000", ...} for per-element colors.
"""

import math
import numpy as np
import svgwrite
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Atom:
    element: str
    x: float
    y: float
    z: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Bond:
    i: int
    j: int
    order: int = 1


@dataclass
class Molecule:
    name: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_xyz(text: str, name: str = "molecule") -> Molecule:
    """Standard XYZ: <N>\\n<comment>\\n<elem x y z> x N"""
    mol = Molecule(name=name)
    lines = [l.strip() for l in text.strip().splitlines()]
    n = int(lines[0])
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) >= 4:
            mol.atoms.append(Atom(p[0], float(p[1]), float(p[2]), float(p[3])))
    return mol


def parse_gaussian(text: str) -> Molecule:
    """
    Gaussian .gjf / .com input file.
    Sections separated by blank lines:
      route  ->  title  ->  charge/mult + coords
    """
    lines = text.strip().splitlines()
    sections, cur = [], []
    for l in lines:
        if l.strip() == "":
            if cur:
                sections.append(cur)
                cur = []
        else:
            cur.append(l.strip())
    if cur:
        sections.append(cur)

    name = sections[1][0] if len(sections) > 1 else "molecule"
    mol = Molecule(name=name)

    if len(sections) >= 3:
        for line in sections[2][1:]:
            p = line.split()
            if len(p) >= 4:
                try:
                    mol.atoms.append(Atom(p[0], float(p[1]), float(p[2]), float(p[3])))
                except ValueError:
                    pass
    return mol


def parse_gaussian_log(text: str) -> Molecule:
    """
    Gaussian .log / .out output file.

    Finds the LAST 'Standard orientation:' block (or 'Input orientation:'
    if standard is absent) — this is the final / optimised geometry.

    The coordinate table columns are:
      Center#  AtomicNum  AtomicType  X  Y  Z
    Atomic number is converted to element symbol via a built-in table.
    """
    _Z_TO_SYM = {
        1:"H",  2:"He", 3:"Li", 4:"Be", 5:"B",  6:"C",  7:"N",  8:"O",
        9:"F", 10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",15:"P", 16:"S",
       17:"Cl",18:"Ar",19:"K", 20:"Ca",26:"Fe",28:"Ni",29:"Cu",30:"Zn",
       35:"Br",53:"I", 79:"Au",80:"Hg",
    }

    lines = text.splitlines()

    # Extract job title from lines following the route card
    name = "Gaussian output"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            for j in range(i + 1, min(i + 30, len(lines))):
                s = lines[j].strip()
                if s and not s.startswith("-") and not s.startswith("#") \
                        and not s.startswith("%") and len(s) > 2:
                    name = s
                    break
            break

    # Find all orientation block header positions
    MARKERS = ["Standard orientation:", "Input orientation:", "Z-Matrix orientation:"]
    block_starts = []
    for i, line in enumerate(lines):
        for marker in MARKERS:
            if marker in line:
                block_starts.append((i, marker))
                break

    if not block_starts:
        raise ValueError(
            "No orientation block found in this file.\n"
            "Make sure it is a Gaussian output (.log/.out) with geometry data."
        )

    # Prefer last Standard orientation; fall back to last of any kind
    std = [b for b in block_starts if "Standard" in b[1]]
    chosen_idx = (std or block_starts)[-1][0]

    # The table starts 5 lines after the marker:
    #  +0  "Standard orientation:"
    #  +1  "------..."
    #  +2  column header line 1
    #  +3  column header line 2
    #  +4  "------..."
    #  +5  first data row
    data_start = chosen_idx + 5

    atoms = []
    for line in lines[data_start:]:
        s = line.strip()
        if s.startswith("-"):
            break
        parts = s.split()
        if len(parts) >= 6:
            try:
                atomic_num = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                elem = _Z_TO_SYM.get(atomic_num, f"X{atomic_num}")
                atoms.append(Atom(elem, x, y, z))
            except ValueError:
                continue

    if not atoms:
        raise ValueError(
            "Could not parse atom coordinates from the orientation block.\n"
            "The file may be truncated or use an unexpected format."
        )

    return Molecule(name=name, atoms=atoms)


# ─────────────────────────────────────────────────────────────────────────────
# BOND INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COVALENT_RADII: Dict[str, float] = {
    "H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,
    "S":1.05,"P":1.07,"Cl":1.02,"Br":1.20,"I":1.39,
    "B":0.82,"Si":1.11,
}

def infer_bonds(mol: Molecule, tol: float = 0.40) -> None:
    atoms = mol.atoms
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ri = COVALENT_RADII.get(atoms[i].element, 0.77)
            rj = COVALENT_RADII.get(atoms[j].element, 0.77)
            if np.linalg.norm(atoms[i].pos - atoms[j].pos) <= ri + rj + tol:
                mol.bonds.append(Bond(i, j))


# ─────────────────────────────────────────────────────────────────────────────
# CPK COLOURS
# ─────────────────────────────────────────────────────────────────────────────

CPK_BASE: Dict[str, str] = {
    "H":"#d4d4d4","C":"#444444","N":"#2050d0","O":"#cc1111",
    "F":"#66cc22","S":"#ddcc00","P":"#ff8800","Cl":"#11bb22",
    "Br":"#882200","I":"#660099","B":"#ffaa33","Si":"#8888aa",
}

CPK_DARK: Dict[str, str] = {
    "H":"#888888","C":"#111111","N":"#0a1a60","O":"#550000",
    "F":"#224400","S":"#554400","P":"#441100","Cl":"#003300",
    "Br":"#330000","I":"#220033","B":"#664400","Si":"#333344",
}

DEFAULT_BASE = "#cc44aa"
DEFAULT_DARK = "#440022"

VDW_RADII: Dict[str, float] = {
    "H":0.53,"C":0.77,"N":0.75,"O":0.73,"F":0.71,
    "S":1.02,"P":1.06,"Cl":0.99,"Br":1.14,"I":1.33,
    "B":0.87,"Si":1.10,
}
DEFAULT_VDW = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

def lighten(hex_color: str, factor: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return rgb_to_hex(r+(255-r)*factor, g+(255-g)*factor, b+(255-b)*factor)

def darken(hex_color: str, factor: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return rgb_to_hex(r*(1-factor), g*(1-factor), b*(1-factor))

def auto_dark(base: str) -> str:
    return darken(base, 0.65)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx,sx = math.cos(rx),math.sin(rx)
    cy,sy = math.cos(ry),math.sin(ry)
    cz,sz = math.cos(rz),math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def center_positions(atoms: List[Atom]) -> np.ndarray:
    pos = np.array([a.pos for a in atoms])
    return pos - pos.mean(axis=0)

def bond_half_polygon(
    ax:float, ay:float, bx:float, by:float,
    atom_r_px:float, half_width:float,
) -> List[Tuple[float,float]]:
    dx, dy = bx-ax, by-ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    ux,uy = dx/length, dy/length
    px,py = -uy, ux
    x0 = ax + ux*atom_r_px;  y0 = ay + uy*atom_r_px
    x1 = (ax+bx)/2;           y1 = (ay+by)/2
    return [
        (x0+px*half_width, y0+py*half_width),
        (x0-px*half_width, y0-py*half_width),
        (x1-px*half_width, y1-py*half_width),
        (x1+px*half_width, y1+py*half_width),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_avogadro(
    mol: Molecule,
    rot_x: float = 0.0,
    rot_y: float = 0.0,
    rot_z: float = 0.0,
    rot_matrix_override: Optional[np.ndarray] = None,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    canvas_w: int = 700,
    canvas_h: int = 600,
    scale: float = 110.0,
    atom_scale: float = 1.1,
    bond_width_px: float = 10.0,
    background: str = "#0a0a12",
    color_overrides: Optional[Dict[str, str]] = None,
    output_path: str = "molecule.svg",
) -> str:

    rot = rot_matrix_override if rot_matrix_override is not None \
          else rotation_matrix(rot_x, rot_y, rot_z)

    base_colors = dict(CPK_BASE)
    dark_colors  = dict(CPK_DARK)
    if color_overrides:
        for elem, col in color_overrides.items():
            base_colors[elem] = col
            dark_colors[elem]  = auto_dark(col)

    centered = center_positions(mol.atoms)
    cx = canvas_w/2 + pan_x
    cy = canvas_h/2 + pan_y

    proj = []
    for i, pos in enumerate(centered):
        rp   = rot @ pos
        elem = mol.atoms[i].element
        vdw  = VDW_RADII.get(elem, DEFAULT_VDW)
        r_px = vdw * scale * atom_scale
        proj.append((cx+rp[0]*scale, cy-rp[1]*scale, rp[2], r_px))

    dwg  = svgwrite.Drawing(output_path, size=(canvas_w, canvas_h))
    defs = dwg.defs
    dwg.add(dwg.rect(insert=(0,0), size=(canvas_w,canvas_h), fill=background))

    registered: set = set()

    def ensure_grad(elem: str) -> str:
        gid = f"sph_{elem}"
        if gid in registered:
            return gid
        base = base_colors.get(elem, DEFAULT_BASE)
        dark = dark_colors.get(elem, DEFAULT_DARK)
        g = dwg.radialGradient(id=gid, center=("33%","28%"), r="68%")
        g["fx"]="33%"; g["fy"]="28%"
        g.add_stop_color("0%",   lighten(base,0.92), 1.0)
        g.add_stop_color("18%",  lighten(base,0.55), 1.0)
        g.add_stop_color("48%",  lighten(base,0.25), 1.0)
        g.add_stop_color("78%",  base,               1.0)
        g.add_stop_color("100%", dark,               1.0)
        defs.add(g)
        registered.add(gid)
        return gid

    for atom in mol.atoms:
        ensure_grad(atom.element)

    draw_list = []
    for bond in mol.bonds:
        ai,aj = bond.i,bond.j
        ax,ay,az,ar = proj[ai]
        bx,by,bz,br = proj[aj]
        for (ox,oy,oz,orr),(tx,ty,_,__),col_e in [
            ((ax,ay,az,ar),(bx,by,bz,br), mol.atoms[ai].element),
            ((bx,by,bz,br),(ax,ay,az,ar), mol.atoms[aj].element),
        ]:
            poly = bond_half_polygon(ox,oy,tx,ty,orr,bond_width_px)
            if poly:
                col = base_colors.get(col_e, DEFAULT_BASE)
                draw_list.append((oz-0.001, 0, ("bond_half",poly,col)))

    for idx,atom in enumerate(mol.atoms):
        ax,ay,az,ar = proj[idx]
        gid  = ensure_grad(atom.element)
        base = base_colors.get(atom.element, DEFAULT_BASE)
        dark = dark_colors.get(atom.element, DEFAULT_DARK)
        draw_list.append((az, 1, ("atom",ax,ay,ar,gid,base,dark,atom.element)))

    draw_list.sort(key=lambda x:(x[0],x[1]))

    for _,__,item in draw_list:
        kind = item[0]
        if kind == "bond_half":
            _,pts,col = item
            dwg.add(dwg.polygon(
                points=pts, fill=darken(col,0.20),
                stroke=darken(col,0.55), stroke_width=0.5, opacity=0.95,
            ))
            if len(pts)==4:
                p0,p1,p2,p3 = pts
                dwg.add(dwg.line(
                    start=((p0[0]+p1[0])/2,(p0[1]+p1[1])/2),
                    end  =((p2[0]+p3[0])/2,(p2[1]+p3[1])/2),
                    stroke=lighten(col,0.55),
                    stroke_width=bond_width_px*0.35,
                    stroke_linecap="round", opacity=0.55,
                ))
        elif kind == "atom":
            _,ax,ay,ar,gid,base,dark,elem = item
            dwg.add(dwg.circle(center=(ax,ay),r=ar*1.04,fill=dark,stroke="none"))
            dwg.add(dwg.circle(center=(ax,ay),r=ar,fill=f"url(#{gid})",stroke="none"))

    dwg.add(dwg.text(
        mol.name, insert=(16,26),
        font_size=15,font_family="'Courier New',monospace",
        fill="#88ccff",opacity=0.75,
    ))
    dwg.save()
    return output_path