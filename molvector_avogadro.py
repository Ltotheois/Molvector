"""
molvector_avogadro.py — Renderer + Parsers library
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
class ExcitedState:
    index: int
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    symmetry: str


@dataclass
class VibrationalMode:
    index: int
    frequency: float  # cm^-1
    intensity: float = 0.0 # IR Intensity
    displacements: np.ndarray = None # (N, 3)



@dataclass
class Molecule:
    name: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    charge: int = 0
    excited_states: List[ExcitedState] = field(default_factory=list)
    vibrational_modes: List[VibrationalMode] = field(default_factory=list)



# ─────────────────────────────────────────────────────────────────────────────
# CHEMISTRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Standard atomic weights (g/mol), most common isotope / IUPAC 2021
ATOMIC_MASSES: Dict[str, float] = {
    "H":1.008,  "He":4.003, "Li":6.941, "Be":9.012, "B":10.811,
    "C":12.011, "N":14.007, "O":15.999, "F":18.998, "Ne":20.180,
    "Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.086,"P":30.974,
    "S":32.065, "Cl":35.453,"Ar":39.948,"K":39.098, "Ca":40.078,
    "Fe":55.845,"Ni":58.693,"Cu":63.546,"Zn":65.38, "Br":79.904,
    "I":126.904,"Au":196.967,"Hg":200.592,
}

# Hill order: C first, H second, then alphabetical
_HILL_PRIORITY = {"C": 0, "H": 1}

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _to_subscript(n: int) -> str:
    return str(n).translate(_SUBSCRIPT)

def chemical_formula(mol: "Molecule") -> str:
    """Return Hill-order chemical formula with Unicode subscript counts, e.g. C₆₀Ca."""
    from collections import Counter
    counts = Counter(a.element for a in mol.atoms)
    elems = sorted(counts.keys(),
                   key=lambda e: (_HILL_PRIORITY.get(e, 2), e))
    formula = "".join(f"{e}{_to_subscript(counts[e]) if counts[e] > 1 else ''}" for e in elems)
    return formula

def molecular_mass(mol: "Molecule") -> float:
    """Return the monoisotopic-approximate molecular mass in Da (g/mol)."""
    return sum(ATOMIC_MASSES.get(a.element, 0.0) for a in mol.atoms)


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_xyz(text: str, name: str = "molecule") -> Molecule:
    """Standard XYZ: <N>\n<comment>\n<elem x y z> x N"""
    lines = [l.strip() for l in text.strip().splitlines()]
    n = int(lines[0])
    comment = lines[1] if len(lines) > 1 else ""
    
    # Try to extract charge from comment line (e.g. "charge=1" or "charge 1")
    import re
    charge = 0
    charge_match = re.search(r"charge\s*[:=]?\s*([+-]?\d+)", comment, re.IGNORECASE)
    if charge_match:
        try:
            charge = int(charge_match.group(1))
        except ValueError:
            pass

    atoms = []
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) >= 4:
            atoms.append(Atom(p[0], float(p[1]), float(p[2]), float(p[3])))
    
    mol = Molecule(name=name, atoms=atoms, charge=charge)
    mol.name = chemical_formula(mol)
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
        # First line of section 3 is "charge multiplicity"
        try:
            charge_mult = sections[2][0].split()
            mol.charge = int(charge_mult[0])
        except (IndexError, ValueError):
            pass
        for line in sections[2][1:]:
            p = line.split()
            if len(p) >= 4:
                try:
                    mol.atoms.append(Atom(p[0], float(p[1]), float(p[2]), float(p[3])))
                except ValueError:
                    pass
    
    # Override name with formula for consistency
    mol.name = chemical_formula(mol)
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

    # Extract charge from "Charge = X  Multiplicity = Y" line
    charge = 0
    for line in lines:
        if "Charge =" in line and "Multiplicity =" in line:
            try:
                charge = int(line.split("Charge =")[1].split()[0])
            except (IndexError, ValueError):
                pass
            break

    # Extract excited states (TDDFT)
    excited_states = []
    import re
    for line in lines:
        # Excited State   1:  2.002-?Sym    0.1463 eV 8473.13 nm  f=0.0000  <S**2>=0.752
        m = re.search(r"Excited State\s+(\d+):\s+(.+?)\s+([\d.]+) eV\s+([\d.]+) nm\s+f=([\d.]+)", line)
        if m:
            idx, sym, ev, nm, f = m.groups()
            excited_states.append(ExcitedState(int(idx), float(ev.strip()), float(nm.strip()), float(f.strip()), sym.strip()))

    # Extract vibrational modes
    vibrational_modes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("Frequencies --"):
            freqs = [float(f) for f in line.split("--")[1].split()]
            n_freqs = len(freqs)
            
            # Find IR intensities if present
            intensities = [0.0] * n_freqs
            j = i + 1
            while j < i + 10 and j < len(lines):
                if "IR Inten    --" in lines[j]:
                    try:
                        intensities = [float(x) for x in lines[j].split("--")[1].split()]
                    except ValueError:
                        pass
                    break
                j += 1

            # Find the start of the displacement table
            while i < len(lines) and "  Atom  AN      X      Y      Z" not in lines[i]:
                i += 1
            i += 1 # skip header
            
            disps = [[] for _ in range(n_freqs)]
            for _ in range(len(atoms)):
                if i >= len(lines): break
                parts = lines[i].split()
                # parts[0]=atom#, parts[1]=atomic#, then 3 coords per freq
                for f_idx in range(n_freqs):
                    dx = float(parts[2 + f_idx*3])
                    dy = float(parts[3 + f_idx*3])
                    dz = float(parts[4 + f_idx*3])
                    disps[f_idx].append([dx, dy, dz])
                i += 1
            
            for f_idx in range(n_freqs):
                vibrational_modes.append(VibrationalMode(
                    index=len(vibrational_modes) + 1,
                    frequency=freqs[f_idx],
                    intensity=intensities[f_idx] if f_idx < len(intensities) else 0.0,
                    displacements=np.array(disps[f_idx])
                ))
            continue
        i += 1

    mol = Molecule(name=name, atoms=atoms, charge=charge, 
                   excited_states=excited_states, 
                   vibrational_modes=vibrational_modes)
    mol.name = chemical_formula(mol)
    return mol


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
            dist = np.linalg.norm(atoms[i].pos - atoms[j].pos)
            if dist <= ri + rj + tol:
                r_single = ri + rj
                if dist <= r_single * 0.85:
                    mol.bonds.append(Bond(i, j, 3))
                elif dist <= r_single * 0.92:
                    mol.bonds.append(Bond(i, j, 2))
                else:
                    mol.bonds.append(Bond(i, j, 1))


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

def interpolate_color(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex(r1 + (r2-r1)*t, g1 + (g2-g1)*t, b1 + (b2-b1)*t)


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

def bond_half_line(
    ax:float, ay:float, bx:float, by:float,
    atom_r_px:float, offset:float = 0.0,
) -> Tuple[Optional[Tuple[Tuple[float,float], Tuple[float,float]]], Tuple[float,float]]:
    dx, dy = bx-ax, by-ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None, (0.0, 0.0)
    ux,uy = dx/length, dy/length
    px,py = -uy, ux
    
    cx_a = ax + px*offset; cy_a = ay + py*offset
    cx_b = bx + px*offset; cy_b = by + py*offset
    
    x0 = cx_a + ux*atom_r_px;  y0 = cy_a + uy*atom_r_px
    x1 = (cx_a+cx_b)/2;        y1 = (cy_a+cy_b)/2
    return ((x0, y0), (x1, y1)), (px, py)


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
    atom_scale: float = 0.8,
    bond_width_px: float = 10.0,
    background: str = "#0a0a12",
    color_overrides: Optional[Dict[str, str]] = None,
    output_path: str = "molecule.svg",
    export_mode: bool = False,
    vectors: Optional[List[Tuple[int, float, float, float, str]]] = None,
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

    CAMERA_Z = 60.0
    
    proj = []
    for i in range(len(mol.atoms)):
        pos  = centered[i]
        rp   = rot @ pos
        elem = mol.atoms[i].element
        vdw  = VDW_RADII.get(elem, DEFAULT_VDW)
        
        z_factor = CAMERA_Z / (CAMERA_Z - rp[2]) if (CAMERA_Z - rp[2]) != 0 else 1.0
        
        r_px = vdw * scale * atom_scale * z_factor
        px_coord = cx + rp[0] * scale * z_factor
        py_coord = cy - rp[1] * scale * z_factor
        
        proj.append((px_coord, py_coord, rp[2], r_px))

    dwg  = svgwrite.Drawing(output_path, size=(canvas_w, canvas_h))
    defs = dwg.defs
    if not export_mode:
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
        g.add_stop_color("0%",   lighten(base,0.70), 1.0)
        g.add_stop_color("18%",  lighten(base,0.40), 1.0)
        g.add_stop_color("48%",  lighten(base,0.15), 1.0)
        g.add_stop_color("78%",  base,               1.0)
        g.add_stop_color("100%", dark,               1.0)
        defs.add(g)
        registered.add(gid)
        return gid

    for atom in mol.atoms:
        ensure_grad(atom.element)

    # SVD to find best-fit molecule plane normal
    if len(centered) >= 3:
        U, S, Vt = np.linalg.svd(centered)
        M_normal = Vt[-1]
    else:
        M_normal = np.array([0.0, 0.0, 1.0])

    draw_list = []
    for bi, bond in enumerate(mol.bonds):
        ai,aj = bond.i,bond.j
        A_orig = centered[ai]
        B_orig = centered[aj]
        u_3D = B_orig - A_orig
        
        rp_A_orig = rot @ A_orig
        rp_B_orig = rot @ B_orig
        orig_az = rp_A_orig[2]
        orig_bz = rp_B_orig[2]
        
        dir_3D = np.cross(M_normal, u_3D)
        if np.linalg.norm(dir_3D) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(u_3D, fallback)) < 1e-6:
                fallback = np.array([0.0, 1.0, 0.0])
            dir_3D = np.cross(fallback, u_3D)
        dir_3D = dir_3D / np.linalg.norm(dir_3D)
        
        # Bond width offset in Angstroms
        hw_angstrom = bond_width_px / 110.0
        
        if bond.order == 1:
            offsets = [0.0]
            indiv_hw_angstrom = hw_angstrom
        elif bond.order == 2:
            offsets = [-hw_angstrom * 1.1, hw_angstrom * 1.1]
            indiv_hw_angstrom = hw_angstrom * 0.6
        else:
            offsets = [-hw_angstrom * 2.0, 0.0, hw_angstrom * 2.0]
            indiv_hw_angstrom = hw_angstrom * 0.5
            
        for o_idx, offset_A in enumerate(offsets):
            A_offset = A_orig + offset_A * dir_3D
            B_offset = B_orig + offset_A * dir_3D
            
            rpA = rot @ A_offset
            rpB = rot @ B_offset
            
            zA_factor = CAMERA_Z / (CAMERA_Z - rpA[2]) if (CAMERA_Z - rpA[2]) != 0 else 1.0
            zB_factor = CAMERA_Z / (CAMERA_Z - rpB[2]) if (CAMERA_Z - rpB[2]) != 0 else 1.0
            
            ax, ay, az = cx + rpA[0]*scale*zA_factor, cy - rpA[1]*scale*zA_factor, rpA[2]
            bx, by, bz = cx + rpB[0]*scale*zB_factor, cy - rpB[1]*scale*zB_factor, rpB[2]
            
            avg_z_factor = (zA_factor + zB_factor) / 2.0
            indiv_hw_px = indiv_hw_angstrom * scale * avg_z_factor
            
            for (ox,oy,oz),(tx,ty,tz),col_e,is_first in [
                ((ax,ay,az),(bx,by,bz), mol.atoms[ai].element, True),
                ((bx,by,bz),(ax,ay,az), mol.atoms[aj].element, False),
            ]:
                line_pts, (px, py) = bond_half_line(ox,oy,tx,ty, 0.0, 0.0)
                if line_pts:
                    col = base_colors.get(col_e, DEFAULT_BASE)
                    b_id = f"b_g_{bi}_{o_idx}_{col_e}_{1 if is_first else 0}"
                    z_sort = min(orig_az, orig_bz) - 0.001
                    draw_list.append((z_sort, 0, ("bond_half", line_pts, px, py, indiv_hw_px, b_id, col_e)))

    for idx,atom in enumerate(mol.atoms):
        ax,ay,az,ar = proj[idx]
        gid  = ensure_grad(atom.element)
        base = base_colors.get(atom.element, DEFAULT_BASE)
        dark = dark_colors.get(atom.element, DEFAULT_DARK)
        draw_list.append((az, 1, ("atom",ax,ay,ar,gid,base,dark,atom.element)))

    # Add vectors (e.g. vibrational displacements)
    if vectors:
        # Vector format: (atom_idx, dx, dy, dz, color)
        for ai, vx, vy, vz, vcol in vectors:
            A_orig = centered[ai]
            B_orig = A_orig + np.array([vx, vy, vz])
            
            rpA = rot @ A_orig
            rpB = rot @ B_orig
            
            zA_factor = CAMERA_Z / (CAMERA_Z - rpA[2]) if (CAMERA_Z - rpA[2]) != 0 else 1.0
            zB_factor = CAMERA_Z / (CAMERA_Z - rpB[2]) if (CAMERA_Z - rpB[2]) != 0 else 1.0
            
            ax, ay, az = cx + rpA[0]*scale*zA_factor, cy - rpA[1]*scale*zA_factor, rpA[2]
            bx, by, bz = cx + rpB[0]*scale*zB_factor, cy - rpB[1]*scale*zB_factor, rpB[2]
            
            # Simple arrow: line + triangle at the tip
            draw_list.append((max(az, bz), 2, ("vector", ax, ay, bx, by, vcol)))

    draw_list.sort(key=lambda x:(x[0],x[1]))

    for _,__,item in draw_list:
        kind = item[0]
        if kind == "bond_half":
            _, pts, px, py, indiv_hw, b_id, col_e = item
            
            Lx, Ly, Lz = -0.34, -0.44, 0.83
            A = px * Lx + py * Ly
            I_max = math.hypot(A, Lz)
            if I_max == 0: I_max = 1e-6
            
            cx_poly = (pts[0][0] + pts[1][0]) / 2
            cy_poly = (pts[0][1] + pts[1][1]) / 2
            
            x1 = cx_poly + px * indiv_hw
            y1 = cy_poly + py * indiv_hw
            x2 = cx_poly - px * indiv_hw
            y2 = cy_poly - py * indiv_hw
            
            g = dwg.linearGradient(
                id=b_id,
                start=(x1, y1), end=(x2, y2),
                gradientUnits="userSpaceOnUse"
            )
            base = base_colors.get(col_e, DEFAULT_BASE)
            dark = dark_colors.get(col_e, DEFAULT_DARK)
            
            def get_color_from_ratio(r: float) -> str:
                d = 1.0 - r
                stops = [
                    (0.00, lighten(base, 0.70)),
                    (0.18, lighten(base, 0.40)),
                    (0.48, lighten(base, 0.15)),
                    (0.78, base),
                    (1.00, dark)
                ]
                if d <= 0: return stops[0][1]
                if d >= 1: return stops[-1][1]
                for i in range(len(stops)-1):
                    if stops[i][0] <= d <= stops[i+1][0]:
                        t_ratio = (d - stops[i][0]) / (stops[i+1][0] - stops[i][0])
                        return interpolate_color(stops[i][1], stops[i+1][1], t_ratio)
                return dark

            for i in range(11):
                v = i / 10.0
                s = 1.0 - 2.0 * v
                intensity = max(0.0, s * A + math.sqrt(max(0.0, 1.0 - s**2)) * Lz)
                ratio = intensity / I_max
                g.add_stop_color(f"{v*100:.1f}%", get_color_from_ratio(ratio), 1.0)
            defs.add(g)
            
            dwg.add(dwg.line(
                start=pts[0], end=pts[1],
                stroke=f"url(#{b_id})", stroke_width=indiv_hw * 2,
                stroke_linecap="round"
            ))
        elif kind == "atom":
            _, ax, ay, ar, gid, base, dark, elem = item
            dwg.add(dwg.circle(center=(ax,ay),r=ar*1.04,fill=dark,stroke="none"))
            dwg.add(dwg.circle(center=(ax,ay),r=ar,fill=f"url(#{gid})",stroke="none"))
        elif kind == "vector":
            _, x0, y0, x1, y1, col = item
            # Main line
            dwg.add(dwg.line(start=(x0, y0), end=(x1, y1), stroke=col, stroke_width=2.5, stroke_linecap="round"))
            # Arrowhead
            dx, dy = x1-x0, y1-y0
            L = math.hypot(dx, dy)
            if L > 1e-3:
                ux, uy = dx/L, dy/L
                px, py = -uy, ux
                # Points for triangle
                p1 = (x1, y1)
                p2 = (x1 - ux*6 + px*3.5, y1 - uy*6 + py*3.5)
                p3 = (x1 - ux*6 - px*3.5, y1 - uy*6 - py*3.5)
                dwg.add(dwg.polygon(points=[p1, p2, p3], fill=col))

    if not export_mode:
        dwg.add(dwg.text(
            mol.name, insert=(16,26),
            font_size=15,font_family="'Courier New',monospace",
            fill="#88ccff",opacity=0.75,
        ))
    dwg.save()
    return output_path