"""
molvector_avogadro.py — Renderer + Parsers library
================================================
Avogadro-style ball-and-stick SVG renderer for molecules.

Parsers:
  parse_xyz(text, name)       — standard XYZ format
  parse_gaussian(text)        — Gaussian .gjf / .com input
  parse_gaussian_log(text)    — Gaussian .log / .out output (last geometry)
  parse_pdb(text)             — standard PDB format
  infer_bonds(mol)            — distance-threshold bond detection

Renderer:
  render_avogadro(mol, ...)   — produces an SVG file

Writers:
  save_xyz(mol)               — returns XYZ string
  save_gaussian_input(mol)    — returns GJF string
  save_pdb(mol)               — returns PDB string
"""

import math, random, string
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

def chemical_formula(mol: "Molecule") -> str:
    """Return Hill-order chemical formula, e.g. C60Ca."""
    from collections import Counter
    counts = Counter(a.element for a in mol.atoms)
    elems = sorted(counts.keys(),
                   key=lambda e: (_HILL_PRIORITY.get(e, 2), e))
    formula = "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in elems)
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


def parse_pdb(text: str, name: str = "PDB Molecule") -> Molecule:
    """Basic PDB parser for ATOM/HETATM and CONECT records."""
    lines = text.splitlines()
    atoms = []
    bonds = []
    serial_to_idx = {}
    import re
    
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                # Fixed-width column parsing
                serial_str = line[6:11].strip()
                if not serial_str: continue
                serial = int(serial_str)
                
                atom_name = line[12:16].strip()
                x_str = line[30:38].strip()
                y_str = line[38:46].strip()
                z_str = line[46:54].strip()
                x, y, z = float(x_str), float(y_str), float(z_str)
                
                # Element symbol at 77-78, or infer from name
                elem = line[76:78].strip()
                if not elem:
                    elem = re.sub(r'[^a-zA-Z]', '', atom_name)
                    if len(elem) > 1 and elem[1].islower():
                        elem = elem[:2]
                    else:
                        elem = elem[:1]
                
                if len(elem) == 1: elem = elem.upper()
                else: elem = elem.capitalize()
                
                serial_to_idx[serial] = len(atoms)
                atoms.append(Atom(elem, x, y, z))
            except (ValueError, IndexError):
                continue
        
        elif line.startswith("CONECT"):
            try:
                # CONECT serial1 serial2 serial3...
                # Columns: 7-11, 12-16, 17-21, 22-26, 27-31
                parts = []
                for i in range(6, len(line), 5):
                    s = line[i:i+5].strip()
                    if s: parts.append(s)
                
                if not parts: continue
                src_serial = int(parts[0])
                if src_serial not in serial_to_idx: continue
                src_idx = serial_to_idx[src_serial]
                
                for target_str in parts[1:]:
                    target_serial = int(target_str)
                    if target_serial not in serial_to_idx: continue
                    target_idx = serial_to_idx[target_serial]
                    
                    if src_idx < target_idx:
                        bonds.append(Bond(src_idx, target_idx))
            except (ValueError, IndexError):
                continue

    if not atoms:
        raise ValueError("No valid ATOM or HETATM records found in PDB.")

    mol = Molecule(name=name, atoms=atoms, bonds=bonds)
    # Refine name from HEADER or TITLE
    for line in lines:
        if line.startswith("HEADER") and len(line) > 10:
            mol.name = line[10:50].strip() or name
            break
        elif line.startswith("TITLE ") and len(line) > 10:
            mol.name = line[10:70].strip() or name
            break
    
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
            # Handle cases with -- or ---
            after_marker = line.split("--", 1)[1].lstrip("-")
            freqs = [float(f) for f in after_marker.split()]
            n_freqs = len(freqs)
            
            # Find IR intensities if present
            intensities = [0.0] * n_freqs
            j = i + 1
            while j < i + 15 and j < len(lines):
                if "IR Inten    --" in lines[j]:
                    try:
                        after_marker_ir = lines[j].split("--", 1)[1].lstrip("-")
                        intensities = [float(x) for x in after_marker_ir.split()]
                    except ValueError:
                        pass
                    break
                j += 1

            # Find the start of the displacement table
            has_coord_atom = False
            while i < len(lines):
                if "  Atom  AN      X      Y      Z" in lines[i]:
                    has_coord_atom = False
                    break
                if " Coord Atom Element:" in lines[i]:
                    has_coord_atom = True
                    break
                i += 1
            i += 1 # skip header
            
            disps = [[] for _ in range(n_freqs)]
            if has_coord_atom:
                # Format: 3 lines per atom (X, then Y, then Z)
                for _ in range(len(atoms)):
                    # X line
                    parts_x = lines[i].split(); i += 1
                    # Y line
                    parts_y = lines[i].split(); i += 1
                    # Z line
                    parts_z = lines[i].split(); i += 1
                    
                    for f_idx in range(n_freqs):
                        dx = float(parts_x[3 + f_idx])
                        dy = float(parts_y[3 + f_idx])
                        dz = float(parts_z[3 + f_idx])
                        disps[f_idx].append([dx, dy, dz])
            else:
                # Standard format: 1 line per atom
                for _ in range(len(atoms)):
                    if i >= len(lines): break
                    parts = lines[i].split()
                    for f_idx in range(n_freqs):
                        dx = float(parts[2 + f_idx*3])
                        dy = float(parts[3 + f_idx*3])
                        dz = float(parts[4 + f_idx*3])
                        disps[f_idx].append([dx, dy, dz])
                    i += 1
            
            # Clear existing modes if we found a "fresh" set (index 1 to 3/5)
            # Gaussian usually restarts numbering for a new frequency block.
            if vibrational_modes and vibrational_modes[-1].index >= freqs[0]:
                vibrational_modes = []
            
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
# WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def save_xyz(mol: "Molecule") -> str:
    """Produce standard XYZ string."""
    lines = [str(len(mol.atoms)), mol.name]
    for a in mol.atoms:
        lines.append(f"{a.element:3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
    return "\n".join(lines)

def save_gaussian_input(mol: "Molecule") -> str:
    """Produce a basic Gaussian input (.gjf) string."""
    lines = [
        "%nprocshared=4",
        "%mem=4GB",
        f"# p opt freq b3lyp/6-31g(d)",
        "",
        mol.name,
        "",
        f"{mol.charge} 1"
    ]
    for a in mol.atoms:
        lines.append(f"{a.element:3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
    lines.append("")  # Gaussian needs trailing blank line
    return "\n".join(lines)

def save_pdb(mol: "Molecule") -> str:
    """Produce a basic PDB string."""
    lines = [f"HEADER    {mol.name[:40]:<40}", f"TITLE     {mol.name}"]
    for i, a in enumerate(mol.atoms):
        serial = (i + 1) % 100000
        line = f"HETATM{serial:5d} {a.element:<3s}  UNL A   1    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          {a.element:>2s}"
        lines.append(line)
    
    for b in mol.bonds:
        lines.append(f"CONECT{b.i+1:5d}{b.j+1:5d}")
        
    lines.append("END")
    return "\n".join(lines)


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
    if not atoms:
        return np.zeros((0, 3))
    pos = np.array([a.pos for a in atoms])
    return pos - pos.mean(axis=0)

def optimize_geometry(mol: Molecule, steps: int = 150, k_bond: float = 8.0, k_rep: float = 1.5):
    """
    Apply a simple force-directed layout (spring-electric) to 'clean' 
    the molecular geometry.
    """
    if not mol.atoms: return
    
    # 1. Prepare positions
    pos = np.array([a.pos for a in mol.atoms], dtype=np.float64)
    n = len(pos)
    if n < 2: return
    
    # 2. Calculate ideal bond lengths
    # Using VDW_RADII as a proxy for covalent radii
    bond_params = []
    for b in mol.bonds:
        r1 = VDW_RADII.get(mol.atoms[b.i].element, 0.8)
        r2 = VDW_RADII.get(mol.atoms[b.j].element, 0.8)
        ideal = (r1 + r2) * (0.95 if b.order > 1 else 1.0)
        if b.order == 3: ideal *= 0.9
        bond_params.append((b.i, b.j, ideal))

    # 3. Simple Iterative Relaxation (Gradient Descent with Momentum)
    vel = np.zeros_like(pos)
    dt = 0.05
    damping = 0.85
    
    for _ in range(steps):
        forces = np.zeros_like(pos)
        
        # Bond Springs
        for i, j, r0 in bond_params:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-4: 
                # Avoid singularity: push apart randomly
                forces[i] += np.random.normal(0, 0.1, 3)
                continue
            
            f_mag = -k_bond * (dist - r0)
            f_vec = (diff / dist) * f_mag
            forces[i] += f_vec
            forces[j] -= f_vec
            
        # Non-bonded Repulsion (Van der Waals overlaps)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-4: dist = 0.1
                
                # We want atoms to be at least r_sum apart
                r_sum = VDW_RADII.get(mol.atoms[i].element, 0.8) + VDW_RADII.get(mol.atoms[j].element, 0.8)
                if dist < r_sum * 1.8:
                    f_mag = k_rep * (r_sum / (dist + 0.1))**4
                    forces[i] += (diff / dist) * f_mag
                    forces[j] -= (diff / dist) * f_mag

        # Update
        vel = (vel + forces * dt) * damping
        pos += vel * dt
        
    # 4. Write back
    for i, a in enumerate(mol.atoms):
        a.x, a.y, a.z = pos[i]

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


def project_molecule(
    mol: Molecule,
    rot: np.ndarray,
    pan_x: float,
    pan_y: float,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    atom_scale: float
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float, float, int]]]:
    """
    Project 3D molecule to 2D canvas coordinates.
    Returns:
      atom_projs: list of (px, py, pz, r_px)
      bond_projs: list of (ax, ay, bx, by, z_avg, bond_idx)
    """
    centered = center_positions(mol.atoms)
    cx = canvas_w/2 + pan_x
    cy = canvas_h/2 + pan_y
    CAMERA_Z = 60.0
    
    atom_projs = []
    for i, atom in enumerate(mol.atoms):
        rp = rot @ centered[i]
        elem = atom.element
        vdw  = VDW_RADII.get(elem, DEFAULT_VDW)
        z_factor = CAMERA_Z / (CAMERA_Z - rp[2]) if (CAMERA_Z - rp[2]) != 0 else 1.0
        r_px = vdw * scale * atom_scale * z_factor
        px = cx + rp[0] * scale * z_factor
        py = cy - rp[1] * scale * z_factor
        atom_projs.append((px, py, rp[2], r_px))
        
    bond_projs = []
    for i, bond in enumerate(mol.bonds):
        pA, pB = atom_projs[bond.i], atom_projs[bond.j]
        bond_projs.append((pA[0], pA[1], pB[0], pB[1], (pA[2] + pB[2])/2, i))
        
    return atom_projs, bond_projs

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
    active_vectors: Optional[np.ndarray] = None,
    animation_phase: float = 0.0,
    animation_amplitude: float = 0.0,
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
    for i, atom in enumerate(mol.atoms):
        pos = centered[i]
        if animation_amplitude > 0 and active_vectors is not None:
            # Oscillation: pos + vector * amp * sin(phase)
            pos = pos + active_vectors[i] * animation_amplitude * math.sin(animation_phase)
            
        rp   = rot @ pos
        elem = atom.element
        vdw  = VDW_RADII.get(elem, DEFAULT_VDW)
        
        z_factor = CAMERA_Z / (CAMERA_Z - rp[2]) if (CAMERA_Z - rp[2]) != 0 else 1.0
        
        r_px = vdw * scale * atom_scale * z_factor
        px_coord = cx + rp[0] * scale * z_factor
        py_coord = cy - rp[1] * scale * z_factor
        
        proj.append((px_coord, py_coord, rp[2], r_px))

    dwg  = svgwrite.Drawing(output_path, size=(canvas_w, canvas_h))
    defs = dwg.defs
    
    # Use a random prefix for all IDs to prevent collisions in Inkscape
    prefix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    if not export_mode:
        dwg.add(dwg.rect(insert=(0,0), size=(canvas_w,canvas_h), fill=background))

    registered: set = set()

    def ensure_grad(elem: str) -> str:
        gid = f"sph_{elem}_{prefix}"
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
        
        if animation_amplitude > 0 and active_vectors is not None:
            A_orig = A_orig + active_vectors[ai] * animation_amplitude * math.sin(animation_phase)
            B_orig = B_orig + active_vectors[aj] * animation_amplitude * math.sin(animation_phase)

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
                    b_id = f"b_g_{bi}_{o_idx}_{col_e}_{prefix}_{1 if is_first else 0}"
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
            if animation_amplitude > 0 and active_vectors is not None:
                A_orig = A_orig + active_vectors[ai] * animation_amplitude * math.sin(animation_phase)
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

    molecule_group = dwg.g(id="molecule")
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
            
            molecule_group.add(dwg.line(
                start=pts[0], end=pts[1],
                stroke=f"url(#{b_id})", stroke_width=indiv_hw * 2,
                stroke_linecap="round"
            ))
        elif kind == "atom":
            _, ax, ay, ar, gid, base, dark, elem = item
            molecule_group.add(dwg.circle(center=(ax,ay),r=ar*1.04,fill=dark,stroke="none"))
            molecule_group.add(dwg.circle(center=(ax,ay),r=ar,fill=f"url(#{gid})",stroke="none"))
        elif kind == "vector":
            _, x0, y0, x1, y1, col = item
            # Main line
            molecule_group.add(dwg.line(start=(x0, y0), end=(x1, y1), stroke=col, stroke_width=2.5, stroke_linecap="round"))
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
                molecule_group.add(dwg.polygon(points=[p1, p2, p3], fill=col))
    
    molecule_group['id'] = f"mol_{prefix}"
    dwg.add(molecule_group)

    if not export_mode:
        dwg.add(dwg.text(
            mol.name, insert=(16,26),
            font_size=15,font_family="'Courier New',monospace",
            fill="#88ccff",opacity=0.75,
        ))
    dwg.save()
    return output_path