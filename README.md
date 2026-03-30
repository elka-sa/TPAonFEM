# tpa_on_fem

Transfer Path Analysis (TPA) and Panel Contribution Analysis for structural, acoustic, and vibro-acoustic Finite Element models.

This repository contains a Python implementation of **Transfer Path Analysis (TPA)** and **Panel Contribution Analysis** using **Finite Element (FE) matrices exported from ANSYS**.  
The framework allows the study of vibro-acoustic systems composed of structural and acoustic subdomains using **Dynamic Substructuring (DS)** and **Component Mode Synthesis (CMS)**.

The methodology enables the identification and quantification of transmission paths between subsystems in coupled structural-acoustic models.

---

# Overview

The code performs the following workflow:

1. Import **Finite Element matrices** (`M`, `D`, `K`) from ANSYS `.full` files.
2. Import **mesh and nodal data** from ANSYS `.rst` files.
3. Define **subdomains** (structures or acoustic cavities).
4. Define **interfaces** between subdomains.
5. Build the **Dynamic Substructuring model**.
6. Assemble the **TPA matrices** using optional **CMS reduction**.
7. Apply **external loads**.
8. Solve the **harmonic vibro-acoustic response**.
9. Extract **transfer paths** and **path contributions**.
10. Visualize results using **matplotlib** or **pyvista**.

The implementation supports:

- Structural subdomains
- Acoustic subdomains
- Vibro-acoustic coupling
- Multi-level Transfer Path Analysis
- Component Mode Synthesis reduction methods

---

# Repository Structure
```text
tpa_on_fem/
├── CMS_TPA_Finite_Elements.py      # Core implementation
├── extra/
│   └── full.py                     # Modified ansys-mapdl-reader file with damping matrix extraction
├── examples/
│   ├── structural_model/
│   ├── acoustic_model/
│   ├── cabin_model/
│   └── vibro-acoustic_model/
└── README.md
```


Each example contains:
```text
example_name/
├── example_script.py               # Python script running the TPA analysis
├── model_data/
│   ├── S1/
│   │   ├── file.full
│   │   └── file.rst
│   ├── S2/
│   ├── F1/
│   └── ...
└── jobcases/
    └── coupling_example/

```

Where:

- **S\*** = structural subdomains
- **F\*** = acoustic subdomains
- `file.full` contains FE matrices
- `file.rst` contains mesh and nodal data

---

# Requirements

The code requires **Python ≥ 3.9**.

Main dependencies:
numpy
scipy
matplotlib
pyvista
joblib
scikit-learn
psutil
pypardiso
sparse_dot_mkl

ANSYS Python interfaces:
ansys-mapdl-reader
ansys-dpf-core
ansys-math-core

These packages allow reading ANSYS files and performing matrix operations.


# Installation

Clone the repository:

```bash
git clone https://gitlab.com/your_username/tpa_on_fem.git
cd tpa_on_fem
```

```bash
Install dependencies:
pip install numpy scipy matplotlib pyvista joblib scikit-learn psutil pypardiso sparse_dot_mkl
```

```bash
Install ANSYS Python packages:
pip install ansys-mapdl-reader ansys-dpf-core ansys-math-core
```

---


# ANSYS Requirements

The framework relies on ANSYS exported data:

Required files:
file.full
file.rst


These files contain:

| File | Content |
|-----|------|
| `.full` | Global FE matrices (M, D, K) |
| `.rst` | Mesh, node coordinates, connectivity |

The ANSYS Python interfaces typically require a **valid ANSYS installation/license**.

## Additional Requirement: Modified ANSYS Reader

This repository includes a modified version of a file from the `ansys-mapdl-reader` package.

The modification enables the extraction of the **damping matrix (D)** from ANSYS `.full` files through the function: load_km()

This functionality is **not available in the original implementation** of the package.


### Required Manual Step

A modified file is provided in this repository: extra/full.py

This file **must overwrite** the `full.py` file inside the installed `ansys-mapdl-reader` package.


### Why This Is Required

The original `ansys-mapdl-reader` implementation only allows extraction of:

- stiffness matrix **K**
- mass matrix **M**

The modified version adds support for extracting the **damping matrix D**, which is required by this framework for vibro-acoustic simulations.


### How to Apply the Modification

#### Locate the installed package

Find the installation directory of ansys-mapdl-reader

#### Replace the file

into the package folder and overwrite the existing file: site-packages/ansys/mapdl/reader/full.py

#### Important Note

If the `ansys-mapdl-reader` package is updated, this step may need to be repeated.


---


# Core Module

The main implementation of the framework is contained in: CMS_TPA_Finite_Elements.py


This module provides all functions required to perform Transfer Path Analysis (TPA) on structural, acoustic and vibro-acoustic finite element models using Dynamic Substructuring (DS) and Component Mode Synthesis (CMS).


# Main Functions

## Build Dynamic Substructuring Model
build_ds_models_vibroacoustic(subdomains, interface_pairs, save_info=(False, '', ''))

Loads the finite element matrices and mesh information from ANSYS files and constructs the data structures required for the DS assembly.

**Inputs**

- `subdomains`  
  Dictionary defining the FE substructures.  
  Each entry contains:
  - folder path containing ANSYS files
  - number of DOF per node
  - rigid mode tolerance
  - interface condition
  - density (to convert mass to volume in acoustic domains)
  - load flag (loads data files from .joblib instead of .full/.rst)

- `interface_pairs`  
  Dictionary defining all interfaces between subdomains.

**Outputs**

- `component_matrices`  
  Dictionary containing:

  - stiffness matrices `K`
  - mass matrices `M`
  - damping matrices `D`
  - constraint matrices
  - nodal coordinates
  - connectivity
  - domain shapes
  - inverse stiffness matrices

- `coupling_info`  
  Dictionary containing interface coupling operators used for DS assembly.

This step imports the FE matrices (`M`, `D`, `K`) from ANSYS `.full` files and mesh information from `.rst` files.



## Assemble Transfer Path Analysis Matrices
TPA_matrices_vibroacoustic(component_matrices, coupling_info, level_components,
    level_interfaces, CMS_method=['', 0], modal_CB_va=False,  save_job_info=(False, '', ''),
    assembly_modes=[False, 0])


Builds the level-wise TPA matrices using Dynamic Substructuring and optional Component Mode Synthesis reduction.

For each TPA level the function:

1. Assembles component matrices
2. Constructs interface coupling operators
3. Applies CMS reduction
4. Builds reduced equations of motion

The function returns reduced matrices and prescription matrices used to solve the TPA problem.

**Outputs**

- `Mat_red`  
  Reduced matrices for each level and interface.

- `Mat_pres`  
  Prescription matrices used to impose interface motion or forces.

- `Coup_info`  
  Detailed coupling information required for reconstruction and post-processing.

---

## Supported CMS Reduction Methods

The implementation supports several CMS reduction techniques.

### Primal Formulation

Fixed-interface methods

- Craig-Bampton (`CBM`)
- Condensed Craig-Bampton (`CCBM`)

Free-interface methods

- Rubin Method (`RM`)
- MacNeal Method (`MNM`)

### Dual Formulation

Fixed-interface methods

- Fixed Dual Craig-Bampton (`FDCBM`)
- Condensed Fixed Dual Craig-Bampton (`CFDCBM`)

Free-interface methods

- Dual Craig-Bampton (`DCBM`)
- Condensed Dual Craig-Bampton (`CDCBM`)



## Solve Vibro-Acoustic Transfer Path Analysis
def TPA_CMS_vibroacoustic(component_matrices, coupling_info, level_components,
    level_interfaces, force_ext, analysis_settings, type_TPA='1L-TPA',
    CMS_method=['', 40],  complex_analysis=True, modal_CB_va=False,
    transient=False,  save_job_info=(False, '', ''), load_matrices_info=(False, '', ''))


This function solves the vibro-acoustic system and extracts path contributions using the matrices generated previously.

The solver performs the following operations:

1. Loads or assembles the reduced matrices
2. Solves the global vibro-acoustic system
3. Computes interface responses
4. Extracts contributions for each transmission path

The function returns the system response and path contributions.



# Analysis Types

The solver supports different analysis modes.

### Harmonic Analysis

Computes frequency response for a set of frequencies.

Example:
analysis_settings = ('HARMONIC', freqs, 'mkl_pardiso')


### Transient Analysis

Time-domain simulation using numerical integration.

### Modal Analysis

Eigenvalue computation of the assembled system.



# Visualization

Results can be visualized using built-in plotting utilities.

## Path contribution plots
def plot_TPA_contribution_results_3(
    X, Coup_info, level, components, path_name, var, freq_set, section=[None, []], nodes_sel=[], descriptions={},
    ylabel='Contribution', cmap_name="jet", plot_title='Overall Path Contributions', xlabel='Frequency [Hz]', ylim=None,
    in_n_octave_bands=[False, 1], fontsize=20, figsize=(16, 9), legend=True, plot_type='contr_strips', savefig=''
)


Generates plots showing:

- magnitude of each path
- path contributions
- frequency response plots

Plots can be generated as:

- strip plots
- line plots
- bar plots


## 3D Field Visualization
def plot_TPA_results_pyvista(
    X, Coup_info, level, components, path_name, var, freq_set, def_factor=0.0,
    show_real_imag_values='real', plot_size=(1600, 900), descriptions={},
    section=[None, [0, 1]], show_min_max=True, share_clim=True,
    orientation=(1, 1, 1), roll_angles=(0, 0, 0), plot_contributions=True,
    show_edges=True, make_zeros_transparent=False, vector_scale=True,
    parallel_projection=False, result_on_node=False, background_color_rgb=(1.0, 1.0, 1.0),
    full_screen=False, animation=[False, ''], domain='frequency'
)


Uses **PyVista** to visualize vibro-acoustic fields on the FE mesh.

Capabilities include:

- pressure fields
- displacement fields
- SPL maps
- interface contributions
- section cuts

---

# Example: Vibro-Acoustic Model

The example `examples/vibro-acoustic_model` demonstrates the analysis of a system composed of:

- four structural panels
- two acoustic cavities



## Step 1 — Define Subdomains

Each structural or acoustic domain must be defined with its location and properties.

Example:
subdomains = {
'S1': ('path/S1', dof_s, None, '', density_s, load_data),
'S2': ('path/S2', dof_s, None, '', density_s, load_data),
'F1': ('path/F1', dof_f, None, '', density_f, load_data)
}



## Step 2 — Define Interfaces

Interfaces describe connectivity between subdomains.

Example:
interface_pairs = {
1: [('INT_1_F1_TO_S1','INT_1_S1_TO_F1'),
('F1','S1'),
(dom_i['F1'], dom_i['S1']),
([],[]),
('NODE-NODE',0.01),
True,
load_data]
}

## Step 3 — Build the DS Model
component_matrices, coupling_info =
CMS.build_ds_models_vibroacoustic(subdomains, interface_pairs)


This step loads FE matrices and constructs coupling operators.



## Step 4 — Assemble TPA Matrices
Mat_red, Mat_pres, Coup_info, time_RAM =
CMS.TPA_matrices_vibroacoustic(
component_matrices,
coupling_info,
level_components,
level_interfaces,
CMS_method=[red_method,n_modes]
)



## Step 5 — Define Load Case

Example:

load_case = {
'S1': ['NODAL_FORCE','FORCE_EXT_STR_S1',1,['X'],True],
'S2': ['NONE']
}

Create load vector:
force_ext = CMS.create_load(subdomains, load_case)



## Step 6 — Solve the Vibro-Acoustic System
X, Coup_info, TPA_CPU_RAM =
CMS.TPA_CMS_vibroacoustic(
component_matrices,
coupling_info,
level_components,
level_interfaces,
force_ext,
analysis_settings
)



## Step 7 — Plot Path Contributions
CMS.plot_TPA_contribution_results_3(...)

or

CMS.plot_TPA_results_pyvista(...)


# Output

The analysis produces:

- nodal displacements
- acoustic pressure fields
- sound pressure level (SPL)
- transfer path contributions
- interface forces (for dual formulations)

Results can be saved as compressed **joblib** files for later reuse.


# Applications

This framework can be applied to:

- automotive NVH analysis
- aircraft cabin noise
- vibro-acoustic panel systems
- noise transmission studies
- structural-acoustic coupling analysis

---

# Acknowledgements

This framework was developed for research on vibro-acoustic modeling and Transfer Path Analysis using Finite Element methods.
