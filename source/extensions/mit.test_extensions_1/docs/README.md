## Digital Twin SLM Control Extension – `mit.test_extensions_1`

This document describes the purpose, architecture, and workflows of the `mit.test_extensions_1` extension within the `omniverse-optics` project. The goal of this extension is to provide a **digital twin** of an optical system (including SLM / PIC style control) that can be driven by the same control algorithms used on the **physical experiment**, inspired by the framework in *“Hardware Co‑Designed Optimal Control for Programmable Atomic Quantum Processors via Reinforcement Learning”* [`arxiv:2504.11737`](https://arxiv.org/pdf/2504.11737).

The extension ties together:

- **USD / CAD models** of the optical bench and components
- **Graph-based optical layouts** defined in JSON
- **Chromatix‑based optical simulations** (JAX‑accelerated)
- A UI in Omniverse Kit for **constructing, simulating, and eventually controlling** the system

This README focuses on this extension; for general Omniverse / Kit information, see the repository‑level `README.md`.

---

## Example Setup Screenshot

![SLM system screenshot](images/Screenshot 2025-12-12 112754.png)

---

## High‑Level Goals

- **Digital twin of the lab optical system**  
  Represent lenses, mirrors, beam splitters, cameras, SLMs, and PIC‑like devices as USD prims linked to a simulation backend.

- **Hardware‑aware optical simulation**  
  Use [`chromatix`](../..../../../../physics/chromatix/src/chromatix/README.md) to simulate beam propagation through 4f systems, Michelson interferometers, and arbitrary graphs of components.

- **Bridge to quantum optimal control / RL**  
  Provide a simulation environment where control voltages \(V(t)\) (for PIC / SLM channels) can be optimized using classical, RL, or differentiable RL algorithms as in [`arxiv:2504.11737`](https://arxiv.org/pdf/2504.11737), and later applied **unchanged** to the real experiment.

- **Tight integration with Omniverse UI and CAD**  
  Use existing CAD USDs (e.g. objective, SLM, PIC, posts/holders) and scene UI to visualize and manipulate the optical layout.

---

## Key Concepts and Components

- **Extension ID**: `mit.test_extensions_1`
- **Primary entrypoint**: `mit.test_extensions_1.extension.MyExtension`
- **Physics backend**: `mit.test_extensions_1.physics` (Chromatix + JAX integration)
- **Optical simulation backends**:
  - `mit.test_extensions_1.physics.optical_simulation`
  - `mit.test_extensions_1.graph_simulation`
- **Graph / CAD bridge**: `mit.test_extensions_1.graph_editor.OpticalLayoutGenerator`
- **Graph data**: JSON files in `source/extensions/mit.test_extensions_1/data/graphs/*.json`

---

## Architecture Overview

### 1. Omniverse UI and Extension Lifecycle

- **File**: `mit/test_extensions_1/extension.py`
- **Class**: `MyExtension(omni.ext.IExt)`

On startup, `MyExtension`:

- Creates a **“Lens Controller”** window with sections for:
  - Scene setup (optical table)
  - 4f optical system
  - Michelson interferometer
  - Physics / simulation configuration
  - Laser control and position control
  - Graph editor and graph‑based simulation
- Imports:
  - `get_physics_status` from `.physics` to detect JAX / Chromatix availability
  - Simulation functions from `.physics.optical_simulation`:
    - `simulate_4f_system_from_positions`
    - `simulate_michelson_interferometer_from_positions`
    - `plot_and_save_results`
- Binds **UI buttons** to high‑level actions:
  - Create 4f system in the scene
  - Create Michelson interferometer in the scene
  - Turn lasers on/off
  - Run simulations and display the resulting intensity images

This layer is responsible for:

- Reading component positions from the USD scene (`get_component_positions`, `get_michelson_component_positions`, etc.)
- Converting them into a compact description for the simulation backend
- Displaying results (saved PNGs) inside the Omniverse UI

### 2. Physics Backend: Chromatix + JAX Integration

- **File**: `mit/test_extensions_1/physics/__init__.py`

Responsibilities:

- Locate the **Chromatix source tree** under `source/physics/chromatix/src` (supporting both the original `kit-app-template` layout and the customized `omniverse-optics` layout).
- Insert that path into `sys.path`.
- Attempt to import:
  - `jax`, `jax.numpy`
  - `chromatix`
- Expose a simple API:

```python
def get_physics_status():
    return {"jax_available": bool, "chromatix_available": bool}
```

This is used by the UI to report whether hardware‑accelerated simulations are available (JAX devices, Chromatix import).

> **Note**: The physics module is intentionally verbose in logging to make it easy to debug path and environment issues.

### 3. Deterministic Optical Simulations

- **File**: `mit/test_extensions_1/physics/optical_simulation.py`

This module implements **direct, deterministic classical optics simulations** using Chromatix:

- **4f Optical System**

  ```python
  def simulate_4f_system_from_positions(
      component_positions,
      wavelength=0.55e-6,
      beam_waist_mm=1.0,
      focal_length=30,
      field_resolution=1000,
  ) -> dict:
      ...
  ```

  - Input: positions of `laser`, `lens1`, `lens2`, `camera` (in cm), plus physical parameters.
  - Builds a Chromatix `OpticalSystem`:
    - `GaussianPlaneWave` (source)
    - `Propagate` → `ThinLens` → `Propagate` → `ThinLens` → `Propagate` → `BasicSensor`
  - Uses configurable `field_resolution` and pixel spacing.
  - Returns a dict with:
    - Input and output fields and intensities
    - Wavelength, focal length, spacing, field shape, system length
    - `success` flag or `error` string

- **Michelson Interferometer**

  ```python
  def simulate_michelson_interferometer_from_positions(
      component_positions,
      mirror_rotations=None,
      wavelength=0.6328e-6,
      beam_waist_mm=0.05,
      field_resolution=1000,
  ) -> dict:
      ...
  ```

  - Input: positions of `laser`, `beamsplitter`, `mirror1`, `mirror2`, `camera` (in mm),
    plus optional per‑mirror rotations.
  - Uses Chromatix functional interface:
    - `gaussian_plane_wave`, `asm_propagate`
  - Explicitly models two interferometer arms and tilt phases from mirror misalignment.
  - Returns fields/intensities and metadata analogous to the 4f case.

- **Plotting & Saving**

  - `plot_and_save_results(result_dict)` (imported by `extension.py`) produces PNGs (for inspection and UI display) from the simulated intensity fields.

These functions form the **classical digital twin** of the table: they propagate scalar fields through lenses and interferometer arms with adjustable wavelength, beam waist, and geometry.

### 4. Graph‑Based Optical Simulation

- **File**: `mit/test_extensions_1/graph_simulation.py`

This module provides a general way to simulate **arbitrary optical layouts** defined as a graph:

- **Input**: graph JSON (see `GRAPH_EDITOR.md` and `data/graphs/*.json`) with:
  - `nodes`: optical components (Laser, Lens, Mirror45, Mirror90, BeamSplitter, Camera, etc.)
  - `edges`: connections between components, including physical lengths.

- **Core class**: `GraphOpticalSimulator`

  High‑level algorithm:

  1. Build an adjacency graph from the JSON.
  2. Identify:
     - **Source nodes**: lasers
     - **Terminal nodes**: cameras and 90° mirrors
  3. **Pass 1** – Source → Terminal:
     - Find paths from each laser to each terminal (shortest path).
     - For each path, build and simulate a Chromatix `OpticalSystem`.
     - Store fields at each terminal node.
  4. **Pass 2** – Mirror(90°) → Camera:
     - Use the terminal fields at 90° mirrors as sources.
     - Propagate to cameras along mirror‑to‑camera paths.
  5. **Combination at cameras**:
     - Coherently sum fields from all contributing paths for each camera.

  Output structure:

  - `terminal_fields`: list of fields and paths for each terminal node.
  - `camera_fields`: list of individual path fields plus a combined field per camera.
  - `paths`: detailed path metadata for debugging/visualization.

This gives you a **general optical network simulator** that can handle branching, multiple beams, and recombination, provided the graph only contains components supported by the Chromatix‑based propagation model.

### 5. Graph Editor and CAD Integration

- **File**: `mit/test_extensions_1/graph_editor.py`

Key responsibilities:

- Launch an **external PySide6 graph editor** (from Chromatix docs) in a separate process:
  - Finds a Python executable with PySide6.
  - Locates the `graphing.py` script in `source/physics/chromatix/docs/experiments`.
  - Starts the process and logs the PID.

- Convert graph JSON into **USD scenes** with CAD prefabs:

  ```python
  class OpticalLayoutGenerator:
      prefab_mapping = {
          "Laser": "LASER_PREFAB.usd",
          "Lens": "LENS_PREFAB.usd",
          "Mirror45": "MIRROR_45_PREFAB.usd",
          "Mirror90": "MIRROR_90_PREFAB.usd",
          "BeamSplitter": "BEAM_SPLITTER_PREFAB.usd",
          "Camera": "CAMERA_PREFAB.usd",
      }
      ...
  ```

  - Locates the repo root (`kit-app-template` / `omniverse-optics`).
  - Uses CAD USDs in `CAD/prefabs` (e.g. your SLM, PIC, objective, posts, etc.).
  - Places and orients instances in the scene based on graph connectivity and edge lengths.

The workflow is:

1. Design an optical layout in the **graph editor UI**.
2. Save to JSON under `data/graphs/`.
3. Use `OpticalLayoutGenerator` to instantiate the corresponding USD scene using CAD models.
4. Optionally, run **graph‑based simulation** (`GraphOpticalSimulator`) on the same JSON to obtain fields and intensities.

---

## Relation to the Quantum Control Paper

The paper [`Hardware Co‑Designed Optimal Control for Programmable Atomic Quantum Processors via Reinforcement Learning`](https://arxiv.org/pdf/2504.11737) introduces:

- A **photonic control stack**: PIC + SLM modeled as unitary transformations with crosstalk and beam leakage.
- A **quantum system model** (neutral atom array, Jaynes–Cummings interaction).
- A **control optimization framework** using classical hybrid SADE–Adam, PPO, and end‑to‑end differentiable RL.

This extension already implements:

- A **classical optical twin** (lenses, mirrors, beam splitters, interferometers) with realistic beam propagation.
- A **graph‑based description** of the optical network, mirroring the way the paper describes hardware as a unitary map from inputs to outputs.
- Integration with **JAX** via Chromatix, which is compatible with differentiable simulations.

To fully realize the paper’s vision within this repo, additional modules are (conceptually) layered on top of this extension:

- **PIC / SLM hardware model**
  - Represent a bank of control channels with voltage‑dependent unitary `U_PIC(V(t))` and crosstalk coefficients \(C_{m,n}\).
  - Represent SLM phase maps as a diagonal unitary `U_SLM`.
  - Couple these into the Chromatix optical simulation stack.

- **Quantum system + fidelity**
  - Implement Hamiltonians \(H_0\) and \(H_{\text{control}}(t)\) corresponding to atom–field interactions.
  - Integrate over time to compute \(U(T_g)\) and fidelity \(F(U(T_g))\).

- **RL / optimal control layer**
  - A JAX / MLP policy that outputs control signals \(V(t)\).
  - Backpropagate through the digital twin to minimize the gate error cost.

This document focuses on what is already implemented and wired into the extension; the RL/quantum layers are conceptually aligned and can be added in additional modules without changing the existing UI and optical simulation structure.

---

## Usage Summary

### Launching the Extension

1. Build and launch the Kit app (see root `README.md` for `repo.bat` / `repo.sh` usage).
2. Enable the `mit.test_extensions_1` extension from the Omniverse Extension Manager.
3. Open the **“Lens Controller”** window.

### Working with the 4f System

- Use **“Set Up Optical Table”** to spawn the base scene.
- Use **“Create 4f System”** to place lenses / camera.
- Optionally adjust wavelength, beam waist, and field resolution.
- Click **“Simulate 4f System”**:
  - The extension reads component positions from the USD scene.
  - Calls `simulate_4f_system_from_positions` in `optical_simulation.py`.
  - Saves and displays intensity plots.

### Working with the Michelson Interferometer

- Use **“Create Michelson Interferometer”** to place components.
- Optionally adjust mirror rotations via the scene.
- Click **“Simulate Michelson Interferometer”**:
  - Positions and rotations are read from USD.
  - `simulate_michelson_interferometer_from_positions` is called.
  - Interference fringes are computed and displayed.

### Graph‑Based Workflows

- From the UI, **launch the graph editor** (PySide6).
- Design an optical graph (lasers, lenses, mirrors, cameras).
- Save to a JSON graph file in `data/graphs/`.
- Use `OpticalLayoutGenerator` (via UI hooks) to:
  - Build a USD scene using your CAD prefabs.
  - Optionally simulate using `GraphOpticalSimulator`.

---

## File Map (Extension‑Specific)

- `mit/test_extensions_1/extension.py`  
  Omniverse UI, USD scene interaction, top‑level wiring into simulations.

- `mit/test_extensions_1/physics/__init__.py`  
  Physics backend discovery; JAX / Chromatix integration and status reporting.

- `mit/test_extensions_1/physics/optical_simulation.py`  
  4f and Michelson simulations using Chromatix; plotting utilities.

- `mit/test_extensions_1/graph_simulation.py`  
  Graph‑based optical simulations over JSON graphs; Chromatix backend.

- `mit/test_extensions_1/graph_editor.py`  
  External PySide6 graph editor launcher; graph → CAD / USD scene generator.

- `mit/test_extensions_1/data/graphs/*.json`  
  Example optical graph definitions (4f, Michelson, and test layouts).

---

## Future Directions

- **Explicit PIC / SLM hardware module**
  - Add a dedicated `slm_hardware.py` / `pic_model.py` implementing unitary maps with crosstalk and beam leakage as described in the paper.

- **Quantum system and fidelity module**
  - Add `quantum_system.py` implementing time‑evolution operators and gate fidelity \(F(U(T_g))\).

- **End‑to‑end differentiable RL integration**
  - Wrap the digital twin into a JAX‑based environment that can be trained with the **end‑to‑end RL** approach from the paper.

- **Tighter round‑trip with hardware**
  - Define a small API layer where the same control vector \(V(t)\) sent to the simulator can also be sent to lab hardware controllers (SLM drivers, DACs, PIC voltage controllers).

These additions build on the existing architecture documented here and are designed to keep the Omniverse UI and optical simulation flows intact while enriching the physics and control stack beneath them.

