"""
Graph-based optical simulation for complex optical systems.

This module implements a two-pass algorithm:
1. Find all paths from light sources to terminal nodes (90° mirrors and cameras)
2. Find all paths from 90° mirrors to cameras
3. Simulate each path as a linear chromatix system
4. Combine fields that end up at the same camera
"""

import json
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
import carb
import datetime

from .utils.path_utils import get_simulation_output_dir

# Import plotting if available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    print("[graph_simulation] Warning: Matplotlib not available")

# Import chromatix components when available
try:
    import jax
    import jax.numpy as jnp
    from chromatix import OpticalSystem
    from chromatix.elements import (
        GaussianPlaneWave,
        Propagate,
        ThinLens,
        BasicSensor
    )
    CHROMATIX_AVAILABLE = True
except ImportError:
    CHROMATIX_AVAILABLE = False
    print("[graph_simulation] Warning: Chromatix not available")


class GraphOpticalSimulator:
    """Handles graph-based optical simulation using two-pass algorithm."""

    def __init__(self):
        self.graph_data = None
        self.adjacency = {}
        self.node_lookup = {}
        self.terminal_nodes = []  # 90° mirrors and cameras
        self.source_nodes = []    # Laser sources
        self.camera_nodes = []    # Cameras only

    def simulate_graph_system(self, graph_data, wavelength_nm=550.0, beam_waist_mm=1.0):
        """
        Main simulation function that implements the two-pass algorithm.

        Args:
            graph_data: JSON graph data with nodes and edges
            wavelength_nm: Laser wavelength in nanometers
            beam_waist_mm: Laser beam waist in millimeters

        Returns:
            Dict with simulation results for each terminal node and camera
        """
        print(f"[graph_simulation] Starting graph simulation - λ={wavelength_nm}nm, waist={beam_waist_mm}mm")

        if not CHROMATIX_AVAILABLE:
            carb.log_warn("[graph_simulation] Chromatix not available - using placeholder graph simulation.")
            return self._simulate_placeholder_results(graph_data, wavelength_nm, beam_waist_mm)

        # Initialize graph structure
        self._initialize_graph(graph_data)

        # Convert units
        wavelength_m = wavelength_nm * 1e-9
        beam_waist_m = beam_waist_mm * 1e-3

        results = {
            "wavelength_nm": wavelength_nm,
            "beam_waist_mm": beam_waist_mm,
            "terminal_fields": {},  # Light at each terminal node
            "camera_fields": {},    # Combined light at each camera
            "paths": {
                "source_to_terminal": [],
                "mirror_to_camera": []
            },
            "success": True
        }

        try:
            # Pass 1: Find all paths from sources to terminal nodes
            print("[graph_simulation] Pass 1: Finding paths from sources to terminals...")
            source_to_terminal_paths = self._find_paths_from_sources_to_terminals()
            results["paths"]["source_to_terminal"] = source_to_terminal_paths

            # Simulate each source-to-terminal path
            for path_info in source_to_terminal_paths:
                field = self._simulate_linear_path(path_info, wavelength_m, beam_waist_m)
                terminal_id = path_info["terminal_node"]

                if terminal_id not in results["terminal_fields"]:
                    results["terminal_fields"][terminal_id] = []
                results["terminal_fields"][terminal_id].append({
                    "field": field,
                    "path": path_info["path"],
                    "source": path_info["source_node"]
                })

            # Pass 2: Find paths from 90° mirrors to cameras
            print("[graph_simulation] Pass 2: Finding paths from mirrors to cameras...")
            mirror_to_camera_paths = self._find_paths_from_mirrors_to_cameras()
            results["paths"]["mirror_to_camera"] = mirror_to_camera_paths

            # Simulate mirror-to-camera paths using terminal fields as sources
            for path_info in mirror_to_camera_paths:
                mirror_id = path_info["source_node"]
                camera_id = path_info["terminal_node"]

                # Get the field at the mirror from Pass 1
                if mirror_id in results["terminal_fields"]:
                    for mirror_field_info in results["terminal_fields"][mirror_id]:
                        # Propagate this field to the camera
                        camera_field = self._propagate_field_along_path(
                            mirror_field_info["field"], path_info, wavelength_m
                        )

                        if camera_id not in results["camera_fields"]:
                            results["camera_fields"][camera_id] = []
                        results["camera_fields"][camera_id].append({
                            "field": camera_field,
                            "path": path_info["path"],
                            "source_mirror": mirror_id,
                            "original_source": mirror_field_info["source"]
                        })

            # Combine fields at each camera
            for camera_id in results["camera_fields"]:
                combined_field = self._combine_fields_at_camera(results["camera_fields"][camera_id])
                results["camera_fields"][camera_id].append({
                    "field": combined_field,
                    "path": ["combined"],
                    "source_mirror": "combined",
                    "original_source": "combined"
                })

            print(f"[graph_simulation] Simulation complete! Found {len(results['terminal_fields'])} terminal nodes, {len(results['camera_fields'])} cameras")
            return results

        except Exception as e:
            print(f"[graph_simulation] Error in graph simulation: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "success": False}

    def _initialize_graph(self, graph_data):
        """Initialize graph data structures."""
        self.graph_data = graph_data
        self.node_lookup = {node['id']: node for node in graph_data['nodes']}
        self.adjacency = self._build_adjacency_list(graph_data)

        # Categorize nodes
        self.source_nodes = [node for node in graph_data['nodes'] if node['type'] == 'Laser']
        self.camera_nodes = [node for node in graph_data['nodes'] if node['type'] == 'Camera']

        # Terminal nodes are 90° mirrors and cameras
        self.terminal_nodes = []
        for node in graph_data['nodes']:
            if node['type'] == 'Camera' or node['type'] == 'Mirror90':
                self.terminal_nodes.append(node)

        print(f"[graph_simulation] Graph initialized: {len(self.source_nodes)} sources, {len(self.terminal_nodes)} terminals, {len(self.camera_nodes)} cameras")

    def _build_adjacency_list(self, graph_data):
        """Build adjacency list from graph data."""
        adjacency = {}
        for node in graph_data['nodes']:
            adjacency[node['id']] = []

        for edge in graph_data['edges']:
            source_id = edge['source']
            target_id = edge['target']
            length_cm = edge.get('attributes', {}).get('length_cm', 10.0)

            # Add bidirectional connections
            adjacency[source_id].append({
                'node_id': target_id,
                'length_cm': length_cm,
                'edge_data': edge
            })
            adjacency[target_id].append({
                'node_id': source_id,
                'length_cm': length_cm,
                'edge_data': edge
            })

        return adjacency

    def _simulate_placeholder_results(self, graph_data, wavelength_nm, beam_waist_mm):
        """Generate deterministic synthetic results when chromatix is unavailable."""
        self._initialize_graph(graph_data)
        resolution = 256
        axis = np.linspace(-1, 1, resolution)
        x, y = np.meshgrid(axis, axis)
        waist = max(beam_waist_mm, 0.1)
        base_field = np.exp(-(x**2 + y**2) / (2 * (waist / 2.0)**2))

        results = {
            "wavelength_nm": wavelength_nm,
            "beam_waist_mm": beam_waist_mm,
            "terminal_fields": {},
            "camera_fields": {},
            "paths": {"source_to_terminal": [], "mirror_to_camera": []},
            "success": True,
        }

        for terminal in self.terminal_nodes:
            rng = np.random.default_rng(terminal["id"])
            intensity = np.clip(base_field * (1 + 0.15 * rng.standard_normal(base_field.shape)), 0, None)
            results["terminal_fields"][terminal["id"]] = [{
                "field": {"field": intensity, "intensity": intensity},
                "path": [terminal["id"]],
                "source": terminal["id"],
            }]

        for camera in self.camera_nodes:
            rng = np.random.default_rng(camera["id"] + 100)
            intensity = np.clip(base_field * (1 + 0.1 * rng.standard_normal(base_field.shape)), 0, None)
            results["camera_fields"][camera["id"]] = [{
                "field": {"field": intensity, "intensity": intensity},
                "path": [camera["id"]],
                "source_mirror": camera["id"],
                "original_source": "placeholder",
            }]

        return results

    def _find_paths_from_sources_to_terminals(self):
        """Find all paths from light sources to terminal nodes."""
        paths = []

        for source_node in self.source_nodes:
            for terminal_node in self.terminal_nodes:
                path = self._find_shortest_path(source_node['id'], terminal_node['id'])
                if path:
                    paths.append({
                        "source_node": source_node['id'],
                        "terminal_node": terminal_node['id'],
                        "path": path,
                        "total_distance": self._calculate_path_distance(path)
                    })

        print(f"[graph_simulation] Found {len(paths)} source-to-terminal paths")
        return paths

    def _find_paths_from_mirrors_to_cameras(self):
        """Find all paths from 90° mirrors to cameras."""
        paths = []
        mirror_nodes = [node for node in self.terminal_nodes if node['type'] == 'Mirror90']

        for mirror_node in mirror_nodes:
            for camera_node in self.camera_nodes:
                path = self._find_shortest_path(mirror_node['id'], camera_node['id'])
                if path:
                    paths.append({
                        "source_node": mirror_node['id'],
                        "terminal_node": camera_node['id'],
                        "path": path,
                        "total_distance": self._calculate_path_distance(path)
                    })

        print(f"[graph_simulation] Found {len(paths)} mirror-to-camera paths")
        return paths

    def _find_shortest_path(self, start_id, end_id):
        """Find shortest path between two nodes using BFS."""
        if start_id == end_id:
            return [start_id]

        queue = deque([(start_id, [start_id])])
        visited = set([start_id])

        while queue:
            current_id, path = queue.popleft()

            for neighbor in self.adjacency[current_id]:
                neighbor_id = neighbor['node_id']

                if neighbor_id == end_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None  # No path found

    def _calculate_path_distance(self, path):
        """Calculate total distance along a path."""
        total_distance = 0.0

        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]

            # Find the edge between these nodes
            for neighbor in self.adjacency[current_id]:
                if neighbor['node_id'] == next_id:
                    total_distance += neighbor['length_cm']
                    break

        return total_distance

    def _simulate_linear_path(self, path_info, wavelength_m, beam_waist_m):
        """Simulate a linear path using chromatix."""
        path = path_info["path"]
        total_distance_cm = path_info["total_distance"]

        print(f"[graph_simulation] Simulating path {' -> '.join(map(str, path))}, distance: {total_distance_cm:.1f}cm")

        # Create a simple linear system for this path
        # For now, treat it as free-space propagation with the total distance
        try:
            # Field parameters
            field_resolution = 256  # Smaller for faster computation
            spacing = 1e-6  # 1 micron spacing
            shape = (field_resolution, field_resolution)

            # Create Gaussian source
            source = GaussianPlaneWave(
                shape=shape,
                dx=spacing,
                spectrum=wavelength_m,
                spectral_density=1.0,
                waist=beam_waist_m,
                power=1.0
            )

            # Create simple propagation system
            system = OpticalSystem([
                source,
                Propagate(z=total_distance_cm * 1e-2, n=1.0),  # Convert cm to m
                BasicSensor(shape=shape, spacing=spacing)
            ])

            # Initialize and run
            key = jax.random.PRNGKey(42)
            params = system.init(key)
            output_field = system.apply(params)

            return {
                "field": output_field,
                "intensity": np.array(jnp.abs(output_field) ** 2),
                "wavelength_m": wavelength_m,
                "path_distance_cm": total_distance_cm,
                "resolution": field_resolution
            }

        except Exception as e:
            print(f"[graph_simulation] Error simulating path: {e}")
            return None

    def _propagate_field_along_path(self, input_field_data, path_info, wavelength_m):
        """Propagate an existing field along a path."""
        # For now, implement as simple propagation
        # In the future, this could use the input field as a source
        return self._simulate_linear_path(path_info, wavelength_m, 1e-3)  # Default beam waist

    def _combine_fields_at_camera(self, camera_field_list):
        """Combine multiple fields that arrive at the same camera."""
        if not camera_field_list:
            return None

        if len(camera_field_list) == 1:
            return camera_field_list[0]

        print(f"[graph_simulation] Combining {len(camera_field_list)} fields at camera")

        # Simple combination: add field amplitudes (coherent addition)
        try:
            combined_field = None
            combined_intensity = None

            for field_info in camera_field_list:
                if field_info["field"] is None:
                    continue

                field_data = field_info["field"]
                if "field" in field_data:
                    field = field_data["field"]
                    intensity = field_data["intensity"]

                    if combined_field is None:
                        combined_field = field
                        combined_intensity = intensity
                    else:
                        # Coherent addition of fields
                        combined_field = combined_field + field
                        combined_intensity = np.array(jnp.abs(combined_field) ** 2)

            return {
                "field": combined_field,
                "intensity": combined_intensity,
                "num_combined": len(camera_field_list)
            }

        except Exception as e:
            print(f"[graph_simulation] Error combining fields: {e}")
            return camera_field_list[0]  # Fallback to first field


def plot_and_save_graph_results(results, output_dir="simulation_results"):
    """
    Plot and save graph simulation results as images.

    Args:
        results: Results dictionary from simulate_graph_system
        output_dir: Directory to save images

    Returns:
        List of saved image paths
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[graph_simulation] Cannot plot - matplotlib not available")
        return []

    if not results.get("success", False):
        print("[graph_simulation] Cannot plot - simulation failed")
        return []

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = get_simulation_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wavelength_nm = results.get("wavelength_nm", 550)
    beam_waist_mm = results.get("beam_waist_mm", 1.0)

    try:
        # Plot terminal node fields
        terminal_fields = results.get("terminal_fields", {})
        if terminal_fields:
            fig, axes = plt.subplots(2, len(terminal_fields), figsize=(4*len(terminal_fields), 8))
            if len(terminal_fields) == 1:
                axes = axes.reshape(2, 1)

            fig.suptitle(f'Graph Simulation - Terminal Node Fields\nλ={wavelength_nm:.0f}nm, waist={beam_waist_mm:.1f}mm')

            for i, (terminal_id, field_list) in enumerate(terminal_fields.items()):
                if not field_list or field_list[0]["field"] is None:
                    continue

                # Get the first (or combined) field for this terminal
                field_data = field_list[0]["field"]
                intensity = field_data["intensity"]

                # Ensure intensity is 2D
                if intensity.ndim > 2:
                    intensity = np.squeeze(intensity)
                    if intensity.ndim > 2:
                        intensity = intensity.sum(axis=0)

                # Plot intensity
                im1 = axes[0, i].imshow(intensity, cmap='hot', origin='lower')
                axes[0, i].set_title(f'Terminal {terminal_id}\nIntensity')
                axes[0, i].set_xlabel('X (pixels)')
                axes[0, i].set_ylabel('Y (pixels)')
                plt.colorbar(im1, ax=axes[0, i])

                # Plot cross-section
                center = intensity.shape[0] // 2
                cross_section = intensity[center, :]
                axes[1, i].plot(cross_section, 'b-', linewidth=2)
                axes[1, i].set_title(f'Cross-section (Y=center)')
                axes[1, i].set_xlabel('X (pixels)')
                axes[1, i].set_ylabel('Intensity')
                axes[1, i].grid(True)

            plt.tight_layout()
            terminal_filename = output_path / f"graph_terminals_{timestamp}.png"
            plt.savefig(terminal_filename, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(str(terminal_filename))
            print(f"[graph_simulation] Saved terminal fields plot: {terminal_filename}")

        # Plot camera fields
        camera_fields = results.get("camera_fields", {})
        if camera_fields:
            for camera_id, field_list in camera_fields.items():
                if not field_list:
                    continue

                # Create figure for this camera
                num_fields = len(field_list)
                fig, axes = plt.subplots(2, num_fields, figsize=(4*num_fields, 8))
                if num_fields == 1:
                    axes = axes.reshape(2, 1)

                fig.suptitle(f'Graph Simulation - Camera {camera_id}\nλ={wavelength_nm:.0f}nm, waist={beam_waist_mm:.1f}mm')

                for i, field_info in enumerate(field_list):
                    if field_info["field"] is None:
                        continue

                    field_data = field_info["field"]
                    if "intensity" not in field_data:
                        continue

                    intensity = field_data["intensity"]

                    # Ensure intensity is 2D
                    if intensity.ndim > 2:
                        intensity = np.squeeze(intensity)
                        if intensity.ndim > 2:
                            intensity = intensity.sum(axis=0)

                    # Label for this field
                    if field_info.get("original_source") == "combined":
                        field_label = "Combined"
                    else:
                        field_label = f"From {field_info.get('original_source', 'unknown')}"

                    # Plot intensity
                    im1 = axes[0, i].imshow(intensity, cmap='hot', origin='lower')
                    axes[0, i].set_title(f'{field_label}\nIntensity')
                    axes[0, i].set_xlabel('X (pixels)')
                    axes[0, i].set_ylabel('Y (pixels)')
                    plt.colorbar(im1, ax=axes[0, i])

                    # Plot cross-section
                    center = intensity.shape[0] // 2
                    cross_section = intensity[center, :]
                    axes[1, i].plot(cross_section, 'r-', linewidth=2)
                    axes[1, i].set_title(f'Cross-section (Y=center)')
                    axes[1, i].set_xlabel('X (pixels)')
                    axes[1, i].set_ylabel('Intensity')
                    axes[1, i].grid(True)

                plt.tight_layout()
                camera_filename = output_path / f"graph_camera_{camera_id}_{timestamp}.png"
                plt.savefig(camera_filename, dpi=150, bbox_inches='tight')
                plt.close()
                saved_files.append(str(camera_filename))
                print(f"[graph_simulation] Saved camera {camera_id} plot: {camera_filename}")

        # Create summary plot with path information
        if results.get("paths"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Graph Simulation Summary\nλ={wavelength_nm:.0f}nm, waist={beam_waist_mm:.1f}mm')

            # Plot paths summary
            source_to_terminal = results["paths"]["source_to_terminal"]
            mirror_to_camera = results["paths"]["mirror_to_camera"]

            ax1.text(0.1, 0.9, "Source → Terminal Paths:", transform=ax1.transAxes, fontweight='bold')
            for i, path_info in enumerate(source_to_terminal[:5]):  # Show first 5
                path_str = " → ".join(map(str, path_info["path"]))
                dist_str = f"{path_info['total_distance']:.1f}cm"
                ax1.text(0.1, 0.8-i*0.1, f"{path_str} ({dist_str})", transform=ax1.transAxes, fontsize=9)

            ax2.text(0.1, 0.9, "Mirror → Camera Paths:", transform=ax2.transAxes, fontweight='bold')
            for i, path_info in enumerate(mirror_to_camera[:5]):  # Show first 5
                path_str = " → ".join(map(str, path_info["path"]))
                dist_str = f"{path_info['total_distance']:.1f}cm"
                ax2.text(0.1, 0.8-i*0.1, f"{path_str} ({dist_str})", transform=ax2.transAxes, fontsize=9)

            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')

            summary_filename = output_path / f"graph_summary_{timestamp}.png"
            plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(str(summary_filename))
            print(f"[graph_simulation] Saved summary plot: {summary_filename}")

        print(f"[graph_simulation] Saved {len(saved_files)} result images")
        return saved_files

    except Exception as e:
        print(f"[graph_simulation] Error plotting results: {e}")
        import traceback
        traceback.print_exc()
        return saved_files


def simulate_graph_optical_system(graph_data, wavelength_nm=550.0, beam_waist_mm=1.0):
    """
    Main entry point for graph-based optical simulation.

    Args:
        graph_data: JSON graph data
        wavelength_nm: Wavelength in nanometers
        beam_waist_mm: Beam waist in millimeters

    Returns:
        Simulation results dictionary
    """
    simulator = GraphOpticalSimulator()
    return simulator.simulate_graph_system(graph_data, wavelength_nm, beam_waist_mm)