#!/usr/bin/env python3
"""
Graph Editor Launcher for Omniverse Extension
---------------------------------------------
Launches the full-featured PySide6 graph editor as a separate process.
Also includes USD scene generation from graph JSON files.
"""

import sys
import json
import os
import subprocess
from pathlib import Path
import omni.ui as ui
import omni.kit.commands
import omni.usd
import carb
import math
from pxr import Usd, UsdGeom, Sdf, Gf

# Global variable to track the graph editor process
_graph_editor_process = None

def check_pyside6_availability():
    """Check if PySide6 is available in the current Python environment."""
    try:
        import PySide6
        carb.log_info(f"PySide6 is available, version: {PySide6.__version__}")
        return True
    except ImportError:
        carb.log_warn("PySide6 is not available in the current Python environment")
        return False

def install_pyside6():
    """Install PySide6 using pip in the current environment."""
    try:
        carb.log_info("Attempting to install PySide6...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "PySide6"
        ], capture_output=True, text=True, check=True)
        carb.log_info(f"PySide6 installation successful: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        carb.log_error(f"Failed to install PySide6: {e.stderr}")
        return False
    except Exception as e:
        carb.log_error(f"Error installing PySide6: {e}")
        return False

def find_system_python_with_pyside6():
    """Find a system Python installation that has PySide6."""
    possible_python_paths = [
        "python",
        "python3",
        "py",
        r"C:\Python311\python.exe",
        r"C:\Python310\python.exe",
        r"C:\Python39\python.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "python.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "python.exe",
    ]

    for python_path in possible_python_paths:
        try:
            result = subprocess.run([
                str(python_path), "-c", "import PySide6; print(PySide6.__version__)"
            ], capture_output=True, text=True, check=True, timeout=10)
            carb.log_info(f"Found Python with PySide6 at {python_path}: version {result.stdout.strip()}")
            return str(python_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return None

def launch_pyside6_graph_editor():
    """Launch the PySide6 graph editor as a separate process."""
    try:
        # First try to find system Python with PySide6
        python_executable = find_system_python_with_pyside6()

        if not python_executable:
            carb.log_warn("No system Python with PySide6 found, trying current environment")
            python_executable = sys.executable

            # Check if PySide6 is available in current environment
            if not check_pyside6_availability():
                carb.log_info("Attempting to install PySide6 in current environment...")
                if not install_pyside6():
                    carb.log_error("Failed to install PySide6. Please install it manually: pip install PySide6")
                    return None

        # Find the path to the graphing.py file
        # Navigate from the extension directory to the chromatix experiments directory
        current_file = Path(__file__).resolve()

        # Go up to kit-app-template directory
        kit_app_template = None
        for parent in current_file.parents:
            if parent.name == "kit-app-template":
                kit_app_template = parent
                break

        if not kit_app_template:
            carb.log_error("Could not find kit-app-template directory")
            return None

        # Path to the graphing.py script
        graphing_script = kit_app_template / "source" / "physics" / "chromatix" / "docs" / "experiments" / "graphing.py"

        if not graphing_script.exists():
            carb.log_error(f"Graphing script not found at: {graphing_script}")
            return None

        carb.log_info(f"Launching PySide6 graph editor from: {graphing_script}")
        carb.log_info(f"Using Python executable: {python_executable}")

        # Start the process
        process = subprocess.Popen(
            [python_executable, str(graphing_script)],
            cwd=str(graphing_script.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        carb.log_info(f"Graph editor process started with PID: {process.pid}")
        return process

    except Exception as e:
        carb.log_error(f"Failed to launch PySide6 graph editor: {e}")
        import traceback
        carb.log_error(f"Traceback: {traceback.format_exc()}")
        return None

def wavelength_to_rgb(wavelength_nm):
    """Convert wavelength in nanometers to RGB color tuple.

    Args:
        wavelength_nm: Wavelength in nanometers (400-700nm range for visible light)

    Returns:
        Tuple of (r, g, b) values in range 0.0-1.0
    """
    # Clamp wavelength to visible range
    wavelength = max(400, min(700, wavelength_nm))

    if wavelength >= 380 and wavelength <= 440:
        # Violet to Blue
        red = -(wavelength - 440) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif wavelength >= 440 and wavelength <= 490:
        # Blue to Cyan
        red = 0.0
        green = (wavelength - 440) / (490 - 440)
        blue = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        # Cyan to Green
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510) / (510 - 490)
    elif wavelength >= 510 and wavelength <= 580:
        # Green to Yellow
        red = (wavelength - 510) / (580 - 510)
        green = 1.0
        blue = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        # Yellow to Red
        red = 1.0
        green = -(wavelength - 645) / (645 - 580)
        blue = 0.0
    elif wavelength >= 645 and wavelength <= 700:
        # Red
        red = 1.0
        green = 0.0
        blue = 0.0
    else:
        # Outside visible range
        red = 0.0
        green = 0.0
        blue = 0.0

    # Apply intensity fade at extremes
    if wavelength >= 380 and wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength >= 420 and wavelength <= 645:
        factor = 1.0
    elif wavelength >= 645 and wavelength <= 700:
        factor = 0.3 + 0.7 * (700 - wavelength) / (700 - 645)
    else:
        factor = 0.0

    return (red * factor, green * factor, blue * factor)

class OpticalLayoutGenerator:
    """Generates USD scenes from optical graph JSON files using CAD prefabs."""

    def __init__(self):
        self.prefab_mapping = {
            "Laser": "LASER_PREFAB.usd",
            "Lens": "LENS_PREFAB.usd",
            "Mirror45": "MIRROR_45_PREFAB.usd",
            "Mirror90": "MIRROR_90_PREFAB.usd",
            "BeamSplitter": "BEAM_SPLITTER_PREFAB.usd",
            "Camera": "CAMERA_PREFAB.usd"
        }

        # Find paths
        current_file = Path(__file__).resolve()
        self.kit_app_template = None
        for parent in current_file.parents:
            if parent.name == "kit-app-template":
                self.kit_app_template = parent
                break

        if not self.kit_app_template:
            raise RuntimeError("Could not find kit-app-template directory")

        self.prefabs_dir = self.kit_app_template / "CAD" / "prefabs"

    def load_graph_json(self, json_path):
        """Load and parse graph JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            carb.log_info(f"Loaded graph JSON with {len(data.get('nodes', []))} nodes and {len(data.get('edges', []))} edges")
            return data
        except Exception as e:
            carb.log_error(f"Failed to load JSON: {e}")
            return None

    def find_laser_reference(self, nodes):
        """Find the first laser node to use as origin reference."""
        for node in nodes:
            if node['type'] == 'Laser':
                return node
        return None

    def build_adjacency_graph(self, nodes, edges):
        """Build adjacency graph from nodes and edges."""
        # Create node lookup
        node_lookup = {node['id']: node for node in nodes}

        # Build adjacency list with edge information
        adjacency = {}
        for node in nodes:
            adjacency[node['id']] = []

        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            length_cm = edge.get('attributes', {}).get('length_cm', 10.0)

            # Add bidirectional connections
            adjacency[source_id].append({
                'node_id': target_id,
                'length_cm': length_cm,
                'source_port': edge['source_port'],
                'target_port': edge['target_port']
            })
            adjacency[target_id].append({
                'node_id': source_id,
                'length_cm': length_cm,
                'source_port': edge['target_port'],
                'target_port': edge['source_port']
            })

        return node_lookup, adjacency

    def calculate_positions(self, nodes, edges):
        """Calculate 3D positions using rectilinear layout algorithm."""
        import random  # Import at function level to avoid issues in loop

        # Find laser reference
        laser_ref = self.find_laser_reference(nodes)
        if not laser_ref:
            raise ValueError("No laser found in graph - cannot establish reference point")

        # Build adjacency graph
        node_lookup, adjacency = self.build_adjacency_graph(nodes, edges)

        # Debug: Log adjacency information
        carb.log_info(f"Node lookup has {len(node_lookup)} nodes")
        carb.log_info(f"Adjacency graph: {dict(adjacency)}")
        carb.log_info(f"Laser reference: {laser_ref}")

        # Initialize positions and visited tracking
        positions = {}
        visited = set()

        # Set laser at origin
        positions[laser_ref['id']] = {
            'position': [0.0, 0.0, 0.0],
            'axis': 'z',  # Initial direction along z-axis
            'reference_rotation': [0.0, 0.0, 0.0],  # No reference rotation for laser
            'component_tilt': [0.0, 0.0, 0.0],      # No tilt for laser
            'total_rotation': [0.0, 0.0, 0.0]       # Total rotation (reference + tilt)
        }

        # BFS to calculate positions
        queue = [(laser_ref['id'], [0.0, 0.0, 0.0], 'z')]
        # Don't add laser to visited yet - let BFS process it normally

        carb.log_info(f"Starting BFS with queue: {queue}")
        carb.log_info(f"Initial visited set: {visited}")

        loop_count = 0
        # Simple explicit BFS loop to avoid any hidden issues
        carb.log_info(f"Starting explicit BFS loop")

        while True:
            carb.log_info(f"BFS loop check: queue length = {len(queue)}")
            if len(queue) == 0:
                carb.log_info(f"Queue is empty, exiting BFS")
                break

            loop_count += 1
            carb.log_info(f"BFS Loop iteration {loop_count}")

            # Get next item from queue
            current_id, current_pos, current_axis = queue.pop(0)
            current_node = node_lookup[current_id]
            carb.log_info(f"Processing node {current_id} ({current_node['name']}) at {current_pos}, axis: {current_axis}")

            # Skip if already processed
            if current_id in visited:
                carb.log_info(f"Skipping node {current_id} - already processed")
                continue

            # Mark as visited now that we're processing it
            visited.add(current_id)

            # Debug: Check if node is already in positions
            if current_id in positions:
                carb.log_info(f"Node {current_id} already has position: {positions[current_id]}")
            else:
                carb.log_info(f"Node {current_id} does not have position yet")

            # Process all connected nodes
            carb.log_info(f"Node {current_id} has {len(adjacency[current_id])} connections")
            for i, connection in enumerate(adjacency[current_id]):
                next_id = connection['node_id']
                carb.log_info(f"  Connection {i}: to node {next_id}")
                if next_id in visited:
                    carb.log_info(f"    Skipping {next_id} - already visited")
                    continue

                carb.log_info(f"    Processing connection to {next_id}")

                next_node = node_lookup[next_id]
                length_cm = connection['length_cm']

                # Get the branch axis for this connection based on component type and port
                if current_node['type'] == 'BeamSplitter':
                    branch_axis = self._get_beam_splitter_branch_axis(current_axis, connection)
                    carb.log_info(f"BeamSplitter {current_node['name']} port {connection.get('source_port', 'unknown')} → axis: {branch_axis}")
                else:
                    branch_axis = current_axis

                # Calculate next position based on branch axis and length
                next_pos = current_pos.copy()

                # Move along branch axis by length_cm
                if branch_axis == 'z':
                    next_pos[2] += length_cm
                elif branch_axis == 'x':
                    next_pos[0] += length_cm
                elif branch_axis == '-z':
                    next_pos[2] -= length_cm
                elif branch_axis == '-x':
                    next_pos[0] -= length_cm

                # The next axis starts as the branch axis, then gets modified by component type
                next_axis = branch_axis

                # Calculate reference rotation based on beam path alignment
                reference_rotation = self._calculate_reference_rotation(branch_axis, next_node['type'])

                # Get component tilt (if any) from node attributes
                component_tilt = self._get_component_tilt(next_node)

                # Handle axis changes for mirrors and beam splitters
                if next_node['type'] == 'Mirror45':
                    # 45-degree mirror changes axis by 90 degrees
                    # Since graph doesn't specify orientation, choose randomly but consistently
                    random.seed(next_id)  # Use node ID as seed for consistency

                    if branch_axis == 'z':
                        next_axis = random.choice(['x', '-x'])
                    elif branch_axis == 'x':
                        next_axis = random.choice(['z', '-z'])
                    elif branch_axis == '-z':
                        next_axis = random.choice(['x', '-x'])
                    elif branch_axis == '-x':
                        next_axis = random.choice(['z', '-z'])

                elif next_node['type'] == 'Mirror90':
                    # 90° mirrors - orient to face the incoming beam (perpendicular to beam axis)
                    # They should be oriented similar to lenses but face the incoming direction
                    if branch_axis == 'z':
                        next_axis = '-z'
                    elif branch_axis == 'x':
                        next_axis = '-x'
                    elif branch_axis == '-z':
                        next_axis = 'z'
                    elif branch_axis == '-x':
                        next_axis = 'x'

                elif next_node['type'] == 'BeamSplitter':
                    # Beam splitter: axis is already handled by branching logic
                    # next_axis remains as branch_axis
                    pass

                # Store position and continue traversal
                positions[next_id] = {
                    'position': next_pos,
                    'axis': next_axis,
                    'reference_rotation': reference_rotation,
                    'component_tilt': component_tilt,
                    'total_rotation': self._combine_rotations(reference_rotation, component_tilt)
                }
                # Don't add to visited yet - only add when actually processing the node
                queue.append((next_id, next_pos, next_axis))
                carb.log_info(f"    Added to queue: node {next_id} at {next_pos}, axis: {next_axis}")

                # Handle beam splitters (multiple outputs)
                if next_node['type'] == 'BeamSplitter':
                    # For beam splitters, we need to handle multiple output paths
                    # This is a simplified approach - in reality beam splitters
                    # would need more sophisticated handling
                    pass

            carb.log_info(f"Finished processing node {current_id}, queue now has {len(queue)} items")
            carb.log_info(f"End of BFS loop iteration {loop_count}")

        carb.log_info(f"BFS completed. Final positions: {positions}")
        carb.log_info(f"Final queue state: {queue}")
        carb.log_info(f"Final visited set: {visited}")
        return positions

    def _get_beam_splitter_branch_axis(self, input_axis, connection):
        """Get branch direction for a beam splitter based on input axis and port."""
        source_port = connection.get('source_port', '')

        # Map beam splitter ports to directions based on standard optical setup
        # For coaxial beam splitters (top-bottom connections), maintain the same axis
        # Only perpendicular ports (left-right) change the beam direction

        if source_port == 'top':
            # Top port: continue forward along input axis (transmitted beam)
            return input_axis
        elif source_port == 'bottom':
            # Bottom port: also continue forward along input axis (coaxial with top port)
            # Don't reverse direction - "bottom" is just a geometric label
            return input_axis
        elif source_port == 'left':
            # Left port: go left perpendicular to input axis (reflected beam)
            if input_axis == 'z':
                return '-x'
            elif input_axis == 'x':
                return 'z'
            elif input_axis == '-z':
                return 'x'
            elif input_axis == '-x':
                return '-z'
        elif source_port == 'right':
            # Right port: go right perpendicular to input axis (reflected beam)
            if input_axis == 'z':
                return 'x'
            elif input_axis == 'x':
                return '-z'
            elif input_axis == '-z':
                return '-x'
            elif input_axis == '-x':
                return 'z'

        # Default: continue along input axis
        return input_axis

    def _get_beam_splitter_branches(self, input_axis, num_branches):
        """Get branch directions for a beam splitter based on input axis (legacy method)."""
        import random

        if num_branches == 1:
            # Only one output, continue along input axis
            return [input_axis]
        elif num_branches == 2:
            # Two outputs: one continues, one branches perpendicular
            if input_axis == 'z':
                return ['z', random.choice(['x', '-x'])]
            elif input_axis == 'x':
                return ['x', random.choice(['z', '-z'])]
            elif input_axis == '-z':
                return ['-z', random.choice(['x', '-x'])]
            elif input_axis == '-x':
                return ['-x', random.choice(['z', '-z'])]
        elif num_branches == 3:
            # Three outputs: one continues, two branch perpendicular
            if input_axis == 'z':
                return ['z', 'x', '-x']
            elif input_axis == 'x':
                return ['x', 'z', '-z']
            elif input_axis == '-z':
                return ['-z', 'x', '-x']
            elif input_axis == '-x':
                return ['-x', 'z', '-z']
        elif num_branches == 4:
            # Four outputs: all cardinal directions
            return ['z', 'x', '-z', '-x']

        # Default: all continue along input axis
        return [input_axis] * num_branches

    def _calculate_reference_rotation(self, beam_axis, component_type):
        """Calculate reference rotation based on beam path alignment axis.

        All rotations are designed to keep components upright on the table (Y=0 plane).
        We primarily use Y-axis rotations to change facing direction while maintaining
        proper table-mounted orientation.
        """

        if component_type in ['Laser', 'Camera']:
            # Sources and detectors align with beam direction
            if beam_axis == 'z':
                return [0.0, 0.0, 0.0]      # Default orientation (facing +Z)
            elif beam_axis == 'x':
                return [0.0, 90.0, 0.0]     # Rotate 90° around Y to face +X
            elif beam_axis == '-z':
                return [0.0, 180.0, 0.0]    # Rotate 180° around Y to face -Z
            elif beam_axis == '-x':
                return [0.0, -90.0, 0.0]    # Rotate -90° around Y to face -X

        elif component_type == 'Lens':
            # Lenses are perpendicular to beam direction, but stay upright
            if beam_axis == 'z':
                return [0.0, 0.0, 0.0]      # Default orientation (perpendicular to Z)
            elif beam_axis == 'x':
                return [0.0, 90.0, 0.0]     # Rotate 90° around Y to be perpendicular to X
            elif beam_axis == '-z':
                return [0.0, 0.0, 0.0]      # Same as +Z (symmetric)
            elif beam_axis == '-x':
                return [0.0, -90.0, 0.0]    # Rotate -90° around Y to be perpendicular to -X

        elif component_type == 'Mirror45':
            # 45° mirrors - use Y-axis rotation to maintain table orientation
            if beam_axis == 'z':
                return [0.0, 45.0, 0.0]     # 45° around Y-axis
            elif beam_axis == 'x':
                return [0.0, 135.0, 0.0]    # 135° around Y-axis (90° + 45°)
            elif beam_axis == '-z':
                return [0.0, -45.0, 0.0]    # -45° around Y-axis
            elif beam_axis == '-x':
                return [0.0, -135.0, 0.0]   # -135° around Y-axis (-90° - 45°)

        elif component_type == 'Mirror90':
            # 90° mirrors - orient to face the incoming beam (perpendicular to beam axis)
            # They should be oriented similar to lenses but face the incoming direction
            if beam_axis == 'z':
                return [0.0, 0.0, 0.0]      # Face incoming beam from -Z direction
            elif beam_axis == 'x':
                return [0.0, 90.0, 0.0]     # Face incoming beam from -X direction
            elif beam_axis == '-z':
                return [0.0, 180.0, 0.0]    # Face incoming beam from +Z direction
            elif beam_axis == '-x':
                return [0.0, -90.0, 0.0]    # Face incoming beam from +X direction

        elif component_type == 'BeamSplitter':
            # Beam splitters - orient perpendicular to beam direction like lenses
            # No 45° rotation needed - the CAD model should have the correct geometry
            if beam_axis == 'z':
                return [0.0, 0.0, 0.0]      # Default orientation (perpendicular to Z)
            elif beam_axis == 'x':
                return [0.0, 90.0, 0.0]     # Rotate 90° around Y to be perpendicular to X
            elif beam_axis == '-z':
                return [0.0, 0.0, 0.0]      # Same as +Z (symmetric)
            elif beam_axis == '-x':
                return [0.0, -90.0, 0.0]    # Rotate -90° around Y to be perpendicular to -X

        # Default case
        return [0.0, 0.0, 0.0]

    def _get_component_tilt(self, node):
        """Extract component tilt from node attributes (for interferometry)."""
        attributes = node.get('attributes', {})

        # Look for tilt attributes (in degrees)
        tilt_x = attributes.get('tilt_x', 0.0)
        tilt_y = attributes.get('tilt_y', 0.0)
        tilt_z = attributes.get('tilt_z', 0.0)

        # For mirrors, also check for common tilt parameter names
        if node['type'] in ['Mirror45', 'Mirror90']:
            # Check for generic 'tilt' parameter (assume it's around Y-axis for mirrors)
            generic_tilt = attributes.get('tilt', 0.0)
            if generic_tilt != 0.0 and tilt_y == 0.0:
                tilt_y = generic_tilt

        return [tilt_x, tilt_y, tilt_z]

    def _combine_rotations(self, reference_rotation, component_tilt):
        """Combine reference rotation (from beam path) with component tilt."""
        # Simple addition for small angles (valid for small tilts in interferometry)
        # For larger rotations, would need proper rotation matrix composition
        return [
            reference_rotation[0] + component_tilt[0],  # X-axis
            reference_rotation[1] + component_tilt[1],  # Y-axis
            reference_rotation[2] + component_tilt[2]   # Z-axis
        ]

    def create_usd_scene(self, graph_data, scene_name="OpticalSystem"):
        """Create USD scene from graph data."""
        try:
            # Get current stage
            context = omni.usd.get_context()
            if not context:
                carb.log_error("No USD context available")
                return False

            stage = context.get_stage()
            if not stage:
                carb.log_error("No active USD stage found")
                return False

            carb.log_info(f"USD stage available: {stage.GetRootLayer().identifier}")

            # Check if stage is editable
            if not stage.GetRootLayer().permissionToEdit:
                carb.log_error("USD stage is not editable")
                return False

            # Calculate positions
            positions = self.calculate_positions(graph_data['nodes'], graph_data['edges'])

            # Clear any existing prim with the same name
            root_path = f"/{scene_name}"
            existing_prim = stage.GetPrimAtPath(root_path)
            if existing_prim.IsValid():
                carb.log_info(f"Removing existing prim at {root_path}")
                stage.RemovePrim(root_path)

            # Create root prim for the optical system
            carb.log_info(f"Creating root prim at: {root_path}")
            try:
                root_prim = UsdGeom.Xform.Define(stage, root_path)
                if not root_prim:
                    carb.log_error(f"Failed to create root prim {root_path}")
                    return False
                carb.log_info(f"Successfully created root prim: {root_prim.GetPath()}")
            except Exception as root_error:
                carb.log_error(f"Failed to create root prim {root_path}: {root_error}")
                return False

            # Create each component
            for node in graph_data['nodes']:
                node_id = node['id']
                if node_id not in positions:
                    carb.log_warn(f"No position calculated for node {node_id}")
                    continue

                pos_data = positions[node_id]
                position = pos_data['position']
                reference_rotation = pos_data['reference_rotation']
                component_tilt = pos_data['component_tilt']
                total_rotation = pos_data['total_rotation']

                # Create component prim with sanitized name
                sanitized_name = self._sanitize_prim_name(node['name'], node_id)
                component_path = f"{root_path}/{sanitized_name}"

                # Get prefab path
                prefab_file = self.prefab_mapping.get(node['type'])
                if not prefab_file:
                    carb.log_warn(f"No prefab mapping for type {node['type']}")
                    continue

                prefab_path = self.prefabs_dir / prefab_file
                if not prefab_path.exists():
                    carb.log_warn(f"Prefab not found: {prefab_path}")
                    continue

                # Clear any existing prim with the same path
                existing_component = stage.GetPrimAtPath(component_path)
                if existing_component.IsValid():
                    carb.log_info(f"Removing existing prim at {component_path}")
                    stage.RemovePrim(component_path)

                # Create reference to prefab
                carb.log_info(f"Creating prim at path: {component_path}")
                try:
                    component_prim = UsdGeom.Xform.Define(stage, component_path)
                    if not component_prim:
                        carb.log_error(f"UsdGeom.Xform.Define returned None for {component_path}")
                        continue
                    carb.log_info(f"Successfully created prim: {component_prim.GetPath()}")
                except Exception as prim_error:
                    carb.log_error(f"Failed to create prim {component_path}: {prim_error}")
                    continue

                # Add reference to prefab
                references = component_prim.GetPrim().GetReferences()
                references.AddReference(str(prefab_path))

                # Set transform using individual operations (more robust)
                # Clear any existing transform ops first
                component_prim.ClearXformOpOrder()

                # Add translate operation (Y position always 0 as requested)
                translate_op = component_prim.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(position[0], 0.0, position[2]))

                # Add rotation operations if needed (using total rotation)
                if any(r != 0.0 for r in total_rotation):
                    # Add rotation operations in XYZ order
                    if total_rotation[0] != 0.0:
                        rotate_x_op = component_prim.AddRotateXOp()
                        rotate_x_op.Set(total_rotation[0])
                    if total_rotation[1] != 0.0:
                        rotate_y_op = component_prim.AddRotateYOp()
                        rotate_y_op.Set(total_rotation[1])
                    if total_rotation[2] != 0.0:
                        rotate_z_op = component_prim.AddRotateZOp()
                        rotate_z_op.Set(total_rotation[2])

                # Log detailed rotation information
                axis_info = f" (axis: {pos_data['axis']})" if 'axis' in pos_data else ""
                carb.log_info(f"Created {node['type']} '{node['name']}' at position {position}{axis_info}")
                if any(r != 0.0 for r in reference_rotation):
                    carb.log_info(f"  Reference rotation (beam path): {reference_rotation}")
                if any(r != 0.0 for r in component_tilt):
                    carb.log_info(f"  Component tilt: {component_tilt}")
                if any(r != 0.0 for r in total_rotation):
                    carb.log_info(f"  Total rotation: {total_rotation}")

            carb.log_info(f"Successfully created USD scene '{scene_name}' with {len(graph_data['nodes'])} components")
            return True

        except Exception as e:
            carb.log_error(f"Failed to create USD scene: {e}")
            import traceback
            carb.log_error(f"Traceback: {traceback.format_exc()}")
            return False

    def _sanitize_prim_name(self, name, node_id):
        """Sanitize prim name to be USD-compliant."""
        import re

        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"_{sanitized}"

        # If empty or just underscores, use node type and ID
        if not sanitized or sanitized.replace('_', '') == '':
            sanitized = f"Component_{node_id}"

        # Ensure uniqueness by appending node ID if needed
        if sanitized.count('_') == 0 or not sanitized.endswith(f"_{node_id}"):
            sanitized = f"{sanitized}_{node_id}"

        return sanitized

def load_and_generate_scene_with_picker():
    """Load a graph JSON file using a file picker and generate USD scene."""
    try:
        # First try using Omniverse file importer, then fallback to tkinter
        try:
            # Use Omniverse's file picker to select JSON file
            def on_file_selected(filename, dirname):
                if filename and filename.endswith('.json'):
                    json_path = os.path.join(dirname, filename)
                    carb.log_info(f"Selected JSON file: {json_path}")
                    _load_json_and_generate_scene(json_path)
                else:
                    carb.log_warn("No JSON file selected or invalid file type")

            # Find the default directory (chromatix physics folder)
            current_file = Path(__file__).resolve()
            kit_app_template = None
            for parent in current_file.parents:
                if parent.name == "kit-app-template":
                    kit_app_template = parent
                    break

            default_dir = str(kit_app_template / "source" / "physics" / "chromatix") if kit_app_template else None

            # Try to open file picker dialog
            import omni.kit.window.file_importer
            file_importer = omni.kit.window.file_importer.get_file_importer()
            file_importer.show_window(
                title="Select Graph JSON File",
                file_extension_options=[("*.json", "JSON Files")],
                select_directory=False,
                apply_button_label="Load JSON",
                filename_url=default_dir,
                validation_fn=lambda path: path.endswith('.json')
            )

            # Set callback for when file is selected
            file_importer.add_filename_changed_fn(on_file_selected)

        except Exception as omni_error:
            carb.log_warn(f"Omniverse file picker failed: {omni_error}, trying tkinter fallback")

            # Fallback to tkinter file dialog
            import tkinter as tk
            from tkinter import filedialog

            # Find the default directory
            current_file = Path(__file__).resolve()
            kit_app_template = None
            for parent in current_file.parents:
                if parent.name == "kit-app-template":
                    kit_app_template = parent
                    break

            default_dir = str(kit_app_template / "source" / "physics" / "chromatix") if kit_app_template else str(Path.home())

            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()  # Hide the main window

            # Open file dialog
            json_path = filedialog.askopenfilename(
                title="Select Graph JSON File",
                initialdir=default_dir,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            root.destroy()  # Clean up

            if json_path:
                carb.log_info(f"Selected JSON file: {json_path}")
                _load_json_and_generate_scene(json_path)
            else:
                carb.log_info("No file selected")

    except Exception as e:
        carb.log_error(f"Error in load_and_generate_scene_with_picker: {e}")
        import traceback
        carb.log_error(f"Traceback: {traceback.format_exc()}")

def _load_json_and_generate_scene(json_path):
    """Helper function to load JSON and generate scene."""
    try:
        generator = OpticalLayoutGenerator()
        graph_data = generator.load_graph_json(json_path)

        if graph_data:
            carb.log_info("JSON loaded successfully, generating USD scene...")
            success = generator.create_usd_scene(graph_data)
            if success:
                carb.log_info("USD scene generation completed successfully!")
            else:
                carb.log_error("USD scene generation failed")
        else:
            carb.log_error("Failed to load graph data from JSON")
    except Exception as e:
        carb.log_error(f"Error loading JSON and generating scene: {e}")
        import traceback
        carb.log_error(f"Traceback: {traceback.format_exc()}")

def load_and_generate_scene():
    """Load a graph JSON file and generate USD scene."""
    try:
        # For now, directly load the four-f-plus-michelson.json file
        # This bypasses the file picker issues and gives you immediate functionality
        carb.log_info("Loading fourfplusmichelson.json directly...")

        # Find the JSON file
        current_file = Path(__file__).resolve()
        kit_app_template = None
        for parent in current_file.parents:
            if parent.name == "kit-app-template":
                kit_app_template = parent
                break

        if kit_app_template:
            json_path = kit_app_template / "source" / "physics" / "chromatix" / "fourfplusmichelson.json"
            if json_path.exists():
                carb.log_info(f"Loading JSON file: {json_path}")
                generator = OpticalLayoutGenerator()
                graph_data = generator.load_graph_json(str(json_path))

                if graph_data:
                    carb.log_info("JSON loaded successfully, generating USD scene...")
                    success = generator.create_usd_scene(graph_data)
                    if success:
                        carb.log_info("USD scene generation completed successfully!")
                    else:
                        carb.log_error("USD scene generation failed")
                else:
                    carb.log_error("Failed to load graph data from JSON")
            else:
                carb.log_error(f"JSON file not found at: {json_path}")
        else:
            carb.log_error("Could not find kit-app-template directory")

    except Exception as e:
        carb.log_error(f"Error in load_and_generate_scene: {e}")
        import traceback
        carb.log_error(f"Traceback: {traceback.format_exc()}")

def get_available_json_files():
    """Get list of available JSON files in the chromatix directory."""
    try:
        current_file = Path(__file__).resolve()
        kit_app_template = None
        for parent in current_file.parents:
            if parent.name == "kit-app-template":
                kit_app_template = parent
                break

        if kit_app_template:
            chromatix_dir = kit_app_template / "source" / "physics" / "chromatix"
            if chromatix_dir.exists():
                json_files = list(chromatix_dir.glob("*.json"))
                return [(f.name, str(f)) for f in json_files]

        return []
    except Exception as e:
        carb.log_error(f"Error getting available JSON files: {e}")
        return []

def load_specific_json_file(json_path):
    """Load a specific JSON file and generate USD scene."""
    try:
        carb.log_info(f"Loading specific JSON file: {json_path}")
        _load_json_and_generate_scene(json_path)
    except Exception as e:
        carb.log_error(f"Error loading specific JSON file: {e}")

def create_minimal_json_loader():
    """Create a minimal JSON loader window - absolutely basic."""
    try:
        carb.log_info("Creating minimal JSON loader...")
        print("[DEBUG] Creating minimal JSON loader...")

        # Create the most basic window possible
        window = ui.Window("JSON Loader", width=250, height=200)
        window.position_x = 200
        window.position_y = 200
        window.visible = True

        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label("JSON Loader")
                ui.Button("Load fourfplusmichelson.json",
                         clicked_fn=lambda: load_specific_json_file(
                             "C:/Users/armmt/OneDrive/Desktop/omniverse/kit-app-template/source/physics/chromatix/fourfplusmichelson.json"
                         ))
                ui.Button("Load fout-f-plus-michelson.json",
                         clicked_fn=lambda: load_specific_json_file(
                             "C:/Users/armmt/OneDrive/Desktop/omniverse/kit-app-template/source/physics/chromatix/fout-f-plus-michelson.json"
                         ))
                ui.Button("Close", clicked_fn=lambda: setattr(window, 'visible', False))

        print(f"[DEBUG] Minimal JSON loader created: {window}")
        return window

    except Exception as e:
        print(f"[DEBUG ERROR] Failed to create minimal JSON loader: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_graph_editor_launcher():
    """Create a simplified graph editor launcher to test visibility."""
    try:
        carb.log_info("Creating simple graph editor launcher...")
        print("[DEBUG] Creating simple graph editor launcher...")

        # Create window similar to the working test window
        window = ui.Window(
            "Simple Graph Editor Launcher",
            width=400,
            height=300
        )

        # Set position explicitly
        window.position_x = 150
        window.position_y = 150

        # Force visibility and focus
        window.visible = True

        # Try to focus if the method exists
        try:
            if hasattr(window, 'focus'):
                window.focus()
                print("[DEBUG] Simple launcher focused")
        except:
            print("[DEBUG] Simple launcher focus failed or not available")

        with window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Simple Graph Editor Launcher", height=25)
                ui.Label("This is a simplified version to test visibility")
                ui.Separator()

                ui.Label("Available JSON Files:", height=20)

                # Try to get available files
                try:
                    available_files = get_available_json_files()
                    if available_files:
                        for filename, filepath in available_files:
                            ui.Button(f"Load {filename}",
                                     clicked_fn=lambda path=filepath: load_specific_json_file(path))
                    else:
                        ui.Label("No JSON files found")
                except Exception as files_error:
                    ui.Label(f"Error finding files: {files_error}")

                ui.Separator()

                # Basic buttons
                ui.Button("Load Default JSON → USD", clicked_fn=load_and_generate_scene)

                def close_launcher():
                    window.visible = False
                    print("[DEBUG] Simple launcher closed")

                ui.Button("Close Launcher", clicked_fn=close_launcher)

        carb.log_info("Simple graph editor launcher created successfully")
        print("[DEBUG] Simple graph editor launcher created successfully")
        print(f"[DEBUG] Simple launcher visible: {window.visible}")
        print(f"[DEBUG] Simple launcher object: {window}")

        return window

    except Exception as e:
        carb.log_error(f"Failed to create simple graph editor launcher: {e}")
        print(f"[DEBUG ERROR] Failed to create simple graph editor launcher: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_graph_editor_launcher_window():
    """Create a simple launcher window for the graph editor."""
    try:
        carb.log_info("Creating graph editor launcher window...")
        print("[DEBUG] Starting to create graph editor launcher window")

        # Try to create the window with error handling
        try:
            window = ui.Window(
                "Graph Editor Launcher",
                width=350,
                height=400
            )

            # Set position after creation
            window.position_x = 200
            window.position_y = 200

            carb.log_info(f"Window object created: {window}")
            print(f"[DEBUG] Window object created: {window}")
        except Exception as window_error:
            carb.log_error(f"Failed to create ui.Window: {window_error}")
            print(f"[DEBUG ERROR] Failed to create ui.Window: {window_error}")
            return None

        # Ensure visibility after creation
        try:
            window.visible = True
            # Try to bring window to front if method exists
            try:
                if hasattr(window, 'focus'):
                    window.focus()
                    print("[DEBUG] Graph editor window focused")
            except:
                print("[DEBUG] Graph editor window focus failed or not available")

            carb.log_info("Window visibility set to True")
            print("[DEBUG] Window visibility set to True")
            print(f"[DEBUG] Window focused and visible: {window.visible}")
        except Exception as visibility_error:
            carb.log_error(f"Failed to set window visibility: {visibility_error}")
            print(f"[DEBUG ERROR] Failed to set window visibility: {visibility_error}")

        def launch_editor():
            """Launch the graph editor."""
            global _graph_editor_process
            try:
                carb.log_info("Launch editor button clicked")
                print("[DEBUG] Launch editor button clicked")

                if _graph_editor_process and _graph_editor_process.poll() is None:
                    carb.log_info("Graph editor is already running")
                    status_label.model.set_value("Graph editor already running")
                    return

                status_label.model.set_value("Launching graph editor...")
                process = launch_pyside6_graph_editor()

                if process:
                    _graph_editor_process = process
                    status_label.model.set_value(f"Graph editor launched (PID: {process.pid})")
                else:
                    status_label.model.set_value("Failed to launch graph editor")

            except Exception as e:
                carb.log_error(f"Error in launch_editor: {e}")
                print(f"[DEBUG ERROR] Error in launch_editor: {e}")
                status_label.model.set_value(f"Error: {str(e)}")

        def check_status():
            """Check if the graph editor is still running."""
            global _graph_editor_process
            try:
                if _graph_editor_process:
                    if _graph_editor_process.poll() is None:
                        status_label.model.set_value(f"Graph editor running (PID: {_graph_editor_process.pid})")
                    else:
                        status_label.model.set_value("Graph editor closed")
                        _graph_editor_process = None
                else:
                    status_label.model.set_value("No graph editor running")
            except Exception as e:
                carb.log_error(f"Error in check_status: {e}")
                print(f"[DEBUG ERROR] Error in check_status: {e}")

        def terminate_editor():
            """Terminate the graph editor if running."""
            global _graph_editor_process
            try:
                if _graph_editor_process and _graph_editor_process.poll() is None:
                    _graph_editor_process.terminate()
                    status_label.model.set_value("Graph editor terminated")
                    _graph_editor_process = None
                else:
                    status_label.model.set_value("No graph editor to terminate")
            except Exception as e:
                carb.log_error(f"Error in terminate_editor: {e}")
                print(f"[DEBUG ERROR] Error in terminate_editor: {e}")

        def check_dependencies():
            """Check if all dependencies are available."""
            try:
                # Check system Python first
                system_python = find_system_python_with_pyside6()
                if system_python:
                    status_label.model.set_value(f"✓ Found system Python with PySide6: {system_python}")
                elif check_pyside6_availability():
                    status_label.model.set_value("✓ PySide6 is available in current environment")
                else:
                    status_label.model.set_value("✗ PySide6 not found. Will attempt auto-install on launch.")
            except Exception as e:
                carb.log_error(f"Error in check_dependencies: {e}")
                print(f"[DEBUG ERROR] Error in check_dependencies: {e}")

        def install_dependencies():
            """Install PySide6 in the current environment."""
            try:
                status_label.model.set_value("Installing PySide6...")
                if install_pyside6():
                    status_label.model.set_value("✓ PySide6 installed successfully")
                else:
                    status_label.model.set_value("✗ Failed to install PySide6. Try manually: pip install PySide6")
            except Exception as e:
                carb.log_error(f"Error in install_dependencies: {e}")
                print(f"[DEBUG ERROR] Error in install_dependencies: {e}")

        try:
            carb.log_info("Creating window frame and UI elements...")
            print("[DEBUG] Creating window frame and UI elements...")

            with window.frame:
                with ui.VStack(spacing=10):
                    ui.Label("PySide6 Graph Editor Launcher", height=30)

                    ui.Separator()

                    # Status display
                    status_model = ui.SimpleStringModel("Ready to launch")
                    status_label = ui.Label("Ready to launch", model=status_model)

                    ui.Separator()

                    # Control buttons
                    with ui.HStack(spacing=5):
                        ui.Button("Launch Graph Editor", clicked_fn=launch_editor)
                        ui.Button("Check Status", clicked_fn=check_status)

                    with ui.HStack(spacing=5):
                        ui.Button("Check Dependencies", clicked_fn=check_dependencies)
                        ui.Button("Install PySide6", clicked_fn=install_dependencies)
                        ui.Button("Terminate", clicked_fn=terminate_editor)

                    ui.Separator()

                    # USD Scene Generation
                    ui.Label("USD Scene Generation", height=20)

                    # Show available JSON files
                    available_files = get_available_json_files()
                    if available_files:
                        ui.Label("Quick Load Available JSON Files:", height=15)
                        with ui.VStack(spacing=2):
                            for filename, filepath in available_files:
                                with ui.HStack(spacing=5):
                                    ui.Button(f"Load {filename}",
                                             width=250,
                                             clicked_fn=lambda path=filepath: load_specific_json_file(path))

                        ui.Separator()

                    with ui.HStack(spacing=5):
                        ui.Button("Load Default JSON → USD", clicked_fn=load_and_generate_scene)
                        ui.Button("Choose JSON File → USD", clicked_fn=load_and_generate_scene_with_picker)

                    ui.Separator()

                    # Instructions
                    with ui.VStack(spacing=5):
                        ui.Label("Instructions:", height=20)
                        ui.Label("Graph Editor:")
                        ui.Label("• Click 'Check Dependencies' first")
                        ui.Label("• Install PySide6 if needed (or it will auto-install)")
                        ui.Label("• Click 'Launch Graph Editor' to open the full PySide6 interface")
                        ui.Label("• The graph editor runs as a separate application")
                        ui.Label("• Features: Visual nodes, wiring, save/load, attributes")
                        ui.Spacer(height=5)
                        ui.Label("USD Scene Generation:")
                        ui.Label("• Quick load buttons for available JSON files")
                        ui.Label("• Click 'Load Default JSON → USD' to load the default JSON file")
                        ui.Label("• Click 'Choose JSON File → USD' to select any JSON file")
                        ui.Label("• Automatically generates 3D scene using CAD prefabs")
                        ui.Label("• Uses rectilinear layout with laser at origin")
                        ui.Label("• Handles mirrors, beam splitters, and optical paths")

            carb.log_info("Graph editor launcher window created successfully")
            print("[DEBUG] Graph editor launcher window created successfully")
            return window

        except Exception as ui_error:
            carb.log_error(f"Error creating UI elements: {ui_error}")
            print(f"[DEBUG ERROR] Error creating UI elements: {ui_error}")
            import traceback
            traceback.print_exc()
            return None

    except Exception as e:
        carb.log_error(f"Error creating launcher window: {e}")
        print(f"[DEBUG ERROR] Error creating launcher window: {e}")
        import traceback
        carb.log_error(f"Window creation traceback: {traceback.format_exc()}")
        traceback.print_exc()
        return None

def create_graph_editor_interface(extension_ref=None):
    """Create the main graph editor interface with the requested functionality."""
    try:
        carb.log_info("Creating graph editor interface...")
        print("[DEBUG] Creating graph editor interface...")

        # Create window
        window = ui.Window(
            "Graph Editor",
            width=400,
            height=400  # Reduced height since text fields take less space
        )

        # Set position explicitly
        window.position_x = 150
        window.position_y = 150

        # Force visibility and focus
        window.visible = True

        # Try to focus if the method exists
        try:
            if hasattr(window, 'focus'):
                window.focus()
                print("[DEBUG] Graph editor focused")
        except:
            print("[DEBUG] Graph editor focus failed or not available")

        # Initialize laser parameters and models
        wavelength_model = ui.SimpleFloatModel(550.0)  # Default 550nm (green)
        beam_waist_model = ui.SimpleFloatModel(1.0)    # Default 1.0mm beam waist
        laser_enabled = False

        with window.frame:
            with ui.VStack(spacing=15):
                ui.Label("Graph Editor", height=25, style={"font_size": 18})
                ui.Separator()

                # Graph Management Section
                ui.Label("Graph Management", height=20, style={"font_size": 14})

                def create_graph_placeholder():
                    print("[DEBUG] Create Graph clicked - placeholder functionality")
                    carb.log_info("Create Graph placeholder called")

                def load_graph_from_file():
                    print("[DEBUG] Load Graph from File clicked")
                    carb.log_info("Loading graph from file...")
                    load_and_generate_scene_with_picker()

                def load_demo_graph():
                    print("[DEBUG] Load Demo Graph clicked")
                    carb.log_info("Loading demo graph (4f plus Michelson system)...")
                    # Load the fourfplusmichelson.json file
                    current_file = Path(__file__).resolve()
                    for parent in current_file.parents:
                        if parent.name == "kit-app-template":
                            demo_path = parent / "source" / "physics" / "chromatix" / "fourfplusmichelson.json"
                            if demo_path.exists():
                                load_specific_json_file(str(demo_path))
                                return
                    carb.log_error("Demo graph file not found")

                with ui.VStack(spacing=5):
                    ui.Button("Create Graph", clicked_fn=create_graph_placeholder)
                    ui.Button("Load Graph from File", clicked_fn=load_graph_from_file)
                    ui.Button("Load Demo Graph", clicked_fn=load_demo_graph)

                ui.Separator()

                # Laser Control Section with Text Fields
                ui.Label("Laser Control", height=20, style={"font_size": 14})

                # Wavelength Control
                with ui.HStack(spacing=10):
                    ui.Label("Wavelength (nm):", width=120)
                    wavelength_field = ui.FloatField(
                        model=wavelength_model,
                        width=100,
                        height=20
                    )

                    # Color preview rectangle
                    color_preview = ui.Rectangle(
                        width=40,
                        height=20,
                        style={"background_color": ui.color(*wavelength_to_rgb(wavelength_model.get_value_as_float()))}
                    )

                # Beam Waist Control
                with ui.HStack(spacing=10):
                    ui.Label("Beam Waist (mm):", width=120)
                    beam_waist_field = ui.FloatField(
                        model=beam_waist_model,
                        width=100,
                        height=20
                    )

                # Laser status label
                laser_status_label = ui.Label("Laser: OFF", height=15, style={"color": ui.color.red})

                # Update functions for text fields
                def on_wavelength_changed(model):
                    wavelength = model.get_value_as_float()
                    # Clamp wavelength to visible range
                    if wavelength < 400.0:
                        wavelength = 400.0
                        model.set_value(wavelength)
                    elif wavelength > 700.0:
                        wavelength = 700.0
                        model.set_value(wavelength)

                    # Update color preview
                    new_color = wavelength_to_rgb(wavelength)
                    color_preview.style = {"background_color": ui.color(*new_color)}
                    print(f"[DEBUG] Wavelength changed to {wavelength:.1f}nm")

                def on_beam_waist_changed(model):
                    beam_waist = model.get_value_as_float()
                    # Clamp beam waist to reasonable range
                    if beam_waist < 0.1:
                        beam_waist = 0.1
                        model.set_value(beam_waist)
                    elif beam_waist > 5.0:
                        beam_waist = 5.0
                        model.set_value(beam_waist)

                    print(f"[DEBUG] Beam waist changed to {beam_waist:.2f}mm")

                # Connect text field callbacks
                wavelength_model.add_value_changed_fn(on_wavelength_changed)
                beam_waist_model.add_value_changed_fn(on_beam_waist_changed)

                # Laser control buttons
                def turn_on_laser():
                    nonlocal laser_enabled
                    current_wavelength = wavelength_model.get_value_as_float()
                    current_beam_waist = beam_waist_model.get_value_as_float()
                    print(f"[DEBUG] Turn On Laser clicked - Wavelength: {current_wavelength:.1f}nm, Beam Waist: {current_beam_waist:.2f}mm")
                    if extension_ref:
                        try:
                            success = extension_ref.turn_on_graph_lasers(
                                wavelength_nm=current_wavelength,
                                beam_waist_mm=current_beam_waist
                            )
                            if success:
                                laser_enabled = True
                                laser_status_label.text = f"Laser: ON ({current_wavelength:.0f}nm, {current_beam_waist:.1f}mm)"
                                laser_status_label.style = {"color": ui.color.green}
                                carb.log_info(f"Graph laser system turned on - {current_wavelength:.1f}nm, {current_beam_waist:.2f}mm waist")
                                print(f"[DEBUG] Graph laser system turned on - {current_wavelength:.1f}nm, {current_beam_waist:.2f}mm waist")
                            else:
                                carb.log_warn("Failed to turn on graph lasers")
                                print("[DEBUG] Failed to turn on graph lasers")
                        except Exception as e:
                            carb.log_error(f"Error turning on graph lasers: {e}")
                            print(f"[DEBUG ERROR] Error turning on graph lasers: {e}")
                    else:
                        carb.log_warn("No extension reference available for laser control")

                def turn_off_laser():
                    nonlocal laser_enabled
                    print("[DEBUG] Turn Off Laser clicked")
                    if extension_ref:
                        try:
                            success = extension_ref.turn_off_graph_lasers()
                            if success:
                                laser_enabled = False
                                laser_status_label.text = "Laser: OFF"
                                laser_status_label.style = {"color": ui.color.red}
                                carb.log_info("Graph laser system turned off")
                                print("[DEBUG] Graph laser system turned off")
                            else:
                                carb.log_warn("Failed to turn off graph lasers")
                                print("[DEBUG] Failed to turn off graph lasers")
                        except Exception as e:
                            carb.log_error(f"Error turning off graph lasers: {e}")
                            print(f"[DEBUG ERROR] Error turning off graph lasers: {e}")
                    else:
                        carb.log_warn("No extension reference available for laser control")

                with ui.HStack(spacing=5):
                    ui.Button("Turn On Laser", clicked_fn=turn_on_laser)
                    ui.Button("Turn Off Laser", clicked_fn=turn_off_laser)

                ui.Separator()

                # Simulation Controls Section
                ui.Label("Simulation Controls", height=20, style={"font_size": 14})

                def simulate_placeholder():
                    current_wavelength = wavelength_model.get_value_as_float()
                    current_beam_waist = beam_waist_model.get_value_as_float()
                    print(f"[DEBUG] Simulate clicked - Wavelength: {current_wavelength:.1f}nm, Beam Waist: {current_beam_waist:.2f}mm")
                    carb.log_info(f"Simulate placeholder called with {current_wavelength:.1f}nm, {current_beam_waist:.2f}mm waist")

                ui.Button("Simulate", clicked_fn=simulate_placeholder)

                ui.Separator()

                # Close button
                def close_editor():
                    window.visible = False
                    print("[DEBUG] Graph editor closed")

                ui.Button("Close", clicked_fn=close_editor)

        carb.log_info("Graph editor interface created successfully")
        print("[DEBUG] Graph editor interface created successfully")
        print(f"[DEBUG] Graph editor visible: {window.visible}")

        return window

    except Exception as e:
        carb.log_error(f"Failed to create graph editor interface: {e}")
        print(f"[DEBUG ERROR] Failed to create graph editor interface: {e}")
        import traceback
        traceback.print_exc()
        return None

def launch_graph_editor(extension_ref=None, save_directory=None):
    """Launch the graph editor interface."""
    try:
        carb.log_info("Launch graph editor requested...")
        print("[DEBUG] Launch graph editor requested...")

        # Create the new streamlined graph editor interface
        carb.log_info("About to create graph editor interface...")
        print("[DEBUG] About to create graph editor interface...")

        window = create_graph_editor_interface(extension_ref)

        if window:
            carb.log_info("Graph editor interface created successfully")
            print("[DEBUG] Graph editor interface created successfully")
            return window
        else:
            carb.log_error("Failed to create graph editor interface")
            print("[DEBUG ERROR] Failed to create graph editor interface")
            return None

    except Exception as e:
        carb.log_error(f"Failed to launch graph editor: {e}")
        print(f"[DEBUG ERROR] Failed to launch graph editor: {e}")
        import traceback
        carb.log_error(f"Traceback: {traceback.format_exc()}")
        traceback.print_exc()
        return None

def create_launcher_window_if_needed():
    """Create the launcher window only when explicitly requested."""
    try:
        carb.log_info("create_launcher_window_if_needed called")
        print("[DEBUG] create_launcher_window_if_needed called")
        return create_graph_editor_interface()
    except Exception as e:
        carb.log_error(f"Failed to create launcher window: {e}")
        print(f"[DEBUG ERROR] Failed to create launcher window: {e}")
        return None

if __name__ == "__main__":
    # Test the graph editor launcher
    window = create_launcher_window_if_needed()