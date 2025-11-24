# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext
import omni.usd
import omni.kit.viewport.utility
import omni.kit.commands
import carb
from pathlib import Path
from pxr import Usd, UsdGeom, Sdf, Gf
import omni.kit.app
import numpy as np
import math
import sys
import os

from .utils.path_utils import find_graph_file, get_extension_root

GRAPH_SAMPLE_FILES = ("fourfplusmichelson.json", "example_4f_system.json")

print("[mit.test_extensions_1] About to import physics module...")
try:
    from .physics import get_physics_status
    print("[mit.test_extensions_1] Physics module imported successfully")
    physics_status = get_physics_status()
    print(f"[mit.test_extensions_1] Physics status: {physics_status}")
except Exception as e:
    print(f"[mit.test_extensions_1] Error importing physics module: {e}")

print("[mit.test_extensions_1] About to import optical_simulation...")
try:
    from .physics.optical_simulation import simulate_4f_system_from_positions, simulate_michelson_interferometer_from_positions, plot_and_save_results
    print("[mit.test_extensions_1] Optical simulation imported successfully")
except Exception as e:
    print(f"[mit.test_extensions_1] Error importing optical_simulation: {e}")

# Note: graph_editor is imported lazily when needed to avoid PySide6 startup issues

# Functions and vars are available to other extensions as usual in python:
# `mit.test_extensions_1.some_public_function(x)`
def some_public_function(x: int):
    """This is a public function that can be called from other extensions."""
    print(f"[mit.test_extensions_1] some_public_function was called with {x}")
    return x**x


# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    """This is a blank extension template."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.
    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        print("[mit.test_extensions_1] Extension startup")

        self.count = 0
        self.lens_count = 0

        # Default lens position
        self.lens_x = 0.0
        self.lens_y = 0.0
        self.lens_z = 0.0

        self.window = omni.ui.Window("Lens Controller", width=400, height=600)
        with self.window.frame:
            # Wrap the entire UI in a scrollable frame
            with omni.ui.ScrollingFrame():
                with omni.ui.VStack():
                    # Setup section
                    omni.ui.Label("Scene Setup:", height=20)

                    def on_setup_optical_table():
                        success = self.setup_optical_table()
                        if success:
                            setup_status_label.text = "Optical table set up!"
                        else:
                            setup_status_label.text = "Failed to set up optical table"

                    setup_status_label = omni.ui.Label("Ready to set up optical table")
                    omni.ui.Button("Set Up Optical Table", clicked_fn=on_setup_optical_table)

                    # 4f System section
                    omni.ui.Separator()
                    with omni.ui.CollapsableFrame("4f Optical System", height=0):
                        with omni.ui.VStack(spacing=5):
                            omni.ui.Label("4f Optical System:", height=20)
                            fourf_status_label = omni.ui.Label("Ready to create 4f system")

                            def on_create_4f_system():
                                self.create_4f_system()
                                fourf_status_label.text = f"4f System created! Lenses in scene: {self.lens_count}"

                            omni.ui.Button("Create 4f System", clicked_fn=on_create_4f_system, width=120, height=28)
                            omni.ui.Button("Turn On 4f Laser", clicked_fn=self.turn_on_laser, width=120, height=28)
                            omni.ui.Button("Turn Off 4f Laser", clicked_fn=self.turn_off_laser, width=120, height=28)

                            # 4f Backend Selection section
                            omni.ui.Separator()
                            omni.ui.Label("4f Simulation Backend:", height=20)
                            fourf_backend_status_label = omni.ui.Label("No backend selected")

                            def on_set_4f_chromatix_backend():
                                fourf_backend_status_label.text = "Backend: Chromatix (placeholder - not implemented)"
                                print("[mit.test_extensions_1] Placeholder: Set 4f backend to Chromatix")

                            def on_set_4f_optiland_backend():
                                fourf_backend_status_label.text = "Backend: OptiLand (placeholder - not implemented)"
                                print("[mit.test_extensions_1] Placeholder: Set 4f backend to OptiLand")

                            with omni.ui.HStack(spacing=10):
                                omni.ui.Button("Set Backend to Chromatix", clicked_fn=on_set_4f_chromatix_backend, width=120, height=24)
                                omni.ui.Button("Set Backend to OptiLand", clicked_fn=on_set_4f_optiland_backend, width=120, height=24)

                            # 4f Simulation section
                            omni.ui.Separator()
                            omni.ui.Label("4f Simulation:", height=20)

                            def on_simulate_4f():
                                try:
                                    fourf_components = self.get_component_positions()
                                    if fourf_components:
                                        # Get current laser parameters if available
                                        wavelength_nm = getattr(self, '_current_wavelength_nm', 550.0)
                                        beam_waist_mm = getattr(self, '_current_beam_waist_mm', 1.0)
                                        wavelength_m = wavelength_nm * 1e-9  # Convert nm to meters

                                        self.turn_on_laser()
                                        simulation_result_label.text = f"Simulating 4f system with {self.selected_resolution}x{self.selected_resolution} field..."
                                        print(f"[mit.test_extensions_1] Simulating with λ={wavelength_nm:.0f}nm, waist={beam_waist_mm:.2f}mm")

                                        result = simulate_4f_system_from_positions(
                                            fourf_components,
                                            wavelength=wavelength_m,
                                            beam_waist_mm=beam_waist_mm,
                                            focal_length=30,  # 30 cm
                                            field_resolution=self.selected_resolution
                                        )
                                        if result.get("success", False):
                                            plot_path = plot_and_save_results(result)
                                            if plot_path:
                                                simulation_result_label.text = f"Simulation complete! Saved to: {Path(plot_path).name}"
                                                abs_plot_path = Path(plot_path).resolve()
                                                if abs_plot_path.exists():
                                                    import time
                                                    time.sleep(0.1)
                                                    try:
                                                        path_str = str(abs_plot_path).replace("\\", "/")
                                                        plot_image.source_url = path_str
                                                        plot_image.visible = True
                                                        print(f"[mit.test_extensions_1] Loading plot from: {path_str}")
                                                    except Exception as img_error:
                                                        print(f"[mit.test_extensions_1] Error loading image: {img_error}")
                                                else:
                                                    simulation_result_label.text = f"Plot saved but file not found: {abs_plot_path}"
                                            else:
                                                simulation_result_label.text = "Simulation completed but plotting failed"
                                        else:
                                            error_msg = result.get("error", "Unknown error")
                                            simulation_result_label.text = f"Simulation failed: {error_msg}"
                                            print(f"[mit.test_extensions_1] Simulation error: {error_msg}")
                                    else:
                                        simulation_result_label.text = "Error: Could not get 4f component positions"
                                        print("[mit.test_extensions_1] Error: Could not get 4f component positions")
                                except Exception as e:
                                    simulation_result_label.text = f"Error: {str(e)}"
                                    print(f"[mit.test_extensions_1] Simulation error: {e}")
                                    import traceback
                                    traceback.print_exc()

                            omni.ui.Button("Simulate 4f System", clicked_fn=on_simulate_4f, width=120, height=28)

                    # Michelson Interferometer section
                    omni.ui.Separator()
                    with omni.ui.CollapsableFrame("Michelson Interferometer", height=0):
                        with omni.ui.VStack(spacing=5):
                            omni.ui.Label("Michelson Interferometer:", height=20)

                            def on_create_michelson_interferometer():
                                self.create_michelson_interferometer()
                                michelson_label.text = f"Michelson Interferometer created! Components in scene."

                            michelson_label = omni.ui.Label("")
                            omni.ui.Button("Create Michelson Interferometer", clicked_fn=on_create_michelson_interferometer, width=120, height=28)
                            omni.ui.Button("Turn On Michelson Laser", clicked_fn=self.turn_on_michelson_laser, width=120, height=28)
                            omni.ui.Button("Turn Off Michelson Laser", clicked_fn=self.turn_off_michelson_laser, width=120, height=28)

                            # Michelson Backend Selection section
                            omni.ui.Separator()
                            omni.ui.Label("Michelson Simulation Backend:", height=20)
                            michelson_backend_status_label = omni.ui.Label("No backend selected")

                            def on_set_michelson_chromatix_backend():
                                michelson_backend_status_label.text = "Backend: Chromatix (placeholder - not implemented)"
                                print("[mit.test_extensions_1] Placeholder: Set Michelson backend to Chromatix")

                            def on_set_michelson_optiland_backend():
                                michelson_backend_status_label.text = "Backend: OptiLand (placeholder - not implemented)"
                                print("[mit.test_extensions_1] Placeholder: Set Michelson backend to OptiLand")

                            with omni.ui.HStack(spacing=10):
                                omni.ui.Button("Set Backend to Chromatix", clicked_fn=on_set_michelson_chromatix_backend, width=120, height=24)
                                omni.ui.Button("Set Backend to OptiLand", clicked_fn=on_set_michelson_optiland_backend, width=120, height=24)

                            # Michelson Simulation section
                            omni.ui.Separator()
                            omni.ui.Label("Michelson Simulation:", height=20)

                            def on_simulate_michelson():
                                try:
                                    michelson_components = self.get_michelson_component_positions()
                                    mirror_rotations = self.get_michelson_mirror_rotations()
                                    if michelson_components:
                                        # Get current laser parameters if available
                                        wavelength_nm = getattr(self, '_current_wavelength_nm', 632.8)  # Default to HeNe
                                        beam_waist_mm = getattr(self, '_current_beam_waist_mm', 0.05)   # Default 0.05mm for Michelson
                                        wavelength_m = wavelength_nm * 1e-9  # Convert nm to meters

                                        self.turn_on_michelson_laser()
                                        simulation_result_label.text = f"Simulating Michelson Interferometer with {self.selected_resolution}x{self.selected_resolution} field..."
                                        print(f"[mit.test_extensions_1] Michelson simulation with λ={wavelength_nm:.0f}nm, waist={beam_waist_mm:.3f}mm")

                                        result = simulate_michelson_interferometer_from_positions(
                                            michelson_components,
                                            mirror_rotations=mirror_rotations,
                                            wavelength=wavelength_m,
                                            beam_waist_mm=beam_waist_mm,
                                            field_resolution=self.selected_resolution
                                        )
                                        if result.get("success", False):
                                            plot_path = plot_and_save_results(result)
                                            if plot_path:
                                                simulation_result_label.text = f"Simulation complete! Saved to: {Path(plot_path).name}"
                                                abs_plot_path = Path(plot_path).resolve()
                                                if abs_plot_path.exists():
                                                    import time
                                                    time.sleep(0.1)
                                                    try:
                                                        path_str = str(abs_plot_path).replace("\\", "/")
                                                        plot_image.source_url = path_str
                                                        plot_image.visible = True
                                                        print(f"[mit.test_extensions_1] Loading Michelson plot from: {path_str}")
                                                    except Exception as img_error:
                                                        print(f"[mit.test_extensions_1] Error loading image: {img_error}")
                                                else:
                                                    simulation_result_label.text = f"Plot saved but file not found: {abs_plot_path}"
                                            else:
                                                simulation_result_label.text = "Simulation completed but plotting failed"
                                        else:
                                            error_msg = result.get("error", "Unknown error")
                                            simulation_result_label.text = f"Michelson simulation failed: {error_msg}"
                                            print(f"[mit.test_extensions_1] Michelson simulation error: {error_msg}")
                                    else:
                                        simulation_result_label.text = "Error: Could not get Michelson component positions"
                                        print("[mit.test_extensions_1] Error: Could not get Michelson component positions")
                                except Exception as e:
                                    simulation_result_label.text = f"Error: {str(e)}"
                                    print(f"[mit.test_extensions_1] Michelson simulation error: {e}")
                                    import traceback
                                    traceback.print_exc()

                            omni.ui.Button("Simulate Michelson Interferometer", clicked_fn=on_simulate_michelson, width=120, height=28)

                    # Physics/Simulation section (shared)
                    omni.ui.Separator()
                    omni.ui.Label("Physics Simulation:", height=20)

                    # Check physics status
                    physics_status = get_physics_status()
                    jax_status = "✓" if physics_status["jax_available"] else "✗"
                    chromatix_status = "✓" if physics_status["chromatix_available"] else "✗"
                    omni.ui.Label(f"JAX: {jax_status}  Chromatix: {chromatix_status}", height=20)

                    # Field resolution selection
                    with omni.ui.HStack(spacing=5):
                        omni.ui.Label("Field Resolution:", width=100)
                        resolution_options = [256, 512, 1000, 2048]
                        self.selected_resolution = 512  # Default resolution
                        resolution_status_label = omni.ui.Label(f"Selected: {self.selected_resolution}x{self.selected_resolution}")
                        def on_resolution_change(resolution):
                            self.selected_resolution = resolution
                            resolution_status_label.text = f"Selected: {resolution}x{resolution}"
                            print(f"[extension] Selected resolution: {resolution}x{resolution}")
                        with omni.ui.HStack(spacing=5):
                            for res in resolution_options:
                                omni.ui.Button(f"{res}", clicked_fn=lambda r=res: on_resolution_change(r), width=60, height=24)
                    omni.ui.Label("Note: Higher resolution = more accurate but slower simulation", style_type_name_override="Label", height=15, style={"color": 0xFF606060})

                    simulation_result_label = omni.ui.Label("Click to simulate system from scene")
                    plot_image = omni.ui.Image("", width=400, height=300, fill_policy=omni.ui.FillPolicy.PRESERVE_ASPECT_FIT)
                    plot_image.visible = False  # Hide initially

                    # Separator
                    omni.ui.Separator()

                    # Backend Selection section
                    omni.ui.Label("Simulation Backend:", height=20)
                    backend_status_label = omni.ui.Label("No backend selected")

                    def on_set_chromatix_backend():
                        backend_status_label.text = "Backend: Chromatix (placeholder - not implemented)"
                        print("[mit.test_extensions_1] Placeholder: Set backend to Chromatix")

                    def on_set_optiland_backend():
                        backend_status_label.text = "Backend: OptiLand (placeholder - not implemented)"
                        print("[mit.test_extensions_1] Placeholder: Set backend to OptiLand")

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Set Backend to Chromatix", clicked_fn=on_set_chromatix_backend)
                        omni.ui.Button("Set Backend to OptiLand", clicked_fn=on_set_optiland_backend)

                    # Separator
                    omni.ui.Separator()

                    # Laser Control section
                    omni.ui.Label("Laser Control:", height=20)
                    laser_status_label = omni.ui.Label("Laser off")

                    def on_turn_on_laser():
                        success = self.turn_on_laser()
                        if success:
                            laser_status_label.text = "Laser on"
                        else:
                            laser_status_label.text = "Failed to turn on laser"

                    def on_turn_off_laser():
                        success = self.turn_off_laser()
                        if success:
                            laser_status_label.text = "Laser off"
                        else:
                            laser_status_label.text = "Failed to turn off laser"

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Turn On Laser", clicked_fn=on_turn_on_laser)
                        omni.ui.Button("Turn Off Laser", clicked_fn=on_turn_off_laser)

                    # Separator
                    omni.ui.Separator()

                    # Position Control section
                    omni.ui.Label("Position Control:", height=20)

                    # X coordinate
                    with omni.ui.HStack(spacing=5):
                        omni.ui.Label("X:", width=20)
                        x_field = omni.ui.FloatField(value=self.lens_x, width=80)
                        x_field.model.add_value_changed_fn(lambda m: setattr(self, 'lens_x', m.get_value_as_float()))

                    # Y coordinate
                    with omni.ui.HStack(spacing=5):
                        omni.ui.Label("Y:", width=20)
                        y_field = omni.ui.FloatField(value=self.lens_y, width=80)
                        y_field.model.add_value_changed_fn(lambda m: setattr(self, 'lens_y', m.get_value_as_float()))

                    # Z coordinate
                    with omni.ui.HStack(spacing=5):
                        omni.ui.Label("Z:", width=20)
                        z_field = omni.ui.FloatField(value=self.lens_z, width=80)
                        z_field.model.add_value_changed_fn(lambda m: setattr(self, 'lens_z', m.get_value_as_float()))

                    # Separator
                    omni.ui.Separator()

                    # Position Reading section
                    omni.ui.Label("Position Reading:", height=20)
                    position_label = omni.ui.Label("Select a lens to read position")

                    def on_read_position():
                        position = self.get_selected_lens_position()
                        if position:
                            x, y, z = position
                            position_label.text = f"Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}"
                        else:
                            position_label.text = "No lens selected or found"

                    def on_update_position():
                        success = self.update_selected_lens_position()
                        if success:
                            position_label.text = f"Updated to: X={self.lens_x:.2f}, Y={self.lens_y:.2f}, Z={self.lens_z:.2f}"
                        else:
                            position_label.text = "No lens selected or update failed"

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Read Position", clicked_fn=on_read_position)
                        omni.ui.Button("Update Position", clicked_fn=on_update_position)

                    # Separator
                    omni.ui.Separator()

                    # Graph Editor section
                    omni.ui.Label("Graph Editor:", height=20)
                    graph_editor_status_label = omni.ui.Label("Graph editor ready")

                    def on_launch_graph_editor():
                        try:
                            # Lazy import graph editor to avoid PySide6 startup issues
                            from .graph_editor import launch_graph_editor
                            # Launch the graph editor with extension reference for laser control
                            window = launch_graph_editor(extension_ref=self)
                            if window:
                                graph_editor_status_label.text = "Graph editor launched"
                                print("[mit.test_extensions_1] Graph editor launched successfully")
                            else:
                                graph_editor_status_label.text = "Failed to launch graph editor"
                                print("[mit.test_extensions_1] Graph editor launch returned None")
                        except Exception as e:
                            graph_editor_status_label.text = f"Failed to launch: {str(e)}"
                            print(f"[mit.test_extensions_1] Error launching graph editor: {e}")
                            carb.log_error(f"[mit.test_extensions_1] Error launching graph editor: {e}")

                    def on_load_json_to_usd():
                        """Load JSON graph and generate USD scene."""
                        try:
                            from .graph_editor import load_and_generate_scene
                            load_and_generate_scene()
                            graph_editor_status_label.text = "JSON to USD conversion initiated"
                            print("[mit.test_extensions_1] JSON to USD conversion initiated")
                        except Exception as e:
                            graph_editor_status_label.text = f"JSON conversion error: {str(e)[:30]}..."
                            print(f"[mit.test_extensions_1] Error in JSON to USD conversion: {e}")

                    def on_turn_on_graph_lasers():
                        """Turn on all graph laser beams."""
                        try:
                            success = self.turn_on_graph_lasers()
                            if success:
                                graph_editor_status_label.text = "Graph laser system turned on"
                                print("[mit.test_extensions_1] Graph laser system turned on")
                            else:
                                graph_editor_status_label.text = "Failed to turn on graph lasers"
                                print("[mit.test_extensions_1] Failed to turn on graph lasers")
                        except Exception as e:
                            graph_editor_status_label.text = f"Laser on error: {str(e)[:30]}..."
                            print(f"[mit.test_extensions_1] Error turning on graph lasers: {e}")

                    def on_turn_off_graph_lasers():
                        """Turn off all graph laser beams."""
                        try:
                            success = self.turn_off_graph_lasers()
                            if success:
                                graph_editor_status_label.text = "Graph laser system turned off"
                                print("[mit.test_extensions_1] Graph laser system turned off")
                            else:
                                graph_editor_status_label.text = "Failed to turn off graph lasers"
                                print("[mit.test_extensions_1] Failed to turn off graph lasers")
                        except Exception as e:
                            graph_editor_status_label.text = f"Laser off error: {str(e)[:30]}..."
                            print(f"[mit.test_extensions_1] Error turning off graph lasers: {e}")

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Launch Graph Editor", clicked_fn=on_launch_graph_editor)
                        omni.ui.Button("Load JSON → USD", clicked_fn=on_load_json_to_usd)

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Turn On Graph Lasers", clicked_fn=on_turn_on_graph_lasers)
                        omni.ui.Button("Turn Off Graph Lasers", clicked_fn=on_turn_off_graph_lasers)

        self._last_4f_positions = None
        # self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)

    # def _on_update(self, e):
    #     # Real-time update for 4f system laser beam
    #     positions = self.get_component_positions()
    #     if positions and positions != self._last_4f_positions:
    #         self.update_laser_beam_from_4f_positions(positions)
    #         self._last_4f_positions = positions

    def update_laser_beam_from_4f_positions(self, positions):
        """Update the laser beam prim to match the current 4f system positions using the laser prefab. Only modify the existing laser, do not instantiate/delete here."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            laser_prim_path = "/World/FourFSystem/LaserBeam"  # Updated path to be child of FourFSystem
            prim = stage.GetPrimAtPath(laser_prim_path)
            if not prim:
                # Do not create a new laser here; only update if it exists
                print(f"[mit.test_extensions_1] Laser beam prim does not exist, skipping update.")
                return
            laser_pos = positions['laser']
            camera_pos = positions['camera']
            import numpy as np
            start = np.array(laser_pos)
            end = np.array(camera_pos)
            mid = (start + end) / 2
            self.set_prim_position(prim, *mid)
            # Set scale (length = distance between laser and camera)
            length = np.linalg.norm(end - start)
            scale_attr = prim.GetAttribute("xformOp:scale")
            if not scale_attr:
                scale_attr = prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
            scale_attr.Set((1.0, 1.0, length))
            # Set rotation to align with the direction
            direction = end - start
            def vector_to_euler(v):
                vx, vy, vz = v / np.linalg.norm(v)
                import math
                ry = math.degrees(math.atan2(vx, vz))
                rx = math.degrees(-math.atan2(vy, math.sqrt(vx**2 + vz**2)))
                return (rx, ry, 0)
            rx, ry, rz = vector_to_euler(direction)
            rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
            if not rotate_attr:
                rotate_attr = prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
            rotate_attr.Set((rx, ry, rz))
            print(f"[mit.test_extensions_1] 4f Laser beam prefab transform updated from {laser_pos} to {camera_pos}")
        except Exception as e:
            print(f"[mit.test_extensions_1] Error updating 4f laser beam: {e}")

    def add_lens_to_scene(self):
        """Add a lens model to the scene."""
        try:
            # Get the USD context
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return

            # Path to the lens USD file - using carb tokens to resolve project root
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            lens_file_path = str(Path(project_root) / "CAD" / "AC127-030-A-ML-Step.usd")
            print(f"[mit.test_extensions_1] Loading lens from: {lens_file_path}")

            # Check if file exists
            if not Path(lens_file_path).exists():
                print(f"[mit.test_extensions_1] ERROR: Lens file not found at {lens_file_path}")
                return

            # Create a unique prim path for each lens
            prim_path = f"/World/Lens_{self.lens_count + 1}"

            # Create a prim and add the lens as a reference
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=prim_path,
                prim_type="Xform",
                select_new_prim=True,
                create_default_xform=True,
            )

            # Add reference using USD Python API
            prim = stage.GetPrimAtPath(prim_path)
            if prim:
                references = prim.GetReferences()
                references.AddReference(lens_file_path)

                # Set the position of the lens
                self.set_prim_position(prim, self.lens_x, self.lens_y, self.lens_z)

                print(f"[mit.test_extensions_1] Added lens at {prim_path} with position ({self.lens_x}, {self.lens_y}, {self.lens_z})")
            else:
                print(f"[mit.test_extensions_1] Failed to get prim at {prim_path}")

        except Exception as e:
            print(f"[mit.test_extensions_1] Error adding lens: {e}")

    def clear_lenses_from_scene(self):
        """Remove all lenses and 4f system components from the scene."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return

            # Find and remove all lens prims
            for i in range(1, self.lens_count + 1):
                prim_path = f"/World/Lens_{i}"
                if stage.GetPrimAtPath(prim_path):
                    omni.kit.commands.execute(
                        "DeletePrims",
                        paths=[prim_path],
                        destructive=True,
                    )
                    print(f"[mit.test_extensions_1] Removed lens at {prim_path}")

            # Also remove 4f system components
            system_components = ["LaserSource", "Lens1", "Lens2", "Camera"]
            for component in system_components:
                prim_path = f"/World/{component}"
                if stage.GetPrimAtPath(prim_path):
                    omni.kit.commands.execute(
                        "DeletePrims",
                        paths=[prim_path],
                        destructive=True,
                    )
                    print(f"[mit.test_extensions_1] Removed {component} at {prim_path}")

        except Exception as e:
            print(f"[mit.test_extensions_1] Error clearing lenses: {e}")

    def create_4f_system(self):
        """Create a complete 4f optical system with laser, two lenses, and camera, grouped under /World/FourFSystem."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return

            # Remove existing FourFSystem group if present
            group_path = "/World/FourFSystem"
            if stage.GetPrimAtPath(group_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[group_path],
                    destructive=True,
                )

            # Clear existing lenses first (optional, for legacy cleanup)
            self.clear_lenses_from_scene()

            # Define the components and their positions (f = 30mm)
            components = [
                {
                    "name": "LaserSource",
                    "file": "prefabs/LASER_PREFAB.usd",
                    "position": (0, 0, 0),
                    "description": "Laser source at origin"
                },
                {
                    "name": "Lens1",
                    "file": "prefabs/LENS_PREFAB.usd",
                    "position": (0, 0, 30),
                    "description": "First lens at f=30mm"
                },
                {
                    "name": "Lens2",
                    "file": "prefabs/LENS_PREFAB.usd",
                    "position": (0, 0, 90),
                    "description": "Second lens at 2f=60mm"
                },
                {
                    "name": "Camera",
                    "file": "prefabs/CAMERA_PREFAB.usd",
                    "position": (0, 0, 120),
                    "description": "Camera at f=30mm from second lens"
                }
            ]

            # Project root path
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")

            # Create the FourFSystem group Xform
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=group_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            for i, component in enumerate(components):
                file_path = str(Path(project_root) / "CAD" / component["file"])
                if not Path(file_path).exists():
                    print(f"[mit.test_extensions_1] ERROR: {component['description']} file not found at {file_path}")
                    continue
                prim_path = f"{group_path}/{component['name']}"
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=prim_path,
                    prim_type="Xform",
                    select_new_prim=False,
                    create_default_xform=True,
                )
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    references = prim.GetReferences()
                    references.AddReference(file_path)
                    x, y, z = component["position"]
                    self.set_prim_position(prim, x, y, z)
                    print(f"[mit.test_extensions_1] Added {component['description']} at position ({x}, {y}, {z})")
                    if "Lens" in component["name"]:
                        self.lens_count += 1
                else:
                    print(f"[mit.test_extensions_1] Failed to create prim for {component['description']}")
            print("[mit.test_extensions_1] 4f optical system created successfully!")
        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating 4f system: {e}")

    def create_michelson_interferometer(self):
        """Create a Michelson interferometer with laser, beam splitter, two mirrors (MIRROR_90_PREFAB), and camera, grouped under /World/MichelsonInterferometer in a rectilinear xz-plane layout."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return

            # Remove existing MichelsonInterferometer group if present
            group_path = "/World/MichelsonInterferometer"
            if stage.GetPrimAtPath(group_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[group_path],
                    destructive=True,
                )

            # Project root path
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")


            components = [
                {
                    "name": "LaserSource",
                    "file": "prefabs/LASER_PREFAB.usd",
                    "position": (0, 0, 0),
                    "rotation": (0, 0, 0),
                    "description": "Laser source at origin"
                },
                {
                    "name": "BeamSplitter",
                    "file": "prefabs/BEAM_SPLITTER_PREFAB.usd",
                    "position": (0, 0, 30),
                    "rotation": (0, 0, 0),
                    "description": "Beam splitter at (0,0,30)"
                },
                {
                    "name": "Mirror1",
                    "file": "prefabs/MIRROR_90_PREFAB.usd",
                    "position": (30, 0, 30),  # Right of BS in +x
                    "rotation": (0, 90, 0),   # 90 deg about y axis
                    "description": "Mirror 1 at (30,0,30)mm (right, rotated 90 deg about y)"
                },
                {
                    "name": "Mirror2",
                    "file": "prefabs/MIRROR_90_PREFAB.usd",
                    "position": (0, 0, 60),  # Forward in +z
                    "rotation": (0, 0, 0),
                    "description": "Mirror 2 at (0,0,60)mm (forward)"
                },
                {
                    "name": "Camera",
                    "file": "prefabs/CAMERA_PREFAB.usd",
                    "position": (-30, 0, 30),  # Left of BS in -x
                    "rotation": (0, 90, 0),   # 90 deg clockwise about y axis
                    "description": "Camera at (-30,0,30)mm (left, rotated 90 deg clockwise)"
                }
            ]

            # Create the MichelsonInterferometer group Xform
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=group_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            for component in components:
                file_path = str(Path(project_root) / "CAD" / component["file"])
                if not Path(file_path).exists():
                    print(f"[mit.test_extensions_1] ERROR: {component['description']} file not found at {file_path}")
                    continue
                prim_path = f"{group_path}/{component['name']}"
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=prim_path,
                    prim_type="Xform",
                    select_new_prim=False,
                    create_default_xform=True,
                )
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    references = prim.GetReferences()
                    references.AddReference(file_path)
                    x, y, z = component["position"]
                    self.set_prim_position(prim, x, y, z)
                    # Set rotation if needed
                    rx, ry, rz = component.get("rotation", (0, 0, 0))
                    if (rx, ry, rz) != (0, 0, 0):
                        rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
                        if not rotate_attr:
                            rotate_attr = prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
                        rotate_attr.Set((rx, ry, rz))
                        # Ensure xformOpOrder includes rotateXYZ
                        order_attr = prim.GetAttribute("xformOpOrder")
                        if not order_attr:
                            order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                            order_attr.Set(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
                        else:
                            order = order_attr.Get()
                            if "xformOp:rotateXYZ" not in order:
                                order = list(order) + ["xformOp:rotateXYZ"]
                                order_attr.Set(order)
                    print(f"[mit.test_extensions_1] Added {component['description']} at position ({x}, {y}, {z}) with rotation ({rx}, {ry}, {rz})")
                else:
                    print(f"[mit.test_extensions_1] Failed to create prim for {component['description']}")
            print("[mit.test_extensions_1] Michelson interferometer created successfully!")
        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating Michelson interferometer: {e}")

    def get_michelson_component_positions(self):
        """Get the positions of all Michelson interferometer components in the scene. Returns dict or None."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return None
            group_path = "/World/MichelsonInterferometer"
            components = {
                "laser": f"{group_path}/LaserSource",
                "beamsplitter": f"{group_path}/BeamSplitter",
                "mirror1": f"{group_path}/Mirror1",
                "mirror2": f"{group_path}/Mirror2",
                "camera": f"{group_path}/Camera"
            }
            positions = {}
            for name, prim_path in components.items():
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    position = self.get_prim_position(prim)
                    if position:
                        positions[name] = position
                        print(f"[mit.test_extensions_1] Found {name} at position {position}")
                    else:
                        print(f"[mit.test_extensions_1] Could not get position for {name}")
                        return None
                else:
                    print(f"[mit.test_extensions_1] Component {name} not found at {prim_path}")
                    return None
            return positions
        except Exception as e:
            print(f"[mit.test_extensions_1] Error getting Michelson component positions: {e}")
            return None

    def get_michelson_mirror_rotations(self):
        """Get the rotations of Michelson interferometer mirrors in the scene. Returns dict or None."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return None
            group_path = "/World/MichelsonInterferometer"
            mirrors = {
                "mirror1": f"{group_path}/Mirror1",
                "mirror2": f"{group_path}/Mirror2"
            }
            rotations = {}
            for name, prim_path in mirrors.items():
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    rotation = self.get_prim_rotation(prim)
                    if rotation is not None:
                        rotations[name] = rotation
                        print(f"[mit.test_extensions_1] Found {name} rotation: {rotation}")
                    else:
                        print(f"[mit.test_extensions_1] Could not get rotation for {name}")
                        return None
                else:
                    print(f"[mit.test_extensions_1] Mirror {name} not found at {prim_path}")
                    return None
            return rotations
        except Exception as e:
            print(f"[mit.test_extensions_1] Error getting Michelson mirror rotations: {e}")
            return None

    def set_prim_position(self, prim, x, y, z):
        """Set the position of a prim using transform operations."""
        try:
            # Get or create the translate attribute
            translate_attr = prim.GetAttribute("xformOp:translate")
            if not translate_attr:
                translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)

                # Set the xformOpOrder if it doesn't exist
                order_attr = prim.GetAttribute("xformOpOrder")
                if not order_attr:
                    order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    order_attr.Set(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])

            # Set the translation
            translate_attr.Set((x, y, z))

        except Exception as e:
            print(f"[mit.test_extensions_1] Error setting position: {e}")

    def get_prim_position(self, prim):
        """Get the position of a prim from its transform operations."""
        try:
            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr:
                translation = translate_attr.Get()
                return translation if translation else (0.0, 0.0, 0.0)
            else:
                return (0.0, 0.0, 0.0)
        except Exception as e:
            print(f"[mit.test_extensions_1] Error getting position: {e}")
            return None

    def get_prim_rotation(self, prim):
        """Get the rotation of a prim from its transform operations."""
        try:
            rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
            if rotate_attr:
                rotation = rotate_attr.Get()
                return rotation if rotation else (0.0, 0.0, 0.0)
            else:
                return (0.0, 0.0, 0.0)
        except Exception as e:
            print(f"[mit.test_extensions_1] Error getting rotation: {e}")
            return None

    def get_selected_lens_position(self):
        """Get the position of the currently selected lens."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            selection = usd_context.get_selection()

            if not stage or not selection:
                return None

            selected_paths = selection.get_selected_prim_paths()
            if not selected_paths:
                return None

            # Find the first selected lens
            for path in selected_paths:
                if "/Lens_" in path:
                    prim = stage.GetPrimAtPath(path)
                    if prim:
                        position = self.get_prim_position(prim)
                        if position:
                            return position

            return None

        except Exception as e:
            print(f"[mit.test_extensions_1] Error reading selected lens position: {e}")
            return None

    def update_selected_lens_position(self):
        """Update the position of the currently selected lens. Also turn off the laser if a 4f component is moved and update the laser beam."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            selection = usd_context.get_selection()

            if not stage or not selection:
                return False

            selected_paths = selection.get_selected_prim_paths()
            if not selected_paths:
                return False

            # Find the first selected lens and update its position
            for path in selected_paths:
                if "/Lens_" in path:
                    prim = stage.GetPrimAtPath(path)
                    if prim:
                        self.set_prim_position(prim, self.lens_x, self.lens_y, self.lens_z)
                        print(f"[mit.test_extensions_1] Updated lens at {path} to position ({self.lens_x}, {self.lens_y}, {self.lens_z})")
                        # Turn off the laser if a 4f component is moved
                        self.turn_off_laser()
                        # Update the laser beam to match new positions
                        positions = self.get_component_positions()
                        if positions:
                            self.update_laser_beam_from_4f_positions(positions)
                        return True

            return False

        except Exception as e:
            print(f"[mit.test_extensions_1] Error updating selected lens position: {e}")
            return False

    def get_component_positions(self):
        """Get the positions of all 4f system components in the scene.

        Returns:
            Dict with component positions or None if not found
        """
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return None

            # Look for the 4f system components under /World/FourFSystem
            group_path = "/World/FourFSystem"
            components = {
                "laser": f"{group_path}/LaserSource",
                "lens1": f"{group_path}/Lens1",
                "lens2": f"{group_path}/Lens2",
                "camera": f"{group_path}/Camera"
            }

            positions = {}

            for component_name, prim_path in components.items():
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    position = self.get_prim_position(prim)
                    if position:
                        positions[component_name] = position
                        print(f"[mit.test_extensions_1] Found {component_name} at position {position}")
                    else:
                        print(f"[mit.test_extensions_1] Could not get position for {component_name}")
                        return None
                else:
                    print(f"[mit.test_extensions_1] Component {component_name} not found at {prim_path}")
                    return None

            # Verify we have a valid 4f system (components should be in order along Z)
            z_positions = [pos[2] for pos in positions.values()]
            if z_positions != sorted(z_positions):
                print("[mit.test_extensions_1] Warning: Components not in expected Z order")

            return positions

        except Exception as e:
            print(f"[mit.test_extensions_1] Error getting component positions: {e}")
            return None

    def turn_on_laser(self):
        """Turn on the laser by instantiating the BEAM_PREFAB as a child of FourFSystem."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False

            # Get component positions
            component_positions = self.get_component_positions()
            if not component_positions:
                print("[mit.test_extensions_1] No 4f system found - create system first")
                return False

            # Get laser and camera positions
            laser_pos = component_positions['laser']  # Use laser position
            camera_pos = component_positions['camera']

            # Calculate beam position and scale factor
            beam_distance = camera_pos[2] - laser_pos[2]  # Distance from laser to camera
            beam_z_pos = beam_distance / 2  # Z position at center between laser and camera

            # Create laser beam prim as child of FourFSystem
            laser_prim_path = "/World/FourFSystem/LaserBeam"

            # Remove existing laser if it exists
            if stage.GetPrimAtPath(laser_prim_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[laser_prim_path],
                    destructive=True,
                )

            # Path to the beam prefab
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            beam_file_path = str(Path(project_root) / "CAD" / "prefabs" / "BEAM_PREFAB.usd")

            # Check if file exists
            if not Path(beam_file_path).exists():
                print(f"[mit.test_extensions_1] ERROR: Beam prefab file not found at {beam_file_path}")
                return False

            # Create prim and add reference to beam prefab
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=laser_prim_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            # Add reference to the beam prefab
            laser_prim = stage.GetPrimAtPath(laser_prim_path)
            if laser_prim:
                references = laser_prim.GetReferences()
                references.AddReference(beam_file_path)

                # Set position at center between laser and camera (relative to FourFSystem)
                self.set_prim_position(laser_prim, 0, 0, beam_z_pos)

                # Set scale - unit size in Z, scaled by distance
                scale_attr = laser_prim.GetAttribute("xformOp:scale")
                if not scale_attr:
                    scale_attr = laser_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
                scale_attr.Set((1.0, 1.0, beam_distance))

                print(f"[mit.test_extensions_1] 4f Laser beam prefab instantiated as child of FourFSystem at z={beam_z_pos:.1f} with scale={beam_distance:.1f}")
                return True
            else:
                print("[mit.test_extensions_1] Failed to create laser beam prim")
                return False

        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating 4f laser: {e}")
            return False

    def turn_off_laser(self):
        """Turn off the 4f laser by removing the beam from FourFSystem."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False

            # Remove laser beam from FourFSystem
            laser_prim_path = "/World/FourFSystem/LaserBeam"
            if stage.GetPrimAtPath(laser_prim_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[laser_prim_path],
                    destructive=True,
                )
                print("[mit.test_extensions_1] 4f Laser beam turned off")
                return True
            else:
                print("[mit.test_extensions_1] No 4f laser beam to turn off")
                return True

        except Exception as e:
            print(f"[mit.test_extensions_1] Error turning off 4f laser: {e}")
            return False

    def turn_on_michelson_laser(self):
        """Turn on the Michelson laser by instantiating two BEAM_PREFABs as children of MichelsonInterferometer."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False

            # Get Michelson component positions
            component_positions = self.get_michelson_component_positions()
            if not component_positions:
                print("[mit.test_extensions_1] No Michelson interferometer found - create system first")
                return False

            # Get component positions
            laser_pos = component_positions['laser']
            beamsplitter_pos = component_positions['beamsplitter']
            mirror1_pos = component_positions['mirror1']
            mirror2_pos = component_positions['mirror2']
            camera_pos = component_positions['camera']

            # Path to the beam prefab
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            beam_file_path = str(Path(project_root) / "CAD" / "prefabs" / "BEAM_PREFAB.usd")

            # Check if file exists
            if not Path(beam_file_path).exists():
                print(f"[mit.test_extensions_1] ERROR: Beam prefab file not found at {beam_file_path}")
                return False

            # Remove existing Michelson laser beams if they exist
            beam1_path = "/World/MichelsonInterferometer/LaserBeam1"
            beam2_path = "/World/MichelsonInterferometer/LaserBeam2"

            for beam_path in [beam1_path, beam2_path]:
                if stage.GetPrimAtPath(beam_path):
                    omni.kit.commands.execute(
                        "DeletePrims",
                        paths=[beam_path],
                        destructive=True,
                    )

            # Create first beam: Laser to Mirror2 (Z direction)
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=beam1_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            beam1_prim = stage.GetPrimAtPath(beam1_path)
            if beam1_prim:
                references = beam1_prim.GetReferences()
                references.AddReference(beam_file_path)

                # Calculate beam1 position and scale (laser to mirror2)
                beam1_start = np.array(laser_pos)
                beam1_end = np.array(mirror2_pos)
                beam1_mid = (beam1_start + beam1_end) / 2
                beam1_length = np.linalg.norm(beam1_end - beam1_start)

                # Set position relative to MichelsonInterferometer
                self.set_prim_position(beam1_prim, beam1_mid[0], beam1_mid[1], beam1_mid[2])

                # Set scale
                scale_attr = beam1_prim.GetAttribute("xformOp:scale")
                if not scale_attr:
                    scale_attr = beam1_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
                scale_attr.Set((1.0, 1.0, beam1_length))

                # Set rotation to align with direction
                direction = beam1_end - beam1_start
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    rx = math.degrees(-math.atan2(direction[1], math.sqrt(direction[0]**2 + direction[2]**2)))
                    ry = math.degrees(math.atan2(direction[0], direction[2]))
                    rotate_attr = beam1_prim.GetAttribute("xformOp:rotateXYZ")
                    if not rotate_attr:
                        rotate_attr = beam1_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
                    rotate_attr.Set((rx, ry, 0))

            # Create second beam: Mirror1 to Camera (X direction, rotated 90 degrees about Y)
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=beam2_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            beam2_prim = stage.GetPrimAtPath(beam2_path)
            if beam2_prim:
                references = beam2_prim.GetReferences()
                references.AddReference(beam_file_path)

                # Calculate beam2 position and scale (mirror1 to camera)
                beam2_start = np.array(mirror1_pos)
                beam2_end = np.array(camera_pos)
                beam2_mid = (beam2_start + beam2_end) / 2
                beam2_length = np.linalg.norm(beam2_end - beam2_start)

                # Set position relative to MichelsonInterferometer
                self.set_prim_position(beam2_prim, beam2_mid[0], beam2_mid[1], beam2_mid[2])

                # Set scale
                scale_attr = beam2_prim.GetAttribute("xformOp:scale")
                if not scale_attr:
                    scale_attr = beam2_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
                scale_attr.Set((1.0, 1.0, beam2_length))

                # Set rotation to align with direction (90 degrees about Y from the first beam)
                direction = beam2_end - beam2_start
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    rx = math.degrees(-math.atan2(direction[1], math.sqrt(direction[0]**2 + direction[2]**2)))
                    ry = math.degrees(math.atan2(direction[0], direction[2]))
                    rotate_attr = beam2_prim.GetAttribute("xformOp:rotateXYZ")
                    if not rotate_attr:
                        rotate_attr = beam2_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
                    rotate_attr.Set((rx, ry, 0))

            print(f"[mit.test_extensions_1] Michelson laser beams instantiated as children of MichelsonInterferometer")
            print(f"[mit.test_extensions_1] Beam1 (Laser->Mirror2): length={beam1_length:.1f}, Beam2 (Mirror1->Camera): length={beam2_length:.1f}")
            return True

        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating Michelson laser: {e}")
            return False

    def turn_off_michelson_laser(self):
        """Turn off the Michelson laser by removing the beams from MichelsonInterferometer."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False

            # Remove Michelson laser beams
            beam1_path = "/World/MichelsonInterferometer/LaserBeam1"
            beam2_path = "/World/MichelsonInterferometer/LaserBeam2"

            for beam_path in [beam1_path, beam2_path]:
                if stage.GetPrimAtPath(beam_path):
                    omni.kit.commands.execute(
                        "DeletePrims",
                        paths=[beam_path],
                        destructive=True,
                    )
                    print(f"[mit.test_extensions_1] Removed Michelson laser beam: {beam_path}")

            print("[mit.test_extensions_1] Michelson laser beams turned off")
            return True

        except Exception as e:
            print(f"[mit.test_extensions_1] Error turning off Michelson laser: {e}")
            return False

    def turn_on_graph_lasers(self, wavelength_nm=550.0, beam_waist_mm=1.0):
        """Turn on lasers for the JSON/graph optical system by traversing the graph and creating laser beams.

        Args:
            wavelength_nm: Laser wavelength in nanometers (400-700nm range)
            beam_waist_mm: Laser beam waist in millimeters (0.1-5.0mm range)
        """
        try:
            import json

            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False

            # Store laser parameters for use in beam creation
            self._current_wavelength_nm = wavelength_nm
            self._current_beam_waist_mm = beam_waist_mm

            print(f"[mit.test_extensions_1] Turning on graph lasers - Wavelength: {wavelength_nm:.1f}nm, Beam Waist: {beam_waist_mm:.2f}mm")

            json_path = find_graph_file(GRAPH_SAMPLE_FILES)
            if not json_path:
                fallback = get_extension_root() / "example_4f_system.json"
                json_path = fallback if fallback.exists() else None

            if not json_path or not Path(json_path).exists():
                print("[mit.test_extensions_1] No graph JSON file available. Add one to the graphs directory.")
                return False

            # Load and parse JSON
            with open(json_path, 'r') as f:
                graph_data = json.load(f)

            print(f"[mit.test_extensions_1] Loaded graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")

            # Build adjacency list from edges
            adjacency = self._build_graph_adjacency(graph_data['nodes'], graph_data['edges'])
            node_lookup = {node['id']: node for node in graph_data['nodes']}

            # Find all laser source nodes
            laser_nodes = [node for node in graph_data['nodes'] if node['type'] == 'Laser']
            if not laser_nodes:
                print("[mit.test_extensions_1] No laser sources found in graph")
                return False

            # Clear existing graph lasers
            self.turn_off_graph_lasers()

            # Create parent group for all graph lasers
            graph_lasers_path = "/OpticalSystem/GraphLasers"
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=graph_lasers_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            # Create laser beams by traversing from each laser source
            beam_count = 0
            for laser_node in laser_nodes:
                beam_count += self._create_laser_beams_from_source(
                    laser_node, adjacency, node_lookup, stage, beam_count, graph_lasers_path
                )

            print(f"[mit.test_extensions_1] Created {beam_count} laser beam segments grouped under {graph_lasers_path}")
            return beam_count > 0

        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating graph lasers: {e}")
            return False

    def turn_off_graph_lasers(self):
        """Turn off all graph lasers by removing the entire laser group."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False

            # Remove the entire GraphLasers group (much cleaner than individual beams)
            graph_lasers_path = "/OpticalSystem/GraphLasers"
            if stage.GetPrimAtPath(graph_lasers_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[graph_lasers_path],
                    destructive=True,
                )
                print(f"[mit.test_extensions_1] Removed entire graph laser group: {graph_lasers_path}")

            print("[mit.test_extensions_1] Graph laser system turned off")
            return True

        except Exception as e:
            print(f"[mit.test_extensions_1] Error turning off graph lasers: {e}")
            return False

    def _build_graph_adjacency(self, nodes, edges):
        """Build adjacency list from graph nodes and edges."""
        adjacency = {}
        for node in nodes:
            adjacency[node['id']] = []

        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            source_port = edge['source_port']
            target_port = edge['target_port']
            length_cm = edge.get('attributes', {}).get('length_cm', 10.0)

            # Add bidirectional connections with port and length info
            adjacency[source_id].append({
                'node_id': target_id,
                'source_port': source_port,
                'target_port': target_port,
                'length_cm': length_cm
            })
            adjacency[target_id].append({
                'node_id': source_id,
                'source_port': target_port,
                'target_port': source_port,
                'length_cm': length_cm
            })

        return adjacency

    def _create_laser_beams_from_source(self, laser_node, adjacency, node_lookup, stage, beam_count_start, parent_path):
        """Create laser beams starting from a laser source using graph traversal."""
        visited_edges = set()  # Track visited edges to avoid duplicates
        beam_count = 0

        # BFS/DFS traversal starting from laser
        queue = [(laser_node['id'], None, None)]  # (current_node_id, parent_node_id, connection_info)

        while queue:
            current_id, parent_id, parent_connection = queue.pop(0)
            current_node = node_lookup[current_id]

            print(f"[mit.test_extensions_1] Processing node {current_id} ({current_node['name']}, type: {current_node['type']})")

            # Process all connections from current node
            for connection in adjacency[current_id]:
                next_id = connection['node_id']
                next_node = node_lookup[next_id]

                # Create edge key to avoid duplicate beams
                edge_key = tuple(sorted([current_id, next_id]))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                # Skip if we came from this node (avoid immediate backtrack)
                if next_id == parent_id:
                    continue

                print(f"[mit.test_extensions_1] Creating laser beam: {current_node['name']} -> {next_node['name']}")

                # Create laser beam between current and next node
                beam_created = self._create_single_laser_beam(
                    current_node, next_node, connection, stage, beam_count_start + beam_count, parent_path
                )

                if beam_created:
                    beam_count += 1

                # Continue traversal unless we hit a terminal node
                if not self._is_terminal_node(next_node):
                    queue.append((next_id, current_id, connection))
                else:
                    print(f"[mit.test_extensions_1] Terminal node reached: {next_node['name']} (type: {next_node['type']})")

        return beam_count

    def _create_single_laser_beam(self, source_node, target_node, connection, stage, beam_index, parent_path):
        """Create a single laser beam between two nodes."""
        try:
            # Get node positions from the USD scene
            source_pos = self._get_graph_node_position(source_node, stage)
            target_pos = self._get_graph_node_position(target_node, stage)

            if not source_pos or not target_pos:
                print(f"[mit.test_extensions_1] Could not get positions for {source_node['name']} or {target_node['name']}")
                return False

            # Calculate beam properties
            beam_start = np.array(source_pos)
            beam_end = np.array(target_pos)
            beam_mid = (beam_start + beam_end) / 2
            beam_length = np.linalg.norm(beam_end - beam_start)

            # Create laser beam prim
            beam_prim_path = f"{parent_path}/GraphLaserBeam_{beam_index}"

            # Path to beam prefab
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            beam_file_path = str(Path(project_root) / "CAD" / "prefabs" / "BEAM_PREFAB.usd")

            if not Path(beam_file_path).exists():
                print(f"[mit.test_extensions_1] Beam prefab not found: {beam_file_path}")
                return False

            # Create prim
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=beam_prim_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            beam_prim = stage.GetPrimAtPath(beam_prim_path)
            if not beam_prim:
                return False

            # Add reference to beam prefab
            references = beam_prim.GetReferences()
            references.AddReference(beam_file_path)

            # Set position at beam midpoint
            self.set_prim_position(beam_prim, beam_mid[0], beam_mid[1], beam_mid[2])

            # Set scale (length)
            scale_attr = beam_prim.GetAttribute("xformOp:scale")
            if not scale_attr:
                scale_attr = beam_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
            scale_attr.Set((1.0, 1.0, beam_length))

            # Set rotation to align with beam direction
            direction = beam_end - beam_start
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                rx = math.degrees(-math.atan2(direction[1], math.sqrt(direction[0]**2 + direction[2]**2)))
                ry = math.degrees(math.atan2(direction[0], direction[2]))

                rotate_attr = beam_prim.GetAttribute("xformOp:rotateXYZ")
                if not rotate_attr:
                    rotate_attr = beam_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
                rotate_attr.Set((rx, ry, 0))

            # Apply wavelength-based color to the beam
            wavelength = getattr(self, '_current_wavelength_nm', 550.0)  # Default to green if not set
            beam_waist = getattr(self, '_current_beam_waist_mm', 1.0)

            # Import wavelength-to-color conversion
            from .graph_editor import wavelength_to_rgb
            beam_color = wavelength_to_rgb(wavelength)

            # Apply color to beam material (if the beam prefab supports material overrides)
            try:
                # Look for material binding in the beam prefab
                # This assumes the beam prefab has a material that can be overridden
                self._apply_beam_color(beam_prim, beam_color, wavelength)
                print(f"[mit.test_extensions_1] Applied color RGB{beam_color} for {wavelength:.0f}nm to beam {beam_index}")
            except Exception as color_error:
                print(f"[mit.test_extensions_1] Warning: Could not apply beam color: {color_error}")

            print(f"[mit.test_extensions_1] Created laser beam {beam_index}: {source_node['name']} -> {target_node['name']} (length: {beam_length:.1f}, λ={wavelength:.0f}nm, waist={beam_waist:.2f}mm)")
            return True

        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating laser beam: {e}")
            return False

    def _apply_beam_color(self, beam_prim, rgb_color, wavelength_nm):
        """Apply wavelength-based color to a laser beam prim by modifying the CylinderLight color."""
        try:
            from pxr import UsdLux, Gf

            # Find the CylinderLight prim within the beam hierarchy
            # Expected path: BeamPrim/Group/CylinderLight
            stage = beam_prim.GetStage()

            # Look for CylinderLight in the hierarchy
            def find_cylinder_light(prim):
                """Recursively find CylinderLight prims in the hierarchy."""
                lights = []

                # Check if this prim is a CylinderLight
                if prim.GetTypeName() == "CylinderLight":
                    lights.append(prim)

                # Recursively check children
                for child in prim.GetChildren():
                    lights.extend(find_cylinder_light(child))

                return lights

            # Find all CylinderLight prims in the beam
            cylinder_lights = find_cylinder_light(beam_prim)

            if not cylinder_lights:
                print(f"[mit.test_extensions_1] No CylinderLight found in beam hierarchy: {beam_prim.GetPath()}")
                return

            # Apply color to all found CylinderLights
            r, g, b = rgb_color
            light_color = Gf.Vec3f(r, g, b)

            for light_prim in cylinder_lights:
                # Get the light schema
                light = UsdLux.CylinderLight(light_prim)

                # Set the color attribute
                if light:
                    color_attr = light.GetColorAttr()
                    if not color_attr:
                        color_attr = light.CreateColorAttr()
                    color_attr.Set(light_color)

                    # Also set intensity to make the color more visible
                    intensity_attr = light.GetIntensityAttr()
                    if not intensity_attr:
                        intensity_attr = light.CreateIntensityAttr()
                    # Use a much higher intensity value for maximum visual impact
                    intensity_attr.Set(5000.0)  # Increased from 1000 to 5000 for vivid colors

                    # Set diffuse multiplier for more intense color spread
                    try:
                        diffuse_attr = light.GetDiffuseAttr()
                        if not diffuse_attr:
                            diffuse_attr = light.CreateDiffuseAttr()
                        diffuse_attr.Set(2.0)  # Boost diffuse lighting
                    except:
                        pass  # Not all light types may have diffuse

                    # Enable specular for extra brilliance
                    try:
                        specular_attr = light.GetSpecularAttr()
                        if not specular_attr:
                            specular_attr = light.CreateSpecularAttr()
                        specular_attr.Set(1.5)  # Boost specular highlights
                    except:
                        pass  # Not all light types may have specular

                    print(f"[mit.test_extensions_1] Applied color RGB({r:.3f}, {g:.3f}, {b:.3f}) to CylinderLight at {light_prim.GetPath()}")
                else:
                    print(f"[mit.test_extensions_1] Warning: Could not create UsdLux.CylinderLight from prim at {light_prim.GetPath()}")

            print(f"[mit.test_extensions_1] Successfully updated {len(cylinder_lights)} CylinderLight(s) with {wavelength_nm:.0f}nm color")

        except Exception as e:
            print(f"[mit.test_extensions_1] Warning: Could not apply beam color to CylinderLight: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: try to set display color attribute on the main prim
            try:
                from pxr import Gf
                color_attr = beam_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
                color_attr.Set([Gf.Vec3f(*rgb_color)])
                print(f"[mit.test_extensions_1] Applied fallback display color to beam prim")
            except Exception as fallback_error:
                print(f"[mit.test_extensions_1] Warning: Could not apply fallback color: {fallback_error}")

    def _get_graph_node_position(self, node, stage):
        """Get the position of a graph node from the USD scene."""
        try:
            # Look for the node in the OpticalSystem group (created by graph editor)
            node_name = self._sanitize_prim_name(node['name'], node['id'])
            prim_path = f"/OpticalSystem/{node_name}"

            prim = stage.GetPrimAtPath(prim_path)
            if prim:
                return self.get_prim_position(prim)
            else:
                print(f"[mit.test_extensions_1] Node prim not found: {prim_path}")
                return None

        except Exception as e:
            print(f"[mit.test_extensions_1] Error getting node position: {e}")
            return None

    def _sanitize_prim_name(self, name, node_id):
        """Sanitize prim name to be USD-compliant (same logic as graph editor)."""
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

    def _is_terminal_node(self, node):
        """Check if a node is terminal (should stop laser beam propagation)."""
        node_type = node['type']
        return node_type in ['Camera', 'Mirror90']

    def setup_optical_table(self):
        """Set up the optical table by deleting /World/Environment (removes sun, sky, ground, etc.), then instantiate OPTICAL_FLOOR_SETUP_PREFAB and ROOM_LIGHT_PREFAB at (0,0,0)."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return

            # Delete the entire environment prim if it exists
            environment_prim_path = "/World/Environment"
            if stage.GetPrimAtPath(environment_prim_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[environment_prim_path],
                    destructive=True,
                )

            # Robustly remove any remaining ground/sky/distant light prims
            self.remove_ground_and_sky()

            # Project root path
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")

            # Add optical floor setup
            floor_file = str(Path(project_root) / "CAD" / "prefabs" / "OPTICAL_FLOOR_SETUP_PREFAB.usd")
            floor_prim_path = "/World/OpticalFloor"
            if not stage.GetPrimAtPath(floor_prim_path):
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=floor_prim_path,
                    prim_type="Xform",
                    select_new_prim=False,
                    create_default_xform=True,
                )
                prim = stage.GetPrimAtPath(floor_prim_path)
                if prim:
                    references = prim.GetReferences()
                    references.AddReference(floor_file)
                    self.set_prim_position(prim, 0, 0, 0)

            # Always add room light prefab after cleanup
            light_file = str(Path(project_root) / "CAD" / "prefabs" / "ROOM_LIGHT_PREFAB.usd")
            light_prim_path = "/World/RoomLight"
            if not stage.GetPrimAtPath(light_prim_path):
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=light_prim_path,
                    prim_type="Xform",
                    select_new_prim=False,
                    create_default_xform=True,
                )
                prim = stage.GetPrimAtPath(light_prim_path)
                if prim:
                    references = prim.GetReferences()
                    references.AddReference(light_file)
                    self.set_prim_position(prim, 0, 0, 0)

            print("[mit.test_extensions_1] Optical table and room light set up at (0,0,0)")
            # Hide the Omniverse viewport ground plane overlay
            try:
                omni.kit.viewport.utility.set_ground_plane_enabled(False)
                print("[mit.test_extensions_1] Viewport ground plane hidden")
            except Exception as e:
                print(f"[mit.test_extensions_1] Could not hide viewport ground plane: {e}")
        except Exception as e:
            print(f"[mit.test_extensions_1] Error setting up optical table: {e}")

    def remove_ground_and_sky(self):
        """Remove ground, ground collider, sky light, and distant light from the scene."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return

            # Common ground, sky, and distant light paths that might exist
            paths_to_remove = [
                "/World/groundPlane",
                "/World/GroundPlane",
                "/World/Ground",
                "/World/ground",
                "/World/defaultGroundPlane",
                "/World/Environment/sky",
                "/World/Environment/Sky",
                "/World/Environment/SkyLight",
                "/World/defaultLight",
                "/World/SkyLight",
                "/World/DistantLight",
                "/World/distantLight",
                "/Environment/sky",
                "/Environment/Sky",
                "/Environment/SkyLight",
                "/Environment/DistantLight",
                "/Environment/distantLight",
            ]

            # Remove each path if it exists
            for path in paths_to_remove:
                if stage.GetPrimAtPath(path):
                    omni.kit.commands.execute(
                        "DeletePrims",
                        paths=[path],
                        destructive=True,
                    )
                    print(f"[mit.test_extensions_1] Removed: {path}")

            # Also try to remove any prims with ground/sky/distant light related names or types
            for prim in stage.Traverse():
                prim_name = prim.GetName().lower()
                prim_path = prim.GetPath().pathString
                prim_type = prim.GetTypeName().lower()
                # Remove ground/sky/sun/distant light related prims
                if any(keyword in prim_name for keyword in ['ground', 'sky', 'sun', 'distantlight', 'distant_light']) or prim_type == 'distantlight':
                    if prim_path != "/World" and prim_path != "/":
                        try:
                            omni.kit.commands.execute(
                                "DeletePrims",
                                paths=[prim_path],
                                destructive=True,
                            )
                            print(f"[mit.test_extensions_1] Removed ground/sky/distant light related prim: {prim_path}")
                        except:
                            pass  # Continue even if deletion fails
                # Remove sky light specifically but not distant light
                elif 'skylight' in prim_name and 'distant' not in prim_name:
                    if prim_path != "/World" and prim_path != "/":
                        try:
                            omni.kit.commands.execute(
                                "DeletePrims",
                                paths=[prim_path],
                                destructive=True,
                            )
                            print(f"[mit.test_extensions_1] Removed sky light: {prim_path}")
                        except:
                            pass  # Continue even if deletion fails

            print("[mit.test_extensions_1] Ground, sky, and distant light cleanup completed")

        except Exception as e:
            print(f"[mit.test_extensions_1] Error removing ground and sky: {e}")

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[mit.test_extensions_1] Extension shutdown")
        # if hasattr(self, '_update_sub') and self._update_sub:
        #     self._update_sub.unsubscribe()
