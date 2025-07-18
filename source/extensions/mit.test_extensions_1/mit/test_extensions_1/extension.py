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
import omni.kit.commands
import carb
from pathlib import Path
from pxr import Usd, UsdGeom, Sdf, Gf

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
    from .physics.optical_simulation import simulate_4f_system_from_positions, plot_and_save_results
    print("[mit.test_extensions_1] Optical simulation imported successfully")
except Exception as e:
    print(f"[mit.test_extensions_1] Error importing optical_simulation: {e}")


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
            with omni.ui.ScrollingFrame():
                with omni.ui.VStack(spacing=10):
                    # Counter section
                    omni.ui.Label("Counter Demo:", height=20)
                    label = omni.ui.Label("Hello World")

                    def on_click():
                        self.count += 1
                        label.text = f"Hello World {self.count}"

                    def on_reset():
                        self.count = 0
                        label.text = "empty"

                    on_reset()

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Click me", clicked_fn=on_click)
                        omni.ui.Button("Reset", clicked_fn=on_reset)
                    
                    # Separator
                    omni.ui.Separator()
                    
                    # Lens section
                    omni.ui.Label("Lens Control:", height=20)
                    lens_label = omni.ui.Label("No lenses in scene")
                    
                    def on_add_lens():
                        self.add_lens_to_scene()
                        self.lens_count += 1
                        lens_label.text = f"Lenses in scene: {self.lens_count}"

                    def on_clear_lenses():
                        self.clear_lenses_from_scene()
                        self.lens_count = 0
                        lens_label.text = "No lenses in scene"

                    with omni.ui.HStack(spacing=10):
                        omni.ui.Button("Add Lens", clicked_fn=on_add_lens)
                        omni.ui.Button("Clear Lenses", clicked_fn=on_clear_lenses)
                    
                    # Setup section
                    omni.ui.Separator()
                    omni.ui.Label("Scene Setup:", height=20)
                    
                    def on_setup_optical_table():
                        success = self.setup_optical_table()
                        if success:
                            lens_label.text = "Optical table set up!"
                        else:
                            lens_label.text = "Failed to set up optical table"
                    
                    omni.ui.Button("Set Up Optical Table", clicked_fn=on_setup_optical_table)
                    
                    # 4f System section
                    omni.ui.Separator()
                    omni.ui.Label("4f Optical System:", height=20)
                    
                    def on_create_4f_system():
                        self.create_4f_system()
                        lens_label.text = f"4f System created! Lenses in scene: {self.lens_count}"
                    
                    omni.ui.Button("Create 4f System", clicked_fn=on_create_4f_system)
                    
                    # Physics/Simulation section
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
                                omni.ui.Button(f"{res}", 
                                             clicked_fn=lambda r=res: on_resolution_change(r),
                                             width=50)
                    
                    omni.ui.Label("Note: Higher resolution = more accurate but slower simulation", 
                                 style_type_name_override="Label", height=15,
                                 style={"color": 0xFF606060})
                    
                    simulation_result_label = omni.ui.Label("Click to simulate 4f system from scene")
                    
                    # Plot display area
                    plot_image = omni.ui.Image("", width=400, height=300, fill_policy=omni.ui.FillPolicy.PRESERVE_ASPECT_FIT)
                    plot_image.visible = False  # Hide initially
                    
                    def on_simulate():
                        try:
                            # Get component positions from the scene
                            component_positions = self.get_component_positions()
                            
                            if not component_positions:
                                simulation_result_label.text = "No 4f system found in scene"
                                plot_image.visible = False
                                return
                            
                            # Run the simulation
                            simulation_result_label.text = f"Simulating with {self.selected_resolution}x{self.selected_resolution} field..."
                            result = simulate_4f_system_from_positions(
                                component_positions,
                                wavelength=0.55e-6,
                                focal_length=30,  # 30 cm
                                field_resolution=self.selected_resolution
                            )
                            
                            if result.get("success", False):
                                # Plot and save results
                                plot_path = plot_and_save_results(result)
                                if plot_path:
                                    simulation_result_label.text = f"Simulation complete! Saved to: {Path(plot_path).name}"
                                    # Show the plot in the UI with absolute path
                                    abs_plot_path = Path(plot_path).resolve()
                                    if abs_plot_path.exists():
                                        # Small delay to ensure file is fully written
                                        import time
                                        time.sleep(0.1)
                                        
                                        # Try different path formats for UI compatibility
                                        try:
                                            path_str = str(abs_plot_path).replace("\\", "/")
                                            plot_image.source_url = path_str
                                            plot_image.visible = True
                                            print(f"[mit.test_extensions_1] Loading plot from: {path_str}")
                                            print(f"[mit.test_extensions_1] File size: {abs_plot_path.stat().st_size} bytes")
                                        except Exception as img_error:
                                            print(f"[mit.test_extensions_1] Error loading image: {img_error}")
                                            simulation_result_label.text = f"Image load error: {str(img_error)[:30]}..."
                                            plot_image.visible = False
                                    else:
                                        simulation_result_label.text = "Plot file not found"
                                        plot_image.visible = False
                                        print(f"[mit.test_extensions_1] Plot file not found at: {abs_plot_path}")
                                else:
                                    simulation_result_label.text = "Simulation complete but plotting failed"
                                    plot_image.visible = False
                            else:
                                simulation_result_label.text = f"Simulation failed: {result.get('error', 'Unknown error')[:50]}..."
                                plot_image.visible = False
                                
                        except Exception as e:
                            simulation_result_label.text = f"Error: {str(e)[:50]}..."
                            plot_image.visible = False
                    
                    omni.ui.Button("Simulate", clicked_fn=on_simulate)
                    
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
        """Create a complete 4f optical system with laser, two lenses, and camera."""
        try:
            # Get the USD context
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return
            
            # Clear existing lenses first
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
            
            # Create each component
            for i, component in enumerate(components):
                # Create file path
                file_path = str(Path(project_root) / "CAD" / component["file"])
                
                # Check if file exists
                if not Path(file_path).exists():
                    print(f"[mit.test_extensions_1] ERROR: {component['description']} file not found at {file_path}")
                    continue
                
                # Create unique prim path
                prim_path = f"/World/{component['name']}"
                
                # Create the prim
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=prim_path,
                    prim_type="Xform",
                    select_new_prim=False,
                    create_default_xform=True,
                )
                
                # Add reference to the component file
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    references = prim.GetReferences()
                    references.AddReference(file_path)
                    
                    # Set the position
                    x, y, z = component["position"]
                    self.set_prim_position(prim, x, y, z)
                    
                    print(f"[mit.test_extensions_1] Added {component['description']} at position ({x}, {y}, {z})")
                    
                    # Count lenses for the counter
                    if "Lens" in component["name"]:
                        self.lens_count += 1
                else:
                    print(f"[mit.test_extensions_1] Failed to create prim for {component['description']}")
            
            print("[mit.test_extensions_1] 4f optical system created successfully!")
            
        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating 4f system: {e}")

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
        """Update the position of the currently selected lens."""
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
            
            # Look for the 4f system components
            components = {
                "laser": "/World/LaserSource",
                "lens1": "/World/Lens1", 
                "lens2": "/World/Lens2",
                "camera": "/World/Camera"
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
        """Turn on the laser by instantiating the BEAM_PREFAB."""
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
            
            # Create laser beam prim
            laser_prim_path = "/World/LaserBeam"
            
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
                
                # Set position at center between laser and camera
                self.set_prim_position(laser_prim, 0, 0, beam_z_pos)
                
                # Set scale - unit size in Z, scaled by distance
                scale_attr = laser_prim.GetAttribute("xformOp:scale")
                if not scale_attr:
                    scale_attr = laser_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
                scale_attr.Set((1.0, 1.0, beam_distance))
                
                print(f"[mit.test_extensions_1] Laser beam prefab instantiated at z={beam_z_pos:.1f} with scale={beam_distance:.1f}")
                return True
            else:
                print("[mit.test_extensions_1] Failed to create laser beam prim")
                return False
            
        except Exception as e:
            print(f"[mit.test_extensions_1] Error creating laser: {e}")
            return False

    def turn_off_laser(self):
        """Turn off the laser by removing the cylinder."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False
            
            # Remove laser cylinder
            laser_prim_path = "/World/LaserBeam"
            if stage.GetPrimAtPath(laser_prim_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[laser_prim_path],
                    destructive=True,
                )
                print("[mit.test_extensions_1] Laser beam turned off")
                return True
            else:
                print("[mit.test_extensions_1] No laser beam to turn off")
                return True
                
        except Exception as e:
            print(f"[mit.test_extensions_1] Error turning off laser: {e}")
            return False

    def setup_optical_table(self):
        """Set up the optical table by instantiating TABLE_PREFAB and removing ground/sky."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return False
            
            # Path to the table prefab
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            table_file_path = str(Path(project_root) / "CAD" / "prefabs" / "TABLE_PREFAB.usd")
            
            # Check if file exists
            if not Path(table_file_path).exists():
                print(f"[mit.test_extensions_1] ERROR: Table prefab file not found at {table_file_path}")
                return False
            
            # Create table prim path
            table_prim_path = "/World/OpticalTable"
            
            # Remove existing table if it exists
            if stage.GetPrimAtPath(table_prim_path):
                omni.kit.commands.execute(
                    "DeletePrims",
                    paths=[table_prim_path],
                    destructive=True,
                )
            
            # Create prim and add reference to table prefab
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=table_prim_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )
            
            # Add reference to the table prefab
            table_prim = stage.GetPrimAtPath(table_prim_path)
            if table_prim:
                references = table_prim.GetReferences()
                references.AddReference(table_file_path)
                
                # Set position at (0, 0, 0) as requested
                self.set_prim_position(table_prim, 0, 0, 0)
                
                print("[mit.test_extensions_1] Optical table prefab instantiated at (0, 0, 0)")
            else:
                print("[mit.test_extensions_1] Failed to create optical table prim")
                return False
            
            # Remove ground, ground collider, and sky light
            self.remove_ground_and_sky()
            
            return True
            
        except Exception as e:
            print(f"[mit.test_extensions_1] Error setting up optical table: {e}")
            return False

    def remove_ground_and_sky(self):
        """Remove ground, ground collider, and sky light from the scene."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[mit.test_extensions_1] No stage available")
                return
            
            # Common ground and sky light paths that might exist
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
                "/Environment/sky",
                "/Environment/Sky",
                "/Environment/SkyLight",
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
            
            # Also try to remove any prims with ground/sky related names (but preserve distant light)
            for prim in stage.Traverse():
                prim_name = prim.GetName().lower()
                prim_path = prim.GetPath().pathString
                # Remove ground/sky related prims but preserve distant light
                if any(keyword in prim_name for keyword in ['ground', 'sky', 'sun']):
                    if prim_path != "/World" and prim_path != "/":
                        try:
                            omni.kit.commands.execute(
                                "DeletePrims",
                                paths=[prim_path],
                                destructive=True,
                            )
                            print(f"[mit.test_extensions_1] Removed ground/sky related prim: {prim_path}")
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
            
            print("[mit.test_extensions_1] Ground and sky cleanup completed")
            
        except Exception as e:
            print(f"[mit.test_extensions_1] Error removing ground and sky: {e}")

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[mit.test_extensions_1] Extension shutdown")
