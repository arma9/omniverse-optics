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
import omni.ui as ui
import carb
from pathlib import Path
from pxr import Usd, UsdGeom, Sdf, Gf
from typing import Dict, List, Optional, Tuple
import threading
import time

# Try to import OmniGraph modules - they may not be available
try:
    import omni.graph.core as og
    import omni.kit.selection
    OMNIGRAPH_AVAILABLE = True
    print("[OmniGraph Optics] OmniGraph modules imported successfully")
except ImportError as e:
    OMNIGRAPH_AVAILABLE = False
    print(f"[OmniGraph Optics] OmniGraph modules not available: {e}")
    print("[OmniGraph Optics] Extension will work without OmniGraph integration")

class ComponentBinding:
    """Manages binding between USD scene objects and OmniGraph nodes (if available)."""
    
    def __init__(self, usd_prim_path: str, component_type: str, graph_node_path: Optional[str] = None):
        self.usd_prim_path = usd_prim_path
        self.component_type = component_type
        self.graph_node_path = graph_node_path
        self.last_position = None
        self._is_valid = True
    
    def is_valid(self) -> bool:
        """Check if the binding is still valid."""
        if not self._is_valid:
            return False
            
        # Check USD prim
        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath(self.usd_prim_path):
            self._is_valid = False
            return False
            
        # Check OmniGraph node if available
        if OMNIGRAPH_AVAILABLE and self.graph_node_path:
            try:
                node = og.get_node_by_path(self.graph_node_path)
                if not node.is_valid():
                    self._is_valid = False
                    return False
            except:
                self._is_valid = False
                return False
                
        return True

class OpticalSystemExtension(omni.ext.IExt):
    """Extension for creating and managing optical systems in OmniGraph."""
    
    def on_startup(self, ext_id):
        """Initialize the extension."""
        print("[OmniGraph Optics] Extension startup")
        
        self._ext_id = ext_id
        self._bindings: Dict[str, ComponentBinding] = {}
        self._component_counter = 0
        self._graph_path = "/World/OpticalSystemGraph"
        self._graph = None
        
        # Create the main UI window
        self._create_ui()
        
        # Create the optical system graph
        self._create_optical_system_graph()
        
        print("[OmniGraph Optics] Extension initialized")
    
    def on_shutdown(self):
        """Clean up the extension."""
        print("[OmniGraph Optics] Extension shutdown")
        
        # Clean up UI
        if hasattr(self, '_window') and self._window is not None:
            self._window.destroy()
            self._window = None
        
        # Clean up bindings
        self._bindings.clear()
        
        print("[OmniGraph Optics] Extension cleanup complete")
    
    def _create_optical_system_graph(self):
        """Create the main optical system graph."""
        if not OMNIGRAPH_AVAILABLE:
            print("[OmniGraph Optics] OmniGraph not available, skipping graph creation")
            return
            
        try:
            # Check if graph already exists
            try:
                existing_graph = og.get_graph_by_path(self._graph_path)
                if existing_graph and existing_graph.is_valid():
                    self._graph = existing_graph
                    print(f"[OmniGraph Optics] Using existing optical system graph at {self._graph_path}")
                    return
            except:
                pass  # Graph doesn't exist, continue with creation
            
            # Create the graph
            (self._graph, _, _, _) = og.Controller.create_graph({
                "graph_path": self._graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            })
            
            if self._graph and self._graph.is_valid():
                print(f"[OmniGraph Optics] Created optical system graph at {self._graph_path}")
            else:
                print("[OmniGraph Optics] Failed to create optical system graph")
                self._graph = None
                
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating optical system graph: {e}")
            self._graph = None
    
    def _create_ui(self):
        """Create the main UI window."""
        self._window = ui.Window("Optical System", width=300, height=500)
        
        with self._window.frame:
            with ui.VStack(spacing=10):
                # Title
                ui.Label("Optical Component Creator", height=30, style={"font_size": 18})
                
                # Individual component buttons
                ui.Label("Individual Components:", height=20)
                
                with ui.HStack(spacing=5):
                    ui.Button("Laser", clicked_fn=self._create_laser, width=70)
                    ui.Button("Lens", clicked_fn=self._create_lens, width=70)
                    ui.Button("Camera", clicked_fn=self._create_camera, width=70)
                
                with ui.HStack(spacing=5):
                    ui.Button("Mirror", clicked_fn=self._create_mirror, width=70)
                    ui.Button("Beam Splitter", clicked_fn=self._create_beam_splitter, width=100)
                
                # Separator
                ui.Separator()
                
                # Interferometer presets
                ui.Label("Interferometer Presets:", height=20)
                
                ui.Button("Michelson Interferometer", clicked_fn=self._create_michelson_interferometer)
                ui.Button("Mach-Zehnder Interferometer", clicked_fn=self._create_mach_zehnder_interferometer)
                ui.Button("Fabry-Perot Cavity", clicked_fn=self._create_fabry_perot_cavity)
                
                # Separator
                ui.Separator()
                
                # Scene Setup
                ui.Label("Scene Setup:", height=20)
                ui.Button("Set Up Optical Table", clicked_fn=self._setup_optical_table)
                
                # Separator
                ui.Separator()
                
                # Control buttons
                ui.Label("Controls:", height=20)
                ui.Button("Clear All Components", clicked_fn=self._clear_all_components)
                ui.Button("Refresh Graph", clicked_fn=self._refresh_graph)
                
                # Separator
                ui.Separator()
                
                # Status section
                ui.Label("Components Status:", height=20)
                
                # Create a scrollable frame for components
                with ui.ScrollingFrame(height=150):
                    self._components_vstack = ui.VStack(spacing=2)
                    with self._components_vstack:
                        ui.Label("No components created", height=20, style={"color": 0xFF666666})
                
                # Graph status
                ui.Separator()
                graph_status = "✓ Available" if OMNIGRAPH_AVAILABLE else "✗ Not Available"
                ui.Label(f"OmniGraph Status: {graph_status}", height=20)
                
                if OMNIGRAPH_AVAILABLE:
                    ui.Label("Components will appear in Action Graph window", height=20, style={"color": 0xFF00AA00})
                else:
                    ui.Label("Install OmniGraph for graph visualization", height=20, style={"color": 0xFFAA0000})
    
    def _create_laser(self):
        """Create a laser component."""
        self._create_component("Laser", "prefabs/LASER_PREFAB.usd", (0, 0, 0))
    
    def _create_lens(self):
        """Create a lens component."""
        self._create_component("Lens", "prefabs/LENS_PREFAB.usd", (0, 0, 50))
    
    def _create_camera(self):
        """Create a camera component."""
        self._create_component("Camera", "prefabs/CAMERA_PREFAB.usd", (0, 0, 100))
    
    def _create_mirror(self):
        """Create a mirror component."""
        self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (50, 0, 50))
    
    def _create_beam_splitter(self):
        """Create a beam splitter component."""
        self._create_component("BeamSplitter", "prefabs/BEAM_SPLITTER_PREFAB.usd", (0, 0, 50))
    
    def _create_component(self, component_type: str, prefab_file: str, default_position: Tuple[float, float, float]):
        """Create a component in USD scene and OmniGraph."""
        try:
            self._component_counter += 1
            component_name = f"{component_type}_{self._component_counter}"
            
            # Create USD component
            usd_prim_path = self._create_usd_component(component_name, prefab_file, default_position)
            if not usd_prim_path:
                return
            
            # Create OmniGraph node
            graph_node_path = None
            if OMNIGRAPH_AVAILABLE and self._graph:
                graph_node_path = self._create_graph_node(component_name, component_type, default_position)
            
            # Create binding
            binding = ComponentBinding(usd_prim_path, component_type, graph_node_path)
            self._bindings[component_name] = binding
            
            # Update UI
            self._refresh_status_ui()
            
            print(f"[OmniGraph Optics] Created {component_type}: {component_name}")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating {component_type}: {e}")
    
    def _create_graph_node(self, component_name: str, component_type: str, position: Tuple[float, float, float]) -> Optional[str]:
        """Create an OmniGraph node for the optical component."""
        if not OMNIGRAPH_AVAILABLE or not self._graph or not self._graph.is_valid():
            return None
            
        try:
            # Create a constant node to represent the optical component
            node_path = f"{component_name}"
            
            # Create the node in the graph
            node = og.Controller.create_node(
                self._graph,
                "omni.graph.nodes.ConstantDouble3",  # Use a constant node to represent position
                name=node_path
            )
            
            if node and node.is_valid():
                # Set some metadata for the node
                try:
                    og.Controller.set_node_attribute_value(node, "inputs:value", position)
                except:
                    pass  # Ignore if setting fails
                
                # You can add more attributes here based on component type
                full_path = f"{self._graph_path}/{node_path}"
                print(f"[OmniGraph Optics] Created graph node: {full_path}")
                return full_path
            else:
                print(f"[OmniGraph Optics] Failed to create graph node: {node_path}")
                return None
                
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating graph node: {e}")
            return None
    
    def _create_usd_component(self, component_name: str, prefab_file: str, position: Tuple[float, float, float]) -> Optional[str]:
        """Create USD component in the scene."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[OmniGraph Optics] No stage available")
                return None
            
            # Get prefab path
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            prefab_path = str(Path(project_root) / "CAD" / prefab_file)
            
            if not Path(prefab_path).exists():
                print(f"[OmniGraph Optics] Prefab file not found: {prefab_path}")
                return None
            
            # Create prim path
            prim_path = f"/World/OpticalComponents/{component_name}"
            
            # Create prim
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=prim_path,
                prim_type="Xform",
                select_new_prim=True,
                create_default_xform=True,
            )
            
            # Add reference
            prim = stage.GetPrimAtPath(prim_path)
            if prim:
                references = prim.GetReferences()
                references.AddReference(prefab_path)
                
                # Set position
                self._set_prim_position(prim, *position)
                
                return prim_path
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating USD component: {e}")
            return None
    
    def _set_prim_position(self, prim, x: float, y: float, z: float):
        """Set position of USD prim."""
        try:
            translate_attr = prim.GetAttribute("xformOp:translate")
            if not translate_attr:
                translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
                
                order_attr = prim.GetAttribute("xformOpOrder")
                if not order_attr:
                    order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    order_attr.Set(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
            
            translate_attr.Set((x, y, z))
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error setting prim position: {e}")
    
    def _refresh_graph(self):
        """Refresh the OmniGraph."""
        if not OMNIGRAPH_AVAILABLE:
            print("[OmniGraph Optics] OmniGraph not available")
            return
            
        try:
            # Recreate the graph if it doesn't exist
            if not self._graph:
                self._create_optical_system_graph()
            
            # Connect existing components
            self._connect_components()
            
            print("[OmniGraph Optics] Graph refreshed")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error refreshing graph: {e}")
    
    def _connect_components(self):
        """Connect components in the graph based on their spatial arrangement."""
        if not OMNIGRAPH_AVAILABLE or not self._graph:
            return
            
        try:
            # Simple connection logic: connect components in order of creation
            component_list = list(self._bindings.values())
            
            for i in range(len(component_list) - 1):
                current = component_list[i]
                next_component = component_list[i + 1]
                
                if current.graph_node_path and next_component.graph_node_path:
                    # Create a connection between nodes
                    # This is a simplified example - in practice, you'd connect specific attributes
                    print(f"[OmniGraph Optics] Connecting {current.graph_node_path} to {next_component.graph_node_path}")
                    
        except Exception as e:
            print(f"[OmniGraph Optics] Error connecting components: {e}")
    
    def _create_michelson_interferometer(self):
        """Create a Michelson interferometer setup in 2D (x-z plane)."""
        try:
            # Clear existing components
            self._clear_all_components()
            
            # Create components for Michelson interferometer in 2D (x-z plane)
            self._create_component("Laser", "prefabs/LASER_PREFAB.usd", (0, 0, 0))
            self._create_component("BeamSplitter", "prefabs/BEAM_SPLITTER_PREFAB.usd", (0, 0, 50))
            self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (100, 0, 50))
            self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (0, 0, 100))
            self._create_component("Camera", "prefabs/CAMERA_PREFAB.usd", (-50, 0, 50))
            
            print("[OmniGraph Optics] Created Michelson interferometer in 2D")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating Michelson interferometer: {e}")
    
    def _create_mach_zehnder_interferometer(self):
        """Create a Mach-Zehnder interferometer setup in 2D (x-z plane)."""
        try:
            # Clear existing components
            self._clear_all_components()
            
            # Create components for Mach-Zehnder interferometer in 2D (x-z plane)
            self._create_component("Laser", "prefabs/LASER_PREFAB.usd", (0, 0, 0))
            self._create_component("BeamSplitter", "prefabs/BEAM_SPLITTER_PREFAB.usd", (0, 0, 50))
            self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (50, 0, 50))
            self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (50, 0, 0))
            self._create_component("BeamSplitter", "prefabs/BEAM_SPLITTER_PREFAB.usd", (100, 0, 50))
            self._create_component("Camera", "prefabs/CAMERA_PREFAB.usd", (150, 0, 50))
            
            print("[OmniGraph Optics] Created Mach-Zehnder interferometer in 2D")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating Mach-Zehnder interferometer: {e}")
    
    def _create_fabry_perot_cavity(self):
        """Create a Fabry-Perot cavity setup in 2D (x-z plane)."""
        try:
            # Clear existing components
            self._clear_all_components()
            
            # Create components for Fabry-Perot cavity in 2D (x-z plane)
            self._create_component("Laser", "prefabs/LASER_PREFAB.usd", (0, 0, 0))
            self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (0, 0, 50))
            self._create_component("Mirror", "prefabs/MIRROR_45_PREFAB.usd", (0, 0, 100))
            self._create_component("Camera", "prefabs/CAMERA_PREFAB.usd", (0, 0, 150))
            
            print("[OmniGraph Optics] Created Fabry-Perot cavity in 2D")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error creating Fabry-Perot cavity: {e}")
    
    def _clear_all_components(self):
        """Clear all components from scene."""
        try:
            # Clear USD components
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if stage:
                # Remove all optical components
                components_prim = stage.GetPrimAtPath("/World/OpticalComponents")
                if components_prim:
                    omni.kit.commands.execute(
                        "DeletePrims",
                        paths=["/World/OpticalComponents"],
                        destructive=True,
                    )
            
            # Clear OmniGraph
            if OMNIGRAPH_AVAILABLE and self._graph:
                try:
                    og.Controller.delete_graph(self._graph)
                    self._graph = None
                    print("[OmniGraph Optics] Cleared OmniGraph")
                except Exception as e:
                    print(f"[OmniGraph Optics] Error clearing graph: {e}")
            
            # Clear bindings
            self._bindings.clear()
            self._component_counter = 0
            
            # Recreate the graph
            self._create_optical_system_graph()
            
            # Update UI
            self._refresh_status_ui()
            
            print("[OmniGraph Optics] Cleared all components")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error clearing components: {e}")
    
    def _refresh_status_ui(self):
        """Refresh the status UI display."""
        try:
            if not hasattr(self, '_components_vstack'):
                return
                
            # Clear existing UI
            self._components_vstack.clear()
            
            # Add component entries
            with self._components_vstack:
                for binding_id, binding in self._bindings.items():
                    status = "✓" if binding.is_valid() else "✗"
                    graph_status = "🔗" if binding.graph_node_path else "❌"
                    ui.Label(f"{status} {graph_status} {binding_id} ({binding.component_type})", height=20)
                    
                if not self._bindings:
                    ui.Label("No components created", height=20, style={"color": 0xFF666666})
                    
        except Exception as e:
            print(f"[OmniGraph Optics] Error refreshing status UI: {e}")
    
    def _setup_optical_table(self):
        """Set up the optical table by instantiating TABLE_PREFAB and removing ground/sky."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[OmniGraph Optics] No stage available")
                return False
            
            # Path to the table prefab
            project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
            table_file_path = str(Path(project_root) / "CAD" / "prefabs" / "TABLE_PREFAB.usd")
            
            # Check if file exists
            if not Path(table_file_path).exists():
                print(f"[OmniGraph Optics] ERROR: Table prefab file not found at {table_file_path}")
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
                self._set_prim_position(table_prim, 0, 0, 0)
                
                print("[OmniGraph Optics] Optical table prefab instantiated at (0, 0, 0)")
            else:
                print("[OmniGraph Optics] Failed to create optical table prim")
                return False
            
            # Remove ground, ground collider, and sky light
            self._remove_ground_and_sky()
            
            return True
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error setting up optical table: {e}")
            return False

    def _remove_ground_and_sky(self):
        """Remove ground, ground collider, and sky light from the scene."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            
            if not stage:
                print("[OmniGraph Optics] No stage available")
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
                    print(f"[OmniGraph Optics] Removed: {path}")
            
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
                            print(f"[OmniGraph Optics] Removed ground/sky related prim: {prim_path}")
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
                            print(f"[OmniGraph Optics] Removed sky light: {prim_path}")
                        except:
                            pass  # Continue even if deletion fails
            
            print("[OmniGraph Optics] Ground and sky cleanup completed")
            
        except Exception as e:
            print(f"[OmniGraph Optics] Error removing ground and sky: {e}")

# Extension implementation
class MyExtension(omni.ext.IExt):
    """Main extension class that Kit will instantiate."""
    
    def on_startup(self, ext_id):
        """Initialize the extension."""
        self._extension_instance = OpticalSystemExtension()
        self._extension_instance.on_startup(ext_id)
    
    def on_shutdown(self):
        """Clean up the extension."""
        if hasattr(self, '_extension_instance') and self._extension_instance is not None:
            self._extension_instance.on_shutdown()
            self._extension_instance = None
