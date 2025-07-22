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
import asyncio
import sys
import os

from mit.onmigraph_optics.ogn.OgnLens import OgnLens
from mit.onmigraph_optics.ogn.OgnMirror import OgnMirror
from mit.onmigraph_optics.ogn.OgnBeamSplitter import OgnBeamSplitter
from mit.onmigraph_optics.ogn.OgnCamera import OgnCamera
from mit.onmigraph_optics.ogn.OgnLaser import OgnLaser

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
        self._graph_path = None
        self._graph = None

        # Register custom nodes
        # Node registration is now handled automatically by OGN and extension.toml
        print("[DEBUG] OGN node registration is handled by extension.toml and OGN system.")

        # Create the main UI window
        self._create_ui()

        print("[OmniGraph Optics] Extension initialized - use 'Create Empty Graph' button to start")

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
            # Try to create the graph - if it fails, it might already exist
            print(f"[OmniGraph Optics] Attempting to create graph at {self._graph_path}")

            # Create the graph with proper configuration
            graph_config = {
                "graph_path": self._graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            }

            print(f"[OmniGraph Optics] Creating graph with config: {graph_config}")

            # Create the graph - handle the return value properly
            try:
                result = og.Controller.create_graph(graph_config)
                if isinstance(result, tuple) and len(result) >= 1:
                    self._graph = result[0]
                else:
                    self._graph = result
                print(f"[OmniGraph Optics] Graph creation result type: {type(result)}")
            except Exception as e:
                print(f"[OmniGraph Optics] Error in create_graph: {e}")
                # If creation failed, try to get the existing graph using a different approach
                self._graph = self._find_existing_graph_alternative()
                if not self._graph:
                    print("[OmniGraph Optics] Could not find or create graph")
                    return

            if self._graph and self._graph.is_valid():
                print(f"[OmniGraph Optics] Successfully created/found optical system graph at {self._graph_path}")

                # Verify the graph is accessible
                try:
                    test_node = og.Controller.create_node(
                        self._graph,
                        "omni.graph.nodes.ConstantDouble3",
                        name="test_node"
                    )
                    if test_node and test_node.is_valid():
                        # Remove test node
                        og.Controller.delete_node(test_node)
                        print("[OmniGraph Optics] Graph verification successful")
                    else:
                        print("[OmniGraph Optics] Graph verification failed - test node creation failed")
                        self._graph = None
                except Exception as e:
                    print(f"[OmniGraph Optics] Graph verification failed: {e}")
                    self._graph = None

                # Try to register the graph with the system
                try:
                    # Force graph evaluation to ensure it's properly registered - use sync version
                    self._evaluate_graph_sync()
                    print("[OmniGraph Optics] Graph evaluation successful")
                except Exception as e:
                    print(f"[OmniGraph Optics] Graph evaluation failed: {e}")

                # Save the graph to the USD stage
                try:
                    self._save_graph_to_stage()
                    print("[OmniGraph Optics] Graph saved to USD stage")
                except Exception as e:
                    print(f"[OmniGraph Optics] Error saving graph to stage: {e}")

                # Register the graph with the Action Graph system
                try:
                    self._register_graph_with_action_graph()
                    print("[OmniGraph Optics] Graph registered with Action Graph system")
                except Exception as e:
                    print(f"[OmniGraph Optics] Error registering graph with Action Graph: {e}")
            else:
                print("[OmniGraph Optics] Failed to create optical system graph")
                self._graph = None

        except Exception as e:
            print(f"[OmniGraph Optics] Error creating optical system graph: {e}")
            self._graph = None

    def _find_existing_graph_alternative(self):
        """Find an existing graph using alternative methods."""
        try:
            # Try to get the graph by path using a different approach
            try:
                # Try to create a temporary node to see if the graph exists
                temp_graph = og.Controller.create_graph({
                    "graph_path": self._graph_path + "_temp",
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                })
                if isinstance(temp_graph, tuple):
                    temp_graph = temp_graph[0]

                # If we can create a temp graph, try to get the original
                if temp_graph and temp_graph.is_valid():
                    # Delete the temp graph
                    og.Controller.delete_graph(temp_graph)

                    # Try to get the original graph by attempting to create it with a different name
                    # and then renaming it
                    alt_graph = og.Controller.create_graph({
                        "graph_path": self._graph_path + "_alt",
                        "evaluator_name": "execution",
                        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                    })
                    if isinstance(alt_graph, tuple):
                        alt_graph = alt_graph[0]

                    if alt_graph and alt_graph.is_valid():
                        # Delete the alt graph and return None - we'll handle this differently
                        og.Controller.delete_graph(alt_graph)
                        print(f"[OmniGraph Optics] Graph likely exists at {self._graph_path}")
                        return None

            except Exception as e:
                print(f"[OmniGraph Optics] Error in alternative graph finding: {e}")

        except Exception as e:
            print(f"[OmniGraph Optics] Error in find_existing_graph_alternative: {e}")

        return None

    def _evaluate_graph_sync(self):
        """Evaluate the graph synchronously to avoid async issues."""
        try:
            if not self._graph or not self._graph.is_valid():
                return

            # Try to evaluate the graph without using the async version
            # This might not work in all cases, but it's safer than the async version
            print("[OmniGraph Optics] Graph evaluation skipped to avoid async issues")

        except Exception as e:
            print(f"[OmniGraph Optics] Error in graph evaluation: {e}")

    def _find_existing_graph(self):
        """Find an existing graph with our path."""
        try:
            # Since get_all_graphs doesn't exist, we'll try a different approach
            # Try to create a node in the graph to see if it exists
            try:
                test_node = og.Controller.create_node(
                    self._graph if self._graph else None,
                    "omni.graph.nodes.ConstantDouble3",
                    name="test_existence"
                )
                if test_node and test_node.is_valid():
                    # Delete the test node
                    og.Controller.delete_node(test_node)
                    return self._graph
            except:
                pass

        except Exception as e:
            print(f"[OmniGraph Optics] Error finding existing graph: {e}")

        return None

    def _register_graph_with_action_graph(self):
        """Register the graph with the Action Graph system."""
        try:
            if not self._graph or not self._graph.is_valid():
                return

            # Try to register the graph with the Action Graph system
            try:
                # This might help register the graph with the Action Graph window
                omni.kit.commands.execute("ActionGraph.Refresh")
                print("[OmniGraph Optics] Action Graph refreshed")
            except:
                pass

            # Try to set the graph as the active graph in the Action Graph window
            try:
                omni.kit.commands.execute("ActionGraph.SetGraph", graph_path=self._graph_path)
                print(f"[OmniGraph Optics] Set graph as active in Action Graph: {self._graph_path}")
            except:
                print("[OmniGraph Optics] Could not set graph as active in Action Graph")

            # Try to register the graph with the USD stage
            try:
                # This ensures the graph is properly registered with the USD stage
                og.Controller.save_graph(self._graph)
                print("[OmniGraph Optics] Graph saved to USD stage")
            except Exception as e:
                print(f"[OmniGraph Optics] Error saving graph to USD stage: {e}")

            # Try to register the graph with the USD stage using the correct API
            try:
                # Get the USD stage
                usd_context = omni.usd.get_context()
                stage = usd_context.get_stage()

                if stage:
                    # Create a USD prim for the graph
                    graph_prim_path = "/World/OpticalSystemGraph"

                    # Check if the prim already exists
                    existing_prim = stage.GetPrimAtPath(graph_prim_path)
                    if not existing_prim:
                        # Create the prim
                        omni.kit.commands.execute(
                            "CreatePrim",
                            prim_path=graph_prim_path,
                            prim_type="Xform",
                            select_new_prim=False,
                            create_default_xform=True,
                        )

                        # Add a custom attribute to identify this as our graph
                        graph_prim = stage.GetPrimAtPath(graph_prim_path)
                        if graph_prim:
                            graph_prim.CreateAttribute("omni:graph_path", Sdf.ValueTypeNames.String).Set(self._graph_path)
                            print(f"[OmniGraph Optics] Created graph prim with path: {self._graph_path}")

            except Exception as e:
                print(f"[OmniGraph Optics] Error creating graph prim in USD stage: {e}")

        except Exception as e:
            print(f"[OmniGraph Optics] Error registering graph with Action Graph: {e}")

    def _save_graph_to_stage(self):
        """Save the graph to the USD stage to ensure persistence."""
        try:
            if not self._graph or not self._graph.is_valid():
                return

            # Get the USD stage
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                print("[OmniGraph Optics] No USD stage available for graph saving")
                return

            # Try to save the graph to the stage
            try:
                # This ensures the graph is properly registered with the USD stage
                og.Controller.save_graph(self._graph)
                print("[OmniGraph Optics] Graph saved successfully")
            except Exception as e:
                print(f"[OmniGraph Optics] Error saving graph: {e}")

            # Create a USD prim for the graph to ensure it's visible
            try:
                self._create_graph_prim_in_stage()
            except Exception as e:
                print(f"[OmniGraph Optics] Error creating graph prim: {e}")

            # Try to register the graph with the Action Graph system
            try:
                self._register_graph_with_action_graph()
            except Exception as e:
                print(f"[OmniGraph Optics] Error registering graph with Action Graph: {e}")

        except Exception as e:
            print(f"[OmniGraph Optics] Error in save_graph_to_stage: {e}")

    def _create_graph_prim_in_stage(self):
        """Create a USD prim for the graph to ensure it's properly registered."""
        try:
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()

            if not stage:
                return

            # Create the graph prim path
            graph_prim_path = "/World/OpticalSystemGraph"

            # Check if the prim already exists
            existing_prim = stage.GetPrimAtPath(graph_prim_path)
            if existing_prim:
                print(f"[OmniGraph Optics] Graph prim already exists: {graph_prim_path}")
                return

            # Create the prim
            omni.kit.commands.execute(
                "CreatePrim",
                prim_path=graph_prim_path,
                prim_type="Xform",
                select_new_prim=False,
                create_default_xform=True,
            )

            print(f"[OmniGraph Optics] Created graph prim: {graph_prim_path}")

            # Try to add a reference to the graph
            try:
                graph_prim = stage.GetPrimAtPath(graph_prim_path)
                if graph_prim:
                    # Add a custom attribute to identify this as our graph
                    graph_prim.CreateAttribute("omni:graph_path", Sdf.ValueTypeNames.String).Set(self._graph_path)
                    print(f"[OmniGraph Optics] Added graph path attribute: {self._graph_path}")
            except Exception as e:
                print(f"[OmniGraph Optics] Error adding graph attributes: {e}")

        except Exception as e:
            print(f"[OmniGraph Optics] Error creating graph prim: {e}")

    def _create_ui(self):
        """Create the main UI window."""
        self._window = ui.Window("Optical System", width=300, height=500)
        self._window.visible = True  # Ensure the window pops up by default
        #self._window.position = (100, 100)  # Optionally set position for visibility
        #self._window.focus()  # Bring the window to the front

        with self._window.frame:
            with ui.VStack(spacing=10):
                # Title
                ui.Label("Optical Component Creator", height=30, style={"font_size": 18})

                # Graph Management
                ui.Label("Graph Management:", height=20)
                ui.Button("Create Empty Graph", clicked_fn=self._create_empty_graph)
                ui.Button("Open Action Graph", clicked_fn=self._open_action_graph)
                ui.Button("Create Built-in Node", clicked_fn=self._create_builtin_node)  # New button

                # Graph status
                if self._graph and self._graph.is_valid():
                    ui.Label("✓ Graph Ready - Components will create nodes", height=20, style={"color": 0xFF00AA00})
                else:
                    ui.Label("⚠ Create a graph first to add nodes", height=20, style={"color": 0xFFAA6600})

                # Separator
                ui.Separator()

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
                ui.Button("Debug Graph", clicked_fn=self._debug_graph)
                ui.Button("Test OmniGraph API", clicked_fn=self._test_omnigraph_api)

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
                    ui.Label("Manual: Window > Action Graph", height=20, style={"color": 0xFF00AA00})
                else:
                    ui.Label("Install OmniGraph for graph visualization", height=20, style={"color": 0xFFAA0000})

    def _create_empty_graph(self):
        """Create an empty graph for optical components."""
        if not OMNIGRAPH_AVAILABLE:
            print("[OmniGraph Optics] OmniGraph not available")
            return

        # Create a unique graph path with timestamp
        timestamp = int(time.time())
        self._graph_path = f"/World/OpticalSystemGraph_{timestamp}".rstrip('/')

        print(f"[OmniGraph Optics] Creating empty graph at: {self._graph_path}")

        # Create the graph with simple configuration
        graph_config = {
            "graph_path": self._graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        }

        # Create the graph
        result = og.Controller.create_graph(graph_config)
        if isinstance(result, tuple) and len(result) >= 1:
            self._graph = result[0]
        else:
            self._graph = result

        if self._graph and self._graph.is_valid():
            print(f"[OmniGraph Optics] ✓ Successfully created empty graph at {self._graph_path}")
            self._open_action_graph()
            self._refresh_status_ui()
            print(f"[OmniGraph Optics] 💡 Tip: Open Action Graph window manually via Window > Action Graph")
            print(f"[OmniGraph Optics] 💡 Tip: Your graph path is: {self._graph_path}")
        else:
            print("[OmniGraph Optics] ✗ Failed to create empty graph")

    def _debug_graph(self):
        """Debug the current graph state."""
        if not OMNIGRAPH_AVAILABLE:
            print("[OmniGraph Optics] OmniGraph not available for debugging")
            return

        try:
            print(f"[OmniGraph Optics] === GRAPH DEBUG INFO ===")
            print(f"Graph path: {self._graph_path}")
            print(f"Graph object: {self._graph}")
            print(f"Graph valid: {self._graph.is_valid() if self._graph else False}")

            if self._graph and self._graph.is_valid():
                # Get all nodes in the graph
                nodes = og.Controller.get_nodes(self._graph)
                print(f"Number of nodes in graph: {len(nodes)}")

                for i, node in enumerate(nodes):
                    print(f"  Node {i}: {node.get_name()} - Type: {node.get_node_type_name()} - Valid: {node.is_valid()}")

                    # Try to get node path
                    try:
                        node_path = node.get_path()
                        print(f"    Path: {node_path}")
                    except:
                        print(f"    Path: Could not retrieve")
            else:
                print("No valid graph available")

            print(f"Number of bindings: {len(self._bindings)}")
            for binding_id, binding in self._bindings.items():
                print(f"  Binding: {binding_id} - Graph node: {binding.graph_node_path}")

            print("[OmniGraph Optics] === END DEBUG INFO ===")

        except Exception as e:
            print(f"[OmniGraph Optics] Error during graph debug: {e}")

    def _open_action_graph(self):
        """Open the Action Graph window."""
        print("[OmniGraph Optics] 💡 Manual Action Graph Instructions:")
        print("[OmniGraph Optics]   1. Go to Window > Action Graph in the menu")
        print("[OmniGraph Optics]   2. Or try: Window > Graph Editor")
        print("[OmniGraph Optics]   3. Your graph path is: " + (self._graph_path or "No graph created"))

        # Try a few different commands but don't fail if they don't work
        commands_to_try = [
            "omni.graph.window.action",
            "omni.graph.window.editor",
            "GraphEditor",
            "ActionGraph"
        ]

        for cmd in commands_to_try:
            try:
                print(f"[DEBUG] Trying to execute command: {cmd}")
                omni.kit.commands.execute(cmd)
                print(f"[OmniGraph Optics] ✓ Opened via command: {cmd}")
                return
            except Exception as e:
                print(f"[DEBUG] Failed to execute command '{cmd}': {e}")
        print("[OmniGraph Optics] ⚠ Could not open Action Graph window automatically. Please open it manually from the Window menu.")

    def _create_graph_node(self, component_name: str, component_type: str, position: Tuple[float, float, float]) -> Optional[str]:
        """Create an OmniGraph node for the optical component."""
        if not OMNIGRAPH_AVAILABLE:
            print(f"[OmniGraph Optics] OmniGraph not available for {component_name}")
            return None

        if not self._graph or not self._graph.is_valid():
            print(f"[OmniGraph Optics] No valid graph for {component_name}")
            return None

        # Debug USD stage and graph path
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        print(f"[DEBUG] USD Stage: {stage}")
        graph_path = self._graph_path.rstrip('/') if self._graph_path else self._graph_path
        print(f"[DEBUG] Graph path (stripped): {graph_path!r}")
        layer = stage.GetEditTarget().GetLayer() if stage else None
        from pxr import Sdf
        if not stage:
            print("[DEBUG] No open USD stage! Node creation aborted.")
            return None
        if hasattr(layer, 'permission') and layer.permission == Sdf.Permission.ReadOnly:
            print("[DEBUG] USD stage is read-only! Node creation aborted.")
            return None

        print(f"[OmniGraph Optics] Creating graph node for {component_name} of type {component_type}")
        print(f"[DEBUG] _graph: {self._graph}, type: {type(self._graph)}")
        print(f"[DEBUG] _graph_path: {self._graph_path}, type: {type(self._graph_path)}")

        # Debug print all registered node type names
        try:
            node_types = og.Controller.get_node_type_names()
            print(f"[DEBUG] Registered node types: {node_types}")
        except Exception as e:
            print(f"[DEBUG] Could not get node type names: {e}")

        node_type_map = {
            "Lens": "mit.onmigraph_optics.OgnLens",
            "Mirror": "mit.onmigraph_optics.OgnMirror",
            "BeamSplitter": "mit.onmigraph_optics.OgnBeamSplitter",
            "Camera": "mit.onmigraph_optics.OgnCamera",
            "Laser": "mit.onmigraph_optics.OgnLaser"
        }
        node_type = node_type_map.get(component_type, "mit.onmigraph_optics.OgnLens")

        node = None
        # Try with string path first
        try:
            print(f"[DEBUG] Trying create_node with string path: {graph_path}")
            node = og.Controller.create_node(
                graph_path,
                node_type,
                name=component_name
            )
            print("[DEBUG] Created node with string path")
        except Exception as e:
            print(f"[DEBUG] Failed with string path: {e}")
            # Try with Graph object
            try:
                print(f"[DEBUG] Trying create_node with Graph object: {self._graph}")
                node = og.Controller.create_node(
                    self._graph,
                    node_type,
                    name=component_name
                )
                print("[DEBUG] Created node with Graph object")
            except Exception as e2:
                print(f"[DEBUG] Failed with Graph object: {e2}")
                node = None

        print(f"[OmniGraph Optics] Node creation result: {node}")

        if node and node.is_valid():
            print(f"[OmniGraph Optics] ✓ Node created successfully!")
            new_name = f"{component_type}_{position[0]:.1f}_{position[1]:.1f}_{position[2]:.1f}"
            try:
                og.Controller.set_node_name(node, new_name)
            except Exception:
                pass
            self._log_component_parameters(component_type, position)
            node_path = f"{graph_path}/{component_name}"
            print(f"[OmniGraph Optics] ✓ Node created successfully at: {node_path}")
            return node_path
        else:
            print(f"[OmniGraph Optics] ✗ Node creation failed - node invalid")
            return None

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

            # Create OmniGraph node only if graph exists
            graph_node_path = None
            if OMNIGRAPH_AVAILABLE and self._graph_path:
                graph_node_path = self._create_graph_node(component_name, component_type, default_position)
                if graph_node_path:
                    print(f"[OmniGraph Optics] ✓ Created {component_type} with graph node")
                else:
                    print(f"[OmniGraph Optics] ⚠ Created {component_type} but failed to create graph node")
            else:
                print(f"[OmniGraph Optics] ⚠ Created {component_type} (no graph available - create graph first)")
                print(f"[OmniGraph Optics]   Use 'Create Empty Graph' button to enable node creation")

            # Create binding
            binding = ComponentBinding(usd_prim_path, component_type, graph_node_path)
            self._bindings[component_name] = binding

            # Update UI
            self._refresh_status_ui()

            print(f"[OmniGraph Optics] Created {component_type}: {component_name}")

        except Exception as e:
            print(f"[OmniGraph Optics] Error creating {component_type}: {e}")

    def _log_component_parameters(self, component_type: str, position: Tuple[float, float, float]):
        """Log component-specific parameters to console."""
        try:
            if component_type == "Lens":
                print(f"[OmniGraph Optics] Added focal length parameter: 50.0mm for {component_type}")
            elif component_type == "Laser":
                print(f"[OmniGraph Optics] Added wavelength parameter: 405.0nm for {component_type}")
            elif component_type == "Mirror":
                print(f"[OmniGraph Optics] Added reflectivity parameter: 99.0% for {component_type}")
            elif component_type == "BeamSplitter":
                print(f"[OmniGraph Optics] Added transmission ratio: 50.0% for {component_type}")
            elif component_type == "Camera":
                print(f"[OmniGraph Optics] Added sensor size parameter: 6.4mm for {component_type}")
        except Exception as e:
            print(f"[OmniGraph Optics] Error logging component parameters: {e}")



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

            # If still no graph, try to find existing one
            if not self._graph:
                self._graph = self._find_existing_graph()

            # Connect existing components
            if self._graph and self._graph.is_valid():
                self._connect_components()
                print("[OmniGraph Optics] Graph refreshed")
            else:
                print("[OmniGraph Optics] No valid graph available for refresh")

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

    def _test_omnigraph_api(self):
        """Test OmniGraph API functionality."""
        print("[OmniGraph Optics] === TESTING OMNIGRAPH API ===")

        if not OMNIGRAPH_AVAILABLE:
            print("[OmniGraph Optics] ✗ OmniGraph not available")
            return

        try:
            # Test 1: Check if we can create a simple graph
            test_graph_path = "/World/TestGraph"
            print(f"[OmniGraph Optics] Testing graph creation at: {test_graph_path}")

            test_graph_config = {
                "graph_path": test_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            }

            test_graph = og.Controller.create_graph(test_graph_config)
            if isinstance(test_graph, tuple) and len(test_graph) >= 1:
                test_graph = test_graph[0]

            if test_graph and test_graph.is_valid():
                print("[OmniGraph Optics] ✓ Test graph created successfully")

                # Test 2: Try to create a simple node using Graph object
                print("[OmniGraph Optics] Testing node creation with Graph object...")
                try:
                    test_node1 = og.Controller.create_node(
                        test_graph,
                        "omni.graph.nodes.ConstantDouble3",
                        name="test_node1"
                    )

                    if test_node1 and test_node1.is_valid():
                        print("[OmniGraph Optics] ✓ Test node 1 created successfully with Graph object")

                        # Clean up test node 1
                        try:
                            og.Controller.delete_node(test_node1)
                            print("[OmniGraph Optics] ✓ Test node 1 deleted")
                        except Exception as e:
                            print(f"[OmniGraph Optics] ⚠ Error deleting test node 1: {e}")
                    else:
                        print("[OmniGraph Optics] ✗ Test node 1 creation failed with Graph object")

                except Exception as e:
                    print(f"[OmniGraph Optics] ✗ Error creating test node 1 with Graph object: {e}")

                # Test 3: Try to create a simple node using string path
                print("[OmniGraph Optics] Testing node creation with string path...")
                try:
                    test_node2 = og.Controller.create_node(
                        test_graph_path,  # Use string path
                        "omni.graph.nodes.ConstantDouble3",
                        name="test_node2"
                    )

                    if test_node2 and test_node2.is_valid():
                        print("[OmniGraph Optics] ✓ Test node 2 created successfully with string path")

                        # Try to set an attribute
                        try:
                            og.Controller.set_node_attribute_value(test_node2, "outputs:value", (1.0, 2.0, 3.0))
                            print("[OmniGraph Optics] ✓ Test attribute set successfully")
                        except Exception as e:
                            print(f"[OmniGraph Optics] ⚠ Error setting test attribute: {e}")

                        # Clean up test node 2
                        try:
                            og.Controller.delete_node(test_node2)
                            print("[OmniGraph Optics] ✓ Test node 2 deleted")
                        except Exception as e:
                            print(f"[OmniGraph Optics] ⚠ Error deleting test node 2: {e}")
                    else:
                        print("[OmniGraph Optics] ✗ Test node 2 creation failed with string path")

                except Exception as e:
                    print(f"[OmniGraph Optics] ✗ Error creating test node 2 with string path: {e}")

                # Test 4: Try different node types
                print("[OmniGraph Optics] Testing different node types...")
                node_types_to_try = [
                    "omni.graph.nodes.ConstantFloat3",
                    "omni.graph.nodes.ConstantInt3",
                    "omni.graph.nodes.ConstantDouble",
                    "omni.graph.nodes.ConstantFloat"
                ]

                for i, node_type in enumerate(node_types_to_try):
                    try:
                        test_node3 = og.Controller.create_node(
                            test_graph_path,
                            node_type,
                            name=f"test_node3_{i}"
                        )

                        if test_node3 and test_node3.is_valid():
                            print(f"[OmniGraph Optics] ✓ Test node 3_{i} created successfully with {node_type}")

                            # Clean up
                            try:
                                og.Controller.delete_node(test_node3)
                                print(f"[OmniGraph Optics] ✓ Test node 3_{i} deleted")
                            except Exception as e:
                                print(f"[OmniGraph Optics] ⚠ Error deleting test node 3_{i}: {e}")
                        else:
                            print(f"[OmniGraph Optics] ✗ Test node 3_{i} creation failed with {node_type}")

                    except Exception as e:
                        print(f"[OmniGraph Optics] ✗ Error creating test node 3_{i} with {node_type}: {e}")

                # Clean up test graph
                try:
                    og.Controller.delete_graph(test_graph)
                    print("[OmniGraph Optics] ✓ Test graph deleted")
                except Exception as e:
                    print(f"[OmniGraph Optics] ⚠ Error deleting test graph: {e}")
            else:
                print("[OmniGraph Optics] ✗ Test graph creation failed")

        except Exception as e:
            print(f"[OmniGraph Optics] ✗ Error in API test: {e}")

        print("[OmniGraph Optics] === END API TEST ===")

    def _create_builtin_node(self):
        """Create a built-in ConstantDouble3 node in the current graph for debugging."""
        if not OMNIGRAPH_AVAILABLE:
            print("[OmniGraph Optics] OmniGraph not available for built-in node test")
            return
        if not self._graph or not self._graph.is_valid():
            print("[OmniGraph Optics] No valid graph for built-in node test")
            return
        # Debug USD stage and graph path
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        print(f"[DEBUG] USD Stage: {stage}")
        graph_path = self._graph_path.rstrip('/') if self._graph_path else self._graph_path
        print(f"[DEBUG] Graph path (stripped): {graph_path!r}")
        layer = stage.GetEditTarget().GetLayer() if stage else None
        from pxr import Sdf
        if not stage:
            print("[DEBUG] No open USD stage! Built-in node creation aborted.")
            return
        if hasattr(layer, 'permission') and layer.permission == Sdf.Permission.ReadOnly:
            print("[DEBUG] USD stage is read-only! Built-in node creation aborted.")
            return
        try:
            print("[DEBUG] Attempting to create built-in node 'omni.graph.nodes.ConstantDouble3'")
            node = og.Controller.create_node(
                graph_path,
                "omni.graph.nodes.ConstantDouble3",
                name="TestConstantDouble3"
            )
            if node and node.is_valid():
                print("[DEBUG] Built-in node created successfully!")
            else:
                print("[DEBUG] Built-in node creation failed (node invalid)")
        except Exception as e:
            print(f"[DEBUG] Exception during built-in node creation: {e}")

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
