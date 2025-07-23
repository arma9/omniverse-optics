"""MIT Test Graph Extension – Kit 107.3‑compatible (final)
================================================================
* Uses `ogn/nodes.json` for custom node registration (no manual code).
* Never indexes `self._graph` with `[]`; uses the returned objects.
* Builds node paths with `self._graph_path` so counters work.
* UI buttons: New Graph, Constant, Add, Subtract, BeamSplitter, Lens,
  Mirror, Refresh.
* Open the Action Graph window manually from *Window ▸ Graph ▸ Action Graph*.
"""

from __future__ import annotations

import omni.ext
import omni.ui as ui
import omni.graph.core as og

# -----------------------------------------------------------------------------
# Public helper (example)
# -----------------------------------------------------------------------------

def some_public_function(x: int) -> int:
    print(f"[mit.test_graph_extension] some_public_function was called with {x}")
    return x ** x


# -----------------------------------------------------------------------------
# Extension class
# -----------------------------------------------------------------------------

class MyExtension(omni.ext.IExt):
    """Simple UI to create an Action Graph and add a few nodes."""

    # ---------------------------------------------------------------------
    # Startup / shutdown
    # ---------------------------------------------------------------------

    def on_startup(self, _ext_id):
        print("[mit.test_graph_extension] Extension startup")

        # counters ---------------------------------------------------------
        self._graph_counter = 1
        self._constant_counter = 1
        self._add_counter = 1
        self._sub_counter = 1
        self._beam_counter = 1
        self._lens_counter = 1
        self._mirror_counter = 1

        # graph handle + path ---------------------------------------------
        self._graph: og.Graph | None = None
        self._graph_path: str | None = None

        # -----------------------------------------------------------------
        # UI
        # -----------------------------------------------------------------
        self._window = ui.Window("Test Graph Extension", width=420, height=480)
        with self._window.frame:
            with ui.VStack():
                self._graph_label = ui.Label("No graph created.")
                self._node_label = ui.Label("Nodes:\n-")

                # buttons --------------------------------------------------
                with ui.HStack():
                    ui.Button("New Graph", clicked_fn=self._create_empty_graph)
                    ui.Button("Constant",  clicked_fn=self._add_constant)
                with ui.HStack():
                    ui.Button("Add",       clicked_fn=self._add_add)
                    ui.Button("Subtract", clicked_fn=self._add_sub)
                with ui.HStack():
                    ui.Button("BeamSplitter", clicked_fn=self._add_beam)
                    ui.Button("Lens",         clicked_fn=self._add_lens)
                    ui.Button("Mirror",       clicked_fn=self._add_mirror)
                ui.Button("Refresh", clicked_fn=self._refresh_graph)
                ui.Spacer(height=8)
                ui.Label("Open *Window ▸ Graph ▸ Action Graph* to wire nodes.",
                         style={"color": 0xFF888888})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ui(self):
        if self._graph and self._graph.is_valid():
            nodes = self._graph.get_nodes()
            info  = "\n".join(f"{n.get_prim_path()} ({n.get_type_name()})" for n in nodes)
            self._graph_label.text = f"Graph Path: {self._graph_path}"
            self._node_label.text  = "Nodes:\n" + info
        else:
            self._graph_label.text = "No graph created."
            self._node_label.text  = "Nodes:\n-"

    # ------------------------------------------------------------------
    # Graph creation
    # ------------------------------------------------------------------

    def _create_empty_graph(self):
        self._graph_path = f"/World/graph{self._graph_counter}"
        self._graph_counter += 1

        cfg = {
            "graph_path":     self._graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        }
        self._graph = og.Controller.create_graph(cfg)
        print(f"Created Action Graph at {self._graph_path}")

        # reset node counters for this graph
        self._constant_counter = 1
        self._add_counter      = 1
        self._sub_counter      = 1
        self._beam_counter     = 1
        self._lens_counter     = 1
        self._mirror_counter   = 1

        self._update_ui()

    # ------------------------------------------------------------------
    # Node adders (stock + custom)
    # ------------------------------------------------------------------

    def _add_constant(self):
        self._add_node("constant", self._constant_counter, "omni.graph.nodes.ConstantDouble")
        self._constant_counter += 1

    def _add_add(self):
        self._add_node("add", self._add_counter, "omni.graph.nodes.Add")
        self._add_counter += 1

    def _add_sub(self):
        self._add_node("subtract", self._sub_counter, "omni.graph.nodes.Subtract")
        self._sub_counter += 1

    def _add_beam(self):
        self._add_node("beam_splitter", self._beam_counter,
                       "mit.test_graph_extension.ogn.OgnBeamSplitter")
        self._beam_counter += 1

    def _add_lens(self):
        self._add_node("lens", self._lens_counter,
                       "mit.test_graph_extension.ogn.OgnLens")
        self._lens_counter += 1

    def _add_mirror(self):
        self._add_node("mirror", self._mirror_counter,
                       "mit.test_graph_extension.ogn.OgnMirror")
        self._mirror_counter += 1

    # ------------------------------------------------------------------
    # Generic node‑adder
    # ------------------------------------------------------------------

    def _add_node(self, base: str, count: int, node_type: str):
        if not (self._graph and self._graph.is_valid()):
            print("[mit.test_graph_extension] No valid graph. Create one first.")
            return

        node_name = f"{self._graph_path}/{base}{count}"
        node      = self._graph.create_node(node_name, node_type, True)

        if node and node.is_valid():
            print(f"Added {base} node: {node.get_prim_path()}")
            self._update_ui()
            self._graph.evaluate()
        else:
            print(f"Failed to add {base} node.')")

    # ------------------------------------------------------------------
    # Manual refresh button
    # ------------------------------------------------------------------

    def _refresh_graph(self):
        if self._graph and self._graph.is_valid():
            self._graph.evaluate()
            print("Graph manually refreshed.")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def on_shutdown(self):
        print("[mit.test_graph_extension] Extension shutdown")
