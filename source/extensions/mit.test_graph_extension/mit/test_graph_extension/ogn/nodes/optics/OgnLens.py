import omni.graph.core as og

class OgnLens:
    @staticmethod
    def compute(db):
        # Define inputs and outputs for the node
        # Example: db.inputs.input_name.get() and db.outputs.output_name.set(value)
        print("[OmniGraph Optics] Lens node compute called")
        return True

    @staticmethod
    def get_node_type_name():
        return "mit.test_graph_extension.ogn.nodes.optics.OgnLens"

    @staticmethod
    def get_node_description():
        return "A custom OmniGraph node representing a lens."

    @staticmethod
    def get_node_metadata():
        return {
            "category": "Optics",
            "description": "A custom OmniGraph node representing a lens.",
            "inputs": {
                "input_beam": {"type": "bundle"},
                "focal_length": {"type": "float", "default": 50.0},
            },
            "outputs": {
                "output_beam": {"type": "bundle"},
            },
        }