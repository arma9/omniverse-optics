import omni.graph.core as og

class OgnMirror:
    @staticmethod
    def compute(db):
        # Placeholder logic for Mirror node
        print("[OmniGraph Optics] Mirror node compute called")
        return True

    @staticmethod
    def get_node_type_name():
        return "mit.test_graph_extension.ogn.nodes.optics.OgnMirror"

    @staticmethod
    def get_node_description():
        return "A custom OmniGraph node representing a mirror."

    @staticmethod
    def get_node_metadata():
        return {
            "category": "Optics",
            "description": "A custom OmniGraph node representing a mirror.",
            "inputs": {
                "input_beam": {"type": "bundle"},
                "reflectivity": {"type": "float", "default": 99.0},
            },
            "outputs": {
                "output_beam": {"type": "bundle"},
            },
        }