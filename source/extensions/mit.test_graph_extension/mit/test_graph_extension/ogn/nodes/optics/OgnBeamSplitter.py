import omni.graph.core as og

class OgnBeamSplitter:
    @staticmethod
    def compute(db):
        # Placeholder logic for BeamSplitter node
        print("[OmniGraph Optics] BeamSplitter node compute called")
        return True

    @staticmethod
    def get_node_type_name():
        return "mit.test_graph_extension.ogn.nodes.optics.OgnBeamSplitter"

    @staticmethod
    def get_node_description():
        return "A custom OmniGraph node representing a beam splitter."

    @staticmethod
    def get_node_metadata():
        return {
            "category": "Optics",
            "description": "A custom OmniGraph node representing a beam splitter.",
            "inputs": {
                "input_beam": {"type": "bundle"},
                "transmission_ratio": {"type": "float", "default": 50.0},
            },
            "outputs": {
                "transmitted_beam": {"type": "bundle"},
                "reflected_beam": {"type": "bundle"},
            },
        }