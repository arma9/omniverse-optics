import omni.graph.core as og
from omni.graph.action_core import get_interface


class OgnBeamSplitter:
    @staticmethod
    def initialize(_, node):
        og.Controller.attribute("state:count", node).set(-1)

    @staticmethod
    def compute(db) -> bool:
        pass

        return True
