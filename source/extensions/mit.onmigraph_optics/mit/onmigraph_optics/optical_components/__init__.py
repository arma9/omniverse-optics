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

"""
Optical Component Management Utilities

This module provides utilities for creating and managing optical components
in the OmniGraph Optics extension.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import carb

class OpticalComponentManager:
    """Manager for optical component creation and configuration."""
    
    # Component type definitions
    COMPONENT_TYPES = {
        "Laser": {
            "description": "Laser source component",
            "default_prefab": "prefabs/LASER_PREFAB.usd",
            "default_position": (0, 0, 0),
            "color": (1.0, 0.0, 0.0),  # Red
            "parameters": ["wavelength", "power", "beam_width"]
        },
        "Lens": {
            "description": "Optical lens component",
            "default_prefab": "prefabs/LENS_PREFAB.usd", 
            "default_position": (0, 0, 50),
            "color": (0.0, 0.0, 1.0),  # Blue
            "parameters": ["focal_length", "diameter", "material"]
        },
        "Camera": {
            "description": "Camera/detector component",
            "default_prefab": "prefabs/CAMERA_PREFAB.usd",
            "default_position": (0, 0, 100),
            "color": (0.0, 1.0, 0.0),  # Green
            "parameters": ["sensor_size", "pixel_size", "sensitivity"]
        },
        "Mirror": {
            "description": "Mirror component",
            "default_prefab": "prefabs/LENS_PREFAB.usd",  # Placeholder
            "default_position": (50, 0, 50),
            "color": (0.7, 0.7, 0.7),  # Gray
            "parameters": ["reflectivity", "surface_quality", "angle"]
        },
        "BeamSplitter": {
            "description": "Beam splitter component",
            "default_prefab": "prefabs/LENS_PREFAB.usd",  # Placeholder
            "default_position": (0, 50, 50),
            "color": (1.0, 1.0, 0.0),  # Yellow
            "parameters": ["split_ratio", "transmission", "angle"]
        }
    }
    
    # Interferometer presets
    INTERFEROMETER_PRESETS = {
        "Michelson": {
            "description": "Michelson interferometer setup",
            "components": [
                ("Laser", (0, 0, 0)),
                ("BeamSplitter", (0, 0, 50)),
                ("Mirror", (100, 0, 50)),
                ("Mirror", (0, 100, 50)),
                ("Camera", (0, 0, 100))
            ]
        },
        "MachZehnder": {
            "description": "Mach-Zehnder interferometer setup",
            "components": [
                ("Laser", (0, 0, 0)),
                ("BeamSplitter", (0, 0, 50)),
                ("Mirror", (50, 50, 50)),
                ("Mirror", (50, -50, 50)),
                ("BeamSplitter", (100, 0, 50)),
                ("Camera", (150, 0, 50))
            ]
        },
        "FabryPerot": {
            "description": "Fabry-Perot cavity setup",
            "components": [
                ("Laser", (0, 0, 0)),
                ("Mirror", (0, 0, 50)),
                ("Mirror", (0, 0, 100)),
                ("Camera", (0, 0, 150))
            ]
        }
    }
    
    @classmethod
    def get_component_info(cls, component_type: str) -> Optional[Dict]:
        """Get information about a component type."""
        return cls.COMPONENT_TYPES.get(component_type)
    
    @classmethod
    def get_available_components(cls) -> List[str]:
        """Get list of available component types."""
        return list(cls.COMPONENT_TYPES.keys())
    
    @classmethod
    def get_interferometer_preset(cls, preset_name: str) -> Optional[Dict]:
        """Get interferometer preset configuration."""
        return cls.INTERFEROMETER_PRESETS.get(preset_name)
    
    @classmethod
    def get_available_presets(cls) -> List[str]:
        """Get list of available interferometer presets."""
        return list(cls.INTERFEROMETER_PRESETS.keys())
    
    @classmethod
    def get_prefab_path(cls, component_type: str) -> Optional[str]:
        """Get the full path to a component's prefab file."""
        component_info = cls.get_component_info(component_type)
        if not component_info:
            return None
        
        # Get project root
        project_root = carb.tokens.get_tokens_interface().resolve("${app}/../../../..")
        prefab_path = str(Path(project_root) / "CAD" / component_info["default_prefab"])
        
        return prefab_path if Path(prefab_path).exists() else None
    
    @classmethod
    def validate_component_placement(cls, component_type: str, position: Tuple[float, float, float]) -> bool:
        """Validate if a component can be placed at the given position."""
        # Basic validation - can be extended with more complex rules
        return True
    
    @classmethod
    def get_optical_path_suggestions(cls, existing_components: List[Dict]) -> List[Dict]:
        """Get suggestions for optical path connections between components."""
        suggestions = []
        
        # Simple suggestion logic - can be made more sophisticated
        for i, comp1 in enumerate(existing_components):
            for j, comp2 in enumerate(existing_components):
                if i != j:
                    # Check if components can be connected optically
                    if cls._can_connect_components(comp1, comp2):
                        suggestions.append({
                            "from": comp1,
                            "to": comp2,
                            "type": "optical_path"
                        })
        
        return suggestions
    
    @classmethod
    def _can_connect_components(cls, comp1: Dict, comp2: Dict) -> bool:
        """Check if two components can be connected optically."""
        # Basic connection rules
        output_components = ["Laser", "BeamSplitter", "Mirror"]
        input_components = ["Camera", "BeamSplitter", "Mirror", "Lens"]
        
        comp1_type = comp1.get("type", "")
        comp2_type = comp2.get("type", "")
        
        return comp1_type in output_components and comp2_type in input_components

def get_component_manager() -> OpticalComponentManager:
    """Get the optical component manager instance."""
    return OpticalComponentManager() 