# Physics module for chromatix optical simulation
"""
Physics module containing chromatix optical simulation functionality.
"""

# Test print at very top level to ensure module loading
print(f"[mit.test_extensions_1.physics] ***** PHYSICS MODULE LOADING STARTED *****")

import sys
import os
from pathlib import Path

# Add chromatix source directory to Python path
print(f"[mit.test_extensions_1.physics] Physics module initializing...")
print(f"[mit.test_extensions_1.physics] Current file: {__file__}")

# Get the path to the kit-app-template directory
current_file = Path(__file__).resolve()
# Navigate from the extension location to the kit-app-template directory
# Current file is in: kit-app-template/source/extensions/mit.test_extensions_1/mit/test_extensions_1/physics/__init__.py
# We need to go up 7 levels to get to kit-app-template

print(f"[mit.test_extensions_1.physics] Current file path: {current_file}")
for i in range(1, 10):
    parent = current_file.parents[i-1]
    print(f"[mit.test_extensions_1.physics] Parent level {i}: {parent}")
    
    # Check if this is the kit-app-template directory
    if parent.name == "kit-app-template":
        kit_app_template_dir = parent
        print(f"[mit.test_extensions_1.physics] Found kit-app-template at level {i}: {kit_app_template_dir}")
        break
else:
    print(f"[mit.test_extensions_1.physics] ERROR: Could not find kit-app-template directory")
    kit_app_template_dir = None

if kit_app_template_dir:
    print(f"[mit.test_extensions_1.physics] Kit-app-template directory: {kit_app_template_dir}")
    
    # Path to chromatix source directory
    chromatix_src_path = kit_app_template_dir / "source" / "physics" / "chromatix" / "src"
    print(f"[mit.test_extensions_1.physics] Looking for chromatix at: {chromatix_src_path}")
    
    if chromatix_src_path.exists():
        sys.path.insert(0, str(chromatix_src_path))
        print(f"[mit.test_extensions_1.physics] Added chromatix source path: {chromatix_src_path}")
    else:
        print(f"[mit.test_extensions_1.physics] ERROR: Chromatix source directory not found at {chromatix_src_path}")

print(f"[mit.test_extensions_1.physics] Current sys.path: {sys.path[:3]}...")

# Try to import JAX and chromatix
print(f"[mit.test_extensions_1.physics] Attempting to import JAX and chromatix...")

# Initialize status variables
JAX_AVAILABLE = False
CHROMATIX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"[mit.test_extensions_1.physics] JAX imported successfully")
    print(f"[mit.test_extensions_1.physics] JAX numpy imported successfully")
    
    try:
        import chromatix
        CHROMATIX_AVAILABLE = True
        print(f"[mit.test_extensions_1.physics] Chromatix imported successfully")
        
        # Print version information if available
        try:
            print(f"[mit.test_extensions_1.physics] JAX version: {jax.__version__}")
        except AttributeError:
            print(f"[mit.test_extensions_1.physics] JAX version: Not available")
        
        try:
            print(f"[mit.test_extensions_1.physics] Chromatix version: {chromatix.__version__}")
        except AttributeError:
            print(f"[mit.test_extensions_1.physics] Chromatix version: Not available")
            
        # Print device information
        try:
            print(f"[mit.test_extensions_1.physics] JAX devices: {jax.devices()}")
        except Exception as e:
            print(f"[mit.test_extensions_1.physics] JAX devices error: {e}")
            
    except ImportError as e:
        print(f"[mit.test_extensions_1.physics] Chromatix import failed: {e}")
        
except ImportError as e:
    print(f"[mit.test_extensions_1.physics] JAX import failed: {e}")

print(f"[mit.test_extensions_1.physics] Final status - JAX: {JAX_AVAILABLE}, Chromatix: {CHROMATIX_AVAILABLE}")

def get_physics_status():
    """Get the status of physics dependencies."""
    return {
        "jax_available": JAX_AVAILABLE,
        "chromatix_available": CHROMATIX_AVAILABLE,
    } 