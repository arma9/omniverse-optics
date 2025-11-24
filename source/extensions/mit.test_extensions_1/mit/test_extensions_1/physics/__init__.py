# Physics module for chromatix optical simulation
"""
Physics module containing chromatix optical simulation functionality.
"""

# Test print at very top level to ensure module loading
print(f"[mit.test_extensions_1.physics] ***** PHYSICS MODULE LOADING STARTED *****")

import sys

from ..utils.path_utils import get_chromatix_src_dir

# Add chromatix source directory to Python path
print(f"[mit.test_extensions_1.physics] Physics module initializing...")
chromatix_src_path = get_chromatix_src_dir()
if chromatix_src_path:
    sys.path.insert(0, str(chromatix_src_path))
    print(f"[mit.test_extensions_1.physics] Added chromatix source path: {chromatix_src_path}")
else:
    print("[mit.test_extensions_1.physics] Chromatix source directory not found. Placeholder simulations will be used.")
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