# Optical simulation using chromatix
"""
Optical simulation functionality using chromatix's OpticalSystem class.
"""

import os
import sys
from pathlib import Path
from . import JAX_AVAILABLE, CHROMATIX_AVAILABLE

# Ensure chromatix source path is available
possible_paths = [
    Path(__file__).parent.parent.parent.parent.parent / "physics" / "chromatix" / "src",  # Original path
    Path(__file__).parent.parent.parent.parent.parent.parent.parent / "source" / "physics" / "chromatix" / "src",  # From build to source
    Path(__file__).resolve().parents[7] / "source" / "physics" / "chromatix" / "src",  # Absolute path approach
]

chromatix_src_path = None
for path in possible_paths:
    if path.exists():
        chromatix_src_path = path
        break

if chromatix_src_path and chromatix_src_path.exists() and str(chromatix_src_path) not in sys.path:
    sys.path.insert(0, str(chromatix_src_path))

if JAX_AVAILABLE and CHROMATIX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from chromatix import OpticalSystem
    from chromatix.elements import (
        GaussianPlaneWave, 
        ThinLens, 
        Propagate, 
        BasicSensor
    )

    def simulate_4f_system_from_positions(component_positions, wavelength=0.55e-6, focal_length=30, field_resolution=1000):
        """Simulate a 4f optical system using chromatix from component positions.
        
        Args:
            component_positions: Dict with keys 'laser', 'lens1', 'lens2', 'camera' 
                                and values as (x, y, z) positions in cm
            wavelength: Wavelength in meters (default: 550nm)
            focal_length: Focal length of lenses in cm (default: 30cm)
            field_resolution: Field resolution in pixels (default: 1000)
            
        Returns:
            Dict with simulation results including output field
        """
        try:
            # Convert positions from cm to meters
            laser_z = component_positions['laser'][2] * 1e-2  # cm to m
            lens1_z = component_positions['lens1'][2] * 1e-2
            lens2_z = component_positions['lens2'][2] * 1e-2
            camera_z = component_positions['camera'][2] * 1e-2
            
            # Convert focal length from cm to meters for chromatix
            focal_length_m = focal_length * 1e-2  # cm to m
            
            # Calculate propagation distances
            dist_laser_to_lens1 = lens1_z - laser_z
            dist_lens1_to_lens2 = lens2_z - lens1_z
            dist_lens2_to_camera = camera_z - lens2_z
            
            print(f"[optical_simulation] Focal length: {focal_length}cm ({focal_length_m}m)")
            print(f"[optical_simulation] Distances: laser->lens1: {dist_laser_to_lens1*100:.1f}cm, lens1->lens2: {dist_lens1_to_lens2*100:.1f}cm, lens2->camera: {dist_lens2_to_camera*100:.1f}cm")
            
            # Define field parameters - scale dx to maintain same field of view
            shape = (field_resolution, field_resolution)  # Configurable field resolution
            # Use 1000x1000 with 1μm spacing as reference, scale dx proportionally
            reference_resolution = 1000
            reference_spacing = 1e-6  # 1 micron
            spacing = reference_spacing * (reference_resolution / field_resolution)
            
            field_of_view_mm = shape[0] * spacing * 1000  # Total field of view in mm
            print(f"[optical_simulation] Field resolution: {field_resolution}x{field_resolution}, pixel spacing: {spacing*1e6:.3f}μm, FOV: {field_of_view_mm:.3f}mm")
            
            # Create initial laser field separately to capture input
            laser_source = GaussianPlaneWave(
                shape=shape,
                dx=spacing,
                spectrum=wavelength,
                spectral_density=1.0,
                waist=1e-3,  # 1mm beam waist - corrected parameter name
                power=1.0
            )
            
            # Initialize and get initial laser field
            key = jax.random.PRNGKey(42)
            laser_params = laser_source.init(key)
            input_field = laser_source.apply(laser_params)
            
            # Create 4f optical system using chromatix
            system = OpticalSystem([
                # Source: Gaussian plane wave with proper parameter name
                laser_source,
                
                # Propagate to first lens
                Propagate(z=dist_laser_to_lens1, n=1.0),
                
                # First lens (f=30cm)
                ThinLens(f=focal_length_m, n=1.5),
                
                # Propagate to second lens
                Propagate(z=dist_lens1_to_lens2, n=1.0),
                
                # Second lens (f=30cm)
                ThinLens(f=focal_length_m, n=1.5),
                
                # Propagate to camera/detector
                Propagate(z=dist_lens2_to_camera, n=1.0),
                
                # Camera/detector
                BasicSensor(
                    shape=shape,
                    spacing=spacing,
                    resampling_method=None
                )
            ])
            
            # Initialize the system
            params = system.init(key)
            
            # Run the simulation
            output_field = system.apply(params)
            
            # Extract intensity from both input and output fields
            if hasattr(output_field, 'intensity'):
                output_intensity = np.array(output_field.intensity)
            else:
                # Fallback: compute intensity from field amplitude
                output_intensity = np.array(jnp.abs(output_field) ** 2)
            
            if hasattr(input_field, 'intensity'):
                input_intensity = np.array(input_field.intensity)
            else:
                # Fallback: compute intensity from field amplitude
                input_intensity = np.array(jnp.abs(input_field) ** 2)
            
            return {
                "input_field": input_field,
                "input_intensity": input_intensity,
                "output_field": output_field,
                "output_intensity": output_intensity,
                "wavelength": wavelength,
                "focal_length": focal_length,  # in cm
                "component_positions": component_positions,
                "field_shape": shape,
                "spacing": spacing,
                "system_length": (camera_z - laser_z) * 100,  # cm
                "success": True
            }
            
        except Exception as e:
            print(f"[optical_simulation] Error in 4f system simulation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def plot_and_save_results(results, output_dir="simulation_results"):
        """Plot and save simulation results.
        
        Args:
            results: Results dictionary from simulate_4f_system_from_positions
            output_dir: Directory to save plots and data
            
        Returns:
            Path to saved plot file
        """
        if not results.get("success", False):
            print(f"[optical_simulation] Cannot plot - simulation failed: {results.get('error', 'Unknown error')}")
            return None
            
        try:
            # Create output directory with absolute path
            output_path = Path(output_dir).resolve()
            output_path.mkdir(exist_ok=True)
            
            # Create the plot with 2x2 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Helper function to process intensity arrays
            def process_intensity(intensity, label):
                print(f"[optical_simulation] {label} intensity shape: {intensity.shape}")
                
                # Handle different intensity array shapes - squeeze out singleton dimensions
                if intensity.ndim > 2:
                    # Remove all singleton dimensions (dimensions with size 1)
                    intensity = np.squeeze(intensity)
                    print(f"[optical_simulation] {label} after squeeze: {intensity.shape}")
                    
                    # If still more than 2D, take the first slice or sum
                    if intensity.ndim > 2:
                        if intensity.shape[0] == 1:
                            intensity = intensity[0]
                        else:
                            intensity = intensity.sum(axis=0)
                        print(f"[optical_simulation] {label} after final reshape: {intensity.shape}")
                
                # Ensure we have a 2D array for plotting
                if intensity.ndim != 2:
                    raise ValueError(f"Unable to convert {label} intensity to 2D array. Final shape: {intensity.shape}")
                
                return intensity
            
            # Process both intensities
            input_intensity = process_intensity(results["input_intensity"], "Input")
            output_intensity = process_intensity(results["output_intensity"], "Output")
            
            # Calculate extent for both plots - use same extent for proper comparison
            # Both input and output should have the same field of view
            extent_mm = np.array(input_intensity.shape) * results["spacing"] * 1000 / 2  # Convert to mm
            extent = [-extent_mm[1], extent_mm[1], -extent_mm[0], extent_mm[0]]
            
            print(f"[optical_simulation] Plot extent: ±{extent_mm[1]:.3f}mm x ±{extent_mm[0]:.3f}mm")
            
            # Plot input intensity
            im1 = ax1.imshow(input_intensity, extent=extent, cmap='hot', origin='lower')
            ax1.set_title('Input Laser Intensity')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            plt.colorbar(im1, ax=ax1)
            
            # Plot output intensity
            im2 = ax2.imshow(output_intensity, extent=extent, cmap='hot', origin='lower')
            ax2.set_title('Output Camera Intensity')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
            plt.colorbar(im2, ax=ax2)
            
            # Plot cross-sections - ensure same x-axis scale for both
            center = input_intensity.shape[0] // 2
            input_cross_section = input_intensity[center, :]
            output_cross_section = output_intensity[center, :]
            
            # Use same x-axis for both cross-sections
            x_axis = np.linspace(-extent_mm[1], extent_mm[1], len(input_cross_section))
            
            # Plot input cross-section
            ax3.plot(x_axis, input_cross_section, 'b-', label='Input', linewidth=2)
            ax3.set_title('Input Cross-section (Y=0)')
            ax3.set_xlabel('X (mm)')
            ax3.set_ylabel('Intensity')
            ax3.grid(True)
            
            # Plot output cross-section
            ax4.plot(x_axis, output_cross_section, 'r-', label='Output', linewidth=2)
            ax4.set_title('Output Cross-section (Y=0)')
            ax4.set_xlabel('X (mm)')
            ax4.set_ylabel('Intensity')
            ax4.grid(True)
            
            # Add system info
            system_info = f"4f System: f={results['focal_length']:.1f}cm, λ={results['wavelength']*1e9:.0f}nm"
            system_info += f"\nTotal length: {results['system_length']:.1f}cm"
            fig.suptitle(system_info)
            
            plt.tight_layout()
            
            # Save the plot
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = output_path / f"4f_simulation_{timestamp}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()

            # Save intensity data
            data_filename = output_path / f"intensity_data_{timestamp}.npy"
            np.save(data_filename, output_intensity)
            
            print(f"[optical_simulation] Results saved to {plot_filename}")
            return str(plot_filename)
            
        except Exception as e:
            print(f"[optical_simulation] Error plotting results: {e}")
            return None

else:
    # Provide dummy implementations when dependencies aren't available
    def simulate_4f_system_from_positions(component_positions, wavelength=0.55e-6, focal_length=30, field_resolution=1000):
        print("[optical_simulation] Cannot simulate - JAX/Chromatix not available")
        return {"success": False, "error": "Dependencies not available"}
        
    def plot_and_save_results(results, output_dir="simulation_results"):
        print("[optical_simulation] Cannot plot - dependencies not available")
        return None 