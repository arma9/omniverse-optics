# Optical simulation using chromatix
"""
Optical simulation functionality using chromatix's OpticalSystem class.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from . import JAX_AVAILABLE, CHROMATIX_AVAILABLE
from ..utils.path_utils import get_simulation_output_dir

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
    from chromatix import OpticalSystem
    from chromatix.elements import (
        GaussianPlaneWave,
        Propagate,
        ThinLens,
        BasicSensor
    )

    def simulate_4f_system_from_positions(component_positions, wavelength=0.55e-6, beam_waist_mm=1.0, focal_length=30, field_resolution=1000):
        """Simulate a 4f optical system using chromatix from component positions.

        Args:
            component_positions: Dict with keys 'laser', 'lens1', 'lens2', 'camera'
                                and values as (x, y, z) positions in cm
            wavelength: Wavelength in meters (default: 550nm)
            beam_waist_mm: Beam waist in millimeters (default: 1.0mm)
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

            # Convert beam waist from mm to meters
            beam_waist_m = beam_waist_mm * 1e-3  # mm to m

            # Calculate propagation distances
            dist_laser_to_lens1 = lens1_z - laser_z
            dist_lens1_to_lens2 = lens2_z - lens1_z
            dist_lens2_to_camera = camera_z - lens2_z

            print(f"[optical_simulation] Focal length: {focal_length}cm ({focal_length_m}m)")
            print(f"[optical_simulation] Beam waist: {beam_waist_mm}mm ({beam_waist_m}m)")
            print(f"[optical_simulation] Wavelength: {wavelength*1e9:.0f}nm")
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
                waist=beam_waist_m,  # Use the provided beam waist parameter
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

    def simulate_michelson_interferometer_from_positions(component_positions, mirror_rotations=None, wavelength=0.6328e-6, beam_waist_mm=0.05, field_resolution=1000):
        """Simulate a Michelson interferometer using chromatix from component positions.
        Args:
            component_positions: Dict with keys 'laser', 'beamsplitter', 'mirror1', 'mirror2', 'camera' and values as (x, y, z) positions in mm
            mirror_rotations: Dict with keys 'mirror1', 'mirror2' and values as (x, y, z) rotations in degrees
            wavelength: Wavelength in meters (default: 632.8nm)
            beam_waist_mm: Beam waist in millimeters (default: 0.05mm for 100µm beam diameter)
            field_resolution: Field resolution in pixels (default: 1000)
        Returns:
            Dict with simulation results including output field
        """
        try:
            # Convert positions from mm to meters
            laser_z = component_positions['laser'][2] * 1e-3
            bs_z = component_positions['beamsplitter'][2] * 1e-3
            mirror1_z = component_positions['mirror1'][2] * 1e-3
            mirror2_z = component_positions['mirror2'][2] * 1e-3
            camera_z = component_positions['camera'][2] * 1e-3

            # Calculate arm lengths
            arm1_length = abs(mirror1_z - bs_z) * 2  # Double for round trip
            arm2_length = abs(mirror2_z - bs_z) * 2  # Double for round trip

            print(f"[optical_simulation] Michelson Interferometer:")
            print(f"[optical_simulation] Arm 1 length: {arm1_length*1e3:.1f}mm (round trip)")
            print(f"[optical_simulation] Arm 2 length: {arm2_length*1e3:.1f}mm (round trip)")
            print(f"[optical_simulation] Wavelength: {wavelength*1e9:.0f}nm")
            print(f"[optical_simulation] Beam waist: {beam_waist_mm:.3f}mm")

            # Get mirror rotations (default to aligned positions)
            if mirror_rotations is None:
                mirror_rotations = {'mirror1': (0, 90, 0), 'mirror2': (0, 0, 0)}

            # Convert rotation deviations from degrees to radians
            mirror1_y_rotation = np.radians(mirror_rotations['mirror1'][1])  # Should be ~90° for alignment
            mirror2_y_rotation = np.radians(mirror_rotations['mirror2'][1])  # Should be ~0° for alignment

            print(f"[optical_simulation] Mirror1 y-rotation: {np.degrees(mirror1_y_rotation):.2f}° ({mirror1_y_rotation:.4f} rad)")
            print(f"[optical_simulation] Mirror2 y-rotation: {np.degrees(mirror2_y_rotation):.2f}° ({mirror2_y_rotation:.4f} rad)")

            # Field parameters - scale dx to maintain same field of view
            shape = (field_resolution, field_resolution)
            # Use 1000x1000 with 0.2μm spacing as reference, scale dx proportionally
            reference_resolution = 1000
            reference_spacing = 0.2e-6  # 0.2 micron
            dx = reference_spacing * (reference_resolution / field_resolution)
            beam_waist_m = beam_waist_mm * 1e-3  # Convert mm to meters
            power = 1.0

            field_of_view_mm = shape[0] * dx * 1000  # Total field of view in mm
            print(f"[optical_simulation] Michelson field resolution: {field_resolution}x{field_resolution}, pixel spacing: {dx*1e6:.3f}μm, FOV: {field_of_view_mm:.3f}mm")
            k = 2 * np.pi / wavelength
            n = 1.0
            N_pad = (0, 0)

            # Coordinate grid
            x_coords = np.arange(shape[1]) - shape[1] // 2
            y_coords = np.arange(shape[0]) - shape[0] // 2
            X, Y = np.meshgrid(x_coords * dx, y_coords * dx)

            # Input beam
            from chromatix.functional import gaussian_plane_wave, asm_propagate
            input_beam = gaussian_plane_wave(
                shape=shape,
                dx=dx,
                spectrum=wavelength,
                spectral_density=1.0,
                power=power,
                waist=beam_waist_m  # Use the provided beam waist parameter
            )

            # Arm 1: propagate to mirror, apply tilt based on mirror1 rotation deviation from 90°, propagate back
            arm1 = asm_propagate(input_beam, arm1_length/2, n, N_pad=N_pad, mode="same")
            # Apply tilt phase for mirror1 (deviation from 90° aligned angle)
            mirror1_tilt = mirror1_y_rotation - np.radians(90)  # Deviation from aligned angle
            tilt_phase1 = k * X * np.sin(mirror1_tilt)
            tilt_phase_expanded1 = tilt_phase1[None, :, :, None, None]
            arm1_u = arm1.u * np.exp(1j * tilt_phase_expanded1)
            arm1 = arm1.replace(u=arm1_u)
            arm1 = asm_propagate(arm1, -arm1_length/2, n, N_pad=N_pad, mode="same")

            # Arm 2: propagate to mirror, apply tilt based on mirror2 rotation deviation from 0°, propagate back
            arm2 = asm_propagate(input_beam, arm2_length/2, n, N_pad=N_pad, mode="same")
            # Apply tilt phase for mirror2 (deviation from 0° aligned angle)
            mirror2_tilt = mirror2_y_rotation - np.radians(0)  # Deviation from aligned angle
            tilt_phase2 = k * X * np.sin(mirror2_tilt)
            tilt_phase_expanded2 = tilt_phase2[None, :, :, None, None]
            arm2_u = arm2.u * np.exp(1j * tilt_phase_expanded2)
            arm2 = arm2.replace(u=arm2_u)
            arm2 = asm_propagate(arm2, -arm2_length/2, n, N_pad=N_pad, mode="same")

            # Combine arms (coherent sum)
            output = arm1 + arm2

            # Extract intensity from both input and output fields
            input_intensity = np.array(input_beam.intensity).squeeze()
            output_intensity = np.array(output.intensity).squeeze()

            return {
                "input_field": input_beam,
                "input_intensity": input_intensity,
                "output_field": output,
                "output_intensity": output_intensity,
                "wavelength": wavelength,
                "arm1_length": arm1_length,
                "arm2_length": arm2_length,
                "mirror1_y_rotation": mirror1_y_rotation,
                "mirror2_y_rotation": mirror2_y_rotation,
                "component_positions": component_positions,
                "field_shape": shape,
                "dx": dx,
                "success": True
            }
        except Exception as e:
            print(f"[optical_simulation] Error in Michelson simulation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

else:
    # Provide analytic placeholder implementations when chromatix is unavailable

    def _generate_placeholder_grid(field_resolution: int, extent_mm: float):
        resolution = int(max(64, min(field_resolution, 2048)))
        axis = np.linspace(-extent_mm / 2.0, extent_mm / 2.0, resolution)
        x_grid, y_grid = np.meshgrid(axis, axis)
        spacing_m = (extent_mm / resolution) * 1e-3  # convert mm to meters
        return x_grid, y_grid, spacing_m

    def simulate_4f_system_from_positions(
        component_positions,
        wavelength=0.55e-6,
        beam_waist_mm=1.0,
        focal_length=30,
        field_resolution=1000
    ):
        """Fallback Gaussian propagation for 4f systems."""
        try:
            x, y, spacing = _generate_placeholder_grid(field_resolution, extent_mm=12.0)
            waist = max(beam_waist_mm, 0.1)
            input_intensity = np.exp(-(x**2 + y**2) / (2 * waist**2))

            lens1_z = component_positions["lens1"][2]
            lens2_z = component_positions["lens2"][2]
            ideal_sep = 2 * focal_length
            separation_error = abs((lens2_z - lens1_z) - ideal_sep)
            blur_scale = 1.0 + separation_error / max(ideal_sep, 1.0)

            output_intensity = np.exp(-(x**2 + y**2) / (2 * (waist * blur_scale)**2))
            fringe_term = np.cos(2 * np.pi * x / max(waist * 4.0, 1e-3))
            output_intensity *= 1 + 0.2 * fringe_term

            return {
                "input_field": {"intensity": input_intensity},
                "input_intensity": input_intensity,
                "output_field": {"intensity": output_intensity},
                "output_intensity": output_intensity,
                "wavelength": wavelength,
                "focal_length": focal_length,
                "component_positions": component_positions,
                "field_shape": input_intensity.shape,
                "spacing": spacing,
                "system_length": component_positions["camera"][2] - component_positions["laser"][2],
                "success": True,
            }
        except Exception as exc:
            print(f"[optical_simulation] Placeholder 4f simulation error: {exc}")
            return {"success": False, "error": str(exc)}

    def simulate_michelson_interferometer_from_positions(
        component_positions,
        mirror_rotations=None,
        wavelength=0.6328e-6,
        beam_waist_mm=0.05,
        field_resolution=1000
    ):
        """Fallback analytic interference pattern for Michelson interferometer."""
        try:
            x, y, spacing = _generate_placeholder_grid(field_resolution, extent_mm=8.0)
            waist = max(beam_waist_mm, 0.02)
            input_intensity = np.exp(-(x**2 + y**2) / (2 * waist**2))

            mirror_rotations = mirror_rotations or {
                "mirror1": (0, 90, 0),
                "mirror2": (0, 0, 0),
            }

            bs_z = component_positions["beamsplitter"][2]
            arm1_length = abs(component_positions["mirror1"][2] - bs_z) * 1e-3 * 2
            arm2_length = abs(component_positions["mirror2"][2] - bs_z) * 1e-3 * 2
            path_difference = arm1_length - arm2_length

            mirror1_tilt = np.radians(mirror_rotations["mirror1"][1] - 90)
            mirror2_tilt = np.radians(mirror_rotations["mirror2"][1])

            phase_offset = 2 * np.pi * path_difference / max(wavelength, 1e-9)
            tilt_phase = (
                (x * 1e-3 * mirror1_tilt) / max(wavelength, 1e-9)
                + (y * 1e-3 * mirror2_tilt) / max(wavelength, 1e-9)
            )
            output_intensity = 0.5 * (1 + np.cos(phase_offset + tilt_phase))
            output_intensity *= input_intensity.max()

            return {
                "input_field": {"intensity": input_intensity},
                "input_intensity": input_intensity,
                "output_field": {"intensity": output_intensity},
                "output_intensity": output_intensity,
                "wavelength": wavelength,
                "arm1_length": arm1_length,
                "arm2_length": arm2_length,
                "mirror1_y_rotation": np.radians(mirror_rotations["mirror1"][1]),
                "mirror2_y_rotation": np.radians(mirror_rotations["mirror2"][1]),
                "component_positions": component_positions,
                "field_shape": input_intensity.shape,
                "dx": spacing,
                "success": True,
            }
        except Exception as exc:
            print(f"[optical_simulation] Placeholder Michelson simulation error: {exc}")
            return {"success": False, "error": str(exc)}


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = get_simulation_output_dir(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_and_save_results(results, output_dir="simulation_results"):
    """Common plotting helper used by both chromatix and placeholder simulations."""
    if not results.get("success", False):
        print(f"[optical_simulation] Cannot plot - simulation failed: {results.get('error', 'Unknown error')}")
        return None

    if not MATPLOTLIB_AVAILABLE:
        print("[optical_simulation] Cannot plot - matplotlib not available in this environment")
        return None

    try:
        output_path = _resolve_output_dir(output_dir)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        def _process_intensity(intensity, label):
            if intensity is None:
                raise ValueError(f"{label} intensity missing")
            if intensity.ndim > 2:
                intensity = np.squeeze(intensity)
                if intensity.ndim > 2:
                    intensity = intensity.sum(axis=0)
            if intensity.ndim != 2:
                raise ValueError(f"Unable to convert {label} intensity to 2D array. Final shape: {intensity.shape}")
            return intensity

        input_intensity = _process_intensity(results.get("input_intensity"), "Input")
        output_intensity = _process_intensity(results.get("output_intensity"), "Output")

        spacing = results.get("dx") or results.get("spacing")
        if spacing is None:
            spacing = 1e-4  # fallback spacing (0.1 mm)
        extent_mm = np.array(input_intensity.shape) * spacing * 1000 / 2
        extent = [-extent_mm[1], extent_mm[1], -extent_mm[0], extent_mm[0]]

        im1 = ax1.imshow(input_intensity, extent=extent, cmap="hot", origin="lower")
        ax1.set_title("Input Laser Intensity")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(output_intensity, extent=extent, cmap="hot", origin="lower")
        ax2.set_title("Output Camera Intensity")
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Y (mm)")
        plt.colorbar(im2, ax=ax2)

        center = input_intensity.shape[0] // 2
        x_axis = np.linspace(-extent_mm[1], extent_mm[1], len(input_intensity[center, :]))
        ax3.plot(x_axis, input_intensity[center, :], "b-", linewidth=2)
        ax3.set_title("Input Cross-section (Y=0)")
        ax3.set_xlabel("X (mm)")
        ax3.set_ylabel("Intensity")
        ax3.grid(True)

        ax4.plot(x_axis, output_intensity[center, :], "r-", linewidth=2)
        ax4.set_title("Output Cross-section (Y=0)")
        ax4.set_xlabel("X (mm)")
        ax4.set_ylabel("Intensity")
        ax4.grid(True)

        if "arm1_length" in results and "arm2_length" in results:
            mirror1_rot_deg = np.degrees(results.get("mirror1_y_rotation", 0.0))
            mirror2_rot_deg = np.degrees(results.get("mirror2_y_rotation", 0.0))
            mirror1_tilt_deg = mirror1_rot_deg - 90
            mirror2_tilt_deg = mirror2_rot_deg
            system_info = (
                f"Michelson Interferometer: Arm1={results['arm1_length']*1e3:.1f}mm, "
                f"Arm2={results['arm2_length']*1e3:.1f}mm\n"
                f"Mirror1 tilt: {mirror1_tilt_deg:+.2f}°, Mirror2 tilt: {mirror2_tilt_deg:+.2f}°, "
                f"λ={results['wavelength']*1e9:.0f}nm"
            )
        else:
            system_info = (
                f"4f System: f={results.get('focal_length', 0):.1f}cm, "
                f"λ={results['wavelength']*1e9:.0f}nm\n"
                f"Total length: {results.get('system_length', 0):.1f}cm"
            )

        fig.suptitle(system_info)
        plt.tight_layout()

        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if "arm1_length" in results:
            plot_filename = output_path / f"michelson_simulation_{timestamp}.png"
        else:
            plot_filename = output_path / f"4f_simulation_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        plt.close()

        data_filename = output_path / f"intensity_data_{timestamp}.npy"
        np.save(data_filename, output_intensity)

        print(f"[optical_simulation] Results saved to {plot_filename}")
        return str(plot_filename)

    except Exception as exc:
        print(f"[optical_simulation] Error plotting results: {exc}")
        return None