# OmniGraph Optics Extension

An advanced optical system visualization extension for NVIDIA Omniverse that uses OmniGraph for real-time binding between USD scene components and computational nodes.

## Features

### Core Capabilities
- **Real-time Component Binding**: Automatically syncs USD scene objects with OmniGraph nodes
- **Position Tracking**: Updates OmniGraph node variables when components move in the scene
- **Optical Component Creation**: One-click creation of lasers, lenses, cameras, mirrors, and beam splitters
- **Interferometer Presets**: Pre-configured setups for common interferometer types

### Supported Components
- **Laser**: Light source with configurable wavelength, power, and beam width
- **Lens**: Optical lens with focal length, diameter, and material properties
- **Camera**: Detector/camera with sensor size, pixel size, and sensitivity
- **Mirror**: Reflective surface with reflectivity, surface quality, and angle
- **Beam Splitter**: Optical component with split ratio, transmission, and angle

### Interferometer Presets
- **Michelson Interferometer**: Classic two-arm interferometer setup
- **Mach-Zehnder Interferometer**: Four-mirror interferometer configuration
- **Fabry-Perot Cavity**: Resonant cavity with two mirrors

## Installation

1. Place the extension in your Kit application's extensions directory:
   ```
   source/extensions/mit.onmigraph_optics/
   ```

2. The extension depends on the following Omniverse extensions:
   - `omni.graph.core`
   - `omni.graph.ui`
   - `omni.graph.nodes`
   - `omni.kit.commands`
   - `omni.usd`

3. Enable the extension in the Extension Manager

## Usage

### Basic Workflow

1. **Open the Extension Window**
   - Go to the Extension Manager and enable "OmniGraph Optics"
   - The extension window will appear with component creation buttons

2. **Create Components**
   - Click any component button (Laser, Lens, Camera, etc.)
   - The component will be created in the USD scene
   - A corresponding OmniGraph node will be automatically created
   - The binding between USD and OmniGraph will be established

3. **Move Components**
   - Move components in the 3D viewport
   - The OmniGraph node positions will update automatically
   - Position changes are tracked in real-time

4. **View OmniGraph**
   - Click "Show OmniGraph Editor" to view the computational graph
   - See how components are represented as nodes
   - Observe position updates in real-time

### Interferometer Creation

1. **Use Presets**
   - Click one of the interferometer preset buttons
   - A complete interferometer setup will be created
   - All components will be properly positioned and connected

2. **Custom Setup**
   - Create individual components
   - Position them manually in the scene
   - Use the binding system to track their relationships

## Architecture

### Component Binding System
The extension uses a sophisticated binding system that:
- Monitors USD scene changes
- Synchronizes position data with OmniGraph nodes
- Maintains real-time connections between visual and computational representations
- Validates binding integrity

### OmniGraph Integration
- Creates computational nodes for each optical component
- Stores component properties and parameters
- Enables real-time optical simulation workflows
- Provides foundation for advanced optical calculations

### Position Monitoring
- Background thread monitors component positions
- Updates OmniGraph nodes when components move
- Maintains synchronization between USD and OmniGraph
- Provides validation and error handling

## Advanced Features

### Component Manager
The extension includes a comprehensive component management system:
- Component type definitions with properties
- Prefab path resolution
- Validation and placement rules
- Optical path suggestions

### Real-time Synchronization
- 10Hz update rate for position tracking
- Automatic binding validation
- Error recovery and cleanup
- Thread-safe operation

### Extensible Design
- Easy to add new component types
- Configurable interferometer presets
- Modular architecture for future enhancements
- Integration with existing optical simulation tools

## API Reference

### ComponentBinding Class
Manages individual component bindings:
```python
binding = ComponentBinding(usd_prim_path, omnigraph_node_path, component_type)
binding.sync_positions()
binding.is_valid()
```

### OpticalComponentManager
Provides component management utilities:
```python
manager = get_component_manager()
info = manager.get_component_info("Laser")
presets = manager.get_available_presets()
```

## Development

### Adding New Components
1. Add component definition to `OpticalComponentManager.COMPONENT_TYPES`
2. Create corresponding prefab file in `CAD/prefabs/`
3. Add component creation method to main extension class
4. Update UI with new component button

### Adding Interferometer Presets
1. Define component layout in `OpticalComponentManager.INTERFEROMETER_PRESETS`
2. Add creation method to main extension class
3. Update UI with preset button

## Troubleshooting

### Common Issues
1. **Components not appearing**: Check that prefab files exist in `CAD/prefabs/`
2. **Binding not working**: Verify OmniGraph is created and accessible
3. **Position updates not syncing**: Check background thread is running
4. **UI not responding**: Ensure extension dependencies are loaded

### Debug Information
The extension provides detailed logging:
- Component creation status
- Binding validation results
- Position synchronization events
- Error messages and recovery actions

## Future Enhancements

- Ray tracing integration for optical path visualization
- Advanced material properties for realistic rendering
- Simulation result visualization in OmniGraph
- Export capabilities for optical design software
- Integration with physics simulation engines

## License

Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under LicenseRef-NvidiaProprietary.
