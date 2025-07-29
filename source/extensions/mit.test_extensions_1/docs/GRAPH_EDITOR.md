# Graph Editor Integration

This extension includes an interactive graph editor for designing optical systems. The graph editor allows you to create, connect, and configure optical components in a visual interface.

## Features

### Node Types
- **Lens**: Optical lens with configurable focal length
- **Mirror 45°**: 45-degree mirror for beam steering
- **Mirror 90°**: 90-degree mirror for beam steering
- **Laser**: Light source
- **Beam Splitter**: Beam splitter with configurable ratio
- **Camera**: Detector/sensor

### Interactive Features
- **Add Nodes**: Use the "Add Node" menu to place components
- **Connect Components**: Click on two ports in succession to create connections
- **Rotate Nodes**: Right-click nodes to rotate clockwise or counterclockwise
- **Edit Attributes**: Right-click nodes to edit their properties (focal length, split ratio, etc.)
- **Delete**: Right-click to delete nodes or connections, or use the Delete key

### File Operations
- **Save**: Save your optical system design to JSON format
- **Load**: Load previously saved designs
- **Auto-save Directory**: Graphs are saved to the extension's graphs directory

## Usage

1. **Launch the Extension**: Enable the extension in Omniverse
2. **Open the UI**: The extension window will appear with various controls
3. **Launch Graph Editor**: Click the "Launch Graph Editor" button
4. **Design Your System**:
   - Add components from the "Add Node" menu
   - Connect components by clicking ports
   - Configure properties by right-clicking nodes
   - Save your design when finished

## File Format

The graph editor saves designs in JSON format with the following structure:

```json
{
  "nodes": [
    {
      "id": 0,
      "type": "Lens",
      "name": "Lens_1",
      "x": 100.0,
      "y": 200.0,
      "rotation": 0,
      "attributes": {
        "focal_length": 100.0
      }
    }
  ],
  "edges": [
    {
      "source": 0,
      "source_port": "output",
      "target": 1,
      "target_port": "input",
      "attributes": {
        "length_cm": 10.0
      }
    }
  ]
}
```

## Integration with Chromatix

The graph editor is designed to work with the Chromatix optical simulation library. The saved graph files can be used to:

1. **Generate Optical Systems**: Convert graph designs to physical optical setups
2. **Simulate Performance**: Use the graph structure for optical simulations
3. **Export Configurations**: Generate configuration files for other optical simulation tools

## Troubleshooting

### Common Issues

1. **Graph Editor Won't Launch**:
   - Check that PySide6 is properly installed
   - Ensure the extension has proper permissions

2. **Save/Load Issues**:
   - Verify the graphs directory exists and is writable
   - Check file permissions in the extension directory

3. **Connection Errors**:
   - Ensure you're clicking on valid ports
   - Check that the connection makes physical sense (e.g., 90-degree mirrors can only connect to other 90-degree mirrors)

### Debug Information

The extension provides debug output in the console:
- `[mit.test_extensions_1] Graph editor imported successfully`
- `[mit.test_extensions_1] Graph editor launched successfully`
- Error messages if something goes wrong

## Future Enhancements

Planned improvements include:
- Integration with 3D scene objects
- Real-time simulation preview
- Export to other optical design formats
- Component library with pre-built optical systems