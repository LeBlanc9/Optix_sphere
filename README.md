# OptiX Sphere - Integrating Sphere Monte Carlo Simulation

High-performance Monte Carlo simulation for integrating spheres using NVIDIA OptiX ray tracing.

## Features

- 🚀 **GPU-Accelerated**: Powered by NVIDIA OptiX 9.0 for maximum performance
- 🎯 **High Accuracy**: Double precision arithmetic, < 1% error vs analytical solutions
- 🐍 **Python Bindings**: Modern Python API using pybind11
- 📊 **Flexible**: Support for custom geometries, detectors, and configurations
- 🔬 **Scientific**: Built for optical research and integrating sphere characterization

## Requirements

### For C++ Usage
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- CUDA Toolkit 11.0+
- OptiX SDK 9.0.0
- CMake 3.18+
- C++17 compatible compiler

### For Python Usage
- Python 3.8+
- Above requirements plus:
  - pybind11
  - scikit-build-core
  - numpy

## Installation

### Python Package (Recommended)

```bash
# Install dependencies
pip install pybind11 scikit-build-core numpy

# Install package
pip install .

# Or for development
pip install -e .
```

### C++ Only

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Quick Start (Python)

```python
import optix_sphere as opt
## Examples
See the `examples/` directory:

- `basic_simulation.py` - Simple integrating sphere simulation
- `parameter_sweep.py` - Reflectance parameter sweep

## Performance

Typical performance on RTX 4090:
- 5M rays, 100 avg bounces: ~100-150ms
- Suitable for inverse problems requiring 1000+ forward simulations

## Theory

Uses Goebel formula for validation:

```
E_avg = P / (A_sphere × (1 - ρ_eff))
```

Where:
- `E_avg`: Average irradiance
- `P`: Incident power
- `A_sphere`: Sphere surface area
- `ρ_eff`: Effective reflectance (accounting for ports)

## Project Structure


## Citation

If you use this software in your research, please cite:

```bibtex
@software{optix_sphere,
  title = {OptiX Sphere: GPU-Accelerated Integrating Sphere Simulation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/optix_sphere}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with NVIDIA OptiX ray tracing framework
- Python bindings powered by pybind11
