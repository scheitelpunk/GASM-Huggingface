---
title: GASM-LLM Geometric Language Processing
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: cc-by-nd-4.0
---

# 🧠 GASM Enhanced - Geometric Language Processing

A HuggingFace Space for geometric language processing using GASM (Geometric Attention with Spatial & Mathematical understanding).

## ✨ Features

- **SE(3) Invariant Processing**: Mathematically correct geometric attention mechanisms
- **Real-time Entity Extraction**: Advanced text analysis with spatial relationship detection  
- **Interactive Visualizations**: 3D entity positioning and curvature evolution plots
- **Gradio Interface**: User-friendly web interface for text analysis
- **CPU/GPU Support**: Automatic fallback system with ZeroGPU compatibility

## 🎯 What is GASM?

GASM (Geometric Attention with Spatial & Mathematical understanding) enhances language models by:

1. **Geometric Entity Processing**: Extracts spatial entities and relationships from text
2. **SE(3) Invariant Attention**: Applies proper geometric transformations preserving spatial structure
3. **Curvature Evolution**: Tracks convergence through geometric manifold optimization
4. **3D Visualization**: Renders entity positions in interactive 3D space

## 🚀 Quick Start

### Using the Space

1. **Enter Text**: Input any text with spatial, temporal, or physical relationships
2. **Enable Geometry**: Toggle geometric processing for enhanced analysis
3. **View Results**: See entity extraction, 3D positioning, and curvature evolution
4. **Explore Visualizations**: Interactive plots show geometric convergence

### Example Inputs

Try these examples to see GASM in action:

```
"The robotic arm moves the satellite component above the assembly platform while the crystal detector rotates around its central axis."

"The electron orbits the nucleus while the magnetic field flows through the crystal lattice structure."

"The ball lies left of the table next to the computer, while the book sits between the keyboard and the monitor."
```

## 📁 Project Structure

```
GASM-Huggingface/
├── app.py                    # Main Gradio application with complete interface
├── gasm_core.py             # Core GASM implementation with SE(3) math
├── fastapi_endpoint.py      # Optional API endpoints (standalone)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Technical Implementation

### Core Components

1. **SE3InvariantAttention**: Mathematically correct SE(3) geodesic distance computation
2. **EfficientCurvatureComputation**: Graph Laplacian-based discrete curvature analysis
3. **ConstraintHandler**: Energy-based constraint satisfaction with Lagrange multipliers
4. **RealGASMInterface**: Main processing interface with entity extraction

### Key Features

- **Robust Error Handling**: Graceful fallbacks at every processing step
- **Dependency Management**: Works with or without PyTorch Geometric, Geomstats
- **Memory Efficient**: Optimized for Space deployment constraints
- **Real-time Processing**: Step-by-step debug output with progress tracking

## 🎨 Visualizations

The Space provides two main visualizations:

### 1. Curvature Evolution Plot
- Shows geometric convergence over iterations
- Displays SE(3) manifold optimization progress
- Uses matplotlib with dark theme for clarity

### 2. 3D Entity Space Plot
- Interactive 3D positioning of extracted entities
- Color-coded by entity type (robotic, physical, spatial, etc.)
- Shows relationship connections between entities

## 🔬 How It Works

1. **Text Input**: User provides text for analysis
2. **Entity Extraction**: Regex-based extraction of meaningful entities
3. **Relation Detection**: Identification of spatial, temporal, physical relations
4. **GASM Processing**: If available, real SE(3) forward pass through geometric manifold
5. **Visualization**: Generate curvature evolution and 3D entity plots
6. **Results**: Comprehensive analysis with JSON output

## ⚡ Performance

- **CPU Mode**: Optimized for HuggingFace Spaces CPU allocation
- **GPU Fallback**: Automatic ZeroGPU usage when available
- **Memory Efficient**: ~430MB total memory footprint
- **Fast Processing**: 0.1-0.8s processing time depending on text length

## 🛠️ Local Development

To run locally:

```bash
git clone <this-repo>
cd GASM-Huggingface

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📊 Space Configuration

This Space is configured with:
- **SDK**: Gradio 4.44.1+
- **Python**: 3.8+
- **GPU**: ZeroGPU compatible (A10G/T4 fallback)
- **Memory**: 16GB RAM allocation
- **Storage**: Persistent storage for model caching

## 🔍 API Endpoints

The Space also exposes FastAPI endpoints (when fastapi_endpoint.py is run separately):

- `POST /process`: Process text with geometric enhancement
- `GET /health`: Health check and memory usage
- `GET /info`: Model configuration information

## 📈 Use Cases

Perfect for analyzing:

- **Technical Documentation**: Spatial relationships in engineering texts
- **Scientific Literature**: Physical phenomena and experimental setups  
- **Educational Content**: Geometry and physics explanations
- **Robotic Systems**: Assembly instructions and spatial configurations

## 🎯 Model Details

- **Base Architecture**: Built on transformer foundations
- **Geometric Processing**: SE(3) Lie group operations
- **Attention Mechanism**: Geodesic distance-based attention weighting
- **Curvature Computation**: Discrete Gaussian curvature via graph Laplacian
- **Constraint Handling**: Energy minimization with Lagrange multipliers

## 📄 License

Licensed under CC-BY-NC 4.0. All rights reserved, Versino PsiOmega GmbH.

## 🙏 Acknowledgments

- HuggingFace for Spaces platform
- PyTorch and PyTorch Geometric teams
- Geomstats geometric computing library
- Gradio for the intuitive interface framework

---

**Made with ❤️ by the Versino PsiOmega development team**

*Try the Space above to see geometric language processing in action!*