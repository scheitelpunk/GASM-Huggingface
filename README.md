---
title: GASM Enhanced - Geometric Language AI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: cc-by-nd-4.0
---

# 🚀 GASM Enhanced - Geometric Attention for Spatial Understanding

> *Bridging natural language and geometric reasoning through SE(3)-invariant neural architectures*

## What Makes This Different?

Traditional AI understands *what* objects are mentioned, but struggles with *where* they are and *how* they relate spatially. GASM changes this.

**GASM** (Geometric Attention for Spatial & Mathematical understanding) represents a breakthrough in AI spatial reasoning:

- **🧠 Advanced NLP**: Goes beyond keywords with spaCy + semantic categorization  
- **📐 Proper 3D Math**: Uses SE(3) Lie groups for mathematically correct spatial relationships
- **🔄 Geometric Optimization**: Minimizes curvature on Riemannian manifolds for optimal layouts
- **✨ Real-time Visualization**: Shows spatial understanding in live 3D geometry

## 🌟 What This Enables

### The Spatial Intelligence Gap
Current language models excel at:
- ✅ "What is a keyboard?" → *An input device*
- ❌ "Where is the keyboard relative to the monitor?" → *Spatial confusion*

GASM bridges this gap through mathematical spatial reasoning.

### Real Applications
This isn't just a demo - GASM addresses actual problems in:
- **🤖 Robotics**: "Move the component above the platform" → Precise 3D coordinates
- **🔬 Scientific Modeling**: "The electron orbits the nucleus" → Proper geometric relationships  
- **🏗️ Engineering**: "Place the support between the beams" → Constraint satisfaction
- **🥽 AR/VR**: Natural language to 3D scene understanding

## 🎯 Try It Yourself

### Watch GASM in Action

Input any sentence with spatial relationships:

> *"The ball lies left of the table next to the computer, while the book sits between the keyboard and the monitor."*

**GASM Output:**
- ✅ **6 entities identified**: ball, table, computer, book, keyboard, monitor
- 🔗 **5 spatial relations**: left_of, next_to, between
- 🌌 **3D geometric layout** with proper SE(3) positioning  
- 📈 **Curvature evolution** showing geometric convergence

### More Examples

**🤖 Robotics**: *"The robotic arm moves the satellite component above the assembly platform."*

**🔬 Scientific**: *"The electron orbits the nucleus while the magnetic field flows through the crystal."*  

**🏠 Everyday**: *"The red car parks between two buildings near the park entrance."*

**🧬 Cutting-Edge Domains**: Try advanced examples from drug design, quantum computing, and manufacturing:
- *"Dock the kinase inhibitor with the phenyl ring parallel to the hinge backbone."*
- *"Embed the fluxonium qubit 5 nm above the ground plane, aligned to the Φ = 0.5 Φ₀ sweet spot."*
- *"Place the aluminum bracket flush against the jig, 5 cm left of the drill bit."*

### What You'll See
1. **Advanced Entity Recognition**: Far beyond simple keyword matching
2. **Spatial Relationship Extraction**: Understands "left of", "between", "above" in context  
3. **3D Visualization**: Real geometric positioning in proper 3D space
4. **Mathematical Convergence**: Curvature evolution showing optimization progress

## 📁 Project Structure

```
GASM-Huggingface/
├── app.py                    # Main Gradio application with complete interface
├── gasm_core.py             # Core GASM implementation with SE(3) math
├── fastapi_endpoint.py      # Optional API endpoints (standalone)
├── utils_weights.py         # Weight persistence utilities (auto-save/load)
├── manage_weights.py        # CLI tool for weight management
├── test_weight_persistence.py # Weight persistence test suite
├── requirements.txt         # Python dependencies
├── gasm_weights.pth         # Auto-generated model weights (gitignored)
├── WEIGHT_PERSISTENCE_README.md # Weight system documentation
└── README.md               # This file
```

## 🧮 The Mathematics Behind GASM

### What Makes It Special

Unlike traditional NLP that treats text as sequences of tokens, GASM understands geometry:

**1. SE(3) Invariant Processing**
- Uses Special Euclidean Group SE(3) for proper 3D transformations
- Maintains mathematical correctness under rotations and translations
- Employs Lie group operations for geometric learning

**2. Advanced Entity Recognition**  
- **spaCy NLP**: Part-of-speech tagging + named entity recognition
- **Semantic Filtering**: Domain-specific vocabularies (robotics, scientific, everyday)
- **Contextual Understanding**: Extracts objects from spatial prepositions

**3. Geometric Optimization**
- **Geodesic Distances**: Shortest paths on SE(3) manifold
- **Discrete Curvature**: Graph Laplacian-based curvature minimization
- **Attention Mechanisms**: SE(3)-invariant geometric relationship learning

**4. Weight Persistence & Reproducibility**
- **Deterministic Weights**: Fixed seed (42) ensures reproducible results
- **Automatic Save/Load**: Persistent model state across sessions
- **Force Regeneration**: Environment variables and CLI flags for control

### Technical Architecture

```
Text → spaCy NLP → Entity Extraction → Semantic Filtering
  ↓
SE(3) Embedding → Attention Mechanism → Geometric Refinement  
  ↓
Constraint Satisfaction → Curvature Optimization → 3D Visualization
```

### Why This Matters

Most AI systems use simple word embeddings that lose spatial meaning. GASM preserves geometric relationships through mathematically principled operations, enabling true spatial understanding.

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

## 🚀 Why This Matters

### Current State of AI
- ✅ Excellent at text understanding and generation
- ✅ Great at image recognition and computer vision  
- ❌ **Struggles with spatial reasoning from language**
- ❌ **Can't bridge text ↔ 3D geometry gap**

### GASM's Contribution
GASM represents a step toward AI that understands space the way humans do - not just as coordinates, but as meaningful geometric relationships between objects in the world.

**Applications on the horizon:**
- 🤖 Robots that understand spatial instructions naturally
- 🏗️ AI architects that reason about 3D spaces from descriptions  
- 🔬 Scientific AI that models physical systems geometrically
- 🎮 Game AI that understands spatial gameplay naturally

## 🛠️ Local Development

### Quick Start

```bash
git clone https://github.com/scheitelpunk/GASM-Huggingface
cd GASM-Huggingface
pip install -r requirements.txt
python app.py
```

### ⚡ Weight Persistence System

GASM now features **automatic weight persistence** for consistent, reproducible results:

**🎯 First Run**: Automatically generates initial weights with deterministic seed
```bash
python app.py
# ✅ Generated initial GASM weights and saved to gasm_weights.pth
```

**🔄 Subsequent Runs**: Loads existing weights for consistent behavior
```bash  
python app.py
# ✅ Loaded GASM weights from gasm_weights.pth
```

**🔧 Weight Management CLI**:
```bash
# Check weight status
python manage_weights.py status

# Force regenerate weights
python manage_weights.py generate --force

# Remove weight file
python manage_weights.py remove
```

**🔄 Force Regeneration Options**:
```bash
# Via environment variable
GASM_FORCE_REGEN=true python app.py

# Via CLI flag  
python app.py --force-regen
```

### 🧪 Testing Weight Persistence
```bash
python test_weight_persistence.py
```

**Benefits**:
- ✅ **Reproducible Results**: Same weights = same outputs across runs
- ⚡ **Faster Startup**: No recomputation after first initialization  
- 🎲 **Deterministic**: Fixed seed (42) ensures identical weights
- 🛡️ **Robust Fallback**: Continues with random weights if persistence fails

The system gracefully handles missing dependencies with intelligent fallbacks.

## 🤝 Contributing

This is active research in spatial AI! We welcome:
- 🐛 Bug reports and edge cases
- 💡 New spatial relationship types  
- 🌍 Additional language support
- 📊 Evaluation datasets
- 🔧 Performance optimizations
- 🧪 Weight persistence improvements and testing

### Development Setup
```bash
# Clone and setup
git clone https://github.com/scheitelpunk/GASM-Huggingface
cd GASM-Huggingface
pip install -r requirements.txt

# Test weight persistence system
python test_weight_persistence.py

# Check current weight status
python manage_weights.py status
```

## 📄 License & Citation

Licensed under CC-BY-NC 4.0. For research use, please cite:

```bibtex
@misc{gasm2025,
  title={GASM: Geometric Attention for Spatial Understanding},
  author={Michael Neuberger, Versino PsiOmega GmbH},
  year={2025},
  url={https://huggingface.co/spaces/scheitelpunk/GASM}
}
```

## 🙏 Built With

- 🤗 **Hugging Face Spaces** - Deployment platform
- 🌐 **spaCy** - Advanced NLP processing
- 🔢 **PyTorch** - Neural network framework with weight persistence
- 📊 **Gradio 4.16.0** - Interactive ML interfaces  
- 📐 **Geomstats** - Geometric computing on manifolds
- ⚡ **FastAPI** - High-performance API endpoints
- 🧪 **Custom Weight Management** - Reproducible model persistence

---

*GASM: Where language meets geometry, and AI begins to understand space.* 🚀

Built by Michael Neuberger, Versino PsiOmega GmbH