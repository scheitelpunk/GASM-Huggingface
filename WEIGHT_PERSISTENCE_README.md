# 🚀 GASM Weight Persistence System

## Overview

The GASM project now includes an **automatic weight persistence system** that generates, saves, and loads model weights consistently across runs. This ensures reproducible behavior and faster startup times after the first initialization.

## 🎯 Key Features

### 1. **Automatic First-Time Weight Generation**
- On first startup, checks if `gasm_weights.pth` exists
- If not found, generates initial weights with deterministic seed (42)
- Saves weights automatically for future use
- Logs: `✅ Generated initial GASM weights and saved to gasm_weights.pth`

### 2. **Always Load Weights After First Generation**
- On subsequent startups, loads existing weights
- Ensures consistent model behavior across sessions
- Maps weights to correct device (CPU/GPU)
- Logs: `✅ Loaded GASM weights from gasm_weights.pth`

### 3. **Force Regeneration Options**
- **Environment variable**: `GASM_FORCE_REGEN=true`
- **CLI flag**: `--force-regen`
- Regenerates weights even if file exists
- Logs: `🔄 Force regeneration requested`

### 4. **Deterministic Weight Generation**
- Uses fixed random seed (42) for reproducibility
- Ensures identical weights across different machines
- Maintains SE(3) math and optimization logic unchanged

## 📁 File Structure

```
GASM-Huggingface/
├── app.py                       # Gradio app with weight persistence
├── fastapi_endpoint.py          # FastAPI with weight persistence  
├── gasm_core.py                # Core GASM implementation (unchanged)
├── utils_weights.py            # Weight persistence utilities (NEW)
├── manage_weights.py           # CLI weight management tool (NEW)
├── test_weight_persistence.py  # Test suite (NEW)
├── gasm_weights.pth            # Auto-generated weight file
└── requirements.txt            # Updated dependencies
```

## 🛠 Usage

### Automatic Usage (Recommended)
Just run the app normally - weights are handled automatically:

```bash
# First run - generates weights
python app.py

# Subsequent runs - loads existing weights  
python app.py
```

### Force Regeneration
```bash
# Via environment variable
GASM_FORCE_REGEN=true python app.py

# Via CLI flag
python app.py --force-regen
```

### Manual Weight Management
```bash
# Check weight file status
python manage_weights.py status

# Generate new weights
python manage_weights.py generate

# Remove weight file
python manage_weights.py remove

# Generate with custom parameters
python manage_weights.py generate --feature-dim 512 --hidden-dim 128 --seed 123
```

## 🔧 Implementation Details

### Core Functions (utils_weights.py)

#### `save_gasm_weights(model, path) -> bool`
- Saves PyTorch model state_dict to file
- Creates directories if needed
- Returns success status
- Logs generation completion

#### `load_gasm_weights(model, path, device) -> bool`  
- Loads state_dict into model with device mapping
- Handles file not found gracefully
- Returns success status
- Logs loading completion

#### `handle_gasm_weights(model, device, path) -> bool`
- Main orchestration function
- Checks for existing weights vs force regeneration
- Calls appropriate save/load functions
- Manages deterministic seed setting

#### `should_force_regenerate() -> bool`
- Checks environment variable `GASM_FORCE_REGEN`
- Checks CLI flag `--force-regen` 
- Returns boolean flag for regeneration

#### `get_weights_info(path) -> dict`
- Returns file status, size, modification time
- Useful for debugging and monitoring
- Safe handling of missing files

### Integration Points

#### app.py Integration
```python
# In _initialize_real_gasm() method:
if WEIGHT_UTILS_AVAILABLE:
    weights_handled = handle_gasm_weights(
        self.gasm_model, 
        self.device, 
        "gasm_weights.pth"
    )
    if not weights_handled:
        logger.warning("⚠️ Weight persistence failed, continuing with random weights")
```

#### FastAPI Integration
```python
# In lifespan() manager:
if (WEIGHT_UTILS_AVAILABLE and 
    hasattr(model_instance, 'gasm_layer') and 
    hasattr(model_instance.gasm_layer, 'gasm_model')):
    
    device = next(model_instance.gasm_layer.gasm_model.parameters()).device
    weights_handled = handle_gasm_weights(
        model_instance.gasm_layer.gasm_model, 
        device, 
        "gasm_weights.pth"
    )
```

## 🚀 Startup Logs

### First Run (Weight Generation)
```
============================================================
🚀 GASM Weight Persistence Status
============================================================
📁 Weight file: gasm_weights.pth
✅ Exists: False
🔄 Force regeneration: False
============================================================
INFO: Initializing real GASM model with weight persistence...
INFO: 🔍 Weight file status: exists=False, size=0MB
INFO: 🎯 First-time startup: generating initial GASM weights
INFO: 🎲 Set deterministic seed (42) for weight generation
INFO: ✅ Generated initial GASM weights and saved to gasm_weights.pth
```

### Subsequent Runs (Weight Loading)
```
============================================================
🚀 GASM Weight Persistence Status  
============================================================
📁 Weight file: gasm_weights.pth
✅ Exists: True
📊 Size: 2.34 MB
🔄 Force regeneration: False
============================================================
INFO: Initializing real GASM model with weight persistence...
INFO: 🔍 Weight file status: exists=True, size=2.34MB
INFO: ✅ Loaded GASM weights from gasm_weights.pth
```

### Force Regeneration
```
============================================================
🚀 GASM Weight Persistence Status
============================================================
📁 Weight file: gasm_weights.pth
✅ Exists: True  
📊 Size: 2.34 MB
🔄 Force regeneration: True
============================================================
INFO: 🔄 Force regeneration requested
INFO: 🔄 Force regenerating weights (removing existing gasm_weights.pth)
INFO: 🎲 Set deterministic seed (42) for weight generation
INFO: ✅ Generated initial GASM weights and saved to gasm_weights.pth
```

## 🧪 Testing

### Run Test Suite
```bash
python test_weight_persistence.py
```

### Expected Output
```
🧪 Testing GASM Weight Persistence System
==================================================
1. Testing import of weight utilities...
✅ Successfully imported weight utilities
2. Testing weight info function...
✅ Weight info function works correctly for non-existent file
3. Testing GASM model import and creation...
✅ Successfully created test GASM model
4. Testing weight saving...
✅ Successfully saved test weights
✅ Weight file info correct: size=0.01MB
5. Testing weight loading...
✅ Successfully loaded test weights
6. Testing handle_gasm_weights function...
✅ handle_gasm_weights successfully generated weights
7. Testing force regeneration detection...
✅ Force regeneration correctly detected via environment variable
✅ Force regeneration correctly detected via CLI flag
8. Cleaning up test files...
✅ Test files cleaned up
==================================================
🎉 All weight persistence tests passed!
```

## ⚙️ Configuration

### Environment Variables
- `GASM_FORCE_REGEN=true` - Force weight regeneration

### CLI Flags  
- `--force-regen` - Force weight regeneration

### File Paths
- Default weight file: `gasm_weights.pth` (in project root)
- Configurable via function parameters
- Auto-gitignored (won't be committed)

## 🎯 Benefits

1. **Reproducible Results**: Same weights = same outputs across runs
2. **Faster Startup**: No weight recomputation after first run  
3. **Consistent Behavior**: Eliminates random initialization variance
4. **Easy Management**: Simple CLI tools for weight management
5. **Backward Compatible**: All existing code works unchanged
6. **Flexible Control**: Environment/CLI override options
7. **Robust Fallback**: Continues working if weight persistence fails

## 🔒 Security & Best Practices

- Weight files are gitignored (not committed to repo)
- Deterministic seed ensures reproducible scientific results
- Graceful fallback to random weights if persistence fails
- Clear logging for debugging and monitoring
- Safe file I/O with error handling
- No modification of core SE(3) mathematics or algorithms

## 🚀 Deployment

### Hugging Face Spaces
- Weight file generated on first HF Space startup
- Persists across Space restarts (if persistent storage available)
- Fallback to random weights if storage unavailable

### Docker/Container Deployment
- Mount persistent volume for weight persistence
- Or use init containers to pre-generate weights
- Environment variables work in containerized environments

### Local Development
- Weights generated in project directory
- Shared across development sessions
- Easy regeneration during development/testing

---

**🎉 The weight persistence system is now fully integrated and ready for production use!**