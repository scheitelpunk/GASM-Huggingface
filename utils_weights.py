"""
GASM Weight Persistence Utilities
Handles automatic weight generation, saving, and loading
"""

import torch
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def save_gasm_weights(model: torch.nn.Module, path: str) -> bool:
    """
    Save GASM model weights to file
    
    Args:
        model: PyTorch model to save
        path: Path to save weights to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), path)
        logger.info(f"✅ Generated initial GASM weights and saved to {path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save GASM weights to {path}: {e}")
        return False

def load_gasm_weights(model: torch.nn.Module, path: str, device: torch.device) -> bool:
    """
    Load GASM model weights from file
    
    Args:
        model: PyTorch model to load weights into
        path: Path to load weights from
        device: Device to map weights to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(path):
            logger.warning(f"⚠️ GASM weights file not found: {path}")
            return False
            
        # Load state dict with device mapping
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"✅ Loaded GASM weights from {path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load GASM weights from {path}: {e}")
        return False

def should_force_regenerate() -> bool:
    """
    Check if weights should be forcefully regenerated
    
    Returns:
        bool: True if weights should be regenerated
    """
    import sys
    
    # Check environment variable
    env_regen = os.getenv('GASM_FORCE_REGEN', '').lower() == 'true'
    
    # Check CLI flag
    cli_regen = '--force-regen' in sys.argv
    
    if env_regen or cli_regen:
        logger.info("🔄 Force regeneration requested")
        return True
        
    return False

def handle_gasm_weights(model: torch.nn.Module, device: torch.device, 
                       weights_path: str = "gasm_weights.pth") -> bool:
    """
    Main weight handling function - generates, saves, or loads weights as needed
    
    Args:
        model: PyTorch GASM model
        device: Device to use
        weights_path: Path to weights file
        
    Returns:
        bool: True if weights were successfully handled
    """
    force_regen = should_force_regenerate()
    weights_exist = os.path.exists(weights_path)
    
    # Case 1: Force regeneration requested
    if force_regen:
        if weights_exist:
            logger.info(f"🔄 Force regenerating weights (removing existing {weights_path})")
            try:
                os.remove(weights_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not remove existing weights file: {e}")
        
        # Generate new weights with deterministic seed
        torch.manual_seed(42)
        logger.info("🎲 Set deterministic seed (42) for weight generation")
        
        # Model is already initialized with random weights, just save them
        return save_gasm_weights(model, weights_path)
    
    # Case 2: Weights exist, load them
    elif weights_exist:
        return load_gasm_weights(model, weights_path, device)
    
    # Case 3: First time - generate and save weights
    else:
        logger.info("🎯 First-time startup: generating initial GASM weights")
        
        # Set deterministic seed for reproducible weights
        torch.manual_seed(42)
        logger.info("🎲 Set deterministic seed (42) for weight generation")
        
        # Model is already initialized with random weights, just save them
        return save_gasm_weights(model, weights_path)

def get_weights_info(weights_path: str = "gasm_weights.pth") -> Dict[str, Any]:
    """
    Get information about the weights file
    
    Args:
        weights_path: Path to weights file
        
    Returns:
        dict: Information about weights file
    """
    info = {
        'exists': os.path.exists(weights_path),
        'path': weights_path,
        'size_mb': 0,
        'modified_time': None
    }
    
    if info['exists']:
        try:
            stat = os.stat(weights_path)
            info['size_mb'] = round(stat.st_size / (1024 * 1024), 2)
            info['modified_time'] = stat.st_mtime
        except Exception as e:
            logger.warning(f"⚠️ Could not get weights file stats: {e}")
    
    return info