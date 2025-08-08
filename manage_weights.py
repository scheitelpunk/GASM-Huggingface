#!/usr/bin/env python3
"""
GASM Weight Management CLI
Simple command-line interface for managing GASM model weights
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print a nice banner"""
    print("=" * 60)
    print("🚀 GASM Weight Management CLI")
    print("=" * 60)

def status_command(args):
    """Show status of weight files"""
    try:
        from utils_weights import get_weights_info, should_force_regenerate
        
        weights_path = args.weights_file
        info = get_weights_info(weights_path)
        force_regen = should_force_regenerate()
        
        print(f"📁 Weight file: {weights_path}")
        print(f"✅ Exists: {info['exists']}")
        
        if info['exists']:
            print(f"📊 Size: {info['size_mb']} MB")
            if info['modified_time']:
                mod_time = datetime.fromtimestamp(info['modified_time'])
                print(f"🕐 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"🔄 Force regeneration flags active: {force_regen}")
        
        # Check if we have CLI flag
        if '--force-regen' in sys.argv:
            print("   - CLI flag --force-regen detected")
        
        # Check environment variable
        env_flag = os.getenv('GASM_FORCE_REGEN', '').lower() == 'true'
        if env_flag:
            print("   - Environment variable GASM_FORCE_REGEN=true")
        
        if not force_regen and not info['exists']:
            print("\n💡 Tip: Run with --generate to create initial weights")
        elif info['exists'] and not force_regen:
            print("\n💡 Weights will be loaded on next app startup")
        
    except ImportError as e:
        logger.error(f"❌ Cannot import weight utilities: {e}")
        return False
    
    return True

def generate_command(args):
    """Generate new weights"""
    try:
        from utils_weights import save_gasm_weights, get_weights_info
        
        weights_path = args.weights_file
        
        # Check if file exists and warn
        if os.path.exists(weights_path) and not args.force:
            print(f"⚠️  Weight file {weights_path} already exists!")
            print("   Use --force to overwrite, or --remove first")
            return False
        
        # Try to import GASM
        try:
            from gasm_core import GASM
            print("📦 Creating GASM model...")
            
            model = GASM(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                output_dim=3,
                num_heads=4,
                max_iterations=6,
                dropout=0.1
            )
            
        except ImportError as e:
            logger.warning(f"⚠️  Cannot import GASM core: {e}")
            print("📦 Creating dummy model for testing...")
            
            class DummyGASM(torch.nn.Module):
                def __init__(self, feature_dim, hidden_dim):
                    super().__init__()
                    self.linear1 = torch.nn.Linear(feature_dim, hidden_dim)
                    self.linear2 = torch.nn.Linear(hidden_dim, 3)
                
                def forward(self, x):
                    return self.linear2(torch.relu(self.linear1(x)))
            
            model = DummyGASM(args.feature_dim, args.hidden_dim)
        
        # Set deterministic seed if specified
        if args.seed is not None:
            torch.manual_seed(args.seed)
            print(f"🎲 Set random seed to {args.seed}")
        
        # Move to CPU
        device = torch.device('cpu')
        model = model.to(device)
        
        # Save weights
        success = save_gasm_weights(model, weights_path)
        
        if success:
            info = get_weights_info(weights_path)
            print(f"✅ Generated weights saved to {weights_path}")
            print(f"📊 File size: {info['size_mb']} MB")
            return True
        else:
            print(f"❌ Failed to generate weights")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Cannot import required modules: {e}")
        return False

def remove_command(args):
    """Remove weight file"""
    weights_path = args.weights_file
    
    if not os.path.exists(weights_path):
        print(f"ℹ️  Weight file {weights_path} does not exist")
        return True
    
    if not args.force:
        response = input(f"⚠️  Are you sure you want to remove {weights_path}? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return False
    
    try:
        os.remove(weights_path)
        print(f"✅ Removed {weights_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to remove {weights_path}: {e}")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GASM Weight Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show weight file status
  %(prog)s generate                  # Generate new weights
  %(prog)s generate --force          # Overwrite existing weights  
  %(prog)s remove                    # Remove weight file
  %(prog)s remove --force            # Remove without confirmation
        """
    )
    
    parser.add_argument(
        '--weights-file', '-w',
        default='gasm_weights.pth',
        help='Path to weights file (default: gasm_weights.pth)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show weight file status')
    
    # Generate command  
    generate_parser = subparsers.add_parser('generate', help='Generate new weights')
    generate_parser.add_argument(
        '--force', '-f', 
        action='store_true',
        help='Overwrite existing weight file'
    )
    generate_parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for weight generation (default: 42)'
    )
    generate_parser.add_argument(
        '--feature-dim',
        type=int,
        default=768,
        help='Feature dimension for model (default: 768)'
    )
    generate_parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension for model (default: 256)'
    )
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove weight file')
    remove_parser.add_argument(
        '--force', '-f',
        action='store_true', 
        help='Remove without confirmation'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    print_banner()
    
    success = False
    if args.command == 'status':
        success = status_command(args)
    elif args.command == 'generate':
        success = generate_command(args)
    elif args.command == 'remove':
        success = remove_command(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())