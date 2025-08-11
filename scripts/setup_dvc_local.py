import os
import subprocess
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dvc_local():
    """Setup DVC with local folder storage"""
    
    project_root = Path.cwd()
    dvc_dir = project_root / ".dvc"
    storage_dir = project_root / "dvc-storage"
    
    try:
        # Initialize DVC if not already done
        if not dvc_dir.exists():
            logger.info("Initializing DVC...")
            subprocess.run(["dvc", "init", "--no-scm"], check=True)
            logger.info("✅ DVC initialized successfully")
        
        # Create storage directory
        storage_dir.mkdir(exist_ok=True)
        logger.info(f"✅ Created DVC storage directory: {storage_dir}")
        
        # Add local remote if it doesn't exist
        try:
            result = subprocess.run(["dvc", "remote", "list"], 
                                  capture_output=True, text=True)
            if "local-storage" not in result.stdout:
                subprocess.run([
                    "dvc", "remote", "add", "-d", "local-storage", "./dvc-storage"
                ], check=True)
                logger.info("✅ Added local-storage remote")
            else:
                logger.info("ℹ️  Local storage remote already exists")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup remote: {e}")
            return False
        
        # Create necessary directories
        directories = [
            "data/raw",
            "data/processed", 
            "models",
            "reports",
            "logs"
        ]
        
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files to ensure directories are tracked
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                logger.info(f"✅ Created directory: {directory}")
        
        # Update .gitignore for DVC
        gitignore_path = project_root / ".gitignore"
        dvc_gitignore_entries = [
            "",
            "# DVC",
            "/dvc-storage/",
            "/data/raw/*",
            "/data/processed/*",
            "/models/*.joblib",
            "/models/*.pkl", 
            "/models/*.h5",
            "/models/*.pt",
            "/models/*.pth",
            "/reports/*",
            "!models/model_metadata.json",
            "!models/.gitkeep",
            "!data/.gitkeep", 
            "!reports/.gitkeep",
            ".dvc/cache",
            ".dvc/tmp"
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                current_content = f.read()
        else:
            current_content = ""
        
        # Only add entries that don't already exist
        new_entries = []
        for entry in dvc_gitignore_entries:
            if entry not in current_content:
                new_entries.append(entry)
        
        if new_entries:
            with open(gitignore_path, 'a') as f:
                f.write("\n".join(new_entries) + "\n")
            logger.info("✅ Updated .gitignore with DVC entries")
        
        logger.info("🎉 DVC local setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DVC setup failed: {e}")
        return False

def validate_dvc_setup():
    """Validate DVC setup"""
    try:
        # Check DVC installation
        subprocess.run(["dvc", "version"], check=True, capture_output=True)
        logger.info("✅ DVC is installed")
        
        # Check if DVC is initialized
        if not Path(".dvc").exists():
            logger.error("❌ DVC not initialized")
            return False
        
        # Check remote configuration
        result = subprocess.run(["dvc", "remote", "list"], 
                              capture_output=True, text=True, check=True)
        
        if "local-storage" not in result.stdout:
            logger.error("❌ Local storage remote not configured")
            return False
        
        logger.info("✅ DVC remote configured correctly")
        
        # Check storage directory exists
        if not Path("dvc-storage").exists():
            logger.error("❌ DVC storage directory not found")
            return False
        
        logger.info("✅ DVC storage directory exists")
        logger.info("🎉 All DVC validations passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ DVC validation failed: {e}")
        return False

def cleanup_dvc():
    """Cleanup DVC setup (for testing)"""
    try:
        paths_to_remove = [
            ".dvc",
            "dvc-storage", 
            "dvc.lock",
            "data.dvc",
            "models.dvc"
        ]
        
        for path in paths_to_remove:
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.is_dir():
                    shutil.rmtree(path_obj)
                else:
                    path_obj.unlink()
                logger.info(f"🗑️  Removed {path}")
        
        logger.info("🧹 DVC cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            if setup_dvc_local():
                print("✅ DVC local setup completed successfully")
                sys.exit(0)
            else:
                print("❌ DVC setup failed")
                sys.exit(1)
                
        elif command == "validate":
            if validate_dvc_setup():
                print("✅ DVC setup is valid")
                sys.exit(0)
            else:
                print("❌ DVC setup validation failed")
                sys.exit(1)
                
        elif command == "cleanup":
            if cleanup_dvc():
                print("✅ DVC cleanup completed")
                sys.exit(0)
            else:
                print("❌ DVC cleanup failed")
                sys.exit(1)
        else:
            print("Usage: python setup_dvc_local.py {setup|validate|cleanup}")
            sys.exit(1)
    else:
        # Default action
        setup_dvc_local()