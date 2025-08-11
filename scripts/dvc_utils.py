import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DVCLocalManager:
    """Manage DVC operations with local folder storage"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dvc_dir = self.project_root / ".dvc"
        self.storage_dir = self.project_root / "dvc-storage"
        
    def is_initialized(self) -> bool:
        """Check if DVC is initialized"""
        return self.dvc_dir.exists() and (self.dvc_dir / "config").exists()
    
    def add_and_track(self, paths: List[str]) -> Dict[str, bool]:
        """Add files/directories to DVC tracking"""
        results = {}
        
        for path in paths:
            try:
                path_obj = Path(path)
                if path_obj.exists():
                    # Check if already tracked
                    dvc_file = path_obj.with_suffix(path_obj.suffix + '.dvc')
                    if dvc_file.exists():
                        logger.info(f"📁 {path} already tracked by DVC")
                        results[path] = True
                        continue
                    
                    # Add to DVC
                    subprocess.run(["dvc", "add", path], 
                                 check=True, cwd=self.project_root)
                    logger.info(f"✅ Added {path} to DVC tracking")
                    results[path] = True
                else:
                    logger.warning(f"⚠️  Path {path} does not exist")
                    results[path] = False
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to add {path} to DVC: {e}")
                results[path] = False
                
        return results
    
    def push_data(self) -> bool:
        """Push tracked data to DVC remote"""
        try:
            subprocess.run(["dvc", "push"], 
                         check=True, cwd=self.project_root)
            logger.info("✅ Successfully pushed data to DVC remote")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Push failed (might be first run): {e}")
            return False
    
    def pull_data(self) -> bool:
        """Pull tracked data from DVC remote"""
        try:
            subprocess.run(["dvc", "pull"], 
                         check=True, cwd=self.project_root)
            logger.info("✅ Successfully pulled data from DVC remote")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Pull failed (might be first run): {e}")
            return False
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get status of DVC-tracked data"""
        try:
            result = subprocess.run(["dvc", "status"], 
                                  capture_output=True, text=True, 
                                  cwd=self.project_root)
            
            return {
                "clean": result.returncode == 0,
                "output": result.stdout.strip(),
                "changes": result.stdout.strip() != "" if result.returncode == 0 else True
            }
        except Exception as e:
            logger.error(f"❌ Failed to get DVC status: {e}")
            return {"clean": False, "error": str(e)}
    
    def run_pipeline(self, force: bool = False) -> bool:
        """Run DVC pipeline if dvc.yaml exists"""
        dvc_yaml = self.project_root / "dvc.yaml"
        
        if not dvc_yaml.exists():
            logger.info("ℹ️  No dvc.yaml found, skipping pipeline run")
            return True
            
        try:
            cmd = ["dvc", "repro"]
            if force:
                cmd.append("--force")
                
            subprocess.run(cmd, check=True, cwd=self.project_root)
            logger.info("✅ DVC pipeline completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ DVC pipeline failed: {e}")
            return False
    
    def create_data_snapshot(self, tag: str = None) -> str:
        """Create a snapshot of current data state"""
        if not tag:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Create git tag for the current state
            subprocess.run(["git", "tag", f"data-v{tag}"], 
                         check=True, cwd=self.project_root)
            logger.info(f"📸 Created data snapshot: data-v{tag}")
            return tag
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create snapshot: {e}")
            return ""
    
    def get_file_hash(self, file_path: str) -> str:
        """Get hash of a file for change detection"""
        try:
            path_obj = Path(file_path)
            if path_obj.is_file():
                with open(path_obj, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
            elif path_obj.is_dir():
                # Hash directory contents
                hash_md5 = hashlib.md5()
                for file_path in sorted(path_obj.rglob('*')):
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            hash_md5.update(f.read())
                return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"❌ Failed to hash {file_path}: {e}")
            return ""

class ModelVersionManager:
    """Enhanced model versioning with DVC integration"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "model_metadata.json"
        self.dvc_manager = DVCLocalManager()
    
    def save_model_with_versioning(self, model_data: Dict[str, Any]) -> bool:
        """Save model with automatic versioning and DVC tracking"""
        try:
            # Add timestamp and version info
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_data.update({
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "git_commit": self._get_git_commit(),
                "dvc_tracked": True
            })
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            
            # Track with DVC
            self.dvc_manager.add_and_track(["models/"])
            
            logger.info(f"✅ Model v{version} saved and tracked with DVC")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return False
    
    def load_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Load current model metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load model metadata: {e}")
            return None
    
    def compare_model_performance(self, new_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare new model performance with current model"""
        current_metadata = self.load_model_metadata()
        
        if not current_metadata:
            return {
                "should_update": True,
                "reason": "No existing model found"
            }
        
        current_metrics = current_metadata.get("metrics", {})
        
        # Compare key metrics (customize based on your needs)
        key_metric = "r2"  # or "accuracy", "f1", etc.
        
        if key_metric in new_metrics and key_metric in current_metrics:
            improvement = new_metrics[key_metric] - current_metrics[key_metric]
            threshold = 0.01  # 1% improvement threshold
            
            return {
                "should_update": improvement > threshold,
                "improvement": improvement,
                "current_score": current_metrics[key_metric],
                "new_score": new_metrics[key_metric],
                "reason": f"Performance {'improved' if improvement > 0 else 'degraded'} by {improvement:.4f}"
            }
        
        return {
            "should_update": True,
            "reason": "Unable to compare metrics"
        }
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"