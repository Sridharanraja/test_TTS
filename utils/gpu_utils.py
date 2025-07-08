import torch
import logging
import psutil
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUUtils:
    """Utilities for GPU detection and monitoring"""
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get comprehensive GPU information"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "current_device": None,
            "device_name": None,
            "memory_total": 0,
            "memory_allocated": 0,
            "memory_reserved": 0,
            "memory_free": 0,
            "compute_capability": None,
            "driver_version": None,
            "pytorch_version": torch.__version__,
            "recommendations": []
        }
        
        try:
            if torch.cuda.is_available():
                info["device_count"] = torch.cuda.device_count()
                info["current_device"] = torch.cuda.current_device()
                info["device_name"] = torch.cuda.get_device_name()
                
                # Memory information (in MB)
                memory_stats = torch.cuda.memory_stats()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                
                info["memory_total"] = total_memory // 1024 // 1024
                info["memory_allocated"] = allocated_memory // 1024 // 1024
                info["memory_reserved"] = reserved_memory // 1024 // 1024
                info["memory_free"] = (total_memory - reserved_memory) // 1024 // 1024
                
                # Compute capability
                props = torch.cuda.get_device_properties(0)
                info["compute_capability"] = f"{props.major}.{props.minor}"
                
                # Driver version
                try:
                    info["driver_version"] = torch.version.cuda
                except:
                    info["driver_version"] = "Unknown"
                
                # Generate recommendations
                info["recommendations"] = GPUUtils._generate_recommendations(info)
                
                logger.info(f"GPU detected: {info['device_name']} with {info['memory_total']}MB")
            else:
                info["recommendations"] = [
                    "No CUDA GPU detected - using CPU (will be slower)",
                    "For better performance, use a CUDA-capable GPU",
                    "Minimum recommended: 4GB VRAM",
                    "Optimal: 8GB+ VRAM"
                ]
                logger.warning("No CUDA GPU available, falling back to CPU")
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            info["error"] = str(e)
        
        return info
    
    @staticmethod
    def _generate_recommendations(gpu_info: Dict) -> list:
        """Generate performance recommendations based on GPU specs"""
        recommendations = []
        
        memory_gb = gpu_info["memory_total"] / 1024
        
        if memory_gb < 4:
            recommendations.extend([
                "⚠️ Low GPU memory detected",
                "Consider using smaller batch sizes",
                "May need to use CPU for large models"
            ])
        elif memory_gb < 8:
            recommendations.extend([
                "✅ Adequate GPU memory for most tasks",
                "Should handle F5-TTS and E2-TTS well",
                "Monitor memory usage during synthesis"
            ])
        else:
            recommendations.extend([
                "✅ Excellent GPU memory capacity",
                "Can handle large models efficiently",
                "Optimal for high-quality synthesis"
            ])
        
        # Compute capability recommendations
        if gpu_info.get("compute_capability"):
            major, minor = map(int, gpu_info["compute_capability"].split("."))
            if major < 6:
                recommendations.append("⚠️ Older GPU architecture - may be slower")
            elif major >= 7:
                recommendations.append("✅ Modern GPU architecture - optimal performance")
        
        return recommendations
    
    @staticmethod
    def optimize_memory():
        """Optimize GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict:
        """Get current memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "allocated_mb": allocated // 1024 // 1024,
            "reserved_mb": reserved // 1024 // 1024,
            "total_mb": total // 1024 // 1024,
            "free_mb": (total - reserved) // 1024 // 1024,
            "utilization_percent": (reserved / total) * 100
        }
    
    @staticmethod
    def check_system_resources() -> Dict:
        """Check overall system resources"""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / 1024 / 1024 / 1024
            memory_available_gb = memory.available / 1024 / 1024 / 1024
            memory_percent = memory.percent
            
            return {
                "cpu_cores": cpu_count,
                "cpu_usage_percent": cpu_percent,
                "memory_total_gb": round(memory_total_gb, 1),
                "memory_available_gb": round(memory_available_gb, 1),
                "memory_usage_percent": memory_percent,
                "system_suitable": memory_total_gb >= 8 and cpu_count >= 4
            }
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_device_recommendation() -> str:
        """Get device recommendation for TTS inference"""
        if torch.cuda.is_available():
            gpu_info = GPUUtils.get_gpu_info()
            memory_gb = gpu_info["memory_total"] / 1024
            
            if memory_gb >= 8:
                return "cuda"  # Use GPU
            elif memory_gb >= 4:
                return "cuda"  # Use GPU but monitor memory
            else:
                return "cpu"   # Use CPU for safety
        else:
            return "cpu"
    
    @staticmethod
    def estimate_inference_time(model_name: str, text_length: int) -> str:
        """Estimate inference time based on hardware"""
        gpu_info = GPUUtils.get_gpu_info()
        
        # Base time estimates (seconds per 100 characters)
        base_times = {
            "F5-TTS_v1": {"gpu_high": 2, "gpu_low": 4, "cpu": 15},
            "E2-TTS": {"gpu_high": 1.5, "gpu_low": 3, "cpu": 12}
        }
        
        if not gpu_info["cuda_available"]:
            device_type = "cpu"
        elif gpu_info["memory_total"] > 8000:
            device_type = "gpu_high"
        else:
            device_type = "gpu_low"
        
        base_time = base_times.get(model_name, base_times["F5-TTS_v1"])[device_type]
        estimated_time = (text_length / 100) * base_time
        
        if estimated_time < 10:
            return f"~{estimated_time:.0f} seconds"
        elif estimated_time < 60:
            return f"~{estimated_time:.0f} seconds"
        else:
            return f"~{estimated_time/60:.1f} minutes"
