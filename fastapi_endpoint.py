"""
FastAPI Endpoint for GASM-LLM Integration

This module provides a FastAPI endpoint that can be used with OpenAI's CustomGPT
to access GASM-enhanced language processing capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import torch
import logging
import asyncio
from datetime import datetime
import json
import os
from contextlib import asynccontextmanager

try:
    from gasm_llm_layer import GASMEnhancedLLM, GASMTokenEmbedding
    from gasm.utils import check_se3_invariance
    from gasm.core import GASM
    GASM_LLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GASM LLM layer not available: {e}")
    GASM_LLM_AVAILABLE = False

# Import weight persistence utilities
try:
    from utils_weights import handle_gasm_weights, get_weights_info, should_force_regenerate
    WEIGHT_UTILS_AVAILABLE = True
    logger.info("✅ Weight persistence utilities loaded")
except ImportError as e:
    logger.warning(f"⚠️ Weight utilities not available: {e}")
    WEIGHT_UTILS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for FastAPI app
    """
    global model_instance
    
    # Startup
    logger.info("Loading GASM-LLM model with weight persistence...")
    
    # Log weight persistence status
    if WEIGHT_UTILS_AVAILABLE:
        weights_info = get_weights_info("gasm_weights.pth")
        force_regen = should_force_regenerate()
        
        logger.info("=" * 60)
        logger.info("🚀 GASM Weight Persistence Status (FastAPI)")
        logger.info("=" * 60)
        logger.info(f"📁 Weight file: gasm_weights.pth")
        logger.info(f"✅ Exists: {weights_info['exists']}")
        if weights_info['exists']:
            logger.info(f"📊 Size: {weights_info['size_mb']} MB")
        logger.info(f"🔄 Force regeneration: {force_regen}")
        logger.info("=" * 60)
    
    try:
        if not GASM_LLM_AVAILABLE:
            logger.warning("GASM LLM layer not available, skipping model initialization")
            model_instance = None
        else:
            model_instance = GASMEnhancedLLM(
                base_model_name="distilbert-base-uncased",
                gasm_hidden_dim=256,
                gasm_output_dim=128,
                enable_geometry=True
            )
            
            # Apply weight persistence to the GASM component if available
            if (WEIGHT_UTILS_AVAILABLE and 
                hasattr(model_instance, 'gasm_layer') and 
                hasattr(model_instance.gasm_layer, 'gasm_model')):
                
                device = next(model_instance.gasm_layer.gasm_model.parameters()).device
                weights_handled = handle_gasm_weights(
                    model_instance.gasm_layer.gasm_model, 
                    device, 
                    "gasm_weights.pth"
                )
                if not weights_handled:
                    logger.warning("⚠️ Weight persistence failed for FastAPI GASM model")
            
            logger.info("Model loaded successfully")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_instance = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    model_instance = None


# Create FastAPI app
app = FastAPI(
    title="GASM-LLM API",
    description="API for GASM-enhanced Large Language Model processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TextProcessingRequest(BaseModel):
    """Request model for text processing"""
    text: str = Field(..., description="Text to process", min_length=1, max_length=10000)
    enable_geometry: bool = Field(True, description="Enable geometric processing")
    return_embeddings: bool = Field(False, description="Return raw embeddings")
    return_geometry: bool = Field(False, description="Return geometric information")
    max_length: int = Field(512, description="Maximum sequence length", ge=1, le=2048)
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration overrides")


class GeometricAnalysisRequest(BaseModel):
    """Request model for geometric analysis"""
    text: str = Field(..., description="Text to analyze geometrically")
    analysis_type: str = Field("full", description="Type of analysis: 'full', 'curvature', 'invariance'")
    num_invariance_tests: int = Field(10, description="Number of invariance tests", ge=1, le=100)
    tolerance: float = Field(1e-3, description="Tolerance for invariance tests", ge=1e-6, le=1e-1)


class ComparisonRequest(BaseModel):
    """Request model for comparing geometric vs standard processing"""
    text: str = Field(..., description="Text to compare")
    metrics: List[str] = Field(["embedding_norm", "attention_patterns", "geometric_consistency"], 
                               description="Metrics to compare")


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing"""
    texts: List[str] = Field(..., description="List of texts to process", min_items=1, max_items=100)
    enable_geometry: bool = Field(True, description="Enable geometric processing")
    return_summary: bool = Field(True, description="Return summary statistics")


class TextProcessingResponse(BaseModel):
    """Response model for text processing"""
    success: bool
    timestamp: str
    processing_time: float
    text_length: int
    model_info: Dict[str, Any]
    embedding_stats: Dict[str, float]
    geometric_stats: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[List[float]]] = None
    geometric_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GeometricAnalysisResponse(BaseModel):
    """Response model for geometric analysis"""
    success: bool
    timestamp: str
    analysis_type: str
    curvature_analysis: Optional[Dict[str, Any]] = None
    invariance_results: Optional[Dict[str, Any]] = None
    geometric_properties: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ComparisonResponse(BaseModel):
    """Response model for comparison"""
    success: bool
    timestamp: str
    geometric_results: Dict[str, Any]
    standard_results: Dict[str, Any]
    comparison_metrics: Dict[str, Any]
    error: Optional[str] = None


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing"""
    success: bool
    timestamp: str
    num_texts: int
    processing_times: List[float]
    batch_summary: Dict[str, Any]
    individual_results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    memory_usage: Dict[str, Any]
    uptime: str


def get_model():
    """
    Dependency to get the model instance
    """
    global model_instance
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint
    """
    return {
        "message": "GASM-LLM API",
        "version": "1.0.0",
        "description": "API for GASM-enhanced Large Language Model processing",
        "endpoints": {
            "process": "POST /process - Process text with geometric enhancement",
            "analyze": "POST /analyze - Perform geometric analysis",
            "compare": "POST /compare - Compare geometric vs standard processing",
            "batch": "POST /batch - Process multiple texts",
            "health": "GET /health - Health check",
            "info": "GET /info - Model information"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    global model_instance
    
    # Check memory usage
    memory_info = {}
    if torch.cuda.is_available():
        memory_info["gpu_memory"] = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }
    
    # Check system memory (simplified)
    import psutil
    memory_info["system_memory"] = {
        "used": psutil.virtual_memory().used,
        "total": psutil.virtual_memory().total,
        "percent": psutil.virtual_memory().percent
    }
    
    return HealthResponse(
        status="healthy" if model_instance is not None else "unhealthy",
        model_loaded=model_instance is not None,
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        memory_usage=memory_info,
        uptime=datetime.now().isoformat()
    )


@app.get("/info", response_model=Dict[str, Any])
async def model_info(model: GASMEnhancedLLM = Depends(get_model)):
    """
    Get model information
    """
    return {
        "model_name": model.base_model_name,
        "geometry_enabled": model.enable_geometry,
        "device": str(next(model.parameters()).device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
        "gasm_config": {
            "hidden_dim": getattr(model.gasm_embedding.gasm, 'hidden_dim', None) if hasattr(model, 'gasm_embedding') else None,
            "output_dim": getattr(model.gasm_embedding.gasm, 'output_dim', None) if hasattr(model, 'gasm_embedding') else None,
            "max_iterations": getattr(model.gasm_embedding.gasm, 'max_iterations', None) if hasattr(model, 'gasm_embedding') else None,
        }
    }


@app.post("/process", response_model=TextProcessingResponse)
async def process_text(
    request: TextProcessingRequest,
    model: GASMEnhancedLLM = Depends(get_model)
):
    """
    Process text with GASM-enhanced LLM
    """
    start_time = datetime.now()
    
    try:
        # Configure model
        model.enable_geometry = request.enable_geometry
        
        # Process text
        outputs = model.encode_text(
            request.text,
            return_geometry=request.return_geometry
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract embeddings
        embeddings = outputs['last_hidden_state']
        embedding_stats = {
            "shape": list(embeddings.shape),
            "mean": float(embeddings.mean()),
            "std": float(embeddings.std()),
            "min": float(embeddings.min()),
            "max": float(embeddings.max()),
            "norm": float(torch.norm(embeddings))
        }
        
        # Prepare response
        response = TextProcessingResponse(
            success=True,
            timestamp=start_time.isoformat(),
            processing_time=processing_time,
            text_length=len(request.text),
            model_info={
                "model_name": model.base_model_name,
                "geometry_enabled": request.enable_geometry,
                "device": str(next(model.parameters()).device)
            },
            embedding_stats=embedding_stats
        )
        
        # Add embeddings if requested
        if request.return_embeddings:
            response.embeddings = embeddings.detach().cpu().numpy().tolist()
        
        # Add geometric information if available
        if request.return_geometry and 'geometric_info' in outputs:
            geometric_info = outputs['geometric_info']
            if geometric_info:
                response.geometric_info = {
                    "num_sequences": len(geometric_info),
                    "has_curvature": any('output' in info for info in geometric_info),
                    "has_constraints": any('constraints' in info for info in geometric_info),
                    "has_relations": any('relations' in info for info in geometric_info)
                }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return TextProcessingResponse(
            success=False,
            timestamp=start_time.isoformat(),
            processing_time=(datetime.now() - start_time).total_seconds(),
            text_length=len(request.text),
            model_info={},
            embedding_stats={},
            error=str(e)
        )


@app.post("/analyze", response_model=GeometricAnalysisResponse)
async def analyze_geometry(
    request: GeometricAnalysisRequest,
    model: GASMEnhancedLLM = Depends(get_model)
):
    """
    Perform geometric analysis of text
    """
    start_time = datetime.now()
    
    try:
        # Enable geometry for analysis
        model.enable_geometry = True
        
        # Process text with geometric information
        outputs = model.encode_text(request.text, return_geometry=True)
        
        response = GeometricAnalysisResponse(
            success=True,
            timestamp=start_time.isoformat(),
            analysis_type=request.analysis_type
        )
        
        # Perform requested analysis
        if request.analysis_type in ["full", "curvature"]:
            # Curvature analysis
            geometric_info = outputs.get('geometric_info', [])
            if geometric_info:
                curvature_stats = []
                for info in geometric_info:
                    if 'output' in info:
                        geo_output = info['output']
                        curvature_norm = torch.norm(geo_output, dim=1)
                        curvature_stats.append({
                            "mean": float(curvature_norm.mean()),
                            "std": float(curvature_norm.std()),
                            "min": float(curvature_norm.min()),
                            "max": float(curvature_norm.max())
                        })
                
                response.curvature_analysis = {
                    "per_sequence": curvature_stats,
                    "global_stats": {
                        "num_sequences": len(curvature_stats),
                        "avg_mean_curvature": sum(s["mean"] for s in curvature_stats) / len(curvature_stats) if curvature_stats else 0
                    }
                }
        
        if request.analysis_type in ["full", "invariance"]:
            # SE(3) invariance analysis
            try:
                # Create simple test data for invariance check
                test_points = torch.randn(10, 3)
                test_features = torch.randn(10, model.base_model.config.hidden_size)
                test_relations = torch.randn(10, 10, 16)
                
                # Test with simplified model for invariance
                gasm_model = GASM(
                    feature_dim=model.base_model.config.hidden_size,
                    hidden_dim=256,
                    output_dim=3
                )
                
                is_invariant = check_se3_invariance(
                    gasm_model,
                    test_points,
                    test_features,
                    test_relations,
                    num_tests=request.num_invariance_tests,
                    tolerance=request.tolerance
                )
                
                response.invariance_results = {
                    "is_invariant": is_invariant,
                    "num_tests": request.num_invariance_tests,
                    "tolerance": request.tolerance,
                    "test_type": "SE(3) invariance"
                }
                
            except Exception as e:
                response.invariance_results = {
                    "is_invariant": None,
                    "error": str(e)
                }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in geometric analysis: {e}")
        return GeometricAnalysisResponse(
            success=False,
            timestamp=start_time.isoformat(),
            analysis_type=request.analysis_type,
            error=str(e)
        )


@app.post("/compare", response_model=ComparisonResponse)
async def compare_processing(
    request: ComparisonRequest,
    model: GASMEnhancedLLM = Depends(get_model)
):
    """
    Compare geometric vs standard processing
    """
    start_time = datetime.now()
    
    try:
        # Process with geometry
        model.enable_geometry = True
        geometric_outputs = model.encode_text(request.text, return_geometry=True)
        
        # Process without geometry
        model.enable_geometry = False
        standard_outputs = model.encode_text(request.text, return_geometry=False)
        
        # Extract results
        geometric_embeddings = geometric_outputs['last_hidden_state']
        standard_embeddings = standard_outputs['last_hidden_state']
        
        # Calculate comparison metrics
        comparison_metrics = {}
        
        if "embedding_norm" in request.metrics:
            comparison_metrics["embedding_norm"] = {
                "geometric": float(torch.norm(geometric_embeddings)),
                "standard": float(torch.norm(standard_embeddings)),
                "ratio": float(torch.norm(geometric_embeddings) / torch.norm(standard_embeddings))
            }
        
        if "attention_patterns" in request.metrics:
            # Simplified attention pattern comparison
            geo_attention = torch.softmax(geometric_embeddings @ geometric_embeddings.transpose(-2, -1), dim=-1)
            std_attention = torch.softmax(standard_embeddings @ standard_embeddings.transpose(-2, -1), dim=-1)
            
            comparison_metrics["attention_patterns"] = {
                "geometric_entropy": float(torch.sum(-geo_attention * torch.log(geo_attention + 1e-9))),
                "standard_entropy": float(torch.sum(-std_attention * torch.log(std_attention + 1e-9))),
                "pattern_difference": float(torch.norm(geo_attention - std_attention))
            }
        
        if "geometric_consistency" in request.metrics:
            comparison_metrics["geometric_consistency"] = {
                "has_geometric_info": 'geometric_info' in geometric_outputs,
                "embedding_difference": float(torch.norm(geometric_embeddings - standard_embeddings)),
                "relative_change": float(torch.norm(geometric_embeddings - standard_embeddings) / torch.norm(standard_embeddings))
            }
        
        return ComparisonResponse(
            success=True,
            timestamp=start_time.isoformat(),
            geometric_results={
                "embedding_stats": {
                    "shape": list(geometric_embeddings.shape),
                    "mean": float(geometric_embeddings.mean()),
                    "std": float(geometric_embeddings.std()),
                    "norm": float(torch.norm(geometric_embeddings))
                }
            },
            standard_results={
                "embedding_stats": {
                    "shape": list(standard_embeddings.shape),
                    "mean": float(standard_embeddings.mean()),
                    "std": float(standard_embeddings.std()),
                    "norm": float(torch.norm(standard_embeddings))
                }
            },
            comparison_metrics=comparison_metrics
        )
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return ComparisonResponse(
            success=False,
            timestamp=start_time.isoformat(),
            geometric_results={},
            standard_results={},
            comparison_metrics={},
            error=str(e)
        )


@app.post("/batch", response_model=BatchProcessingResponse)
async def batch_process(
    request: BatchProcessingRequest,
    model: GASMEnhancedLLM = Depends(get_model)
):
    """
    Process multiple texts in batch
    """
    start_time = datetime.now()
    
    try:
        model.enable_geometry = request.enable_geometry
        
        processing_times = []
        individual_results = []
        
        for i, text in enumerate(request.texts):
            text_start = datetime.now()
            
            outputs = model.encode_text(text, return_geometry=False)
            embeddings = outputs['last_hidden_state']
            
            processing_time = (datetime.now() - text_start).total_seconds()
            processing_times.append(processing_time)
            
            if not request.return_summary:
                individual_results.append({
                    "text_index": i,
                    "text_length": len(text),
                    "processing_time": processing_time,
                    "embedding_norm": float(torch.norm(embeddings))
                })
        
        # Calculate batch summary
        batch_summary = {
            "total_texts": len(request.texts),
            "total_processing_time": sum(processing_times),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "texts_per_second": len(request.texts) / sum(processing_times),
            "geometry_enabled": request.enable_geometry,
            "total_characters": sum(len(text) for text in request.texts),
            "average_text_length": sum(len(text) for text in request.texts) / len(request.texts)
        }
        
        return BatchProcessingResponse(
            success=True,
            timestamp=start_time.isoformat(),
            num_texts=len(request.texts),
            processing_times=processing_times,
            batch_summary=batch_summary,
            individual_results=individual_results if not request.return_summary else None
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return BatchProcessingResponse(
            success=False,
            timestamp=start_time.isoformat(),
            num_texts=len(request.texts),
            processing_times=[],
            batch_summary={},
            error=str(e)
        )


# Development endpoints for weight management
@app.get("/download-weights")
async def download_weights():
    """
    Download the current GASM weight file
    """
    import os
    from fastapi.responses import FileResponse
    
    weight_file = "gasm_weights.pth"
    
    if not os.path.exists(weight_file):
        raise HTTPException(
            status_code=404, 
            detail="Weight file not found. Model may not be initialized yet."
        )
    
    try:
        file_size = os.path.getsize(weight_file)
        logger.info(f"📥 Weight file download requested: {weight_file} ({file_size} bytes)")
        
        return FileResponse(
            weight_file,
            filename="gasm_weights.pth",
            media_type="application/octet-stream",
            headers={
                "Content-Description": "GASM Model Weights",
                "Content-Disposition": "attachment; filename=gasm_weights.pth"
            }
        )
    except Exception as e:
        logger.error(f"Error downloading weights: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading weights: {str(e)}")

@app.get("/debug-info")
async def debug_info():
    """
    Get container debug information
    """
    import os
    import sys
    from datetime import datetime
    
    try:
        debug_data = {
            "container_info": {
                "working_directory": os.getcwd(),
                "python_version": sys.version,
                "weight_file_exists": os.path.exists("gasm_weights.pth")
            },
            "files": [],
            "weight_file": {},
            "system": {
                "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False
            }
        }
        
        # List files in working directory
        try:
            for file in sorted(os.listdir(".")):
                file_path = os.path.join(".", file)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    debug_data["files"].append({
                        "name": file,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        except Exception as e:
            debug_data["files_error"] = str(e)
        
        # Weight file specific info
        weight_file = "gasm_weights.pth"
        if os.path.exists(weight_file):
            stat = os.stat(weight_file)
            debug_data["weight_file"] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "full_path": os.path.abspath(weight_file),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        else:
            debug_data["weight_file"] = {"exists": False}
        
        return debug_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting debug info: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


# OpenAPI customization for CustomGPT
@app.get("/openapi.json")
async def custom_openapi():
    """
    Custom OpenAPI schema for CustomGPT integration
    """
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="GASM-LLM API",
        version="1.0.0",
        description="API for GASM-enhanced Large Language Model processing with geometric inference capabilities",
        routes=app.routes,
    )
    
    # Add custom metadata for CustomGPT
    openapi_schema["info"]["x-logo"] = {
        "url": "https://huggingface.co/spaces/your-username/gasm-llm/resolve/main/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )