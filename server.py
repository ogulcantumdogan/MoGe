from fastapi import FastAPI, UploadFile, File, HTTPException, Response
import torch
import numpy as np
from moge.model import MoGeModel
from PIL import Image
import io
import os
import json
import tempfile
from pathlib import Path
import cv2
import trimesh
import trimesh.visual
import time
import uuid

# Import the utilities used in the scripts
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
import utils3d

app = FastAPI()

# Load pretrained model
try:
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
    if torch.cuda.is_available():
        model = model.eval().cuda()
    else:
        model = model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Still create app but will return error when called
    model = None

def preprocess(img: Image.Image, max_size: int = 800) -> torch.Tensor:
    # Resize large images to maintain performance
    width, height = img.size
    larger_size = max(height, width)
    if larger_size > max_size:
        scale = max_size / larger_size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor, np.array(img)

@app.post("/infer/")
async def infer(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor, _ = preprocess(img)
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        with torch.no_grad():
            output = model.infer(tensor)
        
        # Convert to regular Python types for JSON serialization
        result = {
            "depth": output["depth"].cpu().numpy().tolist() if isinstance(output["depth"], torch.Tensor) else output["depth"].tolist(),
            "mask": output["mask"].cpu().numpy().tolist() if isinstance(output["mask"], torch.Tensor) else output["mask"].tolist()
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/infer3d/")
async def infer3d(
    file: UploadFile = File(...),
    threshold: float = 0.03,
    max_size: int = 800,
    resolution_level: int = 9,
    fov_x: float = None
):
    """
    Advanced inference endpoint that returns 3D model data and camera intrinsics.
    Parameters:
    - threshold: Edge removal threshold (0.03 default, smaller removes more edges)
    - max_size: Maximum image dimension for processing
    - resolution_level: Detail level for inference (0-9)
    - fov_x: Optional manual horizontal field of view in degrees
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor, image_np = preprocess(img, max_size=max_size)
        height, width = image_np.shape[:2]
        
        device = tensor.device if hasattr(tensor, 'device') else torch.device('cpu')
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Run inference with optional parameters
        with torch.no_grad():
            output = model.infer(tensor, resolution_level=resolution_level, fov_x=fov_x)
        
        points = output["points"].cpu().numpy()
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()
        
        # Generate normals for better mesh quality
        normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
        
        # Create mesh with edge removal
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            image_np.astype(np.float32) / 255,
            utils3d.numpy.image_uv(width=width, height=height),
            mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask) & 
                        utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
            tri=True
        )
        
        # Follow OpenGL coordinate conventions
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
        
        # Create temporary directory for output files
        run_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.gettempdir(), 'moge')
        temp_dir.mkdir(exist_ok=True)
        
        # Save .glb file
        glb_path = temp_dir / f"{run_id}.glb"
        save_glb(glb_path, vertices, faces, vertex_uvs, image_np)
        
        # Get FOV data
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
        camera_info = {
            "fov_x": float(np.rad2deg(fov_x)),
            "fov_y": float(np.rad2deg(fov_y)),
            "intrinsics": intrinsics.tolist()
        }
        
        # Return JSON with file paths and camera info
        return {
            "glb_url": f"/download/glb/{run_id}",
            "camera": camera_info,
            "info": {
                "image_dimensions": [width, height],
                "processing": {
                    "threshold": threshold,
                    "resolution_level": resolution_level
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/download/glb/{run_id}")
async def download_glb(run_id: str):
    """Download the generated GLB file"""
    temp_dir = Path(tempfile.gettempdir(), 'moge')
    file_path = temp_dir / f"{run_id}.glb"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    try:
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        return Response(
            content=file_content,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f"attachment; filename={run_id}.glb"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/ui")
async def ui_redirect():
    """Redirect to the Gradio UI"""
    return {
        "message": "Gradio UI is available at port 7860",
        "url": "http://localhost:7860"
    }
