import io
import cv2  # OpenCV
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# --- 1. Parameters ---
HIGH_QUALITY_PALETTE = 256
LOW_QUALITY_PALETTE = 16

# --- 2. Initialize FastAPI App ---
app = FastAPI(title="Saliency Compressor API")

# --- 3. Add CORS Middleware ---
# This allows your Firebase frontend to call this Render backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 4. Saliency Logic ---
def get_saliency_mask(pil_image):
    """Finds the main subject and returns a black-and-white PIL mask."""
    cv_image_rgb = np.array(pil_image)
    cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliency_map) = saliency.computeSaliency(cv_image_bgr)
    
    saliency_map_8bit = (saliency_map * 255).astype("uint8")

    _, binary_mask = cv2.threshold(
        saliency_map_8bit, 0, 255, 
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    
    return Image.fromarray(binary_mask).convert('L')

# --- 5. Quantization Logic ---
def quantize_image(pil_image, palette_size):
    """Reduces the image to a specific number of colors."""
    paletted_img = pil_image.convert('P', palette=Image.ADAPTIVE, colors=palette_size)
    return paletted_img.convert('RGB')

# --- 6. The API Endpoint ---
@app.post("/compress/")
async def compress_image(file: UploadFile = File(...)):
    """
    Accepts an image, runs the novel compression pipeline,
    and returns the compressed image as a file.
    """
    
    contents = await file.read()
    original_pil_img = Image.open(io.BytesIO(contents)).convert('RGB')

    saliency_mask = get_saliency_mask(original_pil_img)
    high_q_version = quantize_image(original_pil_img, HIGH_QUALITY_PALETTE)
    low_q_version = quantize_image(original_pil_img, LOW_QUALITY_PALETTE)
    
    hybrid_rgb_img = Image.composite(high_q_version, low_q_version, saliency_mask)

    final_hybrid_img = hybrid_rgb_img.convert(
        'P', palette=Image.ADAPTIVE, colors=HIGH_QUALITY_PALETTE
    )

    img_buffer = io.BytesIO()
    final_hybrid_img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    return StreamingResponse(
        img_buffer, 
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=compressed.png"}
    )

@app.get("/")
def read_root():
    return {"status": "Saliency Compressor API is online."}
