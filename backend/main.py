import io
import cv2
import numpy as np
from PIL import Image
import zipfile  # Import the zipfile module
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# --- 1. Parameters ---
HIGH_QUALITY_PALETTE = 256
LOW_QUALITY_PALETTE = 16

# --- 2. Initialize FastAPI App ---
app = FastAPI(title="Saliency Compressor API")

# --- 3. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Saliency Logic ---
def get_saliency_mask(pil_image):
    cv_image_rgb = np.array(pil_image)
    cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB_BGR)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliency_map) = saliency.computeSaliency(cv_image_bgr)
    
    saliency_map_8bit = (saliency_map * 255).astype("uint8")

    # --- New Post-Processing Pipeline ---
    
    # 1. Apply a small blur to remove high-frequency noise
    blurred_map = cv2.GaussianBlur(saliency_map_8bit, (5, 5), 0)

    # 2. Use Otsu's thresholding as before to get the initial mask
    _, binary_mask = cv2.threshold(
        blurred_map, 0, 255, 
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # 3. "Open" the mask: This removes small white noise dots
    #    It is an erosion followed by a dilation.
    kernel = np.ones((5, 5), np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. "Close" the mask: This fills small black holes in the subject
    #    It is a dilation followed by an erosion.
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. Dilate the final mask slightly to ensure we capture the
    #    edges of the subject.
    final_mask = cv2.dilate(closed_mask, kernel, iterations=3)
    
    # --- End of New Pipeline ---
    
    return Image.fromarray(final_mask).convert('L')

# --- 5. Quantization Logic ---
def quantize_image(pil_image, palette_size):
    paletted_img = pil_image.convert('P', palette=Image.ADAPTIVE, colors=palette_size)
    return paletted_img.convert('RGB')

# --- 6. Helper Function to Save Image to Buffer ---
def save_image_to_buffer(pil_img, format="PNG"):
    """Saves a PIL image to an in-memory BytesIO buffer."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    return buffer

# --- 7. The Updated API Endpoint ---
@app.post("/compress/")
async def compress_image(file: UploadFile = File(...)):
    
    # 1. Read the uploaded image
    contents = await file.read()
    original_pil_img = Image.open(io.BytesIO(contents)).convert('RGB')

    # 2. --- Generate All 4 Component Images ---
    
    # Image 1: Saliency Mask
    saliency_mask_img = get_saliency_mask(original_pil_img)
    
    # Image 2: High-Quality Full Quantization
    high_q_img = quantize_image(original_pil_img, HIGH_QUALITY_PALETTE)
    
    # Image 3: Low-Quality Full Quantization
    low_q_img = quantize_image(original_pil_img, LOW_QUALITY_PALETTE)
    
    # Image 4: Final Hybrid Image
    hybrid_rgb_img = Image.composite(high_q_img, low_q_img, saliency_mask_img)
    final_hybrid_img = hybrid_rgb_img.convert(
        'P', palette=Image.ADAPTIVE, colors=HIGH_QUALITY_PALETTE
    )

    # 3. --- Create an In-Memory Zip File ---
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all 4 images to the zip
        zipf.writestr("1_saliency.png", save_image_to_buffer(saliency_mask_img).getvalue())
        zipf.writestr("2_high_q.png", save_image_to_buffer(high_q_img).getvalue())
        zipf.writestr("3_low_q.png", save_image_to_buffer(low_q_img).getvalue())
        zipf.writestr("4_final_hybrid.png", save_image_to_buffer(final_hybrid_img).getvalue())

    zip_buffer.seek(0)

    # 4. Send the zip file back
    return StreamingResponse(
        zip_buffer, 
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=compression_package.zip"}
    )

@app.get("/")
def read_root():
    return {"status": "Saliency Compressor API is online."}

