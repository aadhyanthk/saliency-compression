import io
import cv2
import numpy as np
from PIL import Image
import zipfile
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
    try:
        cv_image_rgb = np.array(pil_image)
        # Ensure 3 channels (remove alpha if present)
        if cv_image_rgb.shape[2] == 4:
            cv_image_rgb = cv_image_rgb[..., :3]
            
        cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(cv_image_bgr)
        
        # --- ‼️ CRITICAL FIX: Check for success ‼️ ---
        # If saliency fails, create a default white mask
        if not success or saliency_map is None:
            print("Saliency detection failed, falling back to white mask.")
            # Create a white mask of the same size as the image
            h, w, _ = cv_image_rgb.shape
            final_mask = np.full((h, w), 255, dtype=np.uint8)
            return Image.fromarray(final_mask).convert('L')
        # --- End of Fix ---

        saliency_map_8bit = (saliency_map * 255).astype("uint8")

        # --- Post-Processing Pipeline ---
        blurred_map = cv2.GaussianBlur(saliency_map_8bit, (5, 5), 0)

        # Use Otsu's thresholding
        _, binary_mask = cv2.threshold(
            blurred_map, 0, 255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        kernel = np.ones((5, 5), np.uint8)
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_mask = cv2.dilate(closed_mask, kernel, iterations=3)
        
        return Image.fromarray(final_mask).convert('L')

    except Exception as e:
        print(f"Error in get_saliency_mask: {e}")
        # Fallback to white mask on any unexpected error
        h, w = pil_image.size
        final_mask = np.full((h, w), 255, dtype=np.uint8)
        return Image.fromarray(final_mask).convert('L')

# --- 5. Quantization Logic ---
def quantize_image(pil_image, palette_size):
    paletted_img = pil_image.convert('P', palette=Image.ADAPTIVE, colors=palette_size)
    return paletted_img.convert('RGB')

# --- 6. Helper Function to Save Image to Buffer ---
def save_image_to_buffer(pil_img, format="PNG"):
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    return buffer

# --- 7. The Updated API Endpoint ---
@app.post("/compress/")
async def compress_image(file: UploadFile = File(...)):
    
    try:
        # 1. Read the uploaded image
        contents = await file.read()
        original_pil_img = Image.open(io.BytesIO(contents)).convert('RGB')

        # 2. --- Generate All 4 Component Images ---
        saliency_mask_img = get_saliency_mask(original_pil_img)
        high_q_img = quantize_image(original_pil_img, HIGH_QUALITY_PALETTE)
        low_q_img = quantize_image(original_pil_img, LOW_QUALITY_PALETTE)
        
        # Ensure mask is 'L' mode for composite
        hybrid_rgb_img = Image.composite(high_q_img, low_q_img, saliency_mask_img.convert('L'))
        
        final_hybrid_img = hybrid_rgb_img.convert(
            'P', palette=Image.ADAPTIVE, colors=HIGH_QUALITY_PALETTE
        )

        # 3. --- Create an In-Memory Zip File ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("1_saliency.png", save_image_to_buffer(saliency_mask_img).getvalue())
            zipf.writestr("2_high_q.png", save_image_to_buffer(high_q_img).getvalue())
            zipf.writestr("3_low_q.png", save_image_to_buffer(low_q_img).getvalue())
            zipf.writestr("4.final_hybrid.png", save_image_to_buffer(final_hybrid_img).getvalue())

        zip_buffer.seek(0)

        # 4. Send the zip file back
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=compression_package.zip"}
        )
    
    except Exception as e:
        print(f"Error in /compress/ endpoint: {e}")
        # Return a 500 error explicitly if something fails
        return StreamingResponse(
            io.BytesIO(b"Server processing error."), 
            status_code=500,
            media_type="text/plain"
        )

@app.get("/")
def read_root():
    return {"status": "Saliency Compressor API is online."}

