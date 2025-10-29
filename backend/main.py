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

# --- 4. Saliency Logic (NEW GrabCut PIPELINE) ---
def get_saliency_mask(pil_image):
    try:
        cv_image_rgb = np.array(pil_image)
        # Ensure 3 channels (remove alpha if present)
        if cv_image_rgb.shape[2] == 4:
            cv_image_rgb = cv_image_rgb[..., :3]
            
        cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
        h, w = cv_image_bgr.shape[:2]

        # --- Stage 1: Generate a "Hint" with Spectral Residual ---
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(cv_image_bgr)
        
        if not success or saliency_map is None:
            raise Exception("SpectralResidual failed.")

        saliency_map_8bit = (saliency_map * 255).astype("uint8")
        _, binary_mask = cv2.threshold(
            saliency_map_8bit, 0, 255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # Find the largest contour from the noisy mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise Exception("No contours found in saliency map.")

        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box of this contour
        x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
        
        # Create a 10-pixel buffer for the bounding box
        rect = (
            max(0, x - 10), 
            max(0, y - 10), 
            min(w, w_rect + 20), 
            min(h, h_rect + 20)
        )

        # --- Stage 2: Refine with GrabCut ---
        
        # GrabCut requires a "mask" and temporary arrays
        gc_mask = np.zeros((h, w), np.uint8)
        
        # Initialize the mask with our bounding box as "probable foreground"
        gc_mask.fill(cv2.GC_BGD) # Set all to "definite background"
        gc_mask[y:y+h_rect, x:x+w_rect] = cv2.GC_PR_FGD # Set box to "probable foreground"
        
        # Allocate memory for GrabCut's internal models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Run GrabCut segmentation
        cv2.grabCut(
            cv_image_bgr,   # The original image
            gc_mask,        # The hint mask
            rect,           # The region of interest
            bgdModel,
            fgdModel,
            5,              # 5 iterations
            cv2.GC_INIT_WITH_RECT
        )

        # The final mask will contain 0, 1, 2, or 3
        # We want all "foreground" (1) and "probable foreground" (3)
        final_mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
        
        # --- Post-Processing (as before, but on the new mask) ---
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        final_mask_dilated = cv2.dilate(closed_mask, kernel, iterations=3)

        return Image.fromarray(final_mask_dilated).convert('L')

    except Exception as e:
        print(f"Error in get_saliency_mask: {e}. Falling back to white mask.")
        # Fallback to a full white mask if any step fails
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
            zipf.writestr("4_final_hybrid.png", save_image_to_buffer(final_hybrid_img).getvalue())

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

