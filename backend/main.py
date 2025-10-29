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

# --- 4. Saliency Logic (Advanced Multi-Stage Pipeline) ---
def get_saliency_mask(pil_image):
    try:
        cv_image_rgb = np.array(pil_image)
        if cv_image_rgb.shape[2] == 4:
            cv_image_rgb = cv_image_rgb[..., :3]
            
        cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
        h, w = cv_image_bgr.shape[:2]

        # --- Stage 1: Generate Saliency "Hints" ---
        
        # Hint 1: Spectral Residual (finds high-contrast regions)
        saliency_spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success_spectral, map_spectral) = saliency_spectral.computeSaliency(cv_image_bgr)
        if not success_spectral or map_spectral is None:
            map_spectral = np.zeros((h, w), dtype=np.uint8)
        else:
            map_spectral = (map_spectral * 255).astype("uint8")

        # Hint 2: Fine Grained (finds small, detailed regions)
        saliency_fine = cv2.saliency.StaticSaliencyFineGrained_create()
        (success_fine, map_fine) = saliency_fine.computeSaliency(cv_image_bgr)
        if not success_fine or map_fine is None:
            map_fine = np.zeros((h, w), dtype=np.uint8)
        else:
            map_fine = (map_fine * 255).astype("uint8")

        # Combine the hints
        combined_map = cv2.addWeighted(map_spectral, 0.5, map_fine, 0.5, 0)

        # --- Stage 2: Clean Hints and Create Initial Mask ---
        
        # Adaptive thresholding to get the best binary map
        thresh_map = cv2.adaptiveThreshold(
            combined_map, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )

        # Morphological opening to remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        thresh_map = cv2.morphologyEx(thresh_map, cv2.MORPH_OPEN, kernel_small, iterations=2)

        # Find all contours in the cleaned-up hint map
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("No contours found in saliency maps.")

        # --- Stage 3: Filter Contours and Initialize GrabCut ---
        
        # Create the mask for GrabCut
        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8) # All definite background
        
        # Filter contours by size and aspect ratio
        min_area = (w * h) * 0.01 # At least 1% of the image
        has_probable_fg = False

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            
            x, y, w_rect, h_rect = cv2.boundingRect(c)
            aspect_ratio = w_rect / float(h_rect)
            
            # Filter out things that are too long or too tall
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                continue

            # This is a good candidate, mark it as probable foreground
            cv2.drawContours(gc_mask, [c], -1, cv2.GC_PR_FGD, -1)
            has_probable_fg = True

        # If no contours passed, use the largest one as a last resort
        if not has_probable_fg and contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
            rect = (max(0, x-10), max(0, y-10), min(w, w_rect+20), min(h, h_rect+20))
            gc_mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = cv2.GC_PR_FGD
        
        # Set a 5-pixel border as definite background
        gc_mask[0:5, :] = cv2.GC_BGD
        gc_mask[h-5:h, :] = cv2.GC_BGD
        gc_mask[:, 0:5] = cv2.GC_BGD
        gc_mask[:, w-5:w] = cv2.GC_BGD

        # --- Stage 4: Run GrabCut ---
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Run GrabCut using the mask we just built
        cv2.grabCut(
            cv_image_bgr,
            gc_mask,
            None, # We are not using a rect, we use the mask
            bgdModel,
            fgdModel,
            8, # More iterations
            cv2.GC_INIT_WITH_MASK
        )

        # --- Stage 5: Final Mask Refinement ---
        
        # The final mask is where GrabCut marked it as (1) definite FG or (3) probable FG
        final_mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
        
        # Final closing and dilation to smooth edges and fill holes
        kernel_large = np.ones((7, 7), np.uint8)
        closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        final_mask_dilated = cv2.dilate(closed_mask, kernel_large, iterations=2)

        return Image.fromarray(final_mask_dilated).convert('L')

    except Exception as e:
        print(f"Error in get_saliency_mask: {e}. Falling back to white mask.")
        w, h = pil_image.size
        final_mask = np.full((h, w), 255, dtype=np.uint8) # Correct (h, w) order
        return Image.fromarray(final_mask).convert('L')

# --- 5. Quantization Logic ---
def quantize_image(pil_image, palette_size):
    """Reduces the image to a specific number of colors."""
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

