###Saliency-Based Image Compressor

This project is a full-stack web application that demonstrates a novel, lossy image compression technique.

The compression algorithm identifies the main subject (saliency) of an image and applies a high-quality (256-color) quantization, while aggressively compressing the background (16-color) to save file size.

Project Structure

This is a monorepo containing two separate applications:

/backend: A Python FastAPI server that contains the OpenCV compression logic.

/frontend: A React application (built with Create React App) for the user interface.

Deployment

Backend (FastAPI): Deployed to Render.

Frontend (React): Deployed to Firebase Hosting.
