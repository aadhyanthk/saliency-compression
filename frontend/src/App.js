import React, { useState } from "react";
import "./App.css";
import JSZip from "jszip"; // Import JSZip

// -----------------------------------------------------------------
// ⬇️ !!! IMPORTANT !!! ⬇️
// Your Render API URL must be correct.
const API_URL = "https://saliency-compression.onrender.com/compress/";
// -----------------------------------------------------------------

function App() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // New state for all our image URLs
  const [originalURL, setOriginalURL] = useState(null);
  const [saliencyURL, setSaliencyURL] = useState(null);
  const [highQURL, setHighQURL] = useState(null);
  const [lowQURL, setLowQURL] = useState(null);
  const [finalHybridURL, setFinalHybridURL] = useState(null);

  // This will track if a successful result is available
  const [isResultReady, setIsResultReady] = useState(false);

  const clearImageStates = () => {
    // Revoke old URLs to prevent memory leaks
    if (originalURL) URL.revokeObjectURL(originalURL);
    if (saliencyURL) URL.revokeObjectURL(saliencyURL);
    if (highQURL) URL.revokeObjectURL(highQURL);
    if (lowQURL) URL.revokeObjectURL(lowQURL);
    if (finalHybridURL) URL.revokeObjectURL(finalHybridURL);

    setOriginalURL(null);
    setSaliencyURL(null);
    setHighQURL(null);
    setLowQURL(null);
    setFinalHybridURL(null);
    setError(null);
    setIsResultReady(false);
  };

  const handleFileChange = (e) => {
    clearImageStates();
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      // Create and set the URL for the original image immediately
      setOriginalURL(URL.createObjectURL(selectedFile));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true);
    // Clear previous results (but not the original)
    setSaliencyURL(null);
    setHighQURL(null);
    setLowQURL(null);
    setFinalHybridURL(null);
    setError(null);
    setIsResultReady(false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `Server error. It might be spinning up. Try again in 30 seconds.`
        );
      }

      // 1. Get the response as a zip blob
      const zipBlob = await response.blob();

      // 2. Load the zip file
      const zip = await JSZip.loadAsync(zipBlob);

      // 3. Extract each image, create a URL, and set state
      const saliencyBlob = await zip.file("1_saliency.png").async("blob");
      setSaliencyURL(URL.createObjectURL(saliencyBlob));

      const highQBlob = await zip.file("2_high_q.png").async("blob");
      setHighQURL(URL.createObjectURL(highQBlob));

      const lowQBlob = await zip.file("3_low_q.png").async("blob");
      setLowQURL(URL.createObjectURL(lowQBlob));

      const finalBlob = await zip.file("4_final_hybrid.png").async("blob");
      setFinalHybridURL(URL.createObjectURL(finalBlob));

      // Set flag to show the diagram
      setIsResultReady(true);
    } catch (err) {
      setError(
        err.message ||
          "Compression failed. The server might be busy or the file is invalid."
      );
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Saliency-Based Image Compressor</h1>
        <p>
          This tool compresses the background (16 colors) more than the subject
          (256 colors) to create a smaller file with high perceived quality.
        </p>
      </header>

      <form onSubmit={handleSubmit}>
        <label htmlFor="file-upload" className="file-label">
          {file ? file.name : "Choose an image..."}
        </label>
        <input
          id="file-upload"
          type="file"
          onChange={handleFileChange}
          accept="image/png, image/jpeg"
        />
        <button type="submit" disabled={isLoading || !file}>
          {isLoading ? "Compressing..." : "Compress Image"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {isLoading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>This may take up to 30 seconds as the server wakes up...</p>
        </div>
      )}

      {/* --- This is the new Visual Diagram Section --- */}
      {isResultReady && (
        <section className="diagram-section">
          <h2>Processing Pipeline</h2>

          {/* --- Row 1: Original Image --- */}
          <div className="diagram-row">
            <div className="diagram-node">
              <p>1. Original Image</p>
              <img src={originalURL} alt="Original" />
            </div>
          </div>

          {/* --- Arrow Down --- */}
          <div className="diagram-connector">↓</div>

          {/* --- Row 2: Saliency + High/Low Q --- */}
          <div className="diagram-row">
            <div className="diagram-node">
              <p>2. Saliency Mask</p>
              <img src={saliencyURL} alt="Saliency Mask" />
            </div>

            <div className="diagram-connector-plus">+</div>

            <div className="diagram-node-group">
              <div className="diagram-node small">
                <p>3. High-Q (256)</p>
                <img src={highQURL} alt="High Quality" />
              </div>
              <div className="diagram-node small">
                <p>4. Low-Q (16)</p>
                <img src={lowQURL} alt="Low Quality" />
              </div>
            </div>
          </div>

          {/* --- Arrow Down --- */}
          <div className="diagram-connector">↓</div>

          {/* --- Row 3: Final Image & Download --- */}
          <div className="diagram-row">
            <div className="diagram-node large">
              <p>5. Final Hybrid Image</p>
              <img src={finalHybridURL} alt="Final Hybrid" />
              <a
                href={finalHybridURL}
                download="compressed.png"
                className="download-button"
              >
                Download Compressed Image
              </a>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
