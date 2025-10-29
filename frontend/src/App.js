import React, { useState } from "react";
import "./App.css";

// -----------------------------------------------------------------
// ⬇️ !!! IMPORTANT !!! ⬇️
// PASTE YOUR RENDER API URL HERE. MUST END WITH A SLASH.
const API_URL = "https://saliency-compression.onrender.com/compress/";
// -----------------------------------------------------------------

function App() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setDownloadUrl(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true);
    setDownloadUrl(null);
    setError(null);

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

      const imageBlob = await response.blob();
      const url = URL.createObjectURL(imageBlob);
      setDownloadUrl(url);
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

      {downloadUrl && (
        <div className="result">
          <h3>✅ Compression Complete!</h3>
          <a
            href={downloadUrl}
            download="compressed.png"
            className="download-button"
          >
            Download Compressed Image
          </a>
        </div>
      )}
    </div>
  );
}

export default App;
