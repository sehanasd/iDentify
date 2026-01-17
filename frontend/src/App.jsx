import { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detections, setDetections] = useState([]);

  // 1. Handle File Selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file)); // Show preview immediately
      setResultImage(null); // Reset previous results
      setDetections([]);
    }
  };

  // 2. Send to Python Backend
  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Connect to your FastAPI Backend
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      // 3. Handle Response
      setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
      setDetections(data.detections);
      
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Error connecting to backend. Is Python running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🦷 iDentify: Dental AI Assistant</h1>
      
      {/* Upload Section */}
      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!selectedFile || loading}>
          {loading ? "Analyzing..." : "Analyze X-Ray"}
        </button>
      </div>

      {/* Results Section */}
      <div className="results-area">
        {/* Left: The Visual Result */}
        <div className="image-panel">
          {resultImage ? (
            <img src={resultImage} alt="Analysis Result" className="xray-img" />
          ) : preview ? (
            <img src={preview} alt="Preview" className="xray-img preview" />
          ) : (
            <div className="placeholder">Upload an OPG X-Ray to start</div>
          )}
        </div>

        {/* Right: The Text Report */}
        {detections.length > 0 && (
          <div className="report-panel">
            <h3>📋 Diagnosis Report</h3>
            <ul>
              {detections.map((det, index) => (
                <li key={index} className={det.diagnosis === 'healthy' ? 'healthy' : 'issue'}>
                  <strong>{det.diagnosis.toUpperCase()}</strong>
                  <br />
                  <span className="conf">Confidence: {(det.confidence * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;