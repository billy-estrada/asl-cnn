import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";

async function sendImageToServer(imageSrc) {
  const blob = await fetch(imageSrc).then((res) => res.blob());
  const formData = new FormData();
  formData.append("image", blob);

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log("Prediction:", data.letter);
    return data.letter;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function WebCam() {
  const webcamRef = useRef(null);
  const [predictedLetter, setPredictedLetter] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const captureAndSend = useCallback(async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) {
      setError("No image captured.");
      return;
    }

    setError(null);
    const letter = await sendImageToServer(imageSrc);
    if (letter) {
      setPredictedLetter(letter);
    } else {
      setError("Prediction failed.");
    }
  }, []);

  useEffect(() => {
    if (!isRunning) return;

    const intervalId = setInterval(() => {
      captureAndSend();
    }, 2000); // every 2 seconds

    return () => clearInterval(intervalId);
  }, [isRunning, captureAndSend]);


  return (
    <div style={{ position: "relative", display: "inline-block" }}>

    
      <div style={{ marginTop: 10 }}>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={1280}
        height={720}
        videoConstraints={{
          facingMode: "user",
        }}
      />
      <div style={{ height: 40, marginBottom: 10 }}>
      {predictedLetter ? (
        <h2 style={{ margin: 0 }}>Predicted Letter: {predictedLetter}</h2>
      ) : error ? (
        <h3 style={{ margin: 0, color: "red" }}>{error}</h3>
      ) : (
        <div style={{ height: 24 }} />  // Invisible spacer
      )}
    </div>
        <button onClick={() => setIsRunning(true)} disabled={isRunning}>
          Start
        </button>
        <button onClick={() => setIsRunning(false)} disabled={!isRunning}>
          Stop
        </button>
      </div>
      {/* Centered Box Overlay */}
      <div
        style={{
          position: "absolute",
          top: "calc(50% - 150px)",
          left: "calc(50% - 90px)",
          width: "180px",
          height: "180px",
          border: "3px solid lime",
          zIndex: 2,
          pointerEvents: "none",
        }}
      />
      {/* Prediction / Error Display */}

    </div>
  );
}

export default WebCam;
