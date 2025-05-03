import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";

const FRAME_BATCH_SIZE = 15;
const FRAME_INTERVAL_MS = 100;
const COUNTDOWN_SECONDS = 3;
const COOLDOWN_SECONDS = 5;


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
  const captureIntervalRef = useRef(null);
  const cycleTimeoutRef = useRef(null);


  const captureAndSend = useCallback(async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) return;

    frameBufferRef.current.push(imageSrc);

    if (frameBufferRef.current.length === FRAME_BATCH_SIZE) {
      clearInterval(captureIntervalRef.current);

      const framesToSend = [...frameBufferRef.current];
      frameBufferRef.current = [];

      try {
        setError(null);
        const letter = await sendImageBatchToServer(framesToSend);
        if (letter) {
          setPredictedLetter(letter);
        } else {
          setError("Prediction failed.");
        }
      } catch (err) {
        console.error(err);
        setError("Server error.");
      }

      // Start cooldown before next cycle
      cycleTimeoutRef.current = setTimeout(() => {
        startCountdownAndCapture(); // restart after cooldown
      }, COOLDOWN_MS);
    }
  }, []);

  const startCountdownAndCapture = useCallback(() => {
    let secondsLeft = COUNTDOWN_SECONDS;
    setCountdown(secondsLeft);

    const countdownInterval = setInterval(() => {
      secondsLeft--;
      setCountdown(secondsLeft);
      if (secondsLeft === 0) {
        clearInterval(countdownInterval);
        setCountdown(null);

        // Start capturing frames
        captureIntervalRef.current = setInterval(() => {
          captureAndSend();
        }, FRAME_INTERVAL_MS);
      }
    }, 1000);
  }, [captureAndSend]);

  useEffect(() => {
    if (isRunning) {
      captureAndSend();
    } else {
      clearInterval(captureIntervalRef.current);
      clearTimeout(cycleTimeoutRef.current);
      setCountdown(null);
      frameBufferRef.current = [];
    }

    return () => {
      clearInterval(captureIntervalRef.current);
      clearTimeout(cycleTimeoutRef.current);
    };
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
      <div style={{ minHeight: "3em", marginTop: "1em", textAlign: "center" }}>
        {countdown !== null && <h3>Capturing in: {countdown}</h3>}
        {predictedLetter && <h2>Predicted Letter: {predictedLetter}</h2>}
        {error && <p style={{ color: "red" }}>{error}</p>}
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
