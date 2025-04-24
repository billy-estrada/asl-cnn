import React from "react";
import Webcam from "react-webcam";

async function sendImageToServer(imageSrc) {
  const blob = await fetch(imageSrc).then((res) => res.blob());
  const formData = new FormData();
  formData.append("image", blob); // "image" must match Flask's request.files["image"]

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log("Prediction:", data.letter);
    return data.letter; // Return predicted letter
  } catch (error) {
    console.error("Error:", error);
    return null; // Return null if there's an error
  }
}

function WebCam() {
  const [predictedLetter, setPredictedLetter] = React.useState(null);
  const [error, setError] = React.useState(null);
  const webcamRef = React.useRef(null);

  const capture = React.useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      setError("No image captured. Please try again.");
      return;
    }

    setError(null);

    const letter = await sendImageToServer(imageSrc); // Wait for the prediction
    if (letter) {
      setPredictedLetter(letter);
    } else {
      setError("Prediction failed. Please try again.");
    }
  }, [webcamRef]);

  return (
    <div>
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
      {predictedLetter && <h2>Predicted Letter: {predictedLetter}</h2>}
      {error && <h3 style={{ color: "red" }}>{error}</h3>}
      <button onClick={capture}>Capture photo</button>
    </div>
  );
}

export default WebCam;
