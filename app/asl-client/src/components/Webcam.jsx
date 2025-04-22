import React from "react";
import Webcam from "react-webcam";


function sendImageToServer(imageSrc){
    const blob = fetch(imageSrc).then(res => res.blob()).then(() => {
    const formData = new FormData();
    formData.append("image", blob); // "image" must match Flask's request.files["image"]
    }).then(() => {
        fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
        })
        .then((response) => response.json())
        .then((data) => {
            console.log("Prediction:", data.letter);
            alert("Predicted letter: " + data.letter);
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Error during prediction.");
        });
    });
}

function WebCam() {
    const [predictedLetter, setPredictedLetter] = React.useState(null);
    const [error, setError] = React.useState(null); 
    const webcamRef = React.useRef(null);
    const capture = React.useCallback(
        () => {
        const imageSrc = webcamRef.current.getScreenshot();
        console.log(imageSrc); 

        if (!imageSrc) {
            setError("No image captured. Please try again."); // Set error message
            return;
        }
        setError(null);

        const data = sendImageToServer(imageSrc);
        setPredictedLetter(data.letter);
       
        // Do something with the captured image
        },
        [webcamRef]
    );
    
    return (
        <div>
            <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={1280}
                height={720}
                videoConstraints={{
                    facingMode: "user"
                }}
            />
            {predictedLetter && <h2>Predicted Letter: {predictedLetter}</h2>}
            {error && <h3 style={{ color: "red" }}>{error}</h3>}
            <button onClick={capture}>Capture photo</button>
        </div>
    );
}

export default WebCam;