import React from "react";
import Webcam from "react-webcam";

function WebCam() {
    const webcamRef = React.useRef(null);
    const capture = React.useCallback(
        () => {
        const imageSrc = webcamRef.current.getScreenshot();
        console.log(imageSrc); // Do something with the captured image
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
        <button onClick={capture}>Capture photo</button>
        </div>
    );
}

export default WebCam;