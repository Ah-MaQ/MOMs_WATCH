chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.command === "startDetection") {
      startDetection();
    }
  });
  
  async function startDetection() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
  
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
  
    video.addEventListener('play', () => {
      detect(video);
    });
  }
  
  async function detect(video) {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
  
    if (detections.length > 0) {
      const eyesClosed = isEyesClosed(detections[0]);
      if (eyesClosed) {
        playAlarm();
      }
    }
  
    setTimeout(() => detect(video), 100);
  }
  
  function isEyesClosed(detection) {
    const leftEye = detection.landmarks.getLeftEye();
    const rightEye = detection.landmarks.getRightEye();
  
    const leftEyeClosed = isEyeClosed(leftEye);
    const rightEyeClosed = isEyeClosed(rightEye);
  
    return leftEyeClosed && rightEyeClosed;
  }
  
  function isEyeClosed(eye) {
    const eyeHeight = faceapi.euclideanDistance(eye[1], eye[5]) + faceapi.euclideanDistance(eye[2], eye[4]);
    const eyeWidth = faceapi.euclideanDistance(eye[0], eye[3]);
    const eyeAspectRatio = eyeHeight / (2.0 * eyeWidth);
  
    return eyeAspectRatio < 0.2;
  }
  
  function playAlarm() {
    const audio = new Audio('alarm.mp3');
    audio.play();
  }
  