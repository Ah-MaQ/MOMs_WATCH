chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'openSettings') {
    chrome.tabs.create({ url: chrome.runtime.getURL('options/options.html') });
    sendResponse({ status: 'Settings page opened' });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.command === "startDetection") {
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, message, (response) => {
          sendResponse(response);
        });
      } else {
        sendResponse({status: 'no active tab found'});
      }
    });
    return true;  // Indicates that the response is sent asynchronously.
}
});


// async function startDetection() {
//   const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//   const video = document.createElement('video');
//   video.srcObject = stream;
//   video.play();

//   await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
//   await faceapi.nets.faceLandmark68Net.loadFromUri('/models');

//   video.addEventListener('play', () => {
//     detect(video);
//   });
// }

// async function detect(video) {
//   const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();

//   if (detections.length > 0) {
//     const eyesClosed = isEyesClosed(detections[0]);
//     if (eyesClosed) {
//       playAlarm();
//     }
//   }

//   setTimeout(() => detect(video), 100);
// }

// function isEyesClosed(detection) {
//   const leftEye = detection.landmarks.getLeftEye();
//   const rightEye = detection.landmarks.getRightEye();

//   const leftEyeClosed = isEyeClosed(leftEye);
//   const rightEyeClosed = isEyeClosed(rightEye);

//   return leftEyeClosed && rightEyeClosed;
// }

// function isEyeClosed(eye) {
//   const eyeHeight = faceapi.euclideanDistance(eye[1], eye[5]) + faceapi.euclideanDistance(eye[2], eye[4]);
//   const eyeWidth = faceapi.euclideanDistance(eye[0], eye[3]);
//   const eyeAspectRatio = eyeHeight / (2.0 * eyeWidth);

//   return eyeAspectRatio < 0.2;
// }
