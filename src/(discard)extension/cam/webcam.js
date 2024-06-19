const originalVideo = document.getElementById('originalVideo');
const overlayCanvas = document.getElementById('overlayCanvas');
const statusText = document.getElementById('statusText');

const constraints = {
  video: true
};

navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    originalVideo.srcObject = stream;

    const visualize = new URLSearchParams(window.location.search).get('visualize') === 'true';
    if (visualize) {
      overlayCanvas.width = originalVideo.clientWidth;
      overlayCanvas.height = originalVideo.clientHeight;
      const context = overlayCanvas.getContext('2d');

      setInterval(() => {
        const canvas = document.createElement('canvas');
        canvas.width = originalVideo.videoWidth;
        canvas.height = originalVideo.videoHeight;
        const tempContext = canvas.getContext('2d');
        tempContext.drawImage(originalVideo, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
          if (blob) {
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');
            fetch('http://127.0.0.1:5000/upload_frame', {
              method: 'POST',
              body: formData
            })
            .then(response => response.json())
            .then(data => {
              if (data.status === 'success') {
                statusText.textContent = data.message;
                drawVisualization(data.data, context);
              } else {
                statusText.textContent = 'Error processing frame.';
              }
            })
            .catch(error => {
              console.error('Error uploading frame:', error);
              statusText.textContent = 'Error connecting to server.';
            });
          }
        }, 'image/jpeg');
      }, 500);
    }
  }).catch((error) => {
    console.error('Error accessing the webcam:', error);
    statusText.textContent = 'Error accessing the webcam.';
  });

function drawVisualization(data, context) {
  const { yaw, pitch, face_bbox, eye_landmarks } = data;
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  context.strokeStyle = 'green';
  context.lineWidth = 2;
  context.strokeRect(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]);

  context.fillStyle = 'red';
  for (const key in eye_landmarks) {
    const [x, y, r] = eye_landmarks[key];
    context.beginPath();
    context.arc(x, y, r, 0, 2 * Math.PI);
    context.fill();
  }

  context.fillStyle = 'black';
  context.font = '20px Arial';
  context.fillText(`Yaw: ${yaw.toFixed(2)}`, 10, 30);
  context.fillText(`Pitch: ${pitch.toFixed(2)}`, 10, 60);
}
