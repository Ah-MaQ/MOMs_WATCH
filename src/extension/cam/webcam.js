const originalVideo = document.getElementById('originalVideo');
const processedVideo = document.getElementById('processedVideo');

const constraints = {
  video: true
};

// Start capturing video from the webcam
navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    originalVideo.srcObject = stream;

    // Set up a canvas to capture frames from the webcam
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    // Send frames to the server at regular intervals
    setInterval(() => {
      canvas.width = originalVideo.videoWidth;
      canvas.height = originalVideo.videoHeight;

      // Draw the current frame on the canvas
      context.drawImage(originalVideo, 0, 0, canvas.width, canvas.height);

      // Convert the canvas content to a Blob and send it to the server
      canvas.toBlob((blob) => {
        if (blob) {
          const formData = new FormData();
          formData.append('frame', blob, 'frame.jpg');
          fetch('http://127.0.0.1:5000/upload_frame', {
            method: 'POST',
            body: formData
          })
          .then(response => {
            console.log('Frame uploaded:', response.status);
          })
          .catch(error => console.error('Error uploading frame:', error));
        }
      }, 'image/jpeg');
    }, 500); // Adjust the interval as needed

    // Start fetching and displaying the processed stream
    fetchProcessedStream();

  }).catch((error) => {
    console.error('Error accessing the webcam:', error);
  });

function fetchProcessedStream() {
  // Create a MediaSource object
  const mediaSource = new MediaSource();
  processedVideo.src = URL.createObjectURL(mediaSource);

  mediaSource.addEventListener('sourceopen', () => {
    const sourceBuffer = mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');

    const xhr = new XMLHttpRequest();
    xhr.open('GET', 'http://127.0.0.1:5000/stream', true);
    xhr.responseType = 'blob';

    xhr.onload = () => {
      if (xhr.status === 200) {
        sourceBuffer.appendBuffer(xhr.response);
      } else {
        console.error('Failed to load stream:', xhr.status);
      }
    };

    xhr.onerror = () => {
      console.error('XHR request failed:', xhr.status);
    };

    xhr.send();
  });

  mediaSource.addEventListener('sourceended', () => {
    console.log('Stream ended');
  });
}
