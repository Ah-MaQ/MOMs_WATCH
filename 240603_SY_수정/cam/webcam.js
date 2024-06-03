document.addEventListener("DOMContentLoaded", function() {
    const video = document.getElementById('webcam');

    async function startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            function sendFrame() {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);

                    canvas.toBlob(blob => {
                        const formData = new FormData();
                        formData.append('frame', blob, 'frame.jpg');

                        fetch('http://127.0.0.1:5000/upload_frame', {  // URL 수정
                            method: 'POST',
                            body: formData
                        }).then(response => {
                            if (!response.ok) {
                                throw new Error('Network response was not ok');
                            }
                        }).catch(error => {
                            console.error('There was a problem with the fetch operation:', error);
                        });
                    }, 'image/jpeg');
                }

                requestAnimationFrame(sendFrame);
            }

            sendFrame();
        } catch (error) {
            console.error('Error accessing the webcam:', error);
        }
    }

    startWebcam();
});
