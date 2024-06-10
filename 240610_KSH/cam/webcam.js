document.addEventListener("DOMContentLoaded", function() {
    const video = document.getElementById('webcam');
    const pipButton = document.getElementById('pipButton');
    const fullscreenButton = document.getElementById('fullscreenButton');
    const popupButton = document.getElementById('popupButton');

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

                        fetch('http://127.0.0.1:5000/upload_frame', {
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
            }

            setInterval(sendFrame, 1000 / 30);  // 30 fps로 전송
        } catch (error) {
            console.error('Error accessing the webcam:', error);
        }
    }

    pipButton.addEventListener('click', async () => {
        if (document.pictureInPictureElement) {
            await document.exitPictureInPicture();
        } else {
            await video.requestPictureInPicture();
        }
    });

    fullscreenButton.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            video.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
            });
        } else {
            document.exitFullscreen();
        }
    });

    popupButton.addEventListener('click', () => {
        const popup = window.open("", "popup", "width=640,height=480");
        popup.document.write(`<video id="popupWebcam" autoplay playsinline></video>`);
        const popupVideo = popup.document.getElementById('popupWebcam');
        popupVideo.srcObject = video.srcObject;
    });

    startWebcam();
});