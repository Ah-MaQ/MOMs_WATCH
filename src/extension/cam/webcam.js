document.addEventListener("DOMContentLoaded", function() {
    const processedStream = document.getElementById('processedStream');
    const pipButton = document.getElementById('pipButton');
    const fullscreenButton = document.getElementById('fullscreenButton');
    const popupButton = document.getElementById('popupButton');

    async function startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            async function sendFrame() {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);

                    canvas.toBlob(async blob => {
                        const formData = new FormData();
                        formData.append('frame', blob, 'frame.jpg');

                        try {
                            const response = await fetch('http://127.0.0.1:5000/upload_frame', {
                                method: 'POST',
                                body: formData
                            });
                            if (!response.ok) {
                                throw new Error('Network response was not ok');
                            }
                        } catch (error) {
                            console.error('There was a problem with the fetch operation:', error);
                        }
                    }, 'image/jpeg');
                }
            }

            setInterval(sendFrame, 1000 / 10);  // 30 fps로 전송
        } catch (error) {
            console.error('Error accessing the webcam:', error);
        }
    }

    async function startProcessedStream() {
        try {
            processedStream.src = 'http://127.0.0.1:5000/stream';
        } catch (error) {
            console.error('Error starting processed stream:', error);
        }
    }

    pipButton.addEventListener('click', async () => {
        if (processedStream.readyState >= processedStream.HAVE_METADATA) {
            if (document.pictureInPictureElement) {
                await document.exitPictureInPicture();
            } else {
                await processedStream.requestPictureInPicture();
            }
        } else {
            processedStream.addEventListener('loadedmetadata', async () => {
                if (document.pictureInPictureElement) {
                    await document.exitPictureInPicture();
                } else {
                    await processedStream.requestPictureInPicture();
                }
            }, { once: true });
        }
    });

    fullscreenButton.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            processedStream.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
            });
        } else {
            document.exitFullscreen();
        }
    });

    popupButton.addEventListener('click', () => {
        const popup = window.open("", "popup", "width=640,height=480");
        popup.document.write(`<img id="popupProcessedStream" alt="Processed Stream">`);
        const popupProcessedStream = popup.document.getElementById('popupProcessedStream');
        popupProcessedStream.src = processedStream.src;
    });

    startWebcam();
    startProcessedStream();  // 서버에서 처리된 스트림 시작
});
