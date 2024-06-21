document.addEventListener("DOMContentLoaded", function() {
    const processedStream = document.getElementById('processedStream');
    const pipButton = document.getElementById('pipButton');
    const fullscreenButton = document.getElementById('fullscreenButton');
    const popupButton = document.getElementById('popupButton');
    let alarmCount = 0;

    async function startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });  // 해상도 축소
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

            setInterval(sendFrame, 1000 / 10);  // 전송 주기를 10 fps로 줄임
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

    async function updateStatus() {
        try {
            const response = await fetch('http://127.0.0.1:5000/get_status');
            if (response.ok) {
                const data = await response.json();
                console.log(data);  // 받은 데이터를 콘솔에 출력

                // 필요한 경우 DOM 요소를 업데이트
                const statusElement = document.getElementById('status');
                statusElement.textContent = `Detected: ${data.is_there}, Make Alarm: ${data.make_alarm}`;

                if (data.make_alarm === true) {
                    alarmCount += 1;
                    if (alarmCount >= 10) {
                        chrome.runtime.sendMessage({ action: 'triggerAlarm' });
                        alarmCount = 0; // 메시지를 보낸 후 카운터 초기화
                    }
                } else {
                    alarmCount = 0; // 알람이 false이면 카운터 초기화
                }
            } else {
                console.error('Network response was not ok');
            }
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
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
    setInterval(updateStatus, 1000);  // 상태 업데이트 주기를 1초로 설정
});
