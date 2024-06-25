let isAlarmShowing = false;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Message received in content script:', request);
    if (request.action === 'showAlarm' && !isAlarmShowing) {
        isAlarmShowing = true;

        // 오버레이 배경 추가
        const overlay = document.createElement('div');
        overlay.id = 'custom-overlay';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.1)'; // 반투명 검정색
        overlay.style.zIndex = '9998';
        document.body.appendChild(overlay);

        // 알람 메시지 박스 추가
        const div = document.createElement('div');
        div.id = 'custom-alarm';

        const img = document.createElement('img');
        img.src = chrome.runtime.getURL('alarm.png'); // 'your-image.png'를 실제 이미지 파일명으로 변경하세요.
        img.style.display = 'block';
        img.style.width = '300px'; // 이미지 너비 설정
        img.style.height = '300px'; // 이미지 높이 설정
        img.style.margin = '0 auto'; // 이미지 가운데 정렬

        const text = document.createElement('div');
        text.innerText = '타이머가 종료되었습니다!';
        text.style.textAlign = 'center'; // 글자 가운데 정렬
        text.style.fontSize = '20px'; // 글자 크기 설정
        text.style.color = 'red'; // 글자 색상 설정
        text.style.fontWeight = 'bold'; // 글자 굵기 설정

        // 닫기 버튼 추가
        const closeButton = document.createElement('button');
        closeButton.innerText = 'X';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '10px';
        closeButton.style.right = '10px';
        closeButton.style.background = 'white';
        closeButton.style.border = '2px solid black';
        closeButton.style.borderRadius = '50%';
        closeButton.style.width = '30px';
        closeButton.style.height = '30px';
        closeButton.style.fontSize = '16px';
        closeButton.style.fontWeight = 'bold';
        closeButton.style.cursor = 'pointer';

        closeButton.addEventListener('click', () => {
            if (div && div.parentNode) {
                div.parentNode.removeChild(div);
            }
            if (overlay && overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
            isAlarmShowing = false;
        });

        div.appendChild(closeButton);
        div.appendChild(img);
        div.appendChild(text);

        div.style.position = 'fixed';
        div.style.top = '50%';
        div.style.left = '50%';
        div.style.transform = 'translate(-50%, -50%)';
        div.style.background = 'transparent'; // 배경 투명하게 설정
        div.style.border = 'none'; // 테두리 없애기
        div.style.padding = '20px';
        div.style.zIndex = '9999';
        div.style.textAlign = 'center'; // 전체 내용 가운데 정렬

        document.body.appendChild(div);

        const audio = new Audio(chrome.runtime.getURL('alarm.mp3'));
        audio.play().catch(error => console.log('Error playing audio:', error));
    }

    if (request.action === 'wakeUp' && !isAlarmShowing) {
        isAlarmShowing = true;

        // 오버레이 배경 추가
        const overlay = document.createElement('div');
        overlay.id = 'custom-overlay';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.1)'; // 반투명 검정색
        overlay.style.zIndex = '9998';
        document.body.appendChild(overlay);

        // 알람 메시지 박스 추가
        const div = document.createElement('div');
        div.id = 'custom-alarm';

        const img = document.createElement('img');
        img.src = chrome.runtime.getURL('wakeup.png'); // 'your-image.png'를 실제 이미지 파일명으로 변경하세요.
        img.style.display = 'block';
        img.style.width = '300px'; // 이미지 너비 설정
        img.style.height = '300px'; // 이미지 높이 설정
        img.style.margin = '0 auto'; // 이미지 가운데 정렬

        const text = document.createElement('div');
        text.innerText = 'wake up!';
        text.style.textAlign = 'center'; // 글자 가운데 정렬
        text.style.fontSize = '20px'; // 글자 크기 설정
        text.style.color = 'red'; // 글자 색상 설정
        text.style.fontWeight = 'bold'; // 글자 굵기 설정

        // 닫기 버튼 추가
        const closeButton = document.createElement('button');
        closeButton.innerText = 'X';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '10px';
        closeButton.style.right = '10px';
        closeButton.style.background = 'white';
        closeButton.style.border = '2px solid black';
        closeButton.style.borderRadius = '50%';
        closeButton.style.width = '30px';
        closeButton.style.height = '30px';
        closeButton.style.fontSize = '16px';
        closeButton.style.fontWeight = 'bold';
        closeButton.style.cursor = 'pointer';

        closeButton.addEventListener('click', () => {
            if (div && div.parentNode) {
                div.parentNode.removeChild(div);
            }
            if (overlay && overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
            isAlarmShowing = false;
        });

        div.appendChild(closeButton);
        div.appendChild(img);
        div.appendChild(text);

        div.style.position = 'fixed';
        div.style.top = '50%';
        div.style.left = '50%';
        div.style.transform = 'translate(-50%, -50%)';
        div.style.background = 'transparent'; // 배경 투명하게 설정
        div.style.border = 'none'; // 테두리 없애기
        div.style.padding = '20px';
        div.style.zIndex = '9999';
        div.style.textAlign = 'center'; // 전체 내용 가운데 정렬

        document.body.appendChild(div);

        const audio = new Audio(chrome.runtime.getURL('alarm.mp3'));
        audio.play().catch(error => console.log('Error playing audio:', error));
    }
});
