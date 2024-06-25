window.onload = function() {
    const audio = document.getElementById('alarm-audio');

    // 초기 음량과 음소거 상태를 storage에서 가져옴
    chrome.storage.local.get(['alarmVolume', 'alarmMuted'], result => {
        audio.volume = result.alarmVolume !== undefined ? result.alarmVolume : 0.5;
        audio.muted = result.alarmMuted || false;
        audio.play();
    });
};