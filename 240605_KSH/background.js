chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'openSettings') {
    chrome.tabs.create({ url: chrome.runtime.getURL('options/options.html') });
    sendResponse({ status: 'Settings page opened' });
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ running: false, endTime: 0 });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.command === "start") {
    const duration = parseInt(message.duration, 10) * 60; // 분을 초로 변환
    const endTime = Date.now() + duration * 1000; // 종료 시간을 밀리초로 설정
    chrome.storage.local.set({ running: true, endTime: endTime });

    // 기존 알람 삭제 후 새 알람 설정
    chrome.alarms.clear("timerAlarm", () => {
      chrome.alarms.create("timerAlarm", { delayInMinutes: duration / 60 });
    });
  } else if (message.command === "getRemainingTime") {
    chrome.storage.local.get(["running", "endTime"], (result) => {
      if (result.running) {
        const remainingTime = Math.max(0, Math.floor((result.endTime - Date.now()) / 1000));
        sendResponse({ remainingTime: remainingTime });
      } else {
        sendResponse({ remainingTime: 0 });
      }
    });
    return true; // 비동기 응답을 위해 true를 반환
  }

});
