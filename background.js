chrome.runtime.onInstalled.addListener(details => {
    if (details.reason === "install") {
        chrome.storage.local.set({ switch: false });
    }
});

let timerInterval, stopwatchInterval;

let stopwatchData = {
    time: 0,
    running: false,
    startTime: Date.now()
};

let timerData = {
    time: 600000, // 10분
    running: false,
    startTime: Date.now()
};

// 초기 데이터 로드
chrome.storage.local.get(['stopwatchData', 'timerData'], result => {
    stopwatchData = result.stopwatchData || stopwatchData;
    timerData = result.timerData || timerData;

    if (stopwatchData.running) startStopwatch();
    if (timerData.running) startTimer();
});

const startStopwatch = () => {
    if (stopwatchInterval) clearInterval(stopwatchInterval);
    stopwatchInterval = setInterval(() => {
        if (stopwatchData.running) {
            stopwatchData.time += 1000;
            saveData();
        }
    }, 1000);
};

const startTimer = () => {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        if (timerData.running) {
            timerData.time -= 1000;
            if (timerData.time <= 0) {
                timerData.time = 0;
                stopTimer();
                chrome.windows.create({
                    url: chrome.runtime.getURL('alarm.html'),
                    type: 'popup',
                    width: 300,
                    height: 200
                });
            }
            saveData();
        }
    }, 1000);
};

const stopStopwatch = () => {
    stopwatchData.running = false;
    clearInterval(stopwatchInterval);
    saveData();
};

const stopTimer = () => {
    timerData.running = false;
    clearInterval(timerInterval);
    saveData();
};

const saveData = () => {
    chrome.storage.local.set({ stopwatchData, timerData });
};

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const actions = {
        startStopwatch: () => {
            stopwatchData.running = true;
            stopwatchData.startTime = Date.now() - stopwatchData.time;
            startStopwatch();
        },
        stopStopwatch: stopStopwatch,
        resetStopwatch: () => {
            stopwatchData.time = 0;
            stopwatchData.running = false;
            stopwatchData.startTime = Date.now();
            saveData();
        },
        startTimer: () => {
            timerData.running = true;
            timerData.startTime = Date.now() - (600000 - timerData.time);
            startTimer();
        },
        stopTimer: stopTimer,
        resetTimer: () => {
            timerData.time = 600000;
            timerData.running = false;
            timerData.startTime = Date.now();
            saveData();
        },
        updateTimer: () => {
            timerData.time = request.time;
            saveData();
        },
        getTimerData: () => sendResponse({ stopwatchData, timerData }),
        saveButtonState: () => chrome.storage.local.set({ buttonState: request.buttonState }),
        saveSelectedMenu: () => chrome.storage.local.set({ selectedMenu: request.selectedMenu })
    };
    if (actions[request.action]) actions[request.action]();
});
