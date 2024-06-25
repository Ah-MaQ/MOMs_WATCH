chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.set({ switch: false });
});

let timerInterval, stopwatchInterval;

let stopwatchData = {
    time: 0,
    running: false,
    startTime: Date.now()
};

let timerData = {
    time: 0,
    running: false,
    startTime: Date.now()
};

chrome.storage.local.get(['stopwatchData', 'timerData'], result => {
    stopwatchData = result.stopwatchData || stopwatchData;
    timerData = result.timerData || timerData;

    if (stopwatchData.running) startStopwatch();
    if (timerData.running) startTimer();
});

const startStopwatch = () => {
    if (stopwatchInterval) clearInterval(stopwatchInterval);
    stopwatchData.running = true;
    stopwatchData.startTime = Date.now() - stopwatchData.time;
    stopwatchInterval = setInterval(() => {
        if (stopwatchData.running) {
            stopwatchData.time = Date.now() - stopwatchData.startTime;
            saveData();
            chrome.runtime.sendMessage({ action: 'updateStopwatchTime', time: stopwatchData.time });
        }
    }, 100); // 100밀리초마다 실행되도록 설정
};

const startTimer = () => {
    if (timerInterval) clearInterval(timerInterval);
    timerData.running = true;
    timerData.startTime = Date.now() - timerData.time;
    timerInterval = setInterval(() => {
        if (timerData.running) {
            timerData.time -= 100;
            if (timerData.time <= 0) {
                timerData.time = 0;
                stopTimer();
                chrome.windows.getAll({ populate: true }, (windows) => {
                    windows.forEach((window) => {
                        let activeTab = window.tabs.find(tab => tab.active);
                        if (activeTab && !activeTab.url.startsWith('chrome://')) {
                            chrome.scripting.executeScript(
                                {
                                    target: { tabId: activeTab.id },
                                    files: ['content.js']
                                },
                                () => {
                                    chrome.tabs.sendMessage(activeTab.id, { action: 'showAlarm' });
                                }
                            );
                        }
                    });
                });
            }
            saveData();
            chrome.runtime.sendMessage({ action: 'updateTimerTime', time: timerData.time });
        }
    }, 100); // 100밀리초마다 실행
};

const stopStopwatch = () => {
    stopwatchData.running = false;
    clearInterval(stopwatchInterval);
    saveData();
    chrome.storage.local.set({ stopwatchData });
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
        noDetection: () => {
            handleNoDetection();
        },
        stopStopwatch: () => {
            stopStopwatch();
            chrome.storage.local.set({ running: false, buttonStateStopwatch: 'pause-btn' });
        },
        startStopwatch: () => {
            startStopwatch();
        },
        resetStopwatch: () => {
            stopwatchData.time = 0;
            stopwatchData.running = false;
            stopwatchData.startTime = Date.now();
            clearInterval(stopwatchInterval);
            saveData();
            chrome.runtime.sendMessage({ action: 'updateStopwatchTime', time: 0 });
            chrome.runtime.sendMessage({ action: 'updateButtonState', state: 'pause' });
        },
        startTimer: () => {
            timerData.running = true;
            timerData.startTime = Date.now() - timerData.time;
            startTimer();
        },
        stopTimer: stopTimer,
        resetTimer: () => {
            timerData.time = 0;
            timerData.running = false;
            timerData.startTime = Date.now();
            saveData();
        },
        updateTimer: (request) => {
            timerData.time = request.time;
            saveData();
        },
        getTimerData: () => sendResponse({ stopwatchData, timerData }),
        saveButtonState: (state) => chrome.storage.local.set({ buttonState: state }), // 상태 저장 로직
        saveSelectedMenu: (menu) => chrome.storage.local.set({ selectedMenu: menu }), // 메뉴 저장 로직

        triggerAlarm: () => {
            chrome.windows.getAll({ populate: true }, (windows) => {
                windows.forEach((window) => {
                    let activeTab = window.tabs.find(tab => tab.active);
                    if (activeTab && !activeTab.url.startsWith('chrome://')) {
                        chrome.scripting.executeScript(
                            {
                                target: { tabId: activeTab.id },
                                files: ['content.js']
                            },
                            () => {
                                chrome.tabs.sendMessage(activeTab.id, { action: 'wakeUp' });
                            }
                        );
                    }
                });
            });
        },


        toggleSwitch: (request) => {
            if (request.state) {
                // 스위치가 켜진 경우
                startStopwatch();
            } else {
                // 스위치가 꺼진 경우
                stopStopwatch();
            }
            // 모든 팝업에 상태 업데이트 메시지 전송
            chrome.runtime.sendMessage({ action: 'updateButtonState', state: 'pause' });
        },
        updateButtonState: (request) => {
            const buttonState = request.state === 'start' ? 'start-btn' : 'pause-btn';
            chrome.storage.local.set({ buttonState }, () => {
                // 모든 팝업에 상태 업데이트 메시지 전송
                chrome.runtime.sendMessage({ action: 'updateButtonState', state: buttonState });
            });
        }
    };
    if (actions[request.action]) actions[request.action](request);
});

function handleNoDetection() {
    chrome.storage.local.set({ firstToggle: false, switch: false }, () => {
        chrome.tabs.query({}, function(tabs) {
            tabs.forEach(function(tab) {
                if (tab.url && tab.url.includes('webcam.html')) {
                    chrome.scripting.executeScript(
                        {
                            target: { tabId: tab.id },
                            func: () => {
                                switchMode('stopwatch', stopwatchBtn);
                                setButtonState(pauseBtn); // pauseBtn 상태 저장 함수 사용
                                toggleRunning('pause'); // 이 부분을 추가하여 pause 상태로 전환
                            }
                        }
                    );
                    chrome.tabs.remove(tab.id);
                }
            });
        });

        // popup.html이 열려 있는지 확인하고 메시지 전송
        chrome.runtime.sendMessage({ action: 'updateToggle', state: false }, (response) => {
            if (chrome.runtime.lastError) {
                // popup.html이 열려 있지 않은 경우 상태 업데이트
                chrome.storage.local.set({ switch: false });
            }
        });

        // 확실히 스톱워치를 중지하도록 popup.html에 stopStopwatch 메시지 전송
        chrome.runtime.sendMessage({ action: 'stopStopwatch' }, (response) => {
            if (chrome.runtime.lastError) {
                // 메시지 전송 실패 시 로컬 상태 업데이트
                chrome.storage.local.set({ running: false, buttonStateStopwatch: 'pause-btn' });
            }
        });

        // 백그라운드에서 스톱워치를 중지하고 시간을 저장
        stopStopwatch();
    });
}
