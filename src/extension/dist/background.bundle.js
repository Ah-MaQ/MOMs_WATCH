/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	// The require scope
/******/ 	var __webpack_require__ = {};
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
/*!***********************!*\
  !*** ./background.js ***!
  \***********************/
__webpack_require__.r(__webpack_exports__);
chrome.runtime.onInstalled.addListener(function () {
  chrome.storage.local.set({
    "switch": false
  });
});
var timerInterval, stopwatchInterval;
var stopwatchData = {
  time: 0,
  running: false,
  startTime: Date.now()
};
var timerData = {
  time: 0,
  running: false,
  startTime: Date.now()
};
chrome.storage.local.get(['stopwatchData', 'timerData'], function (result) {
  stopwatchData = result.stopwatchData || stopwatchData;
  timerData = result.timerData || timerData;
  if (stopwatchData.running) _startStopwatch();
  if (timerData.running) _startTimer();
});
var _startStopwatch = function startStopwatch() {
  if (stopwatchInterval) clearInterval(stopwatchInterval);
  stopwatchData.running = true;
  stopwatchData.startTime = Date.now() - stopwatchData.time;
  stopwatchInterval = setInterval(function () {
    if (stopwatchData.running) {
      stopwatchData.time = Date.now() - stopwatchData.startTime;
      saveData();
      chrome.runtime.sendMessage({
        action: 'updateStopwatchTime',
        time: stopwatchData.time
      });
    }
  }, 100); // 100밀리초마다 실행되도록 설정
};
var _startTimer = function startTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerData.running = true;
  timerData.startTime = Date.now() - timerData.time;
  timerInterval = setInterval(function () {
    if (timerData.running) {
      timerData.time -= 100;
      if (timerData.time <= 0) {
        timerData.time = 0;
        stopTimer();
        chrome.windows.getAll({
          populate: true
        }, function (windows) {
          windows.forEach(function (window) {
            var activeTab = window.tabs.find(function (tab) {
              return tab.active;
            });
            if (activeTab && !activeTab.url.startsWith('chrome://')) {
              chrome.scripting.executeScript({
                target: {
                  tabId: activeTab.id
                },
                files: ['content.js']
              }, function () {
                chrome.tabs.sendMessage(activeTab.id, {
                  action: 'showAlarm'
                });
              });
            }
          });
        });
      }
      saveData();
      chrome.runtime.sendMessage({
        action: 'updateTimerTime',
        time: timerData.time
      });
    }
  }, 100); // 100밀리초마다 실행
};
var _stopStopwatch = function stopStopwatch() {
  stopwatchData.running = false;
  clearInterval(stopwatchInterval);
  saveData();
  chrome.storage.local.set({
    stopwatchData: stopwatchData
  });
};
var stopTimer = function stopTimer() {
  timerData.running = false;
  clearInterval(timerInterval);
  saveData();
};
var saveData = function saveData() {
  chrome.storage.local.set({
    stopwatchData: stopwatchData,
    timerData: timerData
  });
};
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  var actions = {
    noDetection: function noDetection() {
      handleNoDetection();
    },
    stopStopwatch: function stopStopwatch() {
      _stopStopwatch();
      chrome.storage.local.set({
        running: false,
        buttonStateStopwatch: 'pause-btn'
      });
    },
    startStopwatch: function startStopwatch() {
      _startStopwatch();
    },
    resetStopwatch: function resetStopwatch() {
      stopwatchData.time = 0;
      stopwatchData.running = false;
      stopwatchData.startTime = Date.now();
      clearInterval(stopwatchInterval);
      saveData();
      chrome.runtime.sendMessage({
        action: 'updateStopwatchTime',
        time: 0
      });
      chrome.runtime.sendMessage({
        action: 'updateButtonState',
        state: 'pause'
      });
    },
    startTimer: function startTimer() {
      timerData.running = true;
      timerData.startTime = Date.now() - timerData.time;
      _startTimer();
    },
    stopTimer: stopTimer,
    resetTimer: function resetTimer() {
      timerData.time = 0;
      timerData.running = false;
      timerData.startTime = Date.now();
      saveData();
    },
    updateTimer: function updateTimer(request) {
      timerData.time = request.time;
      saveData();
    },
    getTimerData: function getTimerData() {
      return sendResponse({
        stopwatchData: stopwatchData,
        timerData: timerData
      });
    },
    saveButtonState: function saveButtonState(state) {
      return chrome.storage.local.set({
        buttonState: state
      });
    },
    // 상태 저장 로직
    saveSelectedMenu: function saveSelectedMenu(menu) {
      return chrome.storage.local.set({
        selectedMenu: menu
      });
    },
    // 메뉴 저장 로직

    triggerAlarm: function triggerAlarm() {
      chrome.windows.getAll({
        populate: true
      }, function (windows) {
        windows.forEach(function (window) {
          var activeTab = window.tabs.find(function (tab) {
            return tab.active;
          });
          if (activeTab && !activeTab.url.startsWith('chrome://')) {
            chrome.scripting.executeScript({
              target: {
                tabId: activeTab.id
              },
              files: ['content.js']
            }, function () {
              chrome.tabs.sendMessage(activeTab.id, {
                action: 'wakeUp'
              });
            });
          }
        });
      });
    },
    toggleSwitch: function toggleSwitch(request) {
      if (request.state) {
        // 스위치가 켜진 경우
        _startStopwatch();
      } else {
        // 스위치가 꺼진 경우
        _stopStopwatch();
      }
      // 모든 팝업에 상태 업데이트 메시지 전송
      chrome.runtime.sendMessage({
        action: 'updateButtonState',
        state: 'pause'
      });
    },
    updateButtonState: function updateButtonState(request) {
      var buttonState = request.state === 'start' ? 'start-btn' : 'pause-btn';
      chrome.storage.local.set({
        buttonState: buttonState
      }, function () {
        // 모든 팝업에 상태 업데이트 메시지 전송
        chrome.runtime.sendMessage({
          action: 'updateButtonState',
          state: buttonState
        });
      });
    }
  };
  if (actions[request.action]) actions[request.action](request);
});
function handleNoDetection() {
  chrome.storage.local.set({
    firstToggle: false,
    "switch": false
  }, function () {
    chrome.tabs.query({}, function (tabs) {
      tabs.forEach(function (tab) {
        if (tab.url && tab.url.includes('webcam.html')) {
          chrome.scripting.executeScript({
            target: {
              tabId: tab.id
            },
            func: function func() {
              switchMode('stopwatch', stopwatchBtn);
              setButtonState(pauseBtn); // pauseBtn 상태 저장 함수 사용
              toggleRunning('pause'); // 이 부분을 추가하여 pause 상태로 전환
            }
          });
          chrome.tabs.remove(tab.id);
        }
      });
    });

    // popup.html이 열려 있는지 확인하고 메시지 전송
    chrome.runtime.sendMessage({
      action: 'updateToggle',
      state: false
    }, function (response) {
      if (chrome.runtime.lastError) {
        // popup.html이 열려 있지 않은 경우 상태 업데이트
        chrome.storage.local.set({
          "switch": false
        });
      }
    });

    // 확실히 스톱워치를 중지하도록 popup.html에 stopStopwatch 메시지 전송
    chrome.runtime.sendMessage({
      action: 'stopStopwatch'
    }, function (response) {
      if (chrome.runtime.lastError) {
        // 메시지 전송 실패 시 로컬 상태 업데이트
        chrome.storage.local.set({
          running: false,
          buttonStateStopwatch: 'pause-btn'
        });
      }
    });

    // 백그라운드에서 스톱워치를 중지하고 시간을 저장
    _stopStopwatch();
  });
}
/******/ })()
;
//# sourceMappingURL=background.bundle.js.map