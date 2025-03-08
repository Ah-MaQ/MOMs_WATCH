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
chrome.runtime.onInstalled.addListener(function (details) {
  if (details.reason === "install") {
    chrome.storage.local.set({
      "switch": false
    });
  }
});
var timerInterval, stopwatchInterval;
var stopwatchData = {
  time: 0,
  running: false,
  startTime: Date.now()
};
var timerData = {
  time: 600000,
  // 10분
  running: false,
  startTime: Date.now()
};

// 초기 데이터 로드
chrome.storage.local.get(['stopwatchData', 'timerData'], function (result) {
  stopwatchData = result.stopwatchData || stopwatchData;
  timerData = result.timerData || timerData;
  if (stopwatchData.running) _startStopwatch();
  if (timerData.running) _startTimer();
});
var _startStopwatch = function startStopwatch() {
  if (stopwatchInterval) clearInterval(stopwatchInterval);
  stopwatchInterval = setInterval(function () {
    if (stopwatchData.running) {
      stopwatchData.time += 1000;
      saveData();
    }
  }, 1000);
};
var _startTimer = function startTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(function () {
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
var stopStopwatch = function stopStopwatch() {
  stopwatchData.running = false;
  clearInterval(stopwatchInterval);
  saveData();
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
    startStopwatch: function startStopwatch() {
      stopwatchData.running = true;
      stopwatchData.startTime = Date.now() - stopwatchData.time;
      _startStopwatch();
    },
    stopStopwatch: stopStopwatch,
    resetStopwatch: function resetStopwatch() {
      stopwatchData.time = 0;
      stopwatchData.running = false;
      stopwatchData.startTime = Date.now();
      saveData();
    },
    startTimer: function startTimer() {
      timerData.running = true;
      timerData.startTime = Date.now() - (600000 - timerData.time);
      _startTimer();
    },
    stopTimer: stopTimer,
    resetTimer: function resetTimer() {
      timerData.time = 600000;
      timerData.running = false;
      timerData.startTime = Date.now();
      saveData();
    },
    updateTimer: function updateTimer() {
      timerData.time = request.time;
      saveData();
    },
    getTimerData: function getTimerData() {
      return sendResponse({
        stopwatchData: stopwatchData,
        timerData: timerData
      });
    },
    saveButtonState: function saveButtonState() {
      return chrome.storage.local.set({
        buttonState: request.buttonState
      });
    },
    saveSelectedMenu: function saveSelectedMenu() {
      return chrome.storage.local.set({
        selectedMenu: request.selectedMenu
      });
    }
  };
  if (actions[request.action]) actions[request.action]();
});
/******/ })()
;
//# sourceMappingURL=background.bundle.js.map