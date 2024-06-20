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
/*!*****************************!*\
  !*** ./page/popup/popup.js ***!
  \*****************************/
__webpack_require__.r(__webpack_exports__);
// popup.js
document.addEventListener("DOMContentLoaded", function () {
  var toggleBody = document.querySelector(".toggleBody");
  var loginButton = document.getElementById('login-button');
  var settingsButton = document.querySelector('.settings-button'); // 클래스 선택자로 선택

  settingsButton.addEventListener('click', function () {
    chrome.tabs.create({
      url: chrome.runtime.getURL("../../webpage/settings.html")
    });
    // window.location.href = "../webpage/settings.html";
  });

  // 로그인 버튼 클릭 이벤트 추가
  loginButton.addEventListener('click', function () {
    // 로그인 페이지로 이동
    window.location.href = "../login/login.html"; // 로그인 페이지 경로로 설정
  });

  // 기존 코드 유지...

  var dropdownButton = document.querySelector('.drop-button');
  var dropdownContent = document.getElementById('dropdown-content');
  var addItemBtn = document.getElementById('add-item-btn');
  var dropdown = document.querySelector('.dropdown');
  var stopwatchBtn = document.getElementById('stopwatch-btn');
  var timerBtn = document.getElementById('timer-btn');
  var startBtn = document.getElementById('start-btn');
  var pauseBtn = document.getElementById('pause-btn');
  var resetBtn = document.getElementById('reset-btn');
  var isDropdownOpen = false;
  var timerMode = false;
  var running = false;
  var interval;
  var time = 0;
  var startTime = 0;

  // 음량 조절을 위한 요소들
  var alarmButton = document.querySelector('.alarm-button');
  var muteButtonIcon = document.getElementById('mute-button');
  var volumePopup = document.getElementById('volume-popup');
  var volumeSlider = document.getElementById('volume-slider');
  var muteButton = document.getElementById('mute-btn');
  var alarmAudio = new Audio(chrome.runtime.getURL('alarm.mp3'));

  // 초기 음량과 음소거 상태를 storage에서 가져옴

  chrome.storage.local.get(['alarmVolume', 'alarmMuted'], function (result) {
    alarmAudio.volume = result.alarmVolume !== undefined ? result.alarmVolume : 0.5;
    alarmAudio.muted = result.alarmMuted || false;
    volumeSlider.value = alarmAudio.volume * 100;
    muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
    updateSliderBackground(volumeSlider);
    updateAlarmIcon(alarmAudio.muted);
  });
  alarmButton.addEventListener('click', function (event) {
    event.stopPropagation();
    volumePopup.style.display = volumePopup.style.display === 'block' ? 'none' : 'block';
  });
  muteButtonIcon.addEventListener('click', function (event) {
    event.stopPropagation();
    volumePopup.style.display = volumePopup.style.display === 'block' ? 'none' : 'block';
  });
  document.addEventListener('click', function (event) {
    if (!event.target.closest('.head-button') && !event.target.closest('#volume-popup')) {
      volumePopup.style.display = 'none';
    }
  });
  volumeSlider.addEventListener('input', function (event) {
    updateSliderBackground(event.target);
    var volume = event.target.value / 100;
    alarmAudio.volume = volume;
    alarmAudio.muted = volume === 0;
    chrome.storage.local.set({
      alarmVolume: volume,
      alarmMuted: alarmAudio.muted
    });
    muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
    updateAlarmIcon(alarmAudio.muted);
  });
  muteButton.addEventListener('click', function () {
    alarmAudio.muted = !alarmAudio.muted;
    if (alarmAudio.muted) {
      volumeSlider.value = 0;
      alarmAudio.volume = 0;
    } else {
      volumeSlider.value = 50;
      alarmAudio.volume = 0.5;
    }
    updateSliderBackground(volumeSlider);
    muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
    chrome.storage.local.set({
      alarmVolume: alarmAudio.volume,
      alarmMuted: alarmAudio.muted
    });
    updateAlarmIcon(alarmAudio.muted);
  });
  function updateSliderBackground(slider) {
    var value = (slider.value - slider.min) / (slider.max - slider.min) * 100;
    slider.style.setProperty('--value', "".concat(value, "%"));
  }
  function updateAlarmIcon(isMuted) {
    if (isMuted) {
      alarmButton.style.display = 'none';
      muteButtonIcon.style.display = 'block';
    } else {
      alarmButton.style.display = 'block';
      muteButtonIcon.style.display = 'none';
    }
  }
  if (toggleBody) {
    chrome.storage.local.get(["switch", "firstToggle"], function (result) {
      if (result["switch"]) {
        toggleBody.classList.add("on");
      }

      // 처음 토글을 켰을 때만 페이지를 엽니다.
      if (result["switch"] && !result.firstToggle) {
        setTimeout(function () {
          chrome.tabs.create({
            url: chrome.runtime.getURL('../cam/webcam.html')
          });
        }, 200); // 200ms 지연
        chrome.storage.local.set({
          firstToggle: true
        });
      }
    });
    toggleBody.addEventListener("click", function () {
      toggleBody.classList.add("transition");
      toggleBody.classList.toggle("on");
      var isOn = toggleBody.classList.contains("on");
      chrome.storage.local.set({
        "switch": isOn
      });
      if (isOn) {
        chrome.storage.local.get("firstToggle", function (result) {
          if (!result.firstToggle) {
            setTimeout(function () {
              chrome.tabs.create({
                url: chrome.runtime.getURL('../cam/webcam.html')
              });
            }, 200); // 200ms 지연
            chrome.storage.local.set({
              firstToggle: true
            });
          }
        });
      } else {
        // 토글이 꺼질 때 firstToggle 상태를 초기화합니다.
        chrome.storage.local.set({
          firstToggle: false
        });
      }
    });
  }
  chrome.storage.local.get(['dropdownItems', 'selectedItem'], function (result) {
    var items = result.dropdownItems || [];
    items.forEach(addDropdownItem);
    if (result.selectedItem) {
      dropdown.querySelector('.dropdown-text').textContent = result.selectedItem;
    }
  });
  addItemBtn.addEventListener('click', createNewItemInput);
  dropdownButton.addEventListener('click', function () {
    isDropdownOpen = !isDropdownOpen;
    dropdownContent.style.display = isDropdownOpen ? 'block' : 'none';
    updateDropdownStyles();
  });
  dropdownButton.addEventListener('mouseover', function () {
    dropdown.style.borderColor = '#FA560C';
    dropdownButton.querySelector('path').style.stroke = '#FA560C';
  });
  dropdownButton.addEventListener('mouseout', function () {
    if (!isDropdownOpen) {
      dropdown.style.borderColor = '#C0C0C0';
      dropdownButton.querySelector('path').style.stroke = '#C0C0C0';
    }
  });
  dropdownContent.addEventListener('mouseover', function () {
    dropdown.style.borderColor = '#FA560C';
  });
  dropdownContent.addEventListener('mouseout', function () {
    if (!isDropdownOpen) {
      dropdown.style.borderColor = '#C0C0C0';
    }
  });
  document.addEventListener('click', function (event) {
    if (!dropdown.contains(event.target) && !dropdownButton.contains(event.target) && !dropdownContent.contains(event.target)) {
      dropdownContent.style.display = 'none';
      isDropdownOpen = false;
      updateDropdownStyles();
    }
  });
  function updateDropdownStyles() {
    if (isDropdownOpen) {
      dropdown.style.borderColor = '#FA560C';
      dropdownButton.querySelector('path').style.stroke = '#FA560C';
    } else {
      dropdown.style.borderColor = '#C0C0C0';
      dropdownButton.querySelector('path').style.stroke = '#C0C0C0';
    }
  }
  function addDropdownItem(itemText) {
    var newItem = document.createElement('button');
    newItem.textContent = itemText;
    newItem.addEventListener('click', function () {
      dropdown.querySelector('.dropdown-text').textContent = itemText;
      chrome.storage.local.set({
        selectedItem: itemText
      });
    });
    var removeBtn = document.createElement('span');
    removeBtn.textContent = 'X';
    removeBtn.className = 'remove-btn';
    removeBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      newItem.remove();
      saveItems();
    });
    newItem.appendChild(removeBtn);
    dropdownContent.insertBefore(newItem, addItemBtn);
  }
  function createNewItemInput() {
    var existingInput = document.querySelector('.input-new-item');
    if (existingInput) existingInput.remove();
    var input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Enter new item';
    input.className = 'input-new-item';
    dropdownContent.insertBefore(input, addItemBtn);
    input.focus();
    input.addEventListener('blur', function () {
      if (input.value.trim() && !isItemExist(input.value.trim())) {
        addDropdownItem(input.value.trim());
        saveItems();
      }
      input.remove();
    });
    input.addEventListener('keypress', function (e) {
      if (e.key === 'Enter' && input.value.trim() && !isItemExist(input.value.trim())) {
        addDropdownItem(input.value.trim());
        saveItems();
        input.remove();
      }
    });
  }
  function isItemExist(itemText) {
    return Array.from(dropdownContent.querySelectorAll('button')).some(function (button) {
      return button.firstChild.textContent === itemText;
    });
  }
  function saveItems() {
    var items = Array.from(dropdownContent.querySelectorAll('button')).filter(function (button) {
      return button !== addItemBtn;
    }).map(function (button) {
      return button.firstChild.textContent;
    });
    chrome.storage.local.set({
      dropdownItems: items
    });
  }
  chrome.runtime.sendMessage({
    action: 'getTimerData'
  }, function (result) {
    var stopwatchData = result.stopwatchData || {
      time: 0,
      running: false,
      startTime: Date.now()
    };
    var timerData = result.timerData || {
      time: 600000,
      running: false,
      startTime: Date.now()
    };
    chrome.storage.local.get(['timerMode', 'buttonStateStopwatch', 'buttonStateTimer'], function (res) {
      timerMode = res.timerMode || false;
      if (timerMode) {
        time = timerData.time;
        running = timerData.running;
        startTime = timerData.startTime;
        loadButtonState(res.buttonStateTimer);
      } else {
        time = stopwatchData.time;
        running = stopwatchData.running;
        startTime = stopwatchData.startTime;
        loadButtonState(res.buttonStateStopwatch);
      }
      if (running) interval = setInterval(updateTimer, 10);
      updateTimerDisplay();
      chrome.storage.local.get('selectedMenu', function (result) {
        selectMenu(document.getElementById(result.selectedMenu || 'stopwatch-btn'));
      });
    });
  });
  stopwatchBtn.addEventListener('click', function () {
    return switchMode('stopwatch', stopwatchBtn);
  });
  timerBtn.addEventListener('click', function () {
    return switchMode('timer', timerBtn);
  });
  startBtn.addEventListener('click', function () {
    return toggleRunning(startBtn, 'start');
  });
  pauseBtn.addEventListener('click', function () {
    return toggleRunning(pauseBtn, 'pause');
  });
  resetBtn.addEventListener('click', resetTimer);
  function switchMode(mode, button) {
    saveCurrentState();
    timerMode = mode === 'timer';
    chrome.storage.local.set({
      timerMode: timerMode
    });
    chrome.runtime.sendMessage({
      action: 'getTimerData'
    }, function (result) {
      var data = timerMode ? result.timerData : result.stopwatchData;
      time = data.time;
      running = data.running;
      startTime = data.startTime;
      updateTimerDisplay();
      if (running) {
        clearInterval(interval);
        interval = setInterval(updateTimer, 10);
      }
    });
    chrome.storage.local.get(timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch', function (res) {
      loadButtonState(res[timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch']);
    });
    selectMenu(button);
  }
  function toggleRunning(button, action) {
    if (action === 'start' && !running || action === 'pause' && running) {
      running = !running;
      startTime = Date.now() - time;
      if (running) {
        interval = setInterval(updateTimer, 10);
        chrome.runtime.sendMessage({
          action: timerMode ? 'startTimer' : 'startStopwatch'
        });
      } else {
        clearInterval(interval);
        chrome.runtime.sendMessage({
          action: timerMode ? 'stopTimer' : 'stopStopwatch'
        });
      }
      setButtonState(button);
    }
  }
  function resetTimer() {
    clearInterval(interval);
    time = timerMode ? 600000 : 0;
    running = false;
    startTime = Date.now();
    updateTimerDisplay();
    setButtonState(null);
    chrome.runtime.sendMessage({
      action: timerMode ? 'resetTimer' : 'resetStopwatch'
    });
  }
  function updateTimer() {
    time += running ? timerMode ? -10 : 10 : 0;
    if (time <= 0) {
      time = 0;
      clearInterval(interval);
      running = false;
      chrome.runtime.sendMessage({
        action: 'timerFinished'
      }); // 타이머 종료 메시지 전송
    }
    updateTimerDisplay();
  }
  function updateTimerDisplay() {
    var milliseconds = Math.floor(time % 1000 / 10);
    var seconds = Math.floor(time / 1000) % 60;
    var minutes = Math.floor(time / (1000 * 60)) % 60;
    var hours = Math.floor(time / (1000 * 60 * 60));
    document.getElementById('milliseconds').textContent = milliseconds.toString().padStart(2, '0');
    document.getElementById('seconds').textContent = seconds.toString().padStart(2, '0');
    document.getElementById('minutes').textContent = minutes.toString().padStart(2, '0');
    document.getElementById('hours').textContent = hours.toString().padStart(2, '0');
  }
  function setButtonState(button) {
    var isLoad = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    startBtn.classList.remove('selected');
    pauseBtn.classList.remove('selected');
    startBtn.querySelector('svg path').style.fill = '#C0C0C0';
    pauseBtn.querySelector('svg path').style.fill = '#C0C0C0';
    if (button) {
      button.classList.add('selected');
      button.querySelector('svg path').style.fill = '#FA560C';
    }
    if (!isLoad) {
      var state = button ? button.id : null;
      chrome.storage.local.set(timerMode ? {
        buttonStateTimer: {
          timer: state
        }
      } : {
        buttonStateStopwatch: {
          stopwatch: state
        }
      });
    }
  }
  function loadButtonState(state) {
    if (state) {
      setButtonState(document.getElementById(state[timerMode ? 'timer' : 'stopwatch']), true);
    }
  }
  function selectMenu(button) {
    stopwatchBtn.classList.remove('selected');
    timerBtn.classList.remove('selected');
    button.classList.add('selected');
    chrome.storage.local.set({
      selectedMenu: button.id
    });
  }
  function saveCurrentState() {
    var state = {
      stopwatch: startBtn.classList.contains('selected') ? 'start-btn' : pauseBtn.classList.contains('selected') ? 'pause-btn' : null,
      timer: startBtn.classList.contains('selected') ? 'start-btn' : pauseBtn.classList.contains('selected') ? 'pause-btn' : null
    };
    chrome.storage.local.set(timerMode ? {
      buttonStateTimer: state
    } : {
      buttonStateStopwatch: state
    });
  }
  function makeEditable(element) {
    if (!timerMode || running) return; // Disable edit mode in stopwatch mode or when the timer is running

    element.classList.add('edit-mode');
    element.contentEditable = true;
    element.style.color = '#d3d3d3';
    element.focus();
    var handleBlur = function handleBlur(event) {
      if (event.type === 'blur' || event.type === 'click' && !element.contains(event.target)) {
        element.contentEditable = false;
        element.classList.remove('edit-mode');
        element.style.color = '';
        updateTime();
        document.removeEventListener('click', handleBlur);
        element.removeEventListener('blur', handleBlur);
        element.removeEventListener('wheel', handleWheelEvent);
        element.removeEventListener('keydown', handleKeydownEvent);
      }
    };
    var handleWheelEvent = function handleWheelEvent(event) {
      var value = parseInt(element.textContent, 10);
      var max = element.id === 'hours' ? 100 : 60;
      value = (value + (event.deltaY < 0 || event.key === 'ArrowUp' ? 1 : -1) + max) % max;
      element.textContent = String(value).padStart(2, '0');
      updateTime();
    };
    var handleKeydownEvent = function handleKeydownEvent(event) {
      if (event.key === 'Enter') {
        element.blur();
        event.preventDefault();
      } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        handleWheelEvent(event);
        event.preventDefault();
      } else if (/^\d$/.test(event.key)) {
        var maxValue = element.id === 'hours' ? 99 : 59;
        var currentValue = parseInt(element.textContent + event.key, 10);
        element.textContent = currentValue > maxValue ? '00' : String(currentValue).padStart(2, '0');
        updateTime();
        event.preventDefault();
      } else if (event.key === 'Backspace') {
        element.textContent = '00';
        updateTime();
        event.preventDefault();
      } else {
        showMessage('Invalid input');
        event.preventDefault();
      }
    };
    element.addEventListener('blur', handleBlur);
    document.addEventListener('click', handleBlur);
    element.addEventListener('wheel', handleWheelEvent);
    element.addEventListener('keydown', handleKeydownEvent);
  }
  var updateTime = function updateTime() {
    var hours = parseInt(document.getElementById('hours').textContent, 10);
    var minutes = parseInt(document.getElementById('minutes').textContent, 10);
    var seconds = parseInt(document.getElementById('seconds').textContent, 10);
    var milliseconds = parseInt(document.getElementById('milliseconds').textContent, 10);
    time = (hours * 60 * 60 + minutes * 60 + seconds) * 1000 + milliseconds * 10;
    chrome.runtime.sendMessage({
      action: timerMode ? 'updateTimer' : 'updateStopwatch',
      time: time
    });
  };
  var setEditableListeners = function setEditableListeners(element) {
    element.addEventListener('click', function (event) {
      event.stopPropagation();
      makeEditable(element);
    });
  };
  ['hours', 'minutes', 'seconds', 'milliseconds'].forEach(function (id) {
    setEditableListeners(document.getElementById(id));
  });
  var showMessage = function showMessage(message) {
    var messageBox = document.getElementById('message-box');
    if (!messageBox) {
      messageBox = document.createElement('div');
      messageBox.id = 'message-box';
      messageBox.textContent = message;
      document.body.appendChild(messageBox);
    } else {
      messageBox.textContent = message;
      messageBox.style.display = 'block';
    }
    setTimeout(function () {
      messageBox.style.display = 'none';
    }, 3000);
  };

  // 알람 소리 재생 메시지 수신
  chrome.runtime.onMessage.addListener(function (message) {
    if (message.action === 'playAlarm') {
      alarmAudio.play();
    }
  });
});

// document.addEventListener("DOMContentLoaded", () => {
//     const API_URL = "https://secret-journey-41438-8aa9540f2edc.herokuapp.com"
//     fetch(API_URL + "/read/test2")
//     .then((response) => {
//         return (response.json())
//     })
//     .then((data) => {
//         console.log(data)
//     })
// })

document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("database-button");
  button.addEventListener("click", function () {
    window.location.href = '../data/data.html';
  });
});
/******/ })()
;
//# sourceMappingURL=popup.bundle.js.map