// popup.js
document.addEventListener("DOMContentLoaded", () => {
    const toggleBody = document.querySelector(".toggleBody");
    const loginButton = document.getElementById('login-button');

    // 로그인 버튼 클릭 이벤트 추가
    loginButton.addEventListener('click', () => {
        // 로그인 페이지로 이동
        window.location.href = "webcam.html";  // 로그인 페이지 경로로 설정
    });

    // 기존 코드 유지...

    const dropdownButton = document.querySelector('.drop-button');
    const dropdownContent = document.getElementById('dropdown-content');
    const addItemBtn = document.getElementById('add-item-btn');
    const dropdown = document.querySelector('.dropdown');
    const stopwatchBtn = document.getElementById('stopwatch-btn');
    const timerBtn = document.getElementById('timer-btn');
    const startBtn = document.getElementById('start-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const resetBtn = document.getElementById('reset-btn');

    let isDropdownOpen = false;
    let timerMode = false;
    let running = false;
    let interval;
    let time = 0;
    let startTime = 0;

    // 음량 조절을 위한 요소들
    const alarmButton = document.querySelector('.alarm-button');
    const muteButtonIcon = document.getElementById('mute-button');
    const volumePopup = document.getElementById('volume-popup');
    const volumeSlider = document.getElementById('volume-slider');
    const muteButton = document.getElementById('mute-btn');
    let alarmAudio = new Audio(chrome.runtime.getURL('alarm.mp3'));

    // 초기 음량과 음소거 상태를 storage에서 가져옴

    chrome.storage.local.get(['alarmVolume', 'alarmMuted'], result => {
        alarmAudio.volume = result.alarmVolume !== undefined ? result.alarmVolume : 0.5;
        alarmAudio.muted = result.alarmMuted || false;
        volumeSlider.value = alarmAudio.volume * 100;
        muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
        updateSliderBackground(volumeSlider);
        updateAlarmIcon(alarmAudio.muted);
    });

    alarmButton.addEventListener('click', (event) => {
        event.stopPropagation();
        volumePopup.style.display = volumePopup.style.display === 'block' ? 'none' : 'block';
    });

    muteButtonIcon.addEventListener('click', (event) => {
        event.stopPropagation();
        volumePopup.style.display = volumePopup.style.display === 'block' ? 'none' : 'block';
    });

    document.addEventListener('click', (event) => {
        if (!event.target.closest('.head-button') && !event.target.closest('#volume-popup')) {
            volumePopup.style.display = 'none';
        }
    });

    volumeSlider.addEventListener('input', (event) => {
        updateSliderBackground(event.target);
        const volume = event.target.value / 100;
        alarmAudio.volume = volume;
        alarmAudio.muted = volume === 0;
        chrome.storage.local.set({ alarmVolume: volume, alarmMuted: alarmAudio.muted });
        muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
        updateAlarmIcon(alarmAudio.muted);
    });

    muteButton.addEventListener('click', () => {
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
        chrome.storage.local.set({ alarmVolume: alarmAudio.volume, alarmMuted: alarmAudio.muted });
        updateAlarmIcon(alarmAudio.muted);
    });

    function updateSliderBackground(slider) {
        const value = (slider.value - slider.min) / (slider.max - slider.min) * 100;
        slider.style.setProperty('--value', `${value}%`);
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
        chrome.storage.local.get(["switch", "firstToggle"], result => {
            if (result.switch) {
                toggleBody.classList.add("on");
            }

            // 처음 토글을 켰을 때만 페이지를 엽니다.
            if (result.switch && !result.firstToggle) {
                setTimeout(() => {
                    chrome.tabs.create({ url: chrome.runtime.getURL('../cam/webcam.html') });
                }, 200); // 200ms 지연
                chrome.storage.local.set({ firstToggle: true });
            }
        });

        toggleBody.addEventListener("click", () => {
            toggleBody.classList.add("transition");
            toggleBody.classList.toggle("on");
            const isOn = toggleBody.classList.contains("on");
            chrome.storage.local.set({ switch: isOn });

            if (isOn) {
                chrome.storage.local.get("firstToggle", result => {
                    if (!result.firstToggle) {
                        setTimeout(() => {
                            chrome.tabs.create({ url: chrome.runtime.getURL('../cam/webcam.html') });
                        }, 200); // 200ms 지연
                        chrome.storage.local.set({ firstToggle: true });
                    }
                });
            } else {
                // 토글이 꺼질 때 firstToggle 상태를 초기화합니다.
                chrome.storage.local.set({ firstToggle: false });
            }
        });
    }

    chrome.storage.local.get(['dropdownItems', 'selectedItem'], result => {
        const items = result.dropdownItems || [];
        items.forEach(addDropdownItem);
        if (result.selectedItem) {
            dropdown.querySelector('.dropdown-text').textContent = result.selectedItem;
        }
    });

    addItemBtn.addEventListener('click', createNewItemInput);

    dropdownButton.addEventListener('click', () => {
        isDropdownOpen = !isDropdownOpen;
        dropdownContent.style.display = isDropdownOpen ? 'block' : 'none';
        updateDropdownStyles();
    });

    dropdownButton.addEventListener('mouseover', () => {
        dropdown.style.borderColor = '#FA560C';
        dropdownButton.querySelector('path').style.stroke = '#FA560C';
    });

    dropdownButton.addEventListener('mouseout', () => {
        if (!isDropdownOpen) {
            dropdown.style.borderColor = '#C0C0C0';
            dropdownButton.querySelector('path').style.stroke = '#C0C0C0';
        }
    });

    dropdownContent.addEventListener('mouseover', () => {
        dropdown.style.borderColor = '#FA560C';
    });

    dropdownContent.addEventListener('mouseout', () => {
        if (!isDropdownOpen) {
            dropdown.style.borderColor = '#C0C0C0';
        }
    });

    document.addEventListener('click', (event) => {
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
        const newItem = document.createElement('button');
        newItem.textContent = itemText;
        newItem.addEventListener('click', () => {
            dropdown.querySelector('.dropdown-text').textContent = itemText;
            chrome.storage.local.set({ selectedItem: itemText });
        });

        const removeBtn = document.createElement('span');
        removeBtn.textContent = 'X';
        removeBtn.className = 'remove-btn';
        removeBtn.addEventListener('click', e => {
            e.stopPropagation();
            newItem.remove();
            saveItems();
        });

        newItem.appendChild(removeBtn);
        dropdownContent.insertBefore(newItem, addItemBtn);
    }

    function createNewItemInput() {
        const existingInput = document.querySelector('.input-new-item');
        if (existingInput) existingInput.remove();

        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = 'Enter new item';
        input.className = 'input-new-item';
        dropdownContent.insertBefore(input, addItemBtn);
        input.focus();

        input.addEventListener('blur', () => {
            if (input.value.trim() && !isItemExist(input.value.trim())) {
                addDropdownItem(input.value.trim());
                saveItems();
            }
            input.remove();
        });

        input.addEventListener('keypress', e => {
            if (e.key === 'Enter' && input.value.trim() && !isItemExist(input.value.trim())) {
                addDropdownItem(input.value.trim());
                saveItems();
                input.remove();
            }
        });
    }

    function isItemExist(itemText) {
        return Array.from(dropdownContent.querySelectorAll('button'))
            .some(button => button.firstChild.textContent === itemText);
    }

    function saveItems() {
        const items = Array.from(dropdownContent.querySelectorAll('button'))
            .filter(button => button !== addItemBtn)
            .map(button => button.firstChild.textContent);
        chrome.storage.local.set({ dropdownItems: items });
    }

    chrome.runtime.sendMessage({ action: 'getTimerData' }, result => {
        const stopwatchData = result.stopwatchData || { time: 0, running: false, startTime: Date.now() };
        const timerData = result.timerData || { time: 600000, running: false, startTime: Date.now() };

        chrome.storage.local.get(['timerMode', 'buttonStateStopwatch', 'buttonStateTimer'], res => {
            timerMode = res.timerMode || false;

            if (timerMode) {
                ({ time, running, startTime } = timerData);
                loadButtonState(res.buttonStateTimer);
            } else {
                ({ time, running, startTime } = stopwatchData);
                loadButtonState(res.buttonStateStopwatch);
            }

            if (running) interval = setInterval(updateTimer, 10);

            updateTimerDisplay();

            chrome.storage.local.get('selectedMenu', result => {
                selectMenu(document.getElementById(result.selectedMenu || 'stopwatch-btn'));
            });
        });
    });

    stopwatchBtn.addEventListener('click', () => switchMode('stopwatch', stopwatchBtn));
    timerBtn.addEventListener('click', () => switchMode('timer', timerBtn));

    startBtn.addEventListener('click', () => toggleRunning(startBtn, 'start'));
    pauseBtn.addEventListener('click', () => toggleRunning(pauseBtn, 'pause'));
    resetBtn.addEventListener('click', resetTimer);

    function switchMode(mode, button) {
        saveCurrentState();
        timerMode = (mode === 'timer');
        chrome.storage.local.set({ timerMode });
        chrome.runtime.sendMessage({ action: 'getTimerData' }, result => {
            const data = timerMode ? result.timerData : result.stopwatchData;
            ({ time, running, startTime } = data);
            updateTimerDisplay();
            if (running) {
                clearInterval(interval);
                interval = setInterval(updateTimer, 10);
            }
        });
        chrome.storage.local.get(timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch', res => {
            loadButtonState(res[timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch']);
        });
        selectMenu(button);
    }

    function toggleRunning(button, action) {
        if ((action === 'start' && !running) || (action === 'pause' && running)) {
            running = !running;
            startTime = Date.now() - time;
            if (running) {
                interval = setInterval(updateTimer, 10);
                chrome.runtime.sendMessage({ action: timerMode ? 'startTimer' : 'startStopwatch' });
            } else {
                clearInterval(interval);
                chrome.runtime.sendMessage({ action: timerMode ? 'stopTimer' : 'stopStopwatch' });
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
        chrome.runtime.sendMessage({ action: timerMode ? 'resetTimer' : 'resetStopwatch' });
    }

    function updateTimer() {
        time += running ? (timerMode ? -10 : 10) : 0;
        if (time <= 0) {
            time = 0;
            clearInterval(interval);
            running = false;
            chrome.runtime.sendMessage({ action: 'timerFinished' }); // 타이머 종료 메시지 전송
        }
        updateTimerDisplay();
    }

    function updateTimerDisplay() {
        const milliseconds = Math.floor(time % 1000 / 10);
        const seconds = Math.floor(time / 1000) % 60;
        const minutes = Math.floor(time / (1000 * 60)) % 60;
        const hours = Math.floor(time / (1000 * 60 * 60));
        document.getElementById('milliseconds').textContent = milliseconds.toString().padStart(2, '0');
        document.getElementById('seconds').textContent = seconds.toString().padStart(2, '0');
        document.getElementById('minutes').textContent = minutes.toString().padStart(2, '0');
        document.getElementById('hours').textContent = hours.toString().padStart(2, '0');
    }

    function setButtonState(button, isLoad = false) {
        startBtn.classList.remove('selected');
        pauseBtn.classList.remove('selected');
        startBtn.querySelector('svg path').style.fill = '#C0C0C0';
        pauseBtn.querySelector('svg path').style.fill = '#C0C0C0';

        if (button) {
            button.classList.add('selected');
            button.querySelector('svg path').style.fill = '#FA560C';
        }

        if (!isLoad) {
            const state = button ? button.id : null;
            chrome.storage.local.set(timerMode ? { buttonStateTimer: { timer: state } } : { buttonStateStopwatch: { stopwatch: state } });
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
        chrome.storage.local.set({ selectedMenu: button.id });
    }

    function saveCurrentState() {
        const state = {
            stopwatch: startBtn.classList.contains('selected') ? 'start-btn' : (pauseBtn.classList.contains('selected') ? 'pause-btn' : null),
            timer: startBtn.classList.contains('selected') ? 'start-btn' : (pauseBtn.classList.contains('selected') ? 'pause-btn' : null)
        };
        chrome.storage.local.set(timerMode ? { buttonStateTimer: state } : { buttonStateStopwatch: state });
    }

    function makeEditable(element) {
        if (!timerMode || running) return; // Disable edit mode in stopwatch mode or when the timer is running

        element.classList.add('edit-mode');
        element.contentEditable = true;
        element.style.color = '#d3d3d3';
        element.focus();

        const handleBlur = event => {
            if (event.type === 'blur' || (event.type === 'click' && !element.contains(event.target))) {
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

        const handleWheelEvent = event => {
            let value = parseInt(element.textContent, 10);
            const max = element.id === 'hours' ? 100 : 60;
            value = (value + (event.deltaY < 0 || event.key === 'ArrowUp' ? 1 : -1) + max) % max;
            element.textContent = String(value).padStart(2, '0');
            updateTime();
        };

        const handleKeydownEvent = event => {
            if (event.key === 'Enter') {
                element.blur();
                event.preventDefault();
            } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
                handleWheelEvent(event);
                event.preventDefault();
            } else if (/^\d$/.test(event.key)) {
                const maxValue = element.id === 'hours' ? 99 : 59;
                const currentValue = parseInt(element.textContent + event.key, 10);
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

    const updateTime = () => {
        const hours = parseInt(document.getElementById('hours').textContent, 10);
        const minutes = parseInt(document.getElementById('minutes').textContent, 10);
        const seconds = parseInt(document.getElementById('seconds').textContent, 10);
        const milliseconds = parseInt(document.getElementById('milliseconds').textContent, 10);

        time = ((hours * 60 * 60) + (minutes * 60) + seconds) * 1000 + (milliseconds * 10);
        chrome.runtime.sendMessage({ action: timerMode ? 'updateTimer' : 'updateStopwatch', time });
    };

    const setEditableListeners = element => {
        element.addEventListener('click', event => {
            event.stopPropagation();
            makeEditable(element);
        });
    };

    ['hours', 'minutes', 'seconds', 'milliseconds'].forEach(id => {
        setEditableListeners(document.getElementById(id));
    });

    const showMessage = message => {
        let messageBox = document.getElementById('message-box');
        if (!messageBox) {
            messageBox = document.createElement('div');
            messageBox.id = 'message-box';
            messageBox.textContent = message;
            document.body.appendChild(messageBox);
        } else {
            messageBox.textContent = message;
            messageBox.style.display = 'block';
        }
        setTimeout(() => { messageBox.style.display = 'none'; }, 3000);
    };

    // 알람 소리 재생 메시지 수신
    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === 'playAlarm') {
            alarmAudio.play();
        }
    });

});
