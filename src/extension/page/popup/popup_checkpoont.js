import { initializeApp } from "firebase/app";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { getFirestore, doc, setDoc, getDoc, updateDoc } from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyCZ_kz-1KLtSUlKiaVRKs7KLjsQLRDmgko",
    authDomain: "mom-swatch.firebaseapp.com",
    projectId: "mom-swatch",
    storageBucket: "mom-swatch.appspot.com",
    messagingSenderId: "663295526234",
    appId: "1:663295526234:web:3782f2e0bd775252511f63"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

document.addEventListener("DOMContentLoaded", () => {
    // 초기 변수 설정
    const toggleBody = document.querySelector(".toggleBody");
    const loginButton = document.getElementById('login-button');
    const settingsButton = document.querySelector('.settings-button');
    const stopwatchBtn = document.getElementById('stopwatch-btn');
    const timerBtn = document.getElementById('timer-btn');
    const startBtn = document.getElementById('start-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const resetBtn = document.getElementById('reset-btn');
    const alarmButton = document.querySelector('.alarm-button');
    const muteButtonIcon = document.getElementById('mute-button');
    const volumePopup = document.getElementById('volume-popup');
    const volumeSlider = document.getElementById('volume-slider');
    const muteButton = document.getElementById('mute-btn');
    const dropdownButton = document.querySelector('.drop-button');
    const dropdownContent = document.querySelector('.dropdown-content');
    const addItemBtn = document.getElementById('add-item-btn');
    const dropdown = document.querySelector('.dropdown');

    let alarmAudio = new Audio(chrome.runtime.getURL('alarm.mp3'));
    let isDropdownOpen = false;
    let timerMode = false; // 기본 모드를 스톱워치로 설정
    let running = false;
    let interval;
    let time = 0;
    let startTime = 0;
    let elapsedTime = 0;
    let usedTimeInMinutes = 0;

    // 로그인 상태 확인 및 UI 업데이트
    onAuthStateChanged(auth, (user) => {
        if (user) {
            const email = user.email;
            const uid = user.uid;
            chrome.storage.local.set({ user: { email: email, uid: uid } }, () => {
                loginButton.textContent = 'Data';
                loginButton.addEventListener('click', () => {
                    window.location.href = "../chart/chart.html";  // chart 페이지로 리디렉션
                });
            });
        } else {
            chrome.storage.local.remove('user', () => {
                loginButton.textContent = 'Login';
                loginButton.addEventListener('click', () => {
                    window.location.href = "../login/login.html";  // Login 페이지로 리디렉션
                });
            });
        }
    });

    // 초기 설정 로드
    chrome.storage.local.get(['timerMode', 'buttonStateStopwatch', 'buttonStateTimer', 'selectedMenu', 'alarmVolume', 'alarmMuted', 'dropdownItems', 'selectedItem', 'stopwatchData', 'timerData', 'running', 'switch'], res => {
        timerMode = res.timerMode !== undefined ? res.timerMode : false; // 초기 모드 설정
        const data = timerMode ? res.timerData : res.stopwatchData;
        ({ time, running, startTime } = data || { time: 0, running: false, startTime: Date.now() });

        if (res.running === false) {
            running = false;
            setButtonState(pauseBtn);  // pauseBtn 상태 설정
        } else if (running) {
            if (timerMode) {
                startTime = Date.now() + time; // 남은 시간에서 시작
            } else {
                startTime = Date.now() - time; // 경과 시간에서 시작
            }
            interval = setInterval(updateTimer, 1000);
        }

        // 버튼 상태 로드
        loadButtonState(res[timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch']);

        selectMenu(document.getElementById(res.selectedMenu || 'stopwatch-btn'));

        // 알람 설정 로드
        alarmAudio.volume = res.alarmVolume !== undefined ? res.alarmVolume : 0.5;
        alarmAudio.muted = res.alarmMuted || false;
        volumeSlider.value = alarmAudio.volume * 100;
        muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
        updateSliderBackground(volumeSlider);
        updateAlarmIcon(alarmAudio.muted);

        // 드롭다운 메뉴 로드
        const items = res.dropdownItems || [];
        items.forEach(addDropdownItem);
        if (res.selectedItem) {
            dropdown.querySelector('.dropdown-text').textContent = res.selectedItem;
        }

        // 초기화 시 타이머를 강제로 업데이트
        if (timerMode) {
            updateTimerDisplay(time);
        } else {
            updateStopwatchDisplay(time);
        }

        // toggleBody 상태 로드
        if (res.switch) {
            toggleBody.classList.add("on");
        } else {
            toggleBody.classList.remove("on");
        }
    });

    // 설정 버튼 클릭 이벤트
    settingsButton.addEventListener('click', () => {
        chrome.tabs.create({ url: chrome.runtime.getURL("../../webpage/settings.html") });
    });

    // 모드 스위치
    stopwatchBtn.addEventListener('click', () => switchMode('stopwatch', stopwatchBtn));
    timerBtn.addEventListener('click', () => switchMode('timer', timerBtn));

    // 타이머 조작 버튼
    startBtn.addEventListener('click', () => toggleRunning('start'));
    pauseBtn.addEventListener('click', () => toggleRunning('pause'));
    resetBtn.addEventListener('click', resetTimer);

    // 볼륨 조절 및 음소거 버튼 이벤트
    alarmButton.addEventListener('click', toggleVolumePopup);
    muteButtonIcon.addEventListener('click', toggleVolumePopup);
    document.addEventListener('click', closeVolumePopupOnClickOutside);
    volumeSlider.addEventListener('input', adjustVolume);
    muteButton.addEventListener('click', toggleMute);

    // 드롭다운 메뉴 이벤트
    dropdownButton.addEventListener('click', toggleDropdown);
    dropdownButton.addEventListener('mouseover', () => changeDropdownStyle(true));
    dropdownButton.addEventListener('mouseout', () => changeDropdownStyle(false));
    dropdownContent.addEventListener('mouseover', () => changeDropdownBorder(true));
    dropdownContent.addEventListener('mouseout', () => changeDropdownBorder(false));
    addItemBtn.addEventListener('click', createNewItemInput);

    // 타이머 업데이트 함수
    function updateTimer() {
        if (running) {
            if (timerMode) {
                time = startTime - Date.now();
                if (time <= 0) {
                    time = 0;
                    clearInterval(interval);
                    running = false;
                    chrome.runtime.sendMessage({ action: 'timerFinished' });
                }
            } else {
                time = Date.now() - startTime;
            }
            chrome.storage.local.set({ [timerMode ? 'timerData' : 'stopwatchData']: { time, running, startTime } });
        }
        updateTimerDisplay();
    }

    // 타이머 화면 업데이트 함수
    function updateTimerDisplay(currentTime) {
        const timeToDisplay = currentTime !== undefined ? currentTime : time;
        const milliseconds = Math.floor(timeToDisplay % 1000 / 10);
        const seconds = Math.floor(timeToDisplay / 1000) % 60;
        const minutes = Math.floor(timeToDisplay / (1000 * 60)) % 60;
        const hours = Math.floor(timeToDisplay / (1000 * 60 * 60));
        document.getElementById('milliseconds').textContent = milliseconds.toString().padStart(2, '0');
        document.getElementById('seconds').textContent = seconds.toString().padStart(2, '0');
        document.getElementById('minutes').textContent = minutes.toString().padStart(2, '0');
        document.getElementById('hours').textContent = hours.toString().padStart(2, '0');
    }

    function updateStopwatchDisplay(currentTime) {
        const timeToDisplay = currentTime !== undefined ? currentTime : 0;
        const milliseconds = Math.floor(timeToDisplay % 1000 / 10);
        const seconds = Math.floor(timeToDisplay / 1000) % 60;
        const minutes = Math.floor(timeToDisplay / (1000 * 60)) % 60;
        const hours = Math.floor(timeToDisplay / (1000 * 60 * 60));
        document.getElementById('milliseconds').textContent = milliseconds.toString().padStart(2, '0');
        document.getElementById('seconds').textContent = seconds.toString().padStart(2, '0');
        document.getElementById('minutes').textContent = minutes.toString().padStart(2, '0');
        document.getElementById('hours').textContent = hours.toString().padStart(2, '0');
    }

    // 타이머/스톱워치 모드 전환 함수
    function switchMode(mode, button) {
        saveCurrentState();
        timerMode = (mode === 'timer');
        chrome.storage.local.set({ timerMode });
        chrome.runtime.sendMessage({ action: 'getTimerData' }, result => {
            const data = timerMode ? result.timerData : result.stopwatchData;
            ({ time, running, startTime } = data || { time: 0, running: false, startTime: Date.now() });
            if (running) {
                clearInterval(interval);
                if (timerMode) {
                    startTime = Date.now() + time; // 남은 시간에서 시작
                } else {
                    startTime = Date.now() - time; // 경과 시간에서 시작
                }
                interval = setInterval(updateTimer, 10);
            }
            if (timerMode) {
                updateTimerDisplay();
            } else {
                updateStopwatchDisplay();
            }
        });
        chrome.storage.local.get(timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch', res => {
            loadButtonState(res[timerMode ? 'buttonStateTimer' : 'buttonStateStopwatch']);
        });
        selectMenu(button);
    }

    // 타이머/스톱워치 시작/중지 함수
    function toggleRunning(action) {
        if ((action === 'start' && !running) || (action === 'pause' && running)) {
            running = !running;
            if (running) {
                startTime = timerMode ? Date.now() + time : Date.now() - time;
                interval = setInterval(updateTimer, 10);
                chrome.runtime.sendMessage({ action: timerMode ? 'startTimer' : 'startStopwatch' });
                setButtonState(startBtn);
            } else {
                clearInterval(interval);
                chrome.runtime.sendMessage({ action: timerMode ? 'stopTimer' : 'stopStopwatch' });
                setButtonState(pauseBtn);
            }
            chrome.runtime.sendMessage({ action: 'updateButtonState', state: running ? 'start' : 'pause' });
        }
    }

    // 타이머 리셋 함수
    function resetTimer() {
        clearInterval(interval);
        time = timerMode ? 0 : 0; // 타이머 모드에서 10분(600,000ms)으로 초기화
        running = false;
        startTime = Date.now();
        updateTimerDisplay();
        setButtonState(null);
        chrome.runtime.sendMessage({ action: timerMode ? 'resetTimer' : 'resetStopwatch' });
        chrome.storage.local.set({ [timerMode ? 'timerData' : 'stopwatchData']: { time, running, startTime } });
    }

    // 버튼 상태 저장 및 로드 함수
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
            chrome.storage.local.set(timerMode ? { buttonStateTimer: state } : { buttonStateStopwatch: state });
        }
    }

    function loadButtonState(state) {
        if (state) {
            setButtonState(document.getElementById(state), true); // isLoad를 true로 설정하여 상태 불러오기
        }
    }

    // 메뉴 선택 함수
    function selectMenu(button) {
        stopwatchBtn.classList.remove('selected');
        timerBtn.classList.remove('selected');
        button.classList.add('selected');
        chrome.storage.local.set({ selectedMenu: button.id });
    }

    // 현재 상태 저장 함수
    function saveCurrentState() {
        const state = startBtn.classList.contains('selected') ? 'start-btn' : (pauseBtn.classList.contains('selected') ? 'pause-btn' : null);
        chrome.storage.local.set(timerMode ? { buttonStateTimer: state } : { buttonStateStopwatch: state });
        chrome.storage.local.set({ [timerMode ? 'timerData' : 'stopwatchData']: { time, running, startTime } });
    }

    // 볼륨 조절 및 음소거 관련 함수들
    function toggleVolumePopup(event) {
        event.stopPropagation();
        volumePopup.style.display = volumePopup.style.display === 'block' ? 'none' : 'block';
    }

    function closeVolumePopupOnClickOutside(event) {
        if (!event.target.closest('.head-button') && !event.target.closest('#volume-popup')) {
            volumePopup.style.display = 'none';
        }
    }

    function adjustVolume(event) {
        updateSliderBackground(event.target);
        const volume = event.target.value / 100;
        alarmAudio.volume = volume;
        alarmAudio.muted = volume === 0;
        chrome.storage.local.set({ alarmVolume: volume, alarmMuted: alarmAudio.muted });
        muteButton.textContent = alarmAudio.muted ? 'Unmute' : 'Mute';
        updateAlarmIcon(alarmAudio.muted);
    }

    function toggleMute() {
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
    }

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

    // 드롭다운 메뉴 관련 함수들
    function toggleDropdown() {
        isDropdownOpen = !isDropdownOpen;
        dropdownContent.style.display = isDropdownOpen ? 'block' : 'none';
        updateDropdownStyles();
    }

    function changeDropdownStyle(isHover) {
        dropdown.style.borderColor = isHover || isDropdownOpen ? '#FA560C' : '#C0C0C0';
        dropdownButton.querySelector('path').style.stroke = isHover || isDropdownOpen ? '#FA560C' : '#C0C0C0';
    }

    function changeDropdownBorder(isHover) {
        if (!isDropdownOpen) {
            dropdown.style.borderColor = isHover ? '#FA560C' : '#C0C0C0';
        }
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

    // editable 요소 설정
    function makeEditable(element) {
        if (!timerMode || running) return;

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

    const setEditableListeners = element => {
        element.addEventListener('click', event => {
            event.stopPropagation();
            makeEditable(element);
        });
    };

    ['hours', 'minutes', 'seconds', 'milliseconds'].forEach(id => {
        setEditableListeners(document.getElementById(id));
    });

    // 메세지 박스 표시 함수
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


    // 초기 토글 상태 설정
    if (toggleBody) {
        chrome.storage.local.get(["switch", "firstToggle"], result => {
            if (result.switch) {
                toggleBody.classList.add("on");
            }

            if (result.switch && !result.firstToggle) {
                setTimeout(() => {
                    switchMode('stopwatch', stopwatchBtn);
                    setButtonState(startBtn);
                    toggleRunning('start');
                    chrome.runtime.sendMessage({ action: 'startStopwatch' });
                    chrome.tabs.create({ url: chrome.runtime.getURL('../cam/webcam.html') });
                }, 200);
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
                        }, 200);
                        switchMode('stopwatch', stopwatchBtn);
                        setButtonState(startBtn);
                        toggleRunning('start');
                        chrome.runtime.sendMessage({ action: 'startStopwatch' });
                        chrome.storage.local.set({ firstToggle: true });
                    }
                });
            } else {
                chrome.storage.local.set({ firstToggle: false });
                chrome.tabs.query({}, function(tabs) {
                    tabs.forEach(function(tab) {
                        if (tab.url && tab.url.includes('webcam.html')) {
                            switchMode('stopwatch', stopwatchBtn);
                            setButtonState(pauseBtn);  // pauseBtn 상태 저장 함수 사용
                            toggleRunning('pause'); // 이 부분을 추가하여 pause 상태로 전환
                            chrome.runtime.sendMessage({ action: 'stopStopwatch' });
                            chrome.tabs.remove(tab.id);
                        }
                    });
                });
            }

            // 백그라운드 스크립트로 상태 변경 메시지 전송
            chrome.runtime.sendMessage({ action: 'toggleSwitch', state: isOn });
        });
    }

    const saveButton = document.getElementById('study-end-button');
    if (saveButton) {
        saveButton.addEventListener('click', async event => {
            event.preventDefault();

            // lack_focus 변수 받아옴
            fetch('http://127.0.0.1:5000/get_status')
            .then(response => response.json())
            .then(data => {
                var lack_focus = data.lack_focus;
                console.log('Lack Focus:', lack_focus);
            })

            const user = (await chrome.storage.local.get(['user'])).user;
            const uid = user.uid;
            const totalMilliseconds = (await chrome.storage.local.get(['stopwatchData'])).stopwatchData.time;
            const totalSeconds = totalMilliseconds / 1000; // 밀리초를 초로 변환
            const usedTimeInMinutes = totalSeconds / 60; // 초를 분으로 변환
            const focus = usedTimeInMinutes-lack_focus; // 나중에 변경하기 일단은 total과 동일하게 설정

            const date = new Date().toISOString().split('T')[0]; // 오늘 날짜

            const userDocRef = doc(db, "users", uid);

            // 문서가 존재하는지 확인
            const userDocSnap = await getDoc(userDocRef);
            if (userDocSnap.exists()) {
                // 문서가 존재하면 기존 값을 가져와서 업데이트
                const data = userDocSnap.data();
                const timerData = data.timer || {};
                const existingData = timerData[date] || { total: 0, focus: 0 };

                const newTotal = existingData.total + usedTimeInMinutes;
                const newFocus = existingData.focus + focus;

                await updateDoc(userDocRef, {
                    [`timer.${date}`]: {
                        total: newTotal,
                        focus: newFocus
                    }
                });
            } else {
                // 문서가 존재하지 않으면 생성
                await setDoc(userDocRef, {
                    email: user.email,
                    timer: {
                        [date]: {
                            total: usedTimeInMinutes,
                            focus: focus
                        }
                    }
                });
            }

            // stopwatch 시간 리셋
            chrome.runtime.sendMessage({ action: 'resetStopwatch' }, () => {
                // 버튼 상태와 화면을 즉각 업데이트
                const startBtn = document.getElementById('start-btn');
                const pauseBtn = document.getElementById('pause-btn');
                startBtn.classList.remove('selected');
                pauseBtn.classList.add('selected');
                updateStopwatchDisplay(0);
            });

            alert('Data saved successfully!');
        });
    }

    // 시간 업데이트 함수
    function updateTime() {
        const hours = parseInt(document.getElementById('hours').textContent, 10);
        const minutes = parseInt(document.getElementById('minutes').textContent, 10);
        const seconds = parseInt(document.getElementById('seconds').textContent, 10);
        const milliseconds = parseInt(document.getElementById('milliseconds').textContent, 10) * 10;

        time = (hours * 3600000) + (minutes * 60000) + (seconds * 1000) + milliseconds;
        chrome.runtime.sendMessage({ action: 'updateTimer', time: time });
    }

    // 백그라운드 스크립트에서 버튼 상태 메시지 수신
    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === 'updateButtonState') {
            loadButtonState(message.state);
        }

        if (message.action === 'updateToggle') {
            const isOn = message.state;
            if (!isOn) {
                toggleBody.classList.remove("on");
                setButtonState(pauseBtn);  // pauseBtn 상태 저장 함수 사용
                toggleRunning('pause'); // 이 부분을 추가하여 pause 상태로 전환
            }
        }

        if (message.action === 'stopStopwatch') {
            clearInterval(interval);  // 스톱워치를 멈춤
            running = false;
            setButtonState(pauseBtn);  // pauseBtn 상태 저장 함수 사용
            chrome.storage.local.get('stopwatchData', (result) => {
                const { time, startTime } = result.stopwatchData || { time: 0, startTime: Date.now() };
                updateStopwatchDisplay(time);
            });
        }

        if (message.action === 'updateStopwatchTime') {
            if (!timerMode) {  // 스톱워치 모드에서만 업데이트
                updateStopwatchDisplay(message.time);
            }
        }

        if (message.action === 'updateTimerTime') {
            if (timerMode) {  // 타이머 모드에서만 업데이트
                updateTimerDisplay(message.time);
            }
        }
    });
});


