document.getElementById('setting_button').addEventListener('click', function() {
  chrome.runtime.openOptionsPage();
});

document.getElementById('account_button').addEventListener('click', function() {
  alert('계정 버튼이 클릭되었습니다!');
  // 계정 버튼 클릭 시 추가 동작을 넣으세요
});


const toggle = document.getElementById('toggle1');

// 페이지 로드 시 로컬 스토리지에서 토글 상태 불러오기
window.addEventListener('load', () => {
const isChecked = localStorage.getItem('toggle1') === 'true';
toggle.checked = isChecked;
toggle.classList.add('no-transition'); // 애니메이션 비활성화
if (isChecked) {
  console.log('Toggle 1 was ON');
}
// 로드 후 잠시 뒤 애니메이션 활성화
setTimeout(() => {
  toggle.classList.remove('no-transition');
}, 0);
});

// 토글 상태 변경 시 로컬 스토리지에 저장하고 동작 수행
toggle.addEventListener('change', (event) => {
if (event.target.checked) {
  console.log('Toggle 1 ON');
  localStorage.setItem('toggle1', 'true');
  setTimeout(() => {
    chrome.tabs.create({ url: chrome.runtime.getURL('../cam/webcam.html') });
  }, 200); // 100ms 지연
} else {
  console.log('Toggle 1 OFF');
  localStorage.setItem('toggle1', 'false');
}
});

document.getElementById('startButton').addEventListener('click', function() {
  const minutesInput = document.getElementById('minutesInput').value;
  chrome.runtime.sendMessage({ command: "start", duration: minutesInput });

  // 타이머를 시작한 후 주기적으로 남은 시간을 업데이트
  startUpdatingTime();
});

function pad(number) {
  return number < 10 ? '0' + number : number;
}

function updateTime(remainingTime) {
  const minutes = Math.floor(remainingTime / 60);
  const seconds = remainingTime % 60;
  document.getElementById('time').innerText = `${pad(minutes)}:${pad(seconds)}`;
}

function startUpdatingTime() {
  function requestRemainingTime() {
    chrome.runtime.sendMessage({ command: "getRemainingTime" }, (response) => {
      if (response.remainingTime === 0) {
        clearInterval(updateInterval);
      }
      updateTime(response.remainingTime);
    });
  }

  requestRemainingTime(); // 즉시 호출하여 타이머 상태를 업데이트
  const updateInterval = setInterval(requestRemainingTime, 1000); // 1초마다 업데이트
}

// 팝업이 열릴 때 남은 시간을 즉시 업데이트
startUpdatingTime();

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message,command === "timerEnded") {
    chrome.notifications.create('studyEndNotification', {
      type: 'basic',
      title: '공부 종료',
      message: '공부가 종료되었습니다!'
    }, function(notificationId) {});
  }
});
