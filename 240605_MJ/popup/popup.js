//document.getElementById('button1').addEventListener('click', () => {
//  alert('Button 1 clicked');
//});
//
//document.getElementById('button2').addEventListener('click', () => {
//  console.log('Button 2 clicked');
//});

//document.getElementById('optionsButton').addEventListener('click', () => {
//  chrome.runtime.openOptionsPage();
//});


document.getElementById('setting_button').addEventListener('click', function() {
    chrome.runtime.openOptionsPage();
});

document.getElementById('account_button').addEventListener('click', function() {
    // alert('계정 버튼이 클릭되었습니다!');
    var audio = new Audio('sample.mp3')
    audio.play();
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
  // setTimeout(() => {
  //   toggle.classList.remove('no-transition');
  // }, 0);
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

const toggle2 = document.getElementById('toggle2');

window.addEventListener('load', () => {
  const isChecked = localStorage.getItem('toggle2') === 'true';
  toggle2.checked = isChecked;
  toggle2.classList.add('no-transition'); // 애니메이션 비활성화
  if (isChecked) {
    console.log('Toggle 2 was ON');
  }
  // 로드 후 잠시 뒤 애니메이션 활성화
  setTimeout(() => {
    toggle2.classList.remove('no-transition');
  }, 0);
});

toggle2.addEventListener('change', (event) => {
  if (event.target.checked) {
    console.log('Toggle 2 ON');
    localStorage.setItem('toggle2', 'true');
    chrome.runtime.sendMessage({ command: "startDetection"},(response) => {console.log(response.status);
    });
  } else {
      console.log('Toggle 2 OFF');
      localStorage.setItem('toggle2', 'false');
  }
});