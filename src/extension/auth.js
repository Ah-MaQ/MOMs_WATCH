import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut, onAuthStateChanged } from "firebase/auth";
import { push } from "firebase/database";
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
const db = getFirestore(app);  // Firestore 인스턴스 생성

document.addEventListener('DOMContentLoaded', () => {
  const loginButton = document.getElementById('login-button');
  const registerButton = document.getElementById('register-button');
  const logoutButton = document.getElementById('logout-button');
  const gotoRegisterButton = document.getElementById('goto-register-button');

  if (loginButton) {
    loginButton.addEventListener('click', event => {
      event.preventDefault();
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      loginUser(email, password);
    });
  }

  if (registerButton) {
    registerButton.addEventListener('click', event => {
      event.preventDefault();
      const email = document.getElementById('email').value;
      const username = document.getElementById('username').value;
      const password = document.getElementById('password').value;
      registerUser(email, username, password);
    });
  }

  if (logoutButton) {
    logoutButton.addEventListener('click', event => {
      event.preventDefault();
      logoutUser();
    });
  }

  if (gotoRegisterButton) {
    gotoRegisterButton.addEventListener('click', event => {
      event.preventDefault();
      window.location.href = '../../page/register/register.html';
    });
  }

  DrawChart();
  checkLoginStatus();


});

async function loginUser(email, password) {
  signInWithEmailAndPassword(auth, email, password)
    .then(async (userCredential) => {
      const user = userCredential.user;
      chrome.storage.local.set({ user: { email: email, uid: user.uid } }, () => {
        alert('Logged in successfully!');
        displayUserInfo(username);
        fetchUserData(user.uid);  // 사용자 데이터 불러오기
      });
      document.getElementById('login-form').style.display = 'none'
    })
    .catch((error) => {
      alert('Login failed: ' + error.message);
    });
}

function registerUser(email, username, password) {
  createUserWithEmailAndPassword(auth, email, password)
    .then(async (userCredential) => {
      const user = userCredential.user;
      await setDoc(doc(db, "users", user.uid), {
        email: email,
        username: username
      });
      alert('Registered successfully!');
      window.location.href = '../../page/login/login.html';
    })
    .catch((error) => {
      alert('Registration failed: ' + error.message);
    });
}

function logoutUser() {
  signOut(auth).then(() => {
    chrome.storage.local.remove('user', () => {
      alert('Logged out successfully!');
      window.location.href = '../../page/login/login.html';
    });
  }).catch((error) => {
    alert('Logout failed: ' + error.message);
  });
}

function displayUserInfo() {
  const userInfo = document.getElementById('user-info');
  if (userInfo) {
    userInfo.textContent = `당신의 총 집중도는 ${focusRatio}% 입니다.`;
    userInfo.style.display = 'block';
  }
}

function checkLoginStatus() {
  onAuthStateChanged(auth, (user) => {
    if (user) {
      const email = user.email;
      const uid = user.uid;
      chrome.storage.local.set({ user: { email: email, uid: uid } }, () => {
        fetchUserData(uid);  // 사용자 데이터 불러오기
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('logout-button').style.display = 'block';
        document.getElementById('myChart').style.display = 'block';
        // 로그인 후 login-title의 텍스트를 "Data"로 변경
        document.querySelector('.login-title').textContent = 'Data';
      });
    } else {
      chrome.storage.local.remove('user', () => {
        document.getElementById('user-info').style.display = 'none';
        document.getElementById('logout-button').style.display = 'none';
        document.getElementById('myChart').style.display = 'none';
        // 로그아웃 후 login-title의 텍스트를 "Login"으로 변경
        document.querySelector('.login-title').textContent = 'Login';
      });
    }
  });
}

async function fetchUserData(uid) {
    try {
    const docRef = doc(db, "users", uid);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      const userData = docSnap.data();
      console.log("User Data:", userData);
      // displayUserData(userData);
    } else {
      console.log("No such document!");
    }
  } catch (error) {
    console.error("error fetching user data:", error);
  }
}



document.addEventListener('DOMContentLoaded', () => {
  const saveButton = document.getElementById('save-button');

  if (saveButton) {
    saveButton.addEventListener('click', async event => {
      event.preventDefault();
      const user = (await chrome.storage.local.get(['user'])).user;
      const uid = user.uid;
      const total = parseInt(document.getElementById('total').value);
      const focus = parseInt(document.getElementById('focus').value);
      const date = new Date().toISOString().split('T')[0]; // 오늘 날짜

      const userDocRef = doc(db, "users", uid);

      // 문서가 존재하는지 확인
      const userDocSnap = await getDoc(userDocRef);
      if (userDocSnap.exists()) {
        // 문서가 존재하면 업데이트
        await updateDoc(userDocRef, {
          [`timer.${date}`]: {
            total: total,
            focus: focus
          }
        });
      } else {
        // 문서가 존재하지 않으면 생성
        await setDoc(userDocRef, {
          email: user.email,
          timer: {
            [date]: {
              total: total,
              focus: focus
            }
          }
        });
      }

      alert('Data saved successfully!');
    });
  }
})

//let totalSum;
//let focusSum;
//let focusRatio;
//// chart
//async function DrawChart() {
//  const canvas = document.getElementById('myChart');
//  const ctx = canvas.getContext('2d');
//
//  const user = (await chrome.storage.local.get(['user'])).user;
//  const uid = user.uid;
//  const userDocRef = doc(db, "users", uid);
//  const userDocSnap = await getDoc(userDocRef);
//
//  let labels = [];
//  let totalData = [];
//  let focusData = [];
//
//  if (userDocSnap.exists()) {
//    const data = userDocSnap.data();
//    const timerData = data.timer;
//
//    // 데이터를 객체 배열로 변환
//    let dataArray = [];
//    for (const [date, values] of Object.entries(timerData)) {
//        dataArray.push({
//            date: date,
//            label: date.substring(5, 7) + date.substring(8, 10),
//            total: values.total,
//            focus: values.focus
//        });
//    }
//
//    // 날짜 기준으로 정렬
//    dataArray.sort((a, b) => new Date(a.date) - new Date(b.date));
//
//    // 정렬된 데이터를 labels와 data 배열로 변환
//    labels = dataArray.map(item => item.label);
//    totalData = dataArray.map(item => item.total);
//    focusData = dataArray.map(item => item.focus);
//  }
//    // focusData와 totalData의 합 구하기
//    totalSum = totalData.reduce((acc, value) => acc + value, 0);
//    focusSum = focusData.reduce((acc, value) => acc + value, 0);
//    focusRatio = Math.floor((focusSum / totalSum) * 100);
//
//    console.log('Total Sum:', totalSum);
//    console.log('Focus Sum:', focusSum);
//    console.log('Focus Ratio:', focusRatio);
//
//  // 그래프 설정
//  const barWidth = 25;
//  const barSpacing = 15;
//  const chartHeight = canvas.height;
//  const chartWidth = canvas.width;
//  const maxValue = Math.max(...totalData);
//  const scale = (chartHeight-30) / maxValue;
//
//  // 범례 그리기 (오른쪽 상단)
//  const legendX = chartWidth - 180; // 오른쪽 여백
//  const legendY = 5; // 상단 여백
//
//  ctx.fillStyle = '#FF9D76';
//  ctx.fillRect(legendX, legendY, 10, 10);
//  ctx.fillStyle = '#000';
//  ctx.font = '10px Arial';
//  ctx.fillText('Total', legendX + 15, legendY + 10);
//
//  ctx.fillStyle = '#76A9FF';
//  ctx.fillRect(legendX + 70, legendY, 10, 10);
//  ctx.fillStyle = '#000';
//  ctx.fillText('Focus', legendX + 85, legendY + 10);
//
//  // 그래프 그리기
//  const bars = [];
//  for (let i = 0; i < totalData.length; i++) {
//      const totalBarHeight = totalData[i] * scale;
//      const focusBarHeight = focusData[i] * scale;
//      const x = i * (barWidth + barSpacing) + barSpacing;
//      const y = chartHeight - totalBarHeight;
//      const focusY = y + totalBarHeight - focusBarHeight;
//
//      // Total 바 그리기
//      ctx.fillStyle = '#FF9D76';
//      ctx.fillRect(x, y, barWidth, totalBarHeight);
//
//      // Focus 바 그리기
//      ctx.fillStyle = '#76A9FF';
//      ctx.fillRect(x, focusY, barWidth, focusBarHeight);
//
//      bars.push({ x, y, width: barWidth, height: totalBarHeight, totalValue: totalData[i], focusValue: focusData[i] });
//  }
//
//  // 레이블 그리기
//  ctx.fillStyle = '#000';
//  ctx.font = '12px Arial';
//  for (let i = 0; i < labels.length; i++) {
//      const x = i * (barWidth + barSpacing) + barSpacing + barWidth / 2;
//      const y = chartHeight - 5;
//      ctx.fillText(labels[i], x - ctx.measureText(labels[i]).width / 2, y);
//  }
//
//  // 마우스 오버 이벤트 처리
//  canvas.addEventListener('mousemove', function(event) {
//    const rect = canvas.getBoundingClientRect();
//    const mouseX = event.clientX - rect.left;
//    const mouseY = event.clientY - rect.top;
//
//    ctx.clearRect(0, 0, canvas.width, canvas.height);
//
//    // 범례 다시 그리기 (오른쪽 상단)
//    ctx.fillStyle = '#FF9D76';
//    ctx.fillRect(legendX, legendY, 10, 10);
//    ctx.fillStyle = '#000';
//    ctx.font = '10px Arial';
//    ctx.fillText('Total', legendX + 15, legendY + 10);
//
//    ctx.fillStyle = '#76A9FF';
//    ctx.fillRect(legendX + 70, legendY, 10, 10);
//    ctx.fillStyle = '#000';
//    ctx.fillText('Focus', legendX + 85, legendY + 10);
//
//    // 그래프 다시 그리기
//    for (let i = 0; i < totalData.length; i++) {
//        const totalBarHeight = totalData[i] * scale;
//        const focusBarHeight = focusData[i] * scale;
//        const x = i * (barWidth + barSpacing) + barSpacing;
//        const y = chartHeight - totalBarHeight;
//        const focusY = y + totalBarHeight - focusBarHeight;
//
//        // Total 바 다시 그리기
//        ctx.fillStyle = '#FF9D76';
//        ctx.fillRect(x, y, barWidth, totalBarHeight);
//
//        // Focus 바 다시 그리기
//        ctx.fillStyle = '#76A9FF';
//        ctx.fillRect(x, focusY, barWidth, focusBarHeight);
//    }
//
//    // 레이블 다시 그리기
//    ctx.fillStyle = '#000';
//    ctx.font = '12px Arial';
//    for (let i = 0; i < labels.length; i++) {
//        const x = i * (barWidth + barSpacing) + barSpacing + barWidth / 2;
//        const y = chartHeight - 5;
//        ctx.fillText(labels[i], x - ctx.measureText(labels[i]).width / 2, y);
//    }
//
//    // 마우스 오버된 바 찾기
//    for (const bar of bars) {
//      if (
//        mouseX >= bar.x &&
//        mouseX <= bar.x + bar.width &&
//        mouseY >= bar.y &&
//        mouseY <= bar.y + bar.height
//      ) {
//        // 데이터 표시
//        ctx.fillStyle = '#000';
//        ctx.fillRect(mouseX, mouseY - 40, 80, 40); // 배경 박스
//        ctx.fillStyle = '#fff';
//        ctx.fillText(`Total: ${bar.totalValue.toFixed(2)}`, mouseX + 5, mouseY - 25);
//        ctx.fillText(`Focus: ${bar.focusValue.toFixed(2)}`, mouseX + 5, mouseY - 10);
//        break;
//      }
//    }
//  });
//  displayUserInfo(user.name);
//}