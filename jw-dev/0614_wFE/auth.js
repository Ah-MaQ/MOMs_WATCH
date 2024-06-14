import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from "firebase/auth";
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
  const dataButton = document.getElementById('data-button'); // Data 버튼 추가

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

  if (dataButton) {
    dataButton.addEventListener('click', event => {
      event.preventDefault();
      window.location.href = '../../page/database/database.html'; // Data 페이지로 이동
    });
  }

  checkLoginStatus();
});

function loginUser(email, password) {
  signInWithEmailAndPassword(auth, email, password)
    .then(async (userCredential) => {
      const user = userCredential.user;
      chrome.storage.local.set({ user: { email: email, uid: user.uid } }, () => {
        alert('Logged in successfully!');
        displayUserInfo(email);
        fetchUserData(user.uid);  // 사용자 데이터 불러오기
      });
      document.getElementById('data-button').style.display = 'block'; // 로그인 후 Data 버튼 표시
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

function displayUserInfo(email) {
  const userInfo = document.getElementById('user-info');
  if (userInfo) {
    userInfo.textContent = `Logged in as: ${email}`;
    userInfo.style.display = 'block';
  }
}

function checkLoginStatus() {
  chrome.storage.local.get(['user'], (result) => {
    if (result.user && result.user.email) {
      displayUserInfo(result.user.email);
      const logoutButton = document.getElementById('logout-button');
      if (logoutButton) {
        logoutButton.style.display = 'block';
      }
    }
  });
}

async function fetchUserData(uid) {
  const docRef = doc(db, "users", uid);
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    const userData = docSnap.data();
    console.log("User Data:", userData);
    // 여기서 필요한 작업 수행
  } else {
    console.log("No such document!");
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
});
