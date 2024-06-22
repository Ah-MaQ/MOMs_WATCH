import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAHVkz3pZReHoeMSB--5NHDfF2Hhv0MsgE",
  authDomain: "moms--watch.firebaseapp.com",
  projectId: "moms--watch",
  storageBucket: "moms--watch.appspot.com",
  messagingSenderId: "337992479265",
  appId: "1:337992479265:web:fe5b4f4c47cfcd72d56960",
  measurementId: "G-RWPS6500Z5"
};
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

document.addEventListener('DOMContentLoaded', () => {
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');
  const logoutButton = document.getElementById('logout-button');

  if (loginForm) {
    loginForm.addEventListener('submit', event => {
      event.preventDefault();
      const email = loginForm.email.value;
      const password = loginForm.password.value;
      loginUser(email, password);
    });
  }

  if (registerForm) {
    registerForm.addEventListener('submit', event => {
      event.preventDefault();
      const email = registerForm.email.value;
      const password = registerForm.password.value;
      registerUser(email, password);
    });
  }

  if (logoutButton) {
    logoutButton.addEventListener('click', event => {
      event.preventDefault();
      logoutUser();
    });
  }

  checkLoginStatus();
});

function loginUser(email, password) {
  signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      chrome.storage.local.set({ user: { email, uid: user.uid } }, () => {
        alert('Logged in successfully!');
        displayUserInfo(user.email);
      });
    })
    .catch((error) => {
      alert('Login failed: ' + error.message);
    });
}

function registerUser(email, password) {
  createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      alert('Registered successfully!');
      window.location.href = 'login.html';
    })
    .catch((error) => {
      alert('Registration failed: ' + error.message);
    });
}

function logoutUser() {
  signOut(auth).then(() => {
    chrome.storage.local.remove('user', () => {
      alert('Logged out successfully!');
      window.location.href = 'login.html';
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