import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js";


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

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

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
});

// auth.js 파일
function loginUser(email, password) {
    signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
        const user = userCredential.user;
        chrome.storage.local.set({ user: { email, uid: user.uid } }, () => {
            alert('Logged in successfully!');
            window.location.href = 'alarm.html';
        });
    })
    .catch((error) => {
        var errorCode = error.code;
        var errorMessage = error.message;
        alert('Login failed: ' + errorMessage);
    });
}

function registerUser(email, password) {
    createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
        const user = userCredential.user;
        alert('Registered successfully!');
        window.location.href = 'login.html';
    })
    .catch((error) => {
        var errorCode = error.code;
        var errorMessage = error.message;
        alert('Registration failed: ' + errorMessage);
    });
}