// firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

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

export { auth, db };
