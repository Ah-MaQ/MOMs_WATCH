import { initializeApp } from "firebase/app";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { getFirestore, doc, getDoc } from "firebase/firestore";

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

document.addEventListener('DOMContentLoaded', () => {
  onAuthStateChanged(auth, async (user) => {
    if (user) {
      const uid = user.uid;
      const userDocRef = doc(db, "users", uid);
      const userDocSnap = await getDoc(userDocRef);

      if (userDocSnap.exists()) {
        const userData = userDocSnap.data();
        drawChart(userData.timer);
      } else {
        console.log("No such document!");
      }
    }
  });
});

/* focus.js */
/* focus.js */
/* focus.js */
/* focus.js */
function drawChart(timerData) {
  const chartContainer = document.getElementById('chart');
  const yAxisContainer = document.getElementById('y-axis');

  const today = new Date();
  const last7Days = new Date();
  last7Days.setDate(today.getDate() - 6);

  const dateArray = [];
  for (let d = new Date(last7Days); d <= today; d.setDate(d.getDate() + 1)) {
    const dateStr = d.toISOString().slice(0, 10);
    dateArray.push(dateStr);
  }

  const dataArray = dateArray.map(date => {
    const values = timerData[date] || { total: 0, focus: 0 };
    return {
      date,
      label: date.substring(5, 7) + '/' + date.substring(8, 10),
      total: values.total,
      focus: values.focus,
      isEmpty: values.total === 0 && values.focus === 0
    };
  });

  const totalData = dataArray.map(item => item.total);
  const maxTotal = Math.max(...totalData);

  dataArray.forEach(item => {
    const rowContainer = document.createElement('div');
    rowContainer.className = 'row-container';

    const yAxisLabel = document.createElement('span');
    yAxisLabel.className = 'label';
    yAxisLabel.textContent = item.label;
    rowContainer.appendChild(yAxisLabel);

    const barContainer = document.createElement('div');
    barContainer.className = 'bar-container';

    if (item.isEmpty) {
      const emptyLabel = document.createElement('div');
      emptyLabel.className = 'empty';
      emptyLabel.textContent = '-';
      barContainer.appendChild(emptyLabel);
    } else {
      // Calculate the height percentage of the bar based on maxTotal
      const totalBarHeightPercent = (item.total / maxTotal) * 85; // 85% of bar-container
      const focusBarHeightPercent = (item.focus / maxTotal) * 85; // 85% of bar-container

      const totalBar = document.createElement('div');
      totalBar.className = 'bar bar-total';
      totalBar.style.height = `${totalBarHeightPercent}%`;

      const focusBar = document.createElement('div');
      focusBar.className = 'bar bar-focus';
      focusBar.style.height = `${focusBarHeightPercent}%`;

      const totalBarText = document.createElement('span');
      totalBarText.className = 'bar-text bar-text-right';
      totalBarText.textContent = formatTime(item.total);

      const focusBarText = document.createElement('span');
      focusBarText.className = 'bar-text bar-text-left';
      focusBarText.textContent = formatTime(item.focus);

      totalBar.appendChild(totalBarText);
      focusBar.appendChild(focusBarText);

      barContainer.appendChild(totalBar);
      barContainer.appendChild(focusBar);
    }

    rowContainer.appendChild(barContainer);
    chartContainer.appendChild(rowContainer);
  });
}

function formatTime(minutes) {
  const h = Math.floor(minutes / 60);
  const m = Math.floor(minutes % 60);
  return `${h}h\n${m}m`;
}

