document.getElementById('start').addEventListener('click', () => {
    chrome.runtime.sendMessage({ command: "startDetection" });
  });
  