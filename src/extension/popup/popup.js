document.getElementById('loadWebcam').addEventListener('click', () => {
  chrome.tabs.create({ url: '../cam/webcam.html' });
});
