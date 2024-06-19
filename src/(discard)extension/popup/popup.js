document.getElementById('startButton').addEventListener('click', () => {
  chrome.tabs.create({ url: '../cam/webcam.html' });
});

document.getElementById('visualizeButton').addEventListener('click', () => {
  chrome.tabs.create({ url: '../cam/webcam.html?visualize=true' });
});
