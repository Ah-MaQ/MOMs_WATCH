chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.command === "startDetection") {
        const audio = new Audio(message.soundUrl);
        audio.volume = message.volume;
        audio.play().then(() => {
          sendResponse({status: 'playing'});
        }).catch((error) => {
          console.error('Error playing sound:', error);
          sendResponse({status: 'error', message: error.message});
        });
        return true;  // Indicates that the response is sent asynchronously.
      }
    // chrome.tabs.create({ url: chrome.runtime.getURL('options/options.html') });
    }
);
  