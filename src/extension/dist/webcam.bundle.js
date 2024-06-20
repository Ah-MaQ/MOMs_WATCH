/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	// The require scope
/******/ 	var __webpack_require__ = {};
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
/*!***********************!*\
  !*** ./cam/webcam.js ***!
  \***********************/
__webpack_require__.r(__webpack_exports__);
document.addEventListener("DOMContentLoaded", function () {
  var video = document.getElementById('webcam');
  var pipButton = document.getElementById('pipButton');
  var fullscreenButton = document.getElementById('fullscreenButton');
  var popupButton = document.getElementById('popupButton');
  pipButton.addEventListener('click', function () {
    if (document.pictureInPictureElement) {
      document.exitPictureInPicture();
    } else if (video.requestPictureInPicture) {
      video.requestPictureInPicture();
    } else {
      alert('Picture-in-Picture not supported');
    }
  });
  fullscreenButton.addEventListener('click', function () {
    if (!document.fullscreenElement) {
      if (video.requestFullscreen) {
        video.requestFullscreen();
      } else if (video.mozRequestFullScreen) {
        /* Firefox */
        video.mozRequestFullScreen();
      } else if (video.webkitRequestFullscreen) {
        /* Chrome, Safari and Opera */
        video.webkitRequestFullscreen();
      } else if (video.msRequestFullscreen) {
        /* IE/Edge */
        video.msRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.mozCancelFullScreen) {
        /* Firefox */
        document.mozCancelFullScreen();
      } else if (document.webkitExitFullscreen) {
        /* Chrome, Safari and Opera */
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) {
        /* IE/Edge */
        document.msExitFullscreen();
      }
    }
  });
  popupButton.addEventListener('click', function () {
    window.open('popup.html', 'Popup', 'width=640,height=480');
  });
});
/******/ })()
;
//# sourceMappingURL=webcam.bundle.js.map