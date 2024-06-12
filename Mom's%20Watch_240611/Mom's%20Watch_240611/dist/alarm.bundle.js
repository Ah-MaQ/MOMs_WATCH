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
/*!******************!*\
  !*** ./alarm.js ***!
  \******************/
__webpack_require__.r(__webpack_exports__);
window.onload = function () {
  var audio = document.getElementById('alarm-audio');

  // 초기 음량과 음소거 상태를 storage에서 가져옴
  chrome.storage.local.get(['alarmVolume', 'alarmMuted'], function (result) {
    audio.volume = result.alarmVolume !== undefined ? result.alarmVolume : 0.5;
    audio.muted = result.alarmMuted || false;
    audio.play();
  });
};
/******/ })()
;
//# sourceMappingURL=alarm.bundle.js.map