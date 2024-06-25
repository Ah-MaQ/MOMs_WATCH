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
function _typeof(o) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (o) { return typeof o; } : function (o) { return o && "function" == typeof Symbol && o.constructor === Symbol && o !== Symbol.prototype ? "symbol" : typeof o; }, _typeof(o); }
function _regeneratorRuntime() { "use strict"; /*! regenerator-runtime -- Copyright (c) 2014-present, Facebook, Inc. -- license (MIT): https://github.com/facebook/regenerator/blob/main/LICENSE */ _regeneratorRuntime = function _regeneratorRuntime() { return e; }; var t, e = {}, r = Object.prototype, n = r.hasOwnProperty, o = Object.defineProperty || function (t, e, r) { t[e] = r.value; }, i = "function" == typeof Symbol ? Symbol : {}, a = i.iterator || "@@iterator", c = i.asyncIterator || "@@asyncIterator", u = i.toStringTag || "@@toStringTag"; function define(t, e, r) { return Object.defineProperty(t, e, { value: r, enumerable: !0, configurable: !0, writable: !0 }), t[e]; } try { define({}, ""); } catch (t) { define = function define(t, e, r) { return t[e] = r; }; } function wrap(t, e, r, n) { var i = e && e.prototype instanceof Generator ? e : Generator, a = Object.create(i.prototype), c = new Context(n || []); return o(a, "_invoke", { value: makeInvokeMethod(t, r, c) }), a; } function tryCatch(t, e, r) { try { return { type: "normal", arg: t.call(e, r) }; } catch (t) { return { type: "throw", arg: t }; } } e.wrap = wrap; var h = "suspendedStart", l = "suspendedYield", f = "executing", s = "completed", y = {}; function Generator() {} function GeneratorFunction() {} function GeneratorFunctionPrototype() {} var p = {}; define(p, a, function () { return this; }); var d = Object.getPrototypeOf, v = d && d(d(values([]))); v && v !== r && n.call(v, a) && (p = v); var g = GeneratorFunctionPrototype.prototype = Generator.prototype = Object.create(p); function defineIteratorMethods(t) { ["next", "throw", "return"].forEach(function (e) { define(t, e, function (t) { return this._invoke(e, t); }); }); } function AsyncIterator(t, e) { function invoke(r, o, i, a) { var c = tryCatch(t[r], t, o); if ("throw" !== c.type) { var u = c.arg, h = u.value; return h && "object" == _typeof(h) && n.call(h, "__await") ? e.resolve(h.__await).then(function (t) { invoke("next", t, i, a); }, function (t) { invoke("throw", t, i, a); }) : e.resolve(h).then(function (t) { u.value = t, i(u); }, function (t) { return invoke("throw", t, i, a); }); } a(c.arg); } var r; o(this, "_invoke", { value: function value(t, n) { function callInvokeWithMethodAndArg() { return new e(function (e, r) { invoke(t, n, e, r); }); } return r = r ? r.then(callInvokeWithMethodAndArg, callInvokeWithMethodAndArg) : callInvokeWithMethodAndArg(); } }); } function makeInvokeMethod(e, r, n) { var o = h; return function (i, a) { if (o === f) throw Error("Generator is already running"); if (o === s) { if ("throw" === i) throw a; return { value: t, done: !0 }; } for (n.method = i, n.arg = a;;) { var c = n.delegate; if (c) { var u = maybeInvokeDelegate(c, n); if (u) { if (u === y) continue; return u; } } if ("next" === n.method) n.sent = n._sent = n.arg;else if ("throw" === n.method) { if (o === h) throw o = s, n.arg; n.dispatchException(n.arg); } else "return" === n.method && n.abrupt("return", n.arg); o = f; var p = tryCatch(e, r, n); if ("normal" === p.type) { if (o = n.done ? s : l, p.arg === y) continue; return { value: p.arg, done: n.done }; } "throw" === p.type && (o = s, n.method = "throw", n.arg = p.arg); } }; } function maybeInvokeDelegate(e, r) { var n = r.method, o = e.iterator[n]; if (o === t) return r.delegate = null, "throw" === n && e.iterator["return"] && (r.method = "return", r.arg = t, maybeInvokeDelegate(e, r), "throw" === r.method) || "return" !== n && (r.method = "throw", r.arg = new TypeError("The iterator does not provide a '" + n + "' method")), y; var i = tryCatch(o, e.iterator, r.arg); if ("throw" === i.type) return r.method = "throw", r.arg = i.arg, r.delegate = null, y; var a = i.arg; return a ? a.done ? (r[e.resultName] = a.value, r.next = e.nextLoc, "return" !== r.method && (r.method = "next", r.arg = t), r.delegate = null, y) : a : (r.method = "throw", r.arg = new TypeError("iterator result is not an object"), r.delegate = null, y); } function pushTryEntry(t) { var e = { tryLoc: t[0] }; 1 in t && (e.catchLoc = t[1]), 2 in t && (e.finallyLoc = t[2], e.afterLoc = t[3]), this.tryEntries.push(e); } function resetTryEntry(t) { var e = t.completion || {}; e.type = "normal", delete e.arg, t.completion = e; } function Context(t) { this.tryEntries = [{ tryLoc: "root" }], t.forEach(pushTryEntry, this), this.reset(!0); } function values(e) { if (e || "" === e) { var r = e[a]; if (r) return r.call(e); if ("function" == typeof e.next) return e; if (!isNaN(e.length)) { var o = -1, i = function next() { for (; ++o < e.length;) if (n.call(e, o)) return next.value = e[o], next.done = !1, next; return next.value = t, next.done = !0, next; }; return i.next = i; } } throw new TypeError(_typeof(e) + " is not iterable"); } return GeneratorFunction.prototype = GeneratorFunctionPrototype, o(g, "constructor", { value: GeneratorFunctionPrototype, configurable: !0 }), o(GeneratorFunctionPrototype, "constructor", { value: GeneratorFunction, configurable: !0 }), GeneratorFunction.displayName = define(GeneratorFunctionPrototype, u, "GeneratorFunction"), e.isGeneratorFunction = function (t) { var e = "function" == typeof t && t.constructor; return !!e && (e === GeneratorFunction || "GeneratorFunction" === (e.displayName || e.name)); }, e.mark = function (t) { return Object.setPrototypeOf ? Object.setPrototypeOf(t, GeneratorFunctionPrototype) : (t.__proto__ = GeneratorFunctionPrototype, define(t, u, "GeneratorFunction")), t.prototype = Object.create(g), t; }, e.awrap = function (t) { return { __await: t }; }, defineIteratorMethods(AsyncIterator.prototype), define(AsyncIterator.prototype, c, function () { return this; }), e.AsyncIterator = AsyncIterator, e.async = function (t, r, n, o, i) { void 0 === i && (i = Promise); var a = new AsyncIterator(wrap(t, r, n, o), i); return e.isGeneratorFunction(r) ? a : a.next().then(function (t) { return t.done ? t.value : a.next(); }); }, defineIteratorMethods(g), define(g, u, "Generator"), define(g, a, function () { return this; }), define(g, "toString", function () { return "[object Generator]"; }), e.keys = function (t) { var e = Object(t), r = []; for (var n in e) r.push(n); return r.reverse(), function next() { for (; r.length;) { var t = r.pop(); if (t in e) return next.value = t, next.done = !1, next; } return next.done = !0, next; }; }, e.values = values, Context.prototype = { constructor: Context, reset: function reset(e) { if (this.prev = 0, this.next = 0, this.sent = this._sent = t, this.done = !1, this.delegate = null, this.method = "next", this.arg = t, this.tryEntries.forEach(resetTryEntry), !e) for (var r in this) "t" === r.charAt(0) && n.call(this, r) && !isNaN(+r.slice(1)) && (this[r] = t); }, stop: function stop() { this.done = !0; var t = this.tryEntries[0].completion; if ("throw" === t.type) throw t.arg; return this.rval; }, dispatchException: function dispatchException(e) { if (this.done) throw e; var r = this; function handle(n, o) { return a.type = "throw", a.arg = e, r.next = n, o && (r.method = "next", r.arg = t), !!o; } for (var o = this.tryEntries.length - 1; o >= 0; --o) { var i = this.tryEntries[o], a = i.completion; if ("root" === i.tryLoc) return handle("end"); if (i.tryLoc <= this.prev) { var c = n.call(i, "catchLoc"), u = n.call(i, "finallyLoc"); if (c && u) { if (this.prev < i.catchLoc) return handle(i.catchLoc, !0); if (this.prev < i.finallyLoc) return handle(i.finallyLoc); } else if (c) { if (this.prev < i.catchLoc) return handle(i.catchLoc, !0); } else { if (!u) throw Error("try statement without catch or finally"); if (this.prev < i.finallyLoc) return handle(i.finallyLoc); } } } }, abrupt: function abrupt(t, e) { for (var r = this.tryEntries.length - 1; r >= 0; --r) { var o = this.tryEntries[r]; if (o.tryLoc <= this.prev && n.call(o, "finallyLoc") && this.prev < o.finallyLoc) { var i = o; break; } } i && ("break" === t || "continue" === t) && i.tryLoc <= e && e <= i.finallyLoc && (i = null); var a = i ? i.completion : {}; return a.type = t, a.arg = e, i ? (this.method = "next", this.next = i.finallyLoc, y) : this.complete(a); }, complete: function complete(t, e) { if ("throw" === t.type) throw t.arg; return "break" === t.type || "continue" === t.type ? this.next = t.arg : "return" === t.type ? (this.rval = this.arg = t.arg, this.method = "return", this.next = "end") : "normal" === t.type && e && (this.next = e), y; }, finish: function finish(t) { for (var e = this.tryEntries.length - 1; e >= 0; --e) { var r = this.tryEntries[e]; if (r.finallyLoc === t) return this.complete(r.completion, r.afterLoc), resetTryEntry(r), y; } }, "catch": function _catch(t) { for (var e = this.tryEntries.length - 1; e >= 0; --e) { var r = this.tryEntries[e]; if (r.tryLoc === t) { var n = r.completion; if ("throw" === n.type) { var o = n.arg; resetTryEntry(r); } return o; } } throw Error("illegal catch attempt"); }, delegateYield: function delegateYield(e, r, n) { return this.delegate = { iterator: values(e), resultName: r, nextLoc: n }, "next" === this.method && (this.arg = t), y; } }, e; }
function asyncGeneratorStep(n, t, e, r, o, a, c) { try { var i = n[a](c), u = i.value; } catch (n) { return void e(n); } i.done ? t(u) : Promise.resolve(u).then(r, o); }
function _asyncToGenerator(n) { return function () { var t = this, e = arguments; return new Promise(function (r, o) { var a = n.apply(t, e); function _next(n) { asyncGeneratorStep(a, r, o, _next, _throw, "next", n); } function _throw(n) { asyncGeneratorStep(a, r, o, _next, _throw, "throw", n); } _next(void 0); }); }; }
document.addEventListener("DOMContentLoaded", function () {
  var processedStream = document.getElementById('processedStream');
  var pipButton = document.getElementById('pipButton');
  var fullscreenButton = document.getElementById('fullscreenButton');
  var popupButton = document.getElementById('popupButton');
  var alarmCount = 0;
  var noDetectionCount = 0;
  function startWebcam() {
    return _startWebcam.apply(this, arguments);
  }
  function _startWebcam() {
    _startWebcam = _asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee5() {
      var sendFrame, stream, video, canvas, context;
      return _regeneratorRuntime().wrap(function _callee5$(_context5) {
        while (1) switch (_context5.prev = _context5.next) {
          case 0:
            _context5.prev = 0;
            sendFrame = /*#__PURE__*/function () {
              var _ref3 = _asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee4() {
                return _regeneratorRuntime().wrap(function _callee4$(_context4) {
                  while (1) switch (_context4.prev = _context4.next) {
                    case 0:
                      if (video.readyState === video.HAVE_ENOUGH_DATA) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                        canvas.toBlob( /*#__PURE__*/function () {
                          var _ref4 = _asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee3(blob) {
                            var formData, response;
                            return _regeneratorRuntime().wrap(function _callee3$(_context3) {
                              while (1) switch (_context3.prev = _context3.next) {
                                case 0:
                                  formData = new FormData();
                                  formData.append('frame', blob, 'frame.jpg');
                                  _context3.prev = 2;
                                  _context3.next = 5;
                                  return fetch('http://127.0.0.1:5000/upload_frame', {
                                    method: 'POST',
                                    body: formData
                                  });
                                case 5:
                                  response = _context3.sent;
                                  if (response.ok) {
                                    _context3.next = 8;
                                    break;
                                  }
                                  throw new Error('Network response was not ok');
                                case 8:
                                  _context3.next = 13;
                                  break;
                                case 10:
                                  _context3.prev = 10;
                                  _context3.t0 = _context3["catch"](2);
                                  console.error('There was a problem with the fetch operation:', _context3.t0);
                                case 13:
                                case "end":
                                  return _context3.stop();
                              }
                            }, _callee3, null, [[2, 10]]);
                          }));
                          return function (_x) {
                            return _ref4.apply(this, arguments);
                          };
                        }(), 'image/jpeg');
                      }
                    case 1:
                    case "end":
                      return _context4.stop();
                  }
                }, _callee4);
              }));
              return function sendFrame() {
                return _ref3.apply(this, arguments);
              };
            }();
            _context5.next = 4;
            return navigator.mediaDevices.getUserMedia({
              video: {
                width: 640,
                height: 480
              }
            });
          case 4:
            stream = _context5.sent;
            // 해상도 축소
            video = document.createElement('video');
            video.srcObject = stream;
            video.play();
            canvas = document.createElement('canvas');
            context = canvas.getContext('2d');
            setInterval(sendFrame, 1000 / 10); // 전송 주기를 10 fps로 줄임
            _context5.next = 16;
            break;
          case 13:
            _context5.prev = 13;
            _context5.t0 = _context5["catch"](0);
            console.error('Error accessing the webcam:', _context5.t0);
          case 16:
          case "end":
            return _context5.stop();
        }
      }, _callee5, null, [[0, 13]]);
    }));
    return _startWebcam.apply(this, arguments);
  }
  function startProcessedStream() {
    return _startProcessedStream.apply(this, arguments);
  }
  function _startProcessedStream() {
    _startProcessedStream = _asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee6() {
      return _regeneratorRuntime().wrap(function _callee6$(_context6) {
        while (1) switch (_context6.prev = _context6.next) {
          case 0:
            try {
              processedStream.src = 'http://127.0.0.1:5000/stream';
            } catch (error) {
              console.error('Error starting processed stream:', error);
            }
          case 1:
          case "end":
            return _context6.stop();
        }
      }, _callee6);
    }));
    return _startProcessedStream.apply(this, arguments);
  }
  function updateStatus() {
    return _updateStatus.apply(this, arguments);
  }
  function _updateStatus() {
    _updateStatus = _asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee7() {
      var response, data, statusElement;
      return _regeneratorRuntime().wrap(function _callee7$(_context7) {
        while (1) switch (_context7.prev = _context7.next) {
          case 0:
            _context7.prev = 0;
            _context7.next = 3;
            return fetch('http://127.0.0.1:5000/get_status');
          case 3:
            response = _context7.sent;
            if (!response.ok) {
              _context7.next = 15;
              break;
            }
            _context7.next = 7;
            return response.json();
          case 7:
            data = _context7.sent;
            console.log(data); // 받은 데이터를 콘솔에 출력

            // 필요한 경우 DOM 요소를 업데이트
            statusElement = document.getElementById('status');
            statusElement.textContent = "Detected: ".concat(data.is_there, ", Make Alarm: ").concat(data.make_alarm);
            if (data.make_alarm === true) {
              alarmCount += 1;
              if (alarmCount >= 7) {
                chrome.runtime.sendMessage({
                  action: 'triggerAlarm'
                });
                alarmCount = 0; // 메시지를 보낸 후 카운터 초기화
              }
            } else {
              alarmCount = 0; // 알람이 false이면 카운터 초기화
            }

            //임시로 true -> 나중에 false로
            if (data.is_there === false) {
              noDetectionCount += 1;
              if (noDetectionCount >= 5) {
                chrome.runtime.sendMessage({
                  action: 'noDetection'
                });
                noDetectionCount = 0; // 메시지를 보낸 후 카운터 초기화
              }
            } else {
              noDetectionCount = 0; // 감지가 true이면 카운터 초기화
            }
            _context7.next = 16;
            break;
          case 15:
            console.error('Network response was not ok');
          case 16:
            _context7.next = 21;
            break;
          case 18:
            _context7.prev = 18;
            _context7.t0 = _context7["catch"](0);
            console.error('There was a problem with the fetch operation:', _context7.t0);
          case 21:
          case "end":
            return _context7.stop();
        }
      }, _callee7, null, [[0, 18]]);
    }));
    return _updateStatus.apply(this, arguments);
  }
  pipButton.addEventListener('click', /*#__PURE__*/_asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee2() {
    return _regeneratorRuntime().wrap(function _callee2$(_context2) {
      while (1) switch (_context2.prev = _context2.next) {
        case 0:
          if (!(processedStream.readyState >= processedStream.HAVE_METADATA)) {
            _context2.next = 10;
            break;
          }
          if (!document.pictureInPictureElement) {
            _context2.next = 6;
            break;
          }
          _context2.next = 4;
          return document.exitPictureInPicture();
        case 4:
          _context2.next = 8;
          break;
        case 6:
          _context2.next = 8;
          return processedStream.requestPictureInPicture();
        case 8:
          _context2.next = 11;
          break;
        case 10:
          processedStream.addEventListener('loadedmetadata', /*#__PURE__*/_asyncToGenerator( /*#__PURE__*/_regeneratorRuntime().mark(function _callee() {
            return _regeneratorRuntime().wrap(function _callee$(_context) {
              while (1) switch (_context.prev = _context.next) {
                case 0:
                  if (!document.pictureInPictureElement) {
                    _context.next = 5;
                    break;
                  }
                  _context.next = 3;
                  return document.exitPictureInPicture();
                case 3:
                  _context.next = 7;
                  break;
                case 5:
                  _context.next = 7;
                  return processedStream.requestPictureInPicture();
                case 7:
                case "end":
                  return _context.stop();
              }
            }, _callee);
          })), {
            once: true
          });
        case 11:
        case "end":
          return _context2.stop();
      }
    }, _callee2);
  })));
  fullscreenButton.addEventListener('click', function () {
    if (!document.fullscreenElement) {
      processedStream.requestFullscreen()["catch"](function (err) {
        console.error("Error attempting to enable full-screen mode: ".concat(err.message, " (").concat(err.name, ")"));
      });
    } else {
      document.exitFullscreen();
    }
  });
  popupButton.addEventListener('click', function () {
    var popup = window.open("", "popup", "width=640,height=480");
    popup.document.write("<img id=\"popupProcessedStream\" alt=\"Processed Stream\">");
    var popupProcessedStream = popup.document.getElementById('popupProcessedStream');
    popupProcessedStream.src = processsedStream.src;
  });
  startWebcam();
  startProcessedStream(); // 서버에서 처리된 스트림 시작
  setInterval(updateStatus, 1000); // 상태 업데이트 주기를 1초로 설정
});
/******/ })()
;
//# sourceMappingURL=webcam.bundle.js.map