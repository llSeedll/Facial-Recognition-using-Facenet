from Quartz import (
    CGEventSourceCreate,
    CGSessionCopyCurrentDictionary,
    kCGEventSourceStateHIDSystemState,
    CGEventCreateKeyboardEvent,
    kCGSessionEventTap,
    CGEventPost,
    kCGHIDEventTap,
    CGEventKeyboardSetUnicodeString,
)
import time


OUTPUT_SOURCE = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)

class Keyboard():

    unicodes = [] #['v','p','1','3','7','$']

    def __init__(self, password):
        self.unicodes = list(set(password))

    def _send_string_press(self, c):
        event = CGEventCreateKeyboardEvent(OUTPUT_SOURCE, 0, True)
        self._set_event_string(event, c)
        CGEventPost(kCGSessionEventTap, event)
        event = CGEventCreateKeyboardEvent(OUTPUT_SOURCE, 0, False)
        self._set_event_string(event, c)
        CGEventPost(kCGSessionEventTap, event)

    def _set_event_string(self, event, s):
        #CGEventKeyboardSetUnicodeString(event, len(s), s)
        bytes = len(s.encode('utf-16-le')) // 2
        CGEventKeyboardSetUnicodeString(event, bytes, s)

    def KeyPress(self, k):
        if k in self.unicodes:
            self._send_string_press(k)
            time.sleep(0.0001)
            return

    def TypeUnicode(self, text):
        for key in text:
            self.KeyPress(key)
