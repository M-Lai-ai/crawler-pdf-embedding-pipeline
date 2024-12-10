# utils/event_manager.py
import threading
import queue

class EventManager:
    def __init__(self):
        self.queue = queue.Queue()

    def emit(self, event_type, data):
        self.queue.put({'type': event_type, 'data': data})

    def get_event(self):
        try:
            return self.queue.get(timeout=1)
        except queue.Empty:
            return None

# Singleton instance
event_manager = EventManager()
