"""AI Color Memory Game: Game sound effect management."""
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
import sys
from threading import Thread

SOUND_FILES = {
    "start": "Start.wav",
    "correct": "Correct.wav",
    "wrong": "Wrong.wav",
    "next": "Next.wav",
    "victory": "Victory.wav",
}


class SoundManager:
    def __init__(self, folder, enabled=True):
        self.folder = Path(folder)
        self._queue = Queue()
        self._closed = False
        self._backend = None
        self._thread = None
        if not enabled:
            print("[SOUND] Muted by --mute")
            return
        missing = [n for n in SOUND_FILES.values() if not (self.folder / n).is_file()]
        if missing:
            print("[SOUND] Missing sound files:", ", ".join(missing))
            print("[SOUND] Game will still run without audio.")
            return
        if sys.platform == "win32":
            import winsound
            self._backend = ("winsound", winsound)
        elif shutil.which("aplay"):
            self._backend = ("aplay", shutil.which("aplay"))
        else:
            print("[SOUND] 'aplay' not found. On Jetson, install/check alsa-utils and sound output.")
            print("[SOUND] Game will still run without audio.")
            return
        self._thread = Thread(target=self._worker, daemon=True, name="SimonAudio")
        self._thread.start()
        print("[SOUND] Ready (single, sequential playback)")

    def play(self, event):
        """Enqueue one event; never start a second playback directly."""
        if self._backend and not self._closed:
            if event not in SOUND_FILES:
                raise ValueError("Unknown sound event: " + str(event))
            self._queue.put(event)

    def clear_pending(self):
        """Discard old queued cues on game reset; currently playing cue ends normally."""
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                return

    def _worker(self):
        while True:
            event = self._queue.get()
            if event is None:
                self._queue.task_done()
                return
            path = self.folder / SOUND_FILES[event]
            try:
                backend, player = self._backend
                if backend == "winsound":
                    # Without SND_ASYNC this blocks ONLY the audio worker.
                    player.PlaySound(str(path), player.SND_FILENAME)
                else:
                    # Blocking call in the SAME worker: effects cannot overlap.
                    result = subprocess.run([player, "-q", str(path)], check=False,
                                            capture_output=True, text=True)
                    if result.returncode:
                        print(f"[SOUND] {event}: output failed: {result.stderr.strip()}")
            except Exception as exc:
                print(f"[SOUND] {event} playback failed: {exc}")
            finally:
                self._queue.task_done()

    def close(self):
        if not self._closed:
            self._closed = True
            self.clear_pending()
            if self._thread:
                self._queue.put(None)
                # Daemon worker allows Q/exit without waiting for a long sound tail.
