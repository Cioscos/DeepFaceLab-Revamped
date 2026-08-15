"""EventTail: tail reader for JSON-lines event file with Qt signals."""
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import json
import os


class EventTail(QObject):
    """
    Asynchronous tail reader for JSON-lines event files.

    Reads events from the end of a file line by line, emitting a Qt signal for each
    complete JSON object. Maintains file position across ticks to resume where it left
    off. Partial lines (without terminating newline) are buffered and completed on
    subsequent ticks. Malformed JSON lines are silently skipped. Missing files emit
    no errors.

    Detects file truncation by monitoring inode and size: if the inode changes
    (file recreated) or size decreases below current position (in-place truncation),
    position resets to 0 and reading resumes from the start.

    Known limitation: in-place file truncation at the same byte size with the same
    inode remains undetectable at reasonable cost. In practice this does not occur
    with the real producer (separate workdir per job).

    Signals:
        event (dict): Emitted when a complete, valid JSON dict is read.
    """

    event = pyqtSignal(dict)

    def __init__(self, path, interval_ms=500, parent=None):
        """
        Initialize EventTail.

        Args:
            path: Path to the JSON-lines file.
            interval_ms: Timer interval in milliseconds (default 500).
            parent: Parent QObject.
        """
        super().__init__(parent)
        self.path = path
        self.interval_ms = interval_ms
        self.file_pos = 0
        self._ino = None  # File inode; lazily initialized on first read
        self._active = True
        self._schedule_tick()

    def _schedule_tick(self):
        """Schedule the next tick."""
        if self._active:
            QTimer.singleShot(self.interval_ms, self._tick)

    def _tick(self):
        """Read new lines from file, emit events, and schedule the next tick."""
        if not self._active:
            return
        self._read_once()
        self._schedule_tick()

    def _read_once(self):
        """Read whatever is newly available in the file and emit events for it."""
        if not os.path.exists(self.path):
            # File disappeared; reset position and inode so file recreation is treated as new
            self.file_pos = 0
            self._ino = None
            return

        # Check if file was truncated/recreated by comparing inode and file size
        try:
            st = os.stat(self.path)
            # Initialize inode on first read
            if self._ino is None:
                self._ino = st.st_ino
            # Detect file recreation (inode changed) or truncation (size decreased)
            if st.st_ino != self._ino or st.st_size < self.file_pos:
                # File was recreated or truncated; reset position and update inode
                self.file_pos = 0
                self._ino = st.st_ino
        except (OSError, IOError):
            return

        try:
            with open(self.path, 'rb') as f:
                f.seek(self.file_pos)
                data = f.read()
        except (OSError, IOError):
            return

        if not data:
            return

        # Find the last newline; we only process complete lines
        last_newline = data.rfind(b'\n')

        if last_newline == -1:
            # No complete line; partial line is buffered until next tick
            return

        # Decode data up to and including the last newline
        complete_data = data[:last_newline + 1]
        text = complete_data.decode('utf-8', errors='ignore')
        lines = text.split('\n')

        # Process each non-empty line
        for line in lines:
            if line.strip():
                try:
                    payload = json.loads(line)
                    # Only emit if payload is a dict (contract requirement)
                    if isinstance(payload, dict):
                        self.event.emit(payload)
                except json.JSONDecodeError:
                    # Silently ignore malformed JSON
                    pass

        # Update file position to right after the last newline
        self.file_pos += last_newline + 1

    def stop(self):
        """Stop reading events.

        Reads once more before deactivating, so any line written after the
        last tick (or before the first one ever fired) is still delivered --
        a job that finishes in under one interval must not lose everything
        its child wrote. Idempotent: a second call finds ``_active`` already
        false and reads nothing further; unlike ``_tick``, this final read
        never reschedules another one.
        """
        self._read_once()
        self._active = False
