"""Turning what a process actually writes into the lines a console shows.

Two things sit between the two, and skipping either one puts spurious line
breaks on screen.

A pipe delivers bytes, not lines: a read can end in the middle of one, and
the rest arrives in the next read. So the tail of a read is held back as a
line still being written, not published as a line of its own.

And a line is not written once. `\\r` returns the cursor to column zero and
what follows overwrites what was there -- which is how every progress bar
in the application redraws itself (tqdm in core/interact, the trainer's
status line). Splitting on it, as `str.splitlines()` does, turns one line
that was rewritten a thousand times into a thousand lines.

So this is a one-line terminal: text plus a cursor column. What it produces
is a list of ("line", text) -- start a new line -- and ("update", text) --
rewrite the last one. A read carrying ten redraws of the same line produces
one event, the final state: the nine in between were never on screen.
"""


class LineAssembler(object):
    def __init__(self):
        self._text = ""
        self._col = 0
        self._shown = False   # the line being built already has a row on screen

    def feed(self, chunk):
        """The events for one read. Empty list when nothing changed."""
        if not chunk:
            return []
        events = []
        parts = chunk.split("\n")
        before = (self._text, self._col)
        for part in parts[:-1]:
            self._write(part)
            events.append((self._close(), self._text))
            self._text = ""
            self._col = 0
            self._shown = False
        self._write(parts[-1])
        if self._text and ((self._text, self._col) != before or not self._shown):
            events.append((self._close_pending(), self._text))
        return events

    def _write(self, part):
        """Write `part` at the cursor, honouring the returns inside it.

        A carriage return moves the cursor to column zero without erasing:
        "abcdef\\rxy" is "xycdef", not "xy". A terminal behaves this way, and
        a progress bar whose new state is shorter than the old one would
        otherwise appear to have lost characters it never touched.
        """
        segments = part.split("\r")
        for index, segment in enumerate(segments):
            if index:
                self._col = 0
            if segment:
                self._text = self._text[:self._col] + segment + self._text[self._col + len(segment):]
                self._col += len(segment)

    def _close(self):
        # A line that already has a row on screen is rewritten in place even
        # when the newline is what completes it: the row is the same row.
        return "update" if self._shown else "line"

    def _close_pending(self):
        kind = self._close()
        self._shown = True
        return kind
