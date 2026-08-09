"""A console buffer that keeps its head and scrolls its tail.

A training run prints for hours; keeping every line would grow without
bound, and keeping only the last N would drop exactly the part worth
having -- the options the run chose and its first error, which are printed
at the very beginning. So the first HEAD_LINES lines are kept forever, the
last TAIL_LINES scroll, and what fell between them is replaced by a single
line saying how many lines are missing.
"""
from collections import deque

HEAD_LINES = 500
TAIL_LINES = 4500
OMITTED_TEMPLATE = "--- %d lines omitted ---"


class ConsoleBuffer(object):
    def __init__(self, head=HEAD_LINES, tail=TAIL_LINES):
        self._head_max = head
        self._tail_max = tail
        self._head = []
        self._tail = deque()
        self.omitted = 0

    def append(self, line):
        if len(self._head) < self._head_max:
            self._head.append(line)
            return
        self._tail.append(line)
        while len(self._tail) > self._tail_max:
            self._tail.popleft()
            self.omitted += 1

    def replace_last(self, line):
        """Rewrite the last line, as a carriage return rewrites a row.

        Appends instead when there is nothing yet, so the caller never has
        to know whether the line it is updating was already stored. Which
        of the two halves holds the last line depends on how full the head
        is -- both cases are the normal one, at different points of a run.
        """
        if self._tail:
            self._tail[-1] = line
        elif self._head:
            self._head[-1] = line
        else:
            self.append(line)

    def lines(self):
        """Head, the omission notice if there is one, then tail."""
        if self.omitted == 0:
            return self._head + list(self._tail)
        return self._head + [OMITTED_TEMPLATE % self.omitted] + list(self._tail)
