"""Dependency-free text progress for an individual export."""
from typing import Literal

import sys
from time import monotonic


class _Progress:
    """Display completed tiles on stderr, followed by finalization status.

    Args:
        total: Number of base-resolution output tiles.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.started = False
        self.last_update = 0.0
        self.stream = sys.stderr
        try:
            self.terminal = self.stream.isatty()
        except (OSError, ValueError):
            self.terminal = False

    def update(self, completed: int,
               stage: Literal['Writing', 'Finalizing', 'Done', 'Stopped'] = 'Writing') -> None:
        """Report completed work, limiting ordinary redraws to ten per second.

        Args:
            completed: Number of tiles already processed by the native writer.
            stage: Writing, Finalizing, Done or Stopped.
        """
        self.completed = completed
        now = monotonic()
        if self.started and stage == 'Writing' and now - self.last_update < 0.1:
            return
        self.started = True
        self.last_update = now
        filled = 24 * completed // self.total
        bar = '#' * filled + '-' * (24 - filled)
        text = f'COG [{bar}] {completed}/{self.total} tiles | {stage}'
        end = '' if self.terminal and stage in ('Writing', 'Finalizing') else '\n'
        # Progress is informational: a closed output stream must not turn a
        # successfully installed COG into a reported export failure.
        try:
            print(('\r' if self.terminal else '') + text.ljust(78),
                  file=self.stream, end=end, flush=True)
        except (OSError, ValueError):
            pass

    def finish(self, success: bool) -> None:
        """Close a started display without claiming success after an exception.

        Args:
            success: Whether finalization and destination installation succeeded.
        """
        if self.started:
            self.update(self.completed, 'Done' if success else 'Stopped')
