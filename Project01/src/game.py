"""AI Color Memory Game: Game rules and sequence validation."""
import random
from config import (
    MAX_ROUND, ROUND_LENGTHS, ROUND_SCORES,
    SHOW_ON_SECONDS, SHOW_OFF_SECONDS, ROUND_PAUSE_SECONDS,
)


class SimonGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "READY"
        self.sequence = []
        self.round = 0
        self.index = 0
        self.score = 0
        self.entered = []
        self.phase_since = 0.0
        self.message = "Show all four colors, then press SPACE"
        self.feedback_until = 0.0
        self.last_choice = None
        self.last_award = 0
        self.last_correct = None

    @staticmethod
    def points_for_input(round_number, input_index):
        """Integer points add up exactly to this round's allotted score."""
        count = ROUND_LENGTHS[round_number - 1]
        total = ROUND_SCORES[round_number - 1]
        base, remainder = divmod(total, count)
        return base + (1 if input_index < remainder else 0)

    def start(self, now):
        if self.state not in ("READY", "GAME_OVER", "GAME_CLEAR"):
            return
        self.sequence = [
            random.choice(("RED", "BLUE", "GREEN", "YELLOW"))
            for _ in range(ROUND_LENGTHS[0])
        ]
        self.round = 1
        self.score = 0
        self._start_show(now)

    def _start_show(self, now):
        self.state = "SHOW"
        self.phase_since = now
        self.index = 0
        self.entered = []
        self.message = "Remember the flashing colors"
        self.feedback_until = 0.0
        self.last_choice = None
        self.last_award = 0
        self.last_correct = None

    def flash_color(self, now):
        if self.state != "SHOW":
            return None
        span = SHOW_ON_SECONDS + SHOW_OFF_SECONDS
        elapsed = max(0.0, now - self.phase_since)
        pos = int(elapsed / span)
        if pos >= len(self.sequence):
            return None
        if elapsed - pos * span < SHOW_ON_SECONDS:
            return self.sequence[pos]
        return None

    def update(self, now, chosen=None):
        if self.state == "SHOW":
            total = len(self.sequence) * (SHOW_ON_SECONDS + SHOW_OFF_SECONDS)
            if now - self.phase_since >= total:
                self.state = "INPUT"
                self.index = 0
                self.message = "Your turn: point and hold, then lift finger"

        elif self.state == "ROUND_CLEAR":
            if now - self.phase_since >= ROUND_PAUSE_SECONDS:
                self.round += 1
                target_length = ROUND_LENGTHS[self.round - 1]
                while len(self.sequence) < target_length:
                    self.sequence.append(
                        random.choice(("RED", "BLUE", "GREEN", "YELLOW"))
                    )
                self._start_show(now)

        elif self.state == "INPUT" and chosen is not None:
            expected = self.sequence[self.index]
            self.last_choice = chosen
            self.last_correct = (chosen == expected)
            self.feedback_until = now + 0.9
            if not self.last_correct:
                self.last_award = 0
                self.state = "GAME_OVER"
                self.message = f"WRONG: {chosen} (expected {expected}). R to restart"
                return

            # Score every confirmed correct selection, as in the earlier version.
            earned = self.points_for_input(self.round, self.index)
            self.score += earned
            self.last_award = earned
            self.entered.append(chosen)
            self.index += 1
            self.message = f"CORRECT: {chosen} +{earned} ({self.index}/{len(self.sequence)})"
            if self.index == len(self.sequence):
                if self.round >= MAX_ROUND:
                    self.state = "GAME_CLEAR"
                    self.message = f"GAME CLEAR! TOTAL {self.score}/100. Press R"
                else:
                    self.state = "ROUND_CLEAR"
                    self.phase_since = now
                    self.message = f"ROUND {self.round} CLEAR! TOTAL {self.score} points"
