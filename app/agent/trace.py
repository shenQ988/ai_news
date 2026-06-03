"""
AgentTrace — step-by-step reasoning log for the FactChecker agent.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    step: int
    thought: str
    action: str
    action_input: Any
    observation: str
    duration_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentTrace:
    def __init__(self):
        self._steps: List[TraceStep] = []
        self._start = time.time()
        self._counter = 0

    def log_step(
        self,
        thought: str,
        action: str,
        action_input: Any,
        observation: str,
        duration_ms: int,
    ) -> None:
        self._counter += 1
        self._steps.append(
            TraceStep(
                step=self._counter,
                thought=thought.strip(),
                action=action,
                action_input=action_input,
                observation=str(observation)[:400],
                duration_ms=duration_ms,
            )
        )

    def get_trace(self) -> List[TraceStep]:
        return list(self._steps)

    def print_trace(self) -> None:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        DIM = "\033[2m"

        print(f"\n{BOLD}{CYAN}{'═' * 52}{RESET}")
        print(f"{BOLD}{CYAN}  FACT CHECKER AGENT — REASONING TRACE{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 52}{RESET}")

        for s in self._steps:
            obs = s.observation.replace("\n", " ").strip()
            if len(obs) > 110:
                obs = obs[:107] + "..."

            print(f"\n{YELLOW}Step {s.step:<2}{RESET} │ {BOLD}THOUGHT:{RESET} {s.thought}")
            print(f"       │ {BOLD}ACTION: {RESET} {GREEN}{s.action}{RESET}({json.dumps(s.action_input, default=str)[:80]})")
            print(f"       │ {BOLD}RESULT: {RESET} {obs}")
            print(f"       │ {DIM}TIME:   {s.duration_ms / 1000:.1f}s{RESET}")

    def save_trace(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_steps": self._counter,
            "total_duration_ms": int((time.time() - self._start) * 1000),
            "steps": [asdict(s) for s in self._steps],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
