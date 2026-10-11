"""Private subprocess IPC entry point. Never load workspace/checkpoint pickles."""

from __future__ import annotations

import pickle
import signal
import sys
from pathlib import Path


def main() -> None:
    # The coordinator handles terminal interruption and cleans up process groups.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with Path(sys.argv[1]).open("rb") as handle:
        function, argument = pickle.load(handle)
    result = function(argument)
    with Path(sys.argv[2]).open("wb") as handle:
        pickle.dump(result, handle)


if __name__ == "__main__":
    main()
