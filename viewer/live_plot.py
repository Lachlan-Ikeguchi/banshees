"""Live line plot for ADC data streamed from the Pi over stdin.

Run on the Linux laptop (not the Pi) — pipe the Pi's streamed output into it:

    ssh pi2 'cd banshees/raspberry-pi && uv run banshee --stream --channels 8 --rate 0.05' \\
        | python3 live_plot.py

Each stdin line is expected to be CSV: `timestamp,ch0,ch1,...,chN`
(this is exactly what `banshee --stream` prints).
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Validated 8-slot categorical palette (light mode) — see dataviz skill.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-plot streamed ADC CSV from stdin")
    parser.add_argument(
        "--window",
        type=float,
        default=20.0,
        help="Seconds of history to keep on screen (default: 20)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Plot redraw interval in milliseconds (default: 50)",
    )
    return parser.parse_args()


def stdin_reader(line_queue: "queue.Queue[str]") -> None:
    for line in sys.stdin:
        line = line.strip()
        if line:
            line_queue.put(line)


def main() -> None:
    args = parse_args()

    line_queue: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=stdin_reader, args=(line_queue,), daemon=True).start()

    times: deque = deque()
    series: list[deque] = []
    lines: list = []
    channel_count = None

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("seconds ago", color=INK_SECONDARY)
    ax.set_ylabel("ADC value", color=INK_SECONDARY)
    ax.set_title("Live ADC stream", color=INK_PRIMARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    def ensure_channels(n: int) -> None:
        nonlocal channel_count
        channel_count = n
        for ch in range(n):
            series.append(deque())
            (line,) = ax.plot(
                [],
                [],
                color=CATEGORICAL[ch % len(CATEGORICAL)],
                linewidth=2,
                label=f"ch{ch}",
            )
            lines.append(line)
        ax.legend(loc="upper right", frameon=False, labelcolor=INK_SECONDARY)

    def update(_frame):
        drained = False
        while True:
            try:
                raw = line_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            parts = raw.split(",")
            try:
                ts = float(parts[0])
                values = [float(v) for v in parts[1:]]
            except ValueError:
                continue

            if channel_count is None:
                ensure_channels(len(values))
            elif len(values) != channel_count:
                continue

            times.append(ts)
            for ch, value in enumerate(values):
                series[ch].append(value)

        if not drained or channel_count is None:
            return lines

        now = times[-1]
        cutoff = now - args.window
        while times and times[0] < cutoff:
            times.popleft()
            for s in series:
                s.popleft()

        x = [t - now for t in times]
        for ch, line in enumerate(lines):
            line.set_data(x, series[ch])

        ax.set_xlim(-args.window, 0)
        all_values = [v for s in series for v in s]
        if all_values:
            lo, hi = min(all_values), max(all_values)
            pad = max((hi - lo) * 0.1, 1)
            ax.set_ylim(lo - pad, hi + pad)

        return lines

    ani = FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
