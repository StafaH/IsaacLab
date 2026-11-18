"""Measure how smoothly the joints move in traces of the evaluation worlds: usage analyse_trace.py <npz> ..."""

import sys

import numpy as np


def measures(pos, dt):
    """Return smoothness measures of joint positions [steps, joints] sampled every ``dt`` seconds."""
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    spectrum = np.abs(np.fft.rfft(vel - vel.mean(0), axis=0)) ** 2
    freq = np.fft.rfftfreq(len(vel), dt)
    high = spectrum[freq > 5.0].sum() / max(spectrum.sum(), 1e-12)
    return np.sqrt((vel**2).mean()), np.sqrt((acc**2).mean()), np.sqrt((jerk**2).mean()), high


def episodes(done):
    """Yield (world, first step, step after the last) of the first episode of each world."""
    for world in range(done.shape[1]):
        ends = np.nonzero(done[:, world])[0]
        yield world, 0, (ends[0] if len(ends) else len(done))


for path in sys.argv[1:]:
    data = np.load(path)
    dt = float(data["step_dt"])
    task = data["tasks"][0]
    clip = data["clip_joint_pos"][task[1] : task[3]]
    clip = clip[:, data["moved"]] if clip.shape[1] != data["joint_pos"].shape[2] else clip
    print(path.split("/")[-1], "dt", dt, "clip frames", len(clip))
    print("  %-22s %8s %8s %9s %7s %6s" % ("", "vel rms", "acc rms", "jerk rms", ">5 Hz", "n"))
    print("  %-22s %8.2f %8.1f %9.0f %7.3f" % ("clip", *measures(clip, dt)))
    for rung in sorted(set(data["rung"])):
        for name in ("joint_pos", "target"):
            rows, track = [], []
            for world, begin, end in episodes(data["done"]):
                if data["rung"][world] != rung or end - begin < 30:
                    continue
                series = data[name][begin:end, world]
                rows.append(measures(series, dt))
                if rung == 0 and name == "joint_pos":
                    steps = min(len(series), len(clip) - 1)
                    track.append(np.sqrt(((series[:steps] - clip[1 : steps + 1]) ** 2).mean()))
            if rows:
                label = f"rung {rung} {name}"
                print("  %-22s %8.2f %8.1f %9.0f %7.3f %6d" % (label, *np.mean(rows, axis=0), len(rows)))
            if track:
                print("  %-22s %8.3f rad rms joint error against the clip" % (f"rung {rung} tracking", np.mean(track)))
