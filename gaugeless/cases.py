"""Case registry: where each river's frames live, frame rate, ground sampling distance, reference set.

Values come from each case's Readme.txt in the Perks et al. (2020) dataset.
`ref_frame` notes which image the reference pixel coordinates refer to; for all cases
listed here, it's the first frame of the folder given in `frames`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .refs import ROOT, load as load_ref


@dataclass(frozen=True)
class Case:
    name: str
    frames: str            # folder relative to ROOT
    fps: float             # frame rate of the frames in `frames`
    gsd: float             # metres per pixel
    ref: str | None        # key into refs.LOADERS
    country: str
    platform: str
    notes: str = ""
    tags: tuple = field(default_factory=tuple)
    water: tuple | None = None   # site setup: water polygon in frame pixels, drawn once (as in KLT-IV)

    @property
    def path(self):
        return os.path.join(ROOT, self.frames)

    def available(self) -> int:
        try:
            return len([f for f in os.listdir(self.path) if f.lower().endswith((".jpg", ".jpeg", ".gif", ".png", ".tif", ".pgm"))])
        except FileNotFoundError:
            return 0

    def water_mask(self, shape):
        """uint8 mask of the water polygon, or None when the site has no outline."""
        if not self.water:
            return None
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.array(self.water, np.int32)], 1)
        return m

    def reference(self):
        return load_ref(self.ref) if self.ref else None


CASES = {c.name: c for c in [
    Case("noce", "RiverNoce/Frames", 12, 0.009, "noce", "IT", "drone", water=((80, 0), (1700, 0), (1760, 400), (1800, 1080), (80, 1080))),
    Case("bradano", "RiverBradano/Stabilised/Frames", 24, 0.009, "bradano", "IT", "drone", "charcoal seeding", ("seeded",),
         water=((360, 0), (1250, 0), (1400, 1080), (380, 1080))),
    Case("alpine", "AlpineRiver/Frames", 12.5, 0.021, "alpine", "AT", "drone", "cornstarch seeding", ("seeded",),
         water=((0, 489), (524, 384), (1571, 244), (3840, 209), (3840, 1640), (2722, 1745), (1745, 1728), (0, 1605))),
    Case("arrow_uas_seeded", "RiverArrow/Stabilised/UAS_stable_trans_seeded/Frames", 5, 0.0174, "arrow_uas", "UK", "drone", tags=("seeded",), water=((0, 25), (464, 95), (464, 460), (0, 305))),
    Case("arrow_uas_unseeded", "RiverArrow/Stabilised/UAS_stable_trans_unseeded/Frames", 5, 0.0174, "arrow_uas", "UK", "drone", tags=("unseeded",), water=((0, 25), (464, 95), (464, 460), (0, 305))),
    Case("arrow_gopro_seeded", "RiverArrow/Stabilised/GoPro_stable_trans_seeded/Frames", 5, 0.0174, "arrow_gopro", "UK", "fixed GoPro", tags=("seeded",), water=((0, 0), (236, 0), (464, 100), (464, 460), (421, 442), (295, 391), (0, 282))),
    Case("arrow_gopro_unseeded", "RiverArrow/Stabilised/GoPro_Stable_trans_Unseeded/Frames", 5, 0.0174, "arrow_gopro", "UK", "fixed GoPro", tags=("unseeded",), water=((0, 0), (236, 0), (464, 100), (464, 460), (421, 442), (295, 391), (0, 282))),
    Case("brenta_flir", "RiverBrenta/FLIR/Frames", 5, 0.009, "brenta_flir", "IT", "thermal camera", "ice-cube tracers", ("thermal",),
         water=((0, 0), (318, 0), (318, 200), (0, 200))),
    Case("thalhofen", "RiverThalhofen/Frames", 30, 0.01, "thalhofen", "DE", "fixed camera", water=((0, 0), (700, 0), (700, 2600), (0, 2600))),
    Case("lavence", "RiverLaVence/Frames", 30, 0.008, "lavence", "FR", "smartphone", water=((0, 0), (320, 0), (320, 590), (290, 690), (15, 690), (0, 610))),
    Case("tiber", "RiverTiber/Frames", 6.95, 0.016, "tiber", "IT", "IP camera", "high flow"),
    Case("murg", "MurgRiver/Frames", 12, 0.0156, "murg", "CH", "drone",
         water=((244, 448), (956, 244), (1221, 814), (1547, 2279), (1913, 3378), (2157, 4477), (1343, 4477), (977, 3053), (448, 1221))),
    Case("castor", "../derived/castor_v1/Frames", 20, 0.02, "castor", "CA", "bridge camera",
         "orthorectified by us from raw video (GCPs + lens model, water plane at 3.772 m)", ("high flow",)),
    Case("alpine_full", "AlpineRiver/Frames", 12.5, 0.021, "alpine", "AT", "drone"),
    Case("lamorge", "LaMorgeRiver/Frames", 5, 0.01, None, "FR", "pole camera", "discharge reference only",
         water=((0, 150), (950, 150), (1150, 195), (1550, 195), (1550, 640), (1290, 870), (0, 870))),
    Case("stjulien", "StJulienTorrent/Frames", 25, 0.03, None, "FR", "storm chaser", "flash flood, no reference"),
]}


def load_frames(case: Case, max_frames: int | None = None, stride: int = 1, color: bool = False):
    """Frames in filename order. Returns a list of uint8 arrays (grey unless color=True)."""
    files = sorted(f for f in os.listdir(case.path) if f.lower().endswith((".jpg", ".jpeg", ".gif", ".png", ".tif", ".pgm")))
    files = files[::stride][:max_frames]
    out = []
    for f in files:
        im = cv2.imread(os.path.join(case.path, f), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if im is None:          # some gifs need the video reader
            cap = cv2.VideoCapture(os.path.join(case.path, f)); ok, im = cap.read(); cap.release()
            if not ok:
                continue
            if not color:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        out.append(im)
    # frame QA: a sequence must have one frame size (Murg has one 4614-px frame among 4482-px ones)
    if out:
        from collections import Counter
        shape = Counter(im.shape for im in out).most_common(1)[0][0]
        dropped = sum(im.shape != shape for im in out)
        out = [im for im in out if im.shape == shape]
        load_frames.last_dropped = dropped
    return out


def verify_downloads():
    """Files listed in data/perks/MANIFEST.tsv that are missing or the wrong size (partial downloads)."""
    bad = []
    with open(os.path.join(ROOT, "MANIFEST.tsv")) as fh:
        for line in fh:
            size, name = line.rstrip("\n").split("\t", 1)
            p = os.path.join(ROOT, name)
            if not os.path.exists(p) or os.path.getsize(p) != int(size):
                bad.append(name)
    return bad
