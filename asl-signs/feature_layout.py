# feature_layout.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for the v2 feature vector layout.
#
# Both phase2b_features.py (builds the data) and phase3b_train.py (augments it)
# import from here so the block boundaries can never drift out of sync.
#
# POSITION layout (per frame), built by concatenation in this exact order:
#   left_hand   : 21 pts x (x,y,z) =  63   -> indices   0 .. 62
#   right_hand  : 21 pts x (x,y,z) =  63   -> indices  63 .. 125
#   pose[:25]   : 25 pts x (x,y,z) =  75   -> indices 126 .. 200
#   lips        : 40 pts x (x,y)   =  80   -> indices 201 .. 280
#   ------------------------------------------------------------------
#   N_POS = 281
#
# The final feature vector is  [position(281) | velocity(281)] = 562 dims,
# where velocity[t] = position[t] - position[t-1].
#
# This is a STRICT SUPERSET of the original 402-dim model (which was
# 201 position + 201 velocity, hands + pose[:25] only). We only *add* the
# 40 lip points, so the original signal is preserved exactly and the lip
# signal (mouth morphemes) is what the new model gains.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# ── MediaPipe face-mesh lip landmark indices (outer + inner contour, 40 pts) ──
# Standard set used across GISLR solutions. Index 0 -> face landmark 61 (left
# mouth corner), index 10 -> face landmark 291 (right mouth corner); those two
# are used as the horizontal scale reference for lip normalization.
LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
assert len(LIPS) == 40 and len(set(LIPS)) == 40, "LIPS must be 40 unique indices"

LIP_CORNER_LEFT  = 0    # position of face-idx 61 inside the LIPS list
LIP_CORNER_RIGHT = 10   # position of face-idx 291 inside the LIPS list

# ── Block metadata: (name, start, n_points, dims_per_point) in POSITION space ─
LH_N, RH_N, POSE_N, LIP_N = 21, 21, 25, 40
LH   = ("left_hand",  0,                     LH_N,   3)
RH   = ("right_hand", LH[1] + LH_N * 3,      RH_N,   3)   # start 63
POSE = ("pose",       RH[1] + RH_N * 3,      POSE_N, 3)   # start 126
LIP  = ("lips",       POSE[1] + POSE_N * 3,  LIP_N,  2)   # start 201

BLOCKS = [LH, RH, POSE, LIP]
N_POS  = LIP[1] + LIP_N * 2                                # 281
FEAT_DIM = N_POS * 2                                       # 562 (pos + velocity)

assert N_POS == 281 and FEAT_DIM == 562


# ── Index helpers ────────────────────────────────────────────────────────────
def _block_xy_pairs(start, n_pts, dims):
    """Yield (x_index, y_index) for every landmark in one block."""
    return [(start + i * dims, start + i * dims + 1) for i in range(n_pts)]


def _block_x_indices(start, n_pts, dims):
    """All x-channel indices in one block."""
    return [start + i * dims for i in range(n_pts)]


def build_mirror_index():
    """
    Build a permutation `perm` and a sign vector `sign` (both length FEAT_DIM)
    such that a horizontal mirror of a normalized frame is:

        mirrored = x[perm] * sign

    A horizontal flip:
      * negates every x channel,
      * swaps left_hand <-> right_hand (a signer flipping dominant hand),
      * applies identically to the velocity half.
    pose and lips keep their channels but have x negated (geometric flip).
    """
    perm = np.arange(FEAT_DIM)
    sign = np.ones(FEAT_DIM, dtype=np.float32)

    def mirror_half(offset):
        # Swap the two hand blocks within [offset, offset+N_POS)
        lh_start = offset + LH[1]
        rh_start = offset + RH[1]
        for i in range(LH_N * 3):
            perm[lh_start + i] = rh_start + i
            perm[rh_start + i] = lh_start + i
        # Negate x on every block (hands, pose, lips) in this half
        for _, start, n_pts, dims in BLOCKS:
            for xi in _block_x_indices(offset + start, n_pts, dims):
                sign[xi] = -1.0

    mirror_half(0)       # position half
    mirror_half(N_POS)   # velocity half
    return perm, sign


def build_xy_index():
    """
    Return four int arrays: (pos_x, pos_y, vel_x, vel_y) giving the x- and y-
    channel indices for every landmark, for both the position half and the
    velocity half. Used to apply 2-D affine (rotate/scale/shift) augmentation.
    """
    pos_x, pos_y, vel_x, vel_y = [], [], [], []
    for _, start, n_pts, dims in BLOCKS:
        for xi, yi in _block_xy_pairs(start, n_pts, dims):
            pos_x.append(xi);          pos_y.append(yi)
            vel_x.append(xi + N_POS);  vel_y.append(yi + N_POS)
    return (np.array(pos_x), np.array(pos_y),
            np.array(vel_x), np.array(vel_y))


if __name__ == "__main__":
    # Self-check: mirroring twice must be the identity.
    perm, sign = build_mirror_index()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, FEAT_DIM)).astype(np.float32)
    once  = x[:, perm] * sign
    twice = once[:, perm] * sign
    assert np.allclose(x, twice), "mirror(mirror(x)) != x"
    px, py, vx, vy = build_xy_index()
    print(f"N_POS={N_POS}  FEAT_DIM={FEAT_DIM}")
    print(f"landmarks with (x,y): {len(px)}  (21+21+25+40 = {21+21+25+40})")
    print("mirror self-inverse: OK")
