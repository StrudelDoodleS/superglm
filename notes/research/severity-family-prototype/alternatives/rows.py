"""Reference rows shared by all scripts: (mu, cv, y/mu, tag)."""
TYPICAL = [
    (1.0, 1.0, 0.3), (1.0, 1.0, 1.7), (1.0, 0.5, 0.8), (1.0, 0.5, 1.4),
    (100.0, 0.3, 0.7), (100.0, 0.3, 1.2), (0.01, 2.0, 0.2), (0.01, 2.0, 3.0),
    (1e4, 0.1, 0.9), (1e4, 0.1, 1.08), (50.0, 1.5, 0.05), (50.0, 1.5, 2.5),
    (2.0, 0.7, 1.001), (2.0, 0.2, 1.3), (7.0, 3.0, 0.45),
]
ADVERSARIAL = [
    (1.0, 0.02, 1e-6), (1.0, 0.02, 1e3), (1.0, 5.0, 1e-6), (1.0, 5.0, 1e3),
    (1.0, 0.02, 1.01),   # near-cancellation in the score, tiny CV
    (1.0, 5.0, 1.001),   # tiny shape a = 0.04, y close to mu
]
ROWS = [(m, c, r, "typ") for m, c, r in TYPICAL] + [(m, c, r, "adv") for m, c, r in ADVERSARIAL]
def arrays():
    import numpy as np
    mu = np.array([r[0] for r in ROWS]); s = np.array([r[1] for r in ROWS])
    y = np.array([r[0] * r[2] for r in ROWS])
    return y, mu, s
