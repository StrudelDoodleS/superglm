from typing import assert_type

from superglm import (
    BoundPredictor,
    BoundTerm,
    GaussianLS,
    SuperLSS,
    TweedieLSS,
    bind_predictor,
    cat,
    s,
)

tweedie = TweedieLSS()
assert_type(s("age"), BoundTerm)
assert_type(tweedie.mu("age", cat("region")), BoundPredictor)
assert_type(bind_predictor(tweedie, "power"), BoundPredictor)
assert_type(SuperLSS(tweedie, tweedie.mu(s("age")), tweedie.phi(), tweedie.p()), SuperLSS)
gaussian = GaussianLS()
assert_type(gaussian.location("age"), BoundPredictor)
assert_type(SuperLSS(gaussian, gaussian.location("age"), gaussian.scale()), SuperLSS)
