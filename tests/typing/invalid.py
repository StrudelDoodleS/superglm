from superglm import GaussianLS, TweedieLSS, s

tweedie = TweedieLSS()
tweedie.muu("age")
tweedie.mu(123)
gaussian = GaussianLS()
gaussian.loction("age")
gaussian.scale(object())
s(123)
