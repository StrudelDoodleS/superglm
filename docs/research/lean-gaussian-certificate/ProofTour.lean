import Mathlib

-- The statement is true for every pair of real numbers, not sampled inputs.
theorem square_sum (a b : ℝ) :
    (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  ring

-- A one-coefficient version of the Gaussian quadratic objective-gap identity.
theorem scalar_quadratic_gap (h b x xStar : ℝ)
    (stationary : h * xStar = b) :
    (h * x ^ 2 / 2 - b * x) - (h * xStar ^ 2 / 2 - b * xStar) =
      h * (x - xStar) ^ 2 / 2 := by
  rw [← stationary]
  ring

#check square_sum
#check scalar_quadratic_gap
#print axioms square_sum
#print axioms scalar_quadratic_gap
