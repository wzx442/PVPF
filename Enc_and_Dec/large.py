import numpy as np
from typing import List

def lagrange_interpolation(
        x_points: List[float], 
        y_points: List[float]
        ) -> List[float]:
    """Construct Lagrange interpolation polynomial coefficients.
    
    Args:
        x_points: List of x coordinates (M+1 points)  
        y_points: List of y coordinates (M+1 points) 
        M: Degree of the polynomial
        
    Returns:
        List of coefficients in descending order (highest degree first)
    """
    n = len(x_points)
    assert len(y_points) == n, "x_points and y_points length must be the same"
    
    # initialize polynomial coefficients to 0
    poly_coeffs = np.zeros(n)
    
    for i in range(n):
        # calculate the i-th Lagrange basis polynomial
        p = np.poly1d([1.0])  # initialize to 1
        for j in range(n):
            if j != i:
                # multiply by (x - x_j)/(x_i - x_j)
                term = np.poly1d([1.0, -x_points[j]]) / (x_points[i] - x_points[j])
                p *= term
        
        # multiply the basis polynomial by the corresponding y value and add to the total polynomial
        poly_coeffs += y_points[i] * p.coeffs
    
    return poly_coeffs

def evaluate_polynomial(coefficients: List[float], x: float) -> float:
    """Evaluate polynomial at point x using Horner's method.
    
    Args:
        coefficients: List of coefficients in descending order (highest degree first)
        x: Point at which to evaluate the polynomial
        
    Returns:
        Value of polynomial at x
    """
    # result = 0.0
    # for coeff in coefficients:
    #     result = result * x + coeff
    # return result
    return np.polyval(coefficients, x)

