using LinearAlgebra: inv, I

"""
    bfgsmin(objective, x0; maxiter=1000, tol=1e-6, verbose=false)

Minimizes a differentiable function `f` using the BFGS quasi-Newton method, 
given its gradient `g` and an initial guess `x0`.

# Arguments
- `objective`: Function to minimize. Returns both a value and its gradient
- `x0`: Initial parameter vector

# Keyword Arguments
- `maxiter`: Maximum number of iterations (default: 1000)
- `tol`: Convergence tolerance on directional gradient (default: 1e-6)
- `verbose`: Print iteration log (default: false)

# Returns
A `NamedTuple` with:
- `convergence`: 0 if converged, 2 if max iterations reached
- `iterations`: Number of iterations
- `f_value`: Final function value
- `solution`: Parameter vector at minimum
"""
function bfgsmin(
    objective::Function,
    gradient::Function,
    x0;
    maxiter::Int=1000,
    tol::Float64=1e-06,
    verbose=false
    )
	
	# Initialize
	x = x0;
	f_val = objective(x)
    grad = gradient(x)

    n = length(x)

    hessian_approx = Matrix{Float64}(I,n,n)     # Approximate Hessian
	c1 = 1e-04                                  # Fixed learning rate of Backtracking algorithm
	lambda = 1.                                 # Adaptive learning rate of Backtracking algorithm
	convergence_flag = false                    # Convergence flag
	iterations = 0                              # Iterations

    if verbose
		println("\nInitial F-Value: ", round(f_val;digits=6))
	end

    # Start algorithm
	for iter = 1:maxiter
		lambda = 1

		# Construct direction vector and relative gradient
		direction = -inv(hessian_approx) * grad         # Direction
        directional_derivative = dot(direction, grad)   # Directional derivative

        # Select step size to satisfy the Armijo-Goldstein condition (Backtracking)
		while true;
			x_trial = x + lambda * direction
			
			f_trial = try
                objective(x_trial)
				catch
					NaN
				end
			
			f_test = f_val + c1 * lambda * directional_derivative

			if isfinite(f_trial) && f_trial <= f_test && f_trial>0
				break
			else
				lambda = lambda / 2
                if lambda < 1e-12
                    error("Line search failed: step size too small.")
                end
			end
		end

		# Construct the improvement and gradient improvement
		x_new = x + lambda * direction
        f_new = objective(x_new)
        grad_new = gradient(x_new)
				
        s = x_new - x
        y = grad_new - grad

        ys = dot(y, s)
        if ys > 1e-10
            term1 = (y * y') / ys
            hy = hessian_approx * s
            term2 = (hy * hy') / dot(s, hy)
            hessian_approx += term1 - term2
        end
		
        x = x_new
        grad = grad_new
        f_val = f_new

        grad_norm = abs(directional_derivative)

        iterations =+ 1
		
        # Show information if verbose == true
        if verbose
            println("Iter: $iter | f(x): $(round(f_val; digits=6)) | |grad' * d|: $(round(grad_norm; digits=6)) | Step: $(round(lambda; digits=6))")
        end
		
        # Check if relative gradient is less than tolerance
        if grad_norm < tol
            convergence_flag = true
            verbose && println("Converged.")
            break
        end
	end
	
    if !convergence_flag
        println("\nMaximum iterations reached. Convergence not achieved.")
    end
	
	results = (
        convergence = convergence_flag,
        iterations = iterations,
        f_value = f_val,
        solution = x
    )
	
	return results
end