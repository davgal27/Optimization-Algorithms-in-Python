###### GRAD DESCENT NEW 
# Definition of Griewangk function pasted from prep
def gwangk(x, a=1, b=5):
    return 1+ x[0]**2/4000 + x[1]**2/4000 - np.cos(x[0])*np.cos(0.5*x[1]*np.sqrt(2))
# Gradient of Grienwangk function 
def g_gwangk(x):
    gradx1 = x[0] / 2000 + np.sin(x[0]) * np.cos((1 / np.sqrt(2)) * x[1])
    gradx2 = x[1] / 2000 + np.cos(x[0]) * np.sin((1 / np.sqrt(2)) * x[1]) * (1 / np.sqrt(2))
    return np.array([gradx1, gradx2]) 


def gradient_descent(alpha = 0.5, N=15, limit=10, runs = 100, x_init = None):#step size adjusted from 0.01 to 0.5
    iter_values = np.zeros((runs, N)) # Generative AI was used for syntax on tracking iterations
    best_gx = float('inf')
    best_run = None
    best_min = None
    
    for run in range(runs):
        if x_init is not None: # modification done so that Q3 can work correctly
            x = x_init.copy()
            
        else:
            x = np.random.uniform(-limit, limit, size=2)
            
        for itr in range(N):
            grad = g_gwangk(x)
            x = x - alpha * grad 
            gwangkval = gwangk(x)
            iter_values[run, itr] = gwangkval
        
        # Finding the Minimizer
        if run == 0 or gwangkval < best_gx:
            best_gx = gwangkval
            best_min = x.copy()
            best_run = run 

    final_iter_vals = iter_values [:, -1] # finding values at final iteration
    std_dev = np.std(final_iter_vals) # Standard deviation of final iteration results
    gx_iter_avg1 = iter_values.mean(axis=0) # Generative AI used for syntax 

    return iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg1, N

iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg1, N = gradient_descent()
# Standard deviation calculation
print ("Standard deviation of G(x) final values is :", std_dev)

# Average value of G(x) at each iteration
print("Average value of G(x) at the last iteration is:", gx_iter_avg1[-1])  

# Minimizer
print("Best G(x) found in run %d is: %.10f,with minimizer coordinates: [%.10f, %.10f]"
      % (best_run, best_gx, best_min[0], best_min[1]))

x = range(1, N+1)
y = gx_iter_avg1
plt.plot(x, y, 'o-')
plt.xlabel("Iteration")
plt.ylabel("Average G(x) value")
plt.title("Average value of G(x) at each iteration")
plt.grid(True)
plt.show()

####### RANDOM SEARCH NEW 
# Definition of Griewangk function pasted from prep
def gwangk(x, a=1, b=5):
    return 1+ x[0]**2/4000 + x[1]**2/4000 - np.cos(x[0])*np.cos(0.5*x[1]*np.sqrt(2))

def random_search(N=15, limit=10, runs=100):
    iter_values = np.zeros((runs, N))
    best_gx = float('inf')
    best_run = None #https://builtin.com/software-engineering-perspectives/define-empty-variables-python
    best_min = None
    
    for run in range(runs):
        x = np.random.uniform(-limit, limit, size=2) # Random starting position 
        best_func_val = float('inf') # syntax from: https://www.geeksforgeeks.org/python-infinity/
        best_x_pos = None
        
        for itr in range(N):
            gwangkval = gwangk(x)
            
            if gwangkval < best_func_val:
                best_func_val = gwangkval
                best_x_pos = x.copy() #copy created to store separate object and not modify best
                                      # position when iteration is re run and gwangkval > best
            iter_values[run, itr] = best_func_val 
            x = np.random.uniform(-limit, limit, size=2) # new random position for next iter
            
            # Finding the minimiser 
            if run == 0 or gwangkval < best_gx: 
                best_gx = gwangkval
                best_min = x.copy()
                best_run = run
        
    final_iter_vals = iter_values [:, -1] # finding values at final iteration
    std_dev = np.std(final_iter_vals) # Standard deviation of final iteration results
    gx_iter_avg2 = iter_values.mean(axis=0) # Generative AI used for syntax 
    
    return iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg2, N

iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg2, N = random_search()

# Standard deviation calculation
print ("Standard deviation of G(x) final values is :", std_dev)

#Average value of best so far G(x) at each iteration
print("Average value of G(x) at the last iteration is:", gx_iter_avg2[-1])  

# Minimizer
print("Best G(x) found in run %d is: %.10f,with minimizer coordinates: [%.10f, %.10f]"
      % (best_run, best_gx, best_min[0], best_min[1]))

x = range(1,N+1)
y = gx_iter_avg2
plt.plot(x, y, 'o-')
plt.xlabel("Iteration")
plt.ylabel("Average G(x) value")
plt.title("Average value of G(x) at each iteration")
plt.grid(True)
plt.show()


######### QUESTION 3 NEW WITH FUNCTION NOT LOOP (WORKING)
# Definition of Griewangk function pasted from prep
def gwangk(x, a=1, b=5):
    return 1+ x[0]**2/4000 + x[1]**2/4000 - np.cos(x[0])*np.cos(0.5*x[1]*np.sqrt(2))
# Gradient of Grienwangk function 
def g_gwangk(x):
    gradx1 = x[0] / 2000 + np.sin(x[0]) * np.cos((1 / np.sqrt(2)) * x[1])
    gradx2 = x[1] / 2000 + np.cos(x[0]) * np.sin((1 / np.sqrt(2)) * x[1]) * (1 / np.sqrt(2))
    return np.array([gradx1, gradx2]) 

# Random Search (Step 1):
def random_search3(limit, N):
    best_func_val = float('inf')
    best_x_pos = None
    
    for itr in range(N):
        x = np.random.uniform(-limit, limit, size=2) # Random starting position
        gwangkval = gwangk(x)
        
        if gwangkval < best_func_val:
            best_func_val = gwangkval
            best_x_pos = x.copy() #copy created to store separate object and not modify best
                                  # position when iteration is re run and gwangkval > best

    return best_x_pos
    
# Gradient descent (Step 2)
def gradient_descent3(limit, N, x_init, alpha):
    x = x_init.copy()
    iter_values = np.zeros(N)
    for itr in range(N):
        grad = g_gwangk(x)
        x = x - alpha * grad #From equation in notes x^k+1 = x^k + alpha x d^k
        iter_values[itr] = gwangk(x)
    return x, iter_values
    


# Combining the two 
def gradient_random(limit=10, N=15, alpha=0.5, runs=100):
    iter_values = np.zeros((runs, N))
    best_gx = float('inf')
    best_min = None
    best_run = None

    for run in range(runs):
        x_init = random_search3(limit, N)
        x_final, iter_values[run] = gradient_descent3(limit, N, x_init, alpha)
        gwangkval = gwangk(x_final)

        if gwangkval < best_gx:
            best_gx = gwangkval
            best_min = x_final
            best_run = run
    
    final_iter_vals = iter_values [:, -1] # finding values at final iteration
    std_dev = np.std(final_iter_vals) # Standard deviation of final iteration results
    gx_iter_avg3 = iter_values.mean(axis=0) # Generative AI used for syntax 
    
    return iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg3, N

        
iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg3, N = gradient_random()
   
# Standard deviation calculation
print ("Standard deviation of G(x) final values is :", std_dev)

#Average value of best so far G(x) at each iteration
print("Average value of G(x) at the last iteration is:", gx_iter_avg3[-1])  

# Minimizer
print("Best G(x) found in run %d is: %.10f,with minimizer coordinates: [%.10f, %.10f]"
      % (best_run, best_gx, best_min[0], best_min[1]))

x = range(1,N+1)
y = gx_iter_avg3
plt.plot(x, y, 'o-')
plt.xlabel("Iteration")
plt.ylabel("Average G(x) value")
plt.title("Average value of G(x) at each iteration")
plt.grid(True)
plt.show()


######## QUESTION 4 
# Definition of Griewangk function pasted from prep
def gwangk(x, a=1, b=5):
    return 1+ x[0]**2/4000 + x[1]**2/4000 - np.cos(x[0])*np.cos(0.5*x[1]*np.sqrt(2))


#initial temp of 10 was used in K&W book, however temp of 5 provided better results with few iterations
def simulated_annealing(limit=10, N=15, init_temp=5, runs=100):
    iter_values = np.zeros((runs, N))# ChatGPT was used for logic on how to track indiv. iterations
    best_min = None
    best_run = None
    gx_best = float('inf')
    x_best = None
    
    for run in range(runs):
        x = np.random.uniform(-limit, limit, 2)
        gwangkval = gwangk(x)
    
        for itr in range(1, N+1): 
            T = init_temp / itr # Using Fast annealing schedule from K&W book 
            x_next = x + np.random.uniform(-1, 1, 2) * T 
            gwangkval_next = gwangk(x_next)
            change_in_val = gwangkval_next - gwangkval
            
            if change_in_val <= 0 or np.random.rand() < np.exp(-change_in_val / T):
                x = x_next
                gwangkval = gwangkval_next
    
            if gwangkval < gx_best:
                gx_best = gwangkval
                x_best = x.copy()
                best_run = run 
                best_min = x_best.copy()
    
            iter_values[run, itr - 1] = gwangkval

            
    final_iter_vals = iter_values [:, -1] # finding values at final iteration
    std_dev = np.std(final_iter_vals) # Standard deviation of final iteration results
    gx_iter_avg4 = iter_values.mean(axis=0) # Generative AI used for syntax

    return iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg4,  N
    
iter_values, best_gx, best_min, best_run, std_dev, gx_iter_avg4, N = simulated_annealing()
   
# Standard deviation calculation
print ("Standard deviation of G(x) final values is :", std_dev)

#Average value of best so far G(x) at each iteration
print("Average value of G(x) at the last iteration is:", gx_iter_avg4[-1])  

# Minimizer
print("Best G(x) found in run %d is: %.10f,with minimizer coordinates: [%.10f, %.10f]"
      % (best_run, best_gx, best_min[0], best_min[1]))

x = range(1,N+1)
y = gx_iter_avg4
plt.plot(x, y, 'o-')
plt.xlabel("Iteration")
plt.ylabel("Average G(x) value")
plt.title("Average value of G(x) at each iteration")
plt.grid(True)
plt.show()    
  

#### QUESTION 5 
N = 15 # iterations 
x = range(1, N+1) # redefining x 
# To run Q5 correctly, previous questions must be run beforehand
# Plotting gx_itr_avg of respective questions eg: q1 = 1)
plt.plot(x, gx_iter_avg1, 'o-', color='blue', label="Gradient Descent")
plt.plot(x, gx_iter_avg2, 'o-', color='green', label="Random Search")
plt.plot(x, gx_iter_avg3, 'o-', color='orange', label="Hybrid Grad+Rand")
plt.plot(x, gx_iter_avg4, 'o-', color='red', label="Simulated Annealing")


plt.xlabel("Iteration")
plt.ylabel("Average G(x) value")
plt.title("Average value of G(x) at each iteration")

plt.minorticks_on() #https://www.pythoncharts.com/matplotlib/customizing-grid-matplotlib/
plt.grid(which='major', linewidth = 0.8)
plt.grid(which='minor', linewidth = 0.5)

plt.legend()
plt.show()


# Convergence 
To compare the algorithms in terms of convergence, I will be taking the convergence rate as the rate at which the algorithm reaches a point where further iterations stop improving the average G(x) value by much 

In terms of convergence, the hybrid 2 step approach of Random search followed by gradient descent showed the fastest convergence rate, not improving much after around the 5th iteration. 

Gradient descent also converged at around the 13th iteration, with little room for minimization left.

With the limited amount of iterations, Random search and Simulated annealing did not seem to converge, as they both showed room for improvement had the iteration count been higher.


# Effectiveness
To compare the algorithms in terms of effectiveness, I will be basing purely on how much the functions managed to minimize the function given the same amount of iterations.

The hybrid 2 step approach was the most effective algorithm at minimizing the Griewangk function, with gradient descent being extremely close to it. 

Random search implemented on its own had worse results overall than Gradient Descent and The hybrid approach. However, it performed well compared to simulated annealing. 

The poor performance of simulated annealing is likely due to inefficient parameter tuning. Parameters like initial temperature, annealing strategy, and probability of accepting a worse value all greatly affect the performance of the algorithm, and choosing ideal test conditions is not a trivial task. (Ben-Ameur, Computing the initial temperature of simulated annealing 2004)

# Robustness 
To compare the algorithms in terms of robustness, I will be comparing how much a change in parameters affects the outcome of the algorithm 

When tested the algorithms with multiple iteration counts (15, 100, 1000, and 10,000), the algorithms still behaved rather similairly. However, some algorithms changed vastly with different parameters.

Gradient descent had the parameter of step size (alpha). With lower values than 0.5, at 15 iterations, the graph descended almost linearly, thus minimizing the function poorly. At a step size of 2, the function performed the best, nearly identical to the hybrid approach. One may thus state that a larger step size yields a better result, however at a step size of 2<, the function stopped decreasing and instead increased. 

Random search acted fairly similairly across several iteration counts and can therefore be considered quite robust

The hybrid approach, naturally, acted similairly to gradient descent, however started slowing down in terms of rate of convergence past a step size of 1.7<

Simulated annealing, as previously discussed in effectiveness, was the least robust, with any change in one of many parameters present, such as initial temperature, annealing strategy, and probability of accepting a worse value, will greatly impact the result of performance. 

# References 
[1] W. Ben-Ameur, “Computing the initial temperature of simulated annealing,” Computational Optimization and Applications, vol. 29, no. 3, pp. 369–385, Dec. 2004. doi:10.1023/b:coap.0000044187.23143.bd 

