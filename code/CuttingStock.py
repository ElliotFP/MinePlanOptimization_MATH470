import pyomo.environ as pyo
from itertools import combinations_with_replacement
import KnapsackSolver as ks
from collections import Counter

# Global Variables for the Cutting Stock Problem

# Testing script to make sure GLPK and pyomo is installed and working
def test_with_glpk():
    model = pyo.ConcreteModel()

    # Variables
    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2], sense=pyo.minimize)
    model.constraint = pyo.Constraint(expr=model.x[1] + model.x[2] == 1)

    # Print the model
    model.pprint()

    # Solve the model
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True, logfile='glpk_test_log.txt')

    # Print the results
    print("Solver Results:")
    print(results)
    print(f"x[1] = {pyo.value(model.x[1])}")
    print(f"x[2] = {pyo.value(model.x[2])}")

    # Print the solver log
    with open('glpk_test_log.txt', 'r') as log_file:
        solver_log = log_file.read()
    print("Solver log:")
    print(solver_log)

def generate_patterns(width, orders):
    """
    Generates all possible patterns for the Cutting Stock Problem given the width of the roll and the orders.

    Input:
        width: the width of the roll
        orders: list of tuples (width, demand) where width is the width of the order and demand is the demand for that order
    Output:
        patterns: list of tuples (pattern, waste) where pattern is a list of the order width of each order used and waste is the amount of waste
    """ 
    patterns = []
    order_widths = [order[0] for order in orders]

    # Recursive function to generate all possible combinations of orders
    def generate_combinations(current_pattern, remaining_width):
        patterns.append((current_pattern, width - remaining_width))
        for order_width in order_widths:
            if remaining_width >= order_width:
                generate_combinations(current_pattern + [order_width], remaining_width - order_width)

    generate_combinations([], width) # Start with an empty pattern and the full width of the roll

    # Make the second index of the tuple the waste
    patterns_with_waste = [(pattern[0], width - sum(pattern[0])) for pattern in patterns]
    
    return patterns_with_waste

def CuttingStockSolver(orders, patterns, width=50):
    """
    Solves the Cutting Stock Problem using the given orders.

    Input:
        orders: list of tuples (width, demand) where width is the width of the order and demand is the demand for that order
        width: the width of the roll of material
    Output:
        waste: the minimal amount of waste
        rolls: the number of rolls needed
    """
    print("patterns:", patterns)
    # Create linear programming model
    model = pyo.ConcreteModel()

    # Variables are named as the index of the pattern
    pattern_indices = range(len(patterns))
    model.x = pyo.Var(pattern_indices, domain=pyo.NonNegativeIntegers)

    # make sure length of variables is the same as the number of patterns
    assert len(list(model.x)) == len(patterns)

    # Objective function is to minimize the number of rolls
    # Our objective function is the waste of every pattern multiplied by the number of times that pattern is used
    objective_function = sum(model.x[i]*patterns[i][1] for i in pattern_indices)
    model.obj = pyo.Objective(expr=objective_function, sense=pyo.minimize)

    # print the objective function
    #print("Objective Function: ", model.obj.expr)
    
    # Create the list of constraints based with the demand for each order
    model.constraints = pyo.ConstraintList()
    for order_width, demand in combined_orders.items():
        # the sum of the width of each order used in a pattern multiplied by the number of times that pattern is used must be greater than or equal to the demand for that order
        constraint_expr = sum(model.x[i] * patterns[i][0].count(order_width) for i in pattern_indices if patterns[i][0].count(order_width))
        #print("Constraint Expression: ", constraint_expr, "Demand: ", demand)
        model.constraints.add(constraint_expr >= demand)
    

    # print the constraints
    #print("Constraints:")
    #model.constraints.pprint()

    # Solve the model
    solver = pyo.SolverFactory('glpk') # Use the GLPK solver, other options are 'cplex', 'gurobi', cbc, etc.
    results = solver.solve(model, tee=True)

    return results, model


#################################################
## Column Generation for Cutting Stock Problem ##
#################################################

def generate_initial_patterns(orders, width):
    """
    Generate initial patterns for the Cutting Stock Problem using single order and greedy patterns.
    
    Args:
        orders (list of tuple): A list of tuples (width, demand) where width is the width of the order and demand is the demand for that order.
        width (int): The width of the roll of material.
    
    Returns:
        patterns (list of list): A list of initial patterns where each pattern is a list of integers indicating the count of each order width used in the pattern.
    """
    single_patterns = []
    
    # Single order patterns
    for order in orders: 
        pattern = []
        while sum(pattern) < width - order[0]: # For each order, fill a pattern with as many of that order as possible
            pattern.append(order[0])
        
        single_patterns.append(pattern)

    print("Single Order Patterns:", single_patterns)
    
    # Greedy patterns
    greedy_patterns = []
    sorted_orders = sorted(orders, key=lambda x: -x[0])  # Sort orders by width in descending order
    for i in range(len(orders)):
        j = i
        pattern = []
        while sum(pattern) < width:
            if sum(pattern) > width - sorted_orders[j][0]: # check if adding another order will exceed the remaining width
                j +=1
                if j == len(orders):
                    break
                continue
            pattern.append(sorted_orders[j][0])
        
        greedy_patterns.append(pattern)

    print("Greedy Patterns:", greedy_patterns)

    # remove duplicates
    patterns = single_patterns + greedy_patterns
    patterns = [list(x) for i, x in enumerate(patterns) if patterns.index(x) == i] # Remove duplicates

    # get the waste for each pattern
    pattern_waste = [width - sum(pattern) for pattern in patterns]
    patterns = list(zip(patterns, pattern_waste))
    print("Initial Patterns:", patterns)

    return patterns



# Cutting Stock Problem using delayed column generation and the Dantzig-Wolfe decomposition
def CuttingStockColumnGenSolver(orders, width):

    # Generate initial patterns
    initial_patterns = generate_initial_patterns(orders, width)

   
    print("Initial Patterns:", initial_patterns)

    # solve the initial master problem
    results, model = CuttingStockSolver(orders, initial_patterns, width)
    print("Initial Master Problem Results:", results)
    print("Number of Rolls:", sum(pyo.value(model.x[i]) for i in model.x))
    print("Waste:", pyo.value(model.obj))

    # check if the solution is optimal (0 waste)
    if pyo.value(model.obj) == 0:
        print("Optimal Solution Found")
        return results, model
    
    # otherwise, we need to generate columns
    return results, model


if __name__ == "__main__":
    # define parameters
    width = 43

    # Test if pyomo and glpk are correctly installed
    # test_with_glpk()


    # test the pattern generation
    # orders = [(10, 25), (20, 36), (30, 20)]
    # patterns = generate_patterns(width, orders)
    # print("Patterns:", patterns)

    #Test the Cutting Stock Problem
    orders = [(10, 7), (23, 1), (30, 1)]

    # start by combining orders have the same width
    combined_orders = {}
    for order_width, demand in orders:
        if order_width in combined_orders:
            combined_orders[order_width] += demand
        else:
            combined_orders[order_width] = demand

    # Create a list of possible patterns, patterns are tuples of the form (pattern, waste) and pattern is a list of the width of each order used
    patterns = generate_patterns(width, list(combined_orders.items()))
    results, model = CuttingStockSolver(orders, patterns, width)
    print("Solver Results:", results)
    print("Number of Rolls:", sum(pyo.value(model.x[i]) for i in model.x))
    print("Waste:", pyo.value(model.obj))

    # test initial pattern generation
    # initial_patterns = generate_initial_patterns(orders, width)
    # print("Initial Patterns:", initial_patterns)

    # Test the Cutting Stock Problem with Column Generation
    results, model = CuttingStockColumnGenSolver(orders, width)
    print("Solver Results:", results)
    print("Number of Rolls:", sum(pyo.value(model.x[pattern]) for pattern in model.x))
    print("Waste:", pyo.value(model.obj))




    