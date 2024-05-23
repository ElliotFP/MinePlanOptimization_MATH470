import pyomo.environ as pyo
from itertools import combinations_with_replacement 

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
    # initialize the list of patterns and the list of order widths
    patterns = []
    order_widths = [order[0] for order in orders]

    # Generate all combinations of orders that can fit within the roll width
    for i in range(1, len(order_widths) + 1):
        for combination in combinations_with_replacement(order_widths, i):
            if sum(combination) <= width:
                patterns.append((combination, width - sum(combination)))

    return patterns

def CuttingStockSolver(orders, width=100):
    """
    Solves the Cutting Stock Problem using the given orders.

    Input:
        orders: list of tuples (width, demand) where width is the width of the order and demand is the demand for that order
        width: the width of the roll of material
    Output:
        waste: the minimal amount of waste
        rolls: the number of rolls needed
    """

    # start by combining orders have the same width
    combined_orders = {}
    for order_width, demand in orders:
        if order_width in combined_orders:
            combined_orders[order_width] += demand
        else:
            combined_orders[order_width] = demand

    # Create a list of possible patterns, patterns are tuples of the form (pattern, waste) and pattern is a list of the width of each order used
    patterns = generate_patterns(width, list(combined_orders.items()))

    # Create a concrete model
    # solve optimization problem

    return patterns

if __name__ == "__main__":
    test_with_glpk()
