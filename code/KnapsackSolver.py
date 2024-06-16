# Solver for knapsack subproblem to be used in other files
import numpy as np
import gurobipy as grb

def solve_knapsack(items, capacity):
    """
    Solves the knapsack problem with the given weights, values, and capacity

    INPUTS:
    items: list of tuples, where each tuple is (weight, value).
    capacity: int, the maximum weight the knapsack can hold.

    OUTPUTS:
    total_value: float, the total value of the items in the knapsack.
    selected: list of booleans, indicating whether each item is selected.
    
    """

    # Initialize model
    m = grb.Model("knapsack")

    # Create variables
    selected = []
    for i in range(len(items)):
        selected.append(m.addVar(vtype=grb.GRB.BINARY, name="x%d" % i)
                        )
    m.update()

    # Set objective
    # maximize sum of values of selected items
    m.setObjective(grb.quicksum([items[i][1] * selected[i] for i in range(len(items))]), grb.GRB.MAXIMIZE)

    # Add constraint
    # total weight of selected items <= capacity
    m.addConstr(grb.quicksum([items[i][0] * selected[i] for i in range(len(items))]) <= capacity)

    # Solve
    m.optimize()

    # Get solution
    total_value = m.objVal
    selected = [selected[i].x for i in range(len(items))]

    return total_value, selected

def main():
    items = [(1, 1), (2, 6), (3, 9), (4, 12)]
    capacity = 5
    total_value, selected = solve_knapsack(items, capacity)
    print("Total value:", total_value)
    print("Selected items:", selected)

if __name__ == "__main__":
    main()  