import numpy as np
import pulp

def LP_solve(objective, constraints):
    """
    Basic linear programming solver using PuLP.
    Input:
        objective: list of coefficients for the objective function.
        constraints: List of tuples, where each tuple contains:
            - A list of coefficients for the constraint.
            - A right-hand side value for the constraint.
            - A string indicating the type of constraint ('<=' or '==').: 
    Output:
        results: dictionary with status, objective value, and variable values.
    """
    # Number of variables
    num_vars = len(objective)

    # Define the problem
    prob = pulp.LpProblem("Block_Angular_Problem", pulp.LpMaximize)

    # Define decision variables
    x = pulp.LpVariable.dicts("x", range(num_vars), lowBound=0)

    # Define the objective function
    prob += pulp.lpSum([objective[i] * x[i] for i in range(num_vars)]), "Objective"

    # Define the constraints
    for idx, (coeffs, rhs, sense) in enumerate(constraints):
        if sense == '<=':
            prob += (pulp.lpSum([coeffs[i] * x[i] for i in range(num_vars)]) <= rhs), f"Constraint_{idx+1}"
        elif sense == '==':
            prob += (pulp.lpSum([coeffs[i] * x[i] for i in range(num_vars)]) == rhs), f"Constraint_{idx+1}"
        else:
            raise ValueError("Unsupported constraint type. Use '<=' or '=='.")

    # Solve the problem
    prob.solve()

    # Get the results
    results = {
        "Status": pulp.LpStatus[prob.status],
        "Objective Value": pulp.value(prob.objective),
        "Variable Values": {v.name: v.varValue for v in prob.variables()}
    }

    return results


def solve_master(objective, sub_columns):
    """
    Solves the master problem using the generated columns from the subproblems.
    """
    master_prob = pulp.LpProblem("MasterProblem", pulp.LpMaximize)
    x_vars = pulp.LpVariable.dicts("x", range(len(sub_columns)), lowBound=0, cat='Continuous')
    
    # Master problem objective
    master_prob += pulp.lpSum([objective[i] * pulp.lpSum([x_vars[j] * sub_columns[j][i] for j in range(len(sub_columns))]) for i in range(len(objective))])
    
    # Constraints ensuring that each column sum up to 1
    for i in range(len(sub_columns[0])):
        master_prob += pulp.lpSum([x_vars[j] * sub_columns[j][i] for j in range(len(sub_columns))]) <= 1, f"Constraint_{i+1}"
    
    master_prob.solve()
    
    return master_prob, {v.name: v.varValue for v in master_prob.variables()}

def solve_subproblem(coeffs, duals, rhs):
    """
    Solves a subproblem defined by coefficients, dual values from the master problem, and RHS value.
    Returns a column (if beneficial) to add to the master problem.
    """
    sub_prob = pulp.LpProblem("Subproblem", pulp.LpMaximize)
    y_vars = pulp.LpVariable.dicts("y", range(len(coeffs)), lowBound=0)
    
    # Objective function of subproblem: Maximize reduced profits
    sub_prob += pulp.lpSum([(coeffs[i] - duals[i]) * y_vars[i] for i in range(len(coeffs))]), "ReducedProfitObjective"
    
    # Constraint of subproblem
    sub_prob += pulp.lpSum([y_vars[i] for i in range(len(coeffs))]) <= rhs, "Constraint"
    
    # Solve the subproblem
    status = sub_prob.solve()
    
    # Debug prints
    print(f"Subproblem status: {pulp.LpStatus[status]}")
    print(f"Subproblem objective: {sub_prob.objective}")
    print(f"Subproblem objective value: {pulp.value(sub_prob.objective)}")
    print(f"Subproblem variable values: {[y_vars[i].varValue for i in range(len(coeffs))]}")
    
    # Check if the solution is beneficial (reduced profit is positive)
    if status == pulp.LpStatusOptimal:
        obj_value = pulp.value(sub_prob.objective)
        if obj_value is not None and obj_value > 1e-5:  # Use a small tolerance to avoid numerical issues
            return [y_vars[i].varValue for i in range(len(coeffs))]
    return None

def dantzig_wolfe_decomposition(objective, constraints, max_iterations=1000, tolerance=1e-8):
    num_vars = len(objective)
    num_constraints = len(constraints)

    # Initialize with identity columns
    sub_columns = [[1 if i == j else 0 for i in range(num_vars)] for j in range(num_vars)]

    for iteration in range(max_iterations):
        # Solve master problem
        master_prob = pulp.LpProblem("Master", pulp.LpMaximize)
        lambda_vars = pulp.LpVariable.dicts("lambda", range(len(sub_columns)), lowBound=0)

        # Objective
        master_prob += pulp.lpSum(objective[i] * pulp.lpSum(lambda_vars[j] * sub_columns[j][i] for j in range(len(sub_columns))) for i in range(num_vars))

        # Constraints
        for i, constraint in enumerate(constraints):
            master_prob += pulp.lpSum(constraint[0][k] * pulp.lpSum(lambda_vars[j] * sub_columns[j][k] for j in range(len(sub_columns))) for k in range(num_vars)) <= constraint[1], f"Constraint_{i}"

        master_prob.solve()

        # Get dual values
        duals = [master_prob.constraints[f"Constraint_{i}"].pi for i in range(num_constraints)]

        # Solve subproblem (pricing problem)
        sub_prob = pulp.LpProblem("Subproblem", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(num_vars), lowBound=0)

        # Subproblem objective: Maximize reduced cost
        sub_prob += pulp.lpSum((objective[i] - pulp.lpSum(duals[j] * constraints[j][0][i] for j in range(num_constraints))) * x[i] for i in range(num_vars))

        # Subproblem constraints
        for i, constraint in enumerate(constraints):
            sub_prob += pulp.lpSum(constraint[0][j] * x[j] for j in range(num_vars)) <= constraint[1], f"SubConstraint_{i}"

        sub_prob.solve()

        # Check if we found a column with positive reduced cost
        if pulp.value(sub_prob.objective) <= tolerance:
            break

        # Add new column to master problem
        new_column = [pulp.value(x[i]) for i in range(num_vars)]
        sub_columns.append(new_column)

    # Recover solution
    solution = [0] * num_vars
    for j, var in lambda_vars.items():
        for i in range(num_vars):
            solution[i] += sub_columns[j][i] * pulp.value(var)

    return {f"x_{i}": solution[i] for i in range(num_vars)}, pulp.value(master_prob.objective)


# Simple Column Generation
def simple_column_generation(objective, constraints, max_iterations=1000, tolerance=1e-8):
    """
    Simple Column Generation. This is a simple column generation method that is used to solve the LP problem. 
    We start with identity columns and then iteratively add new columns to the basis until the objective value does not improve.
    """
    num_vars = len(objective)
    
    # Initialize with identity columns
    # Create an initial set of columns, which are identity columns. Each column corresponds to a variable in the LP.
    columns = [[1 if i == j else 0 for i in range(num_vars)] for j in range(num_vars)]
    
    while True:
        # Solve restricted master problem
        # Create the master problem, which is a linear programming problem to maximize the objective function. The lambda variables are the coefficients of the columns in the objective function.
        master_prob = pulp.LpProblem("Master", pulp.LpMaximize)
        lambda_vars = pulp.LpVariable.dicts("lambda", range(len(columns)), lowBound=0)
        
        # Objective
        # Define the objective function of the master problem as the sum of the objective coefficients times the lambda variables.
        master_prob += pulp.lpSum(objective[i] * pulp.lpSum(lambda_vars[j] * columns[j][i] for j in range(len(columns))) for i in range(num_vars))
        
        # Constraints
        # Add the constraints to the master problem. Each constraint is defined as the sum of the constraint coefficients times the lambda variables.
        for i, constraint in enumerate(constraints):
            master_prob += pulp.lpSum(constraint[0][k] * pulp.lpSum(lambda_vars[j] * columns[j][k] for j in range(len(columns))) for k in range(num_vars)) <= constraint[1], f"Constraint_{i}"
        
        # Solve the master problem
        master_prob.solve()
        
        # Get dual values
        # Extract the dual values (shadow prices) for each constraint in the master problem.
        duals = [master_prob.constraints[f"Constraint_{i}"].pi for i in range(len(constraints))]
        
        # Solve pricing problem
        # Create the pricing problem, which is a linear programming problem to maximize the reduced cost.
        pricing_prob = pulp.LpProblem("Pricing", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(num_vars), lowBound=0)
        
        # Define the objective function of the pricing problem as the sum of the reduced costs times the variables.
        pricing_prob += pulp.lpSum((objective[i] - pulp.lpSum(duals[j] * constraints[j][0][i] for j in range(len(constraints)))) * x[i] for i in range(num_vars))
        
        # Add the constraints to the pricing problem.
        for i, constraint in enumerate(constraints):
            pricing_prob += pulp.lpSum(constraint[0][j] * x[j] for j in range(num_vars)) <= constraint[1]
        
        # Solve the pricing problem
        pricing_prob.solve()
        
        # Calculate the reduced cost
        reduced_cost = pulp.value(pricing_prob.objective)
        
        # If the reduced cost is less than the tolerance, stop the iteration.
        if reduced_cost < tolerance:
            break
        
        # Create a new column from the solution of the pricing problem and add it to the columns list.
        new_column = [pulp.value(x[i]) for i in range(num_vars)]
        columns.append(new_column)
    
    # Recover solution
    # Initialize the solution vector with zeros.
    solution = [0] * num_vars
    # Calculate the final solution by summing the columns weighted by the lambda variables.
    for j, var in lambda_vars.items():
        for i in range(num_vars):
            solution[i] += columns[j][i] * pulp.value(var)
    
    # Return the final solution and the objective value of the master problem.
    return {f"x_{i}": solution[i] for i in range(num_vars)}, pulp.value(master_prob.objective)


def generate_and_compare_LPs(num_LPs):
    """
    Generate a number of linear programs of the block angular format and then solve them using LP_solve and then compare the results to the simple column generation method.
    """
    import random

    # Define the number of variables and constraints for each LP
    num_vars = 10
    num_constraints = 5

    # Counter for LPs with the same objective value
    same_objective_count = 0

    # Generate random LPs
    for lp_index in range(num_LPs):
        # Generate random objective coefficients
        objective_coeffs = [random.randint(1, 10) for _ in range(num_vars)]

        # Generate random constraints
        constraints = []
        for _ in range(num_constraints):
            coeffs = [random.randint(0, 10) for _ in range(num_vars)]
            rhs = random.randint(10, 50)
            constraints.append((coeffs, rhs))

        # Solve the LP using LP_solve
        solution_1 = LP_solve(objective_coeffs, [(c[0], c[1], '<=') for c in constraints])
        objective_1 = solution_1['Objective Value']

        # Solve the LP using simple column generation method
        solution_2, objective_2 = simple_column_generation(objective_coeffs, constraints)
        # Solve the LP using Dantzig-Wolfe decomposition
        solution_3= dantzig_wolfe_decomposition(objective_coeffs, constraints)
        objective_3 = solution_3[1]

        # Print the results
        print(f"LP {lp_index + 1}:")
        print("LP_solve solution:", solution_1)
        print("LP_solve objective:", objective_1)
        print("Simple column generation solution:", solution_2)
        print("Simple column generation objective:", objective_2)
        print("Dantzig-Wolfe decomposition solution:", solution_3)
        print("Dantzig-Wolfe decomposition objective:", objective_3)

        # Check if the objective values are the same (within a small tolerance)
        if abs(objective_1 - objective_2) < 1e-6 and abs(objective_1 - objective_3) < 1e-6:
            same_objective_count += 1

        print()


    # Print the number of LPs with the same objective value
    print(f"Number of LPs with the same objective value: {same_objective_count} out of {num_LPs}")

# Example usage
# Example usage:
objective_coeffs = [8, 5, 6, 9, 7, 9, 6, 5]
constraints_ = [
    ([5, 3, 4, 2, 7, 3, 4, 6], 30, '<='),
    ([7, 3, 6, 0, 0, 0, 0, 0], 15, '<='),
    ([0, 0, 0, 3, 1, 2, 0, 0], 7, '<='),
    ([0, 0, 0, 0, 0, 0, 8, 5], 25, '<=')
]
constraints = [
    ([5, 3, 4, 2, 7, 3, 4, 6], 30),
    ([7, 3, 6, 0, 0, 0, 0, 0], 15),
    ([0, 0, 0, 3, 1, 2, 0, 0], 7),
    ([0, 0, 0, 0, 0, 0, 8, 5], 25)
]

# use the testing function
generate_and_compare_LPs(4)

# solution_1 = LP_solve(objective_coeffs, constraints_)
# solution_2 = dantzig_wolfe_decomposition(objective_coeffs, constraints)
# solution_3, objective_3 = simple_column_generation(objective_coeffs, constraints)
# print("Simple column generation solution:", solution_3)
# print("Simple column generation objective:", objective_3)
# print(solution_1, '\n')
# print(solution_2, '\n')



