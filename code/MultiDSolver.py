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
    # Initial step: create a trivial solution
    sub_columns = []
    for i, constraint in enumerate(constraints):
        initial_column = [0] * len(constraint[0])
        initial_column[constraint[0].index(max(constraint[0]))] = constraint[1]
        sub_columns.append(initial_column)
        print(f"Initial column for constraint {i + 1}: {initial_column}")
    
    master_prob, master_solution = solve_master(objective, sub_columns)
    print(f"Initial master problem objective value: {pulp.value(master_prob.objective)}")
    
    # Loop to add new columns until no beneficial columns can be added
    iteration = 0
    while iteration < max_iterations:
        # Get dual values for each variable
        duals = [pulp.value(master_prob.constraints[f"Constraint_{i+1}"].pi) for i in range(len(objective))]
        print(f"\nIteration {iteration + 1}")
        print(f"Current dual values: {duals}")
        print(f"Current master objective value: {pulp.value(master_prob.objective)}")
        
        new_columns = []
        for i, constraint in enumerate(constraints):
            print(f"\nSolving subproblem for constraint {i + 1}")
            new_column = solve_subproblem(objective, duals, constraint[1])
            if new_column is not None:
                new_columns.append(new_column)
                print(f"New column found for constraint {i + 1}: {new_column}")
            else:
                print(f"No beneficial column found for constraint {i + 1}")
        
        if not new_columns:
            print("No beneficial columns found. Terminating.")
            break
        
        sub_columns.extend(new_columns)
        new_master_prob, new_master_solution = solve_master(objective, sub_columns)
        
        # Check for convergence
        if abs(pulp.value(new_master_prob.objective) - pulp.value(master_prob.objective)) < tolerance:
            print("Converged. Terminating.")
            break
        
        master_prob, master_solution = new_master_prob, new_master_solution
        iteration += 1

    if iteration == max_iterations:
        print("Maximum iterations reached. Terminating.")
    
    print(f"Final master problem objective value: {pulp.value(master_prob.objective)}")
    print(f"Final master problem status: {pulp.LpStatus[master_prob.status]}")
    
    # Recover original variable values
    original_solution = recover_original_solution(objective, constraints, sub_columns, master_solution)
    
    # Post-processing step: Solve a small LP with the generated columns
    post_prob = pulp.LpProblem("Post-processing", pulp.LpMaximize)
    post_vars = pulp.LpVariable.dicts("x", range(len(objective)), lowBound=0)
    
    # Objective function
    post_prob += pulp.lpSum(objective[i] * post_vars[i] for i in range(len(objective)))
    
    # Constraints
    for i, constraint in enumerate(constraints):
        post_prob += pulp.lpSum(constraint[0][j] * post_vars[j] for j in range(len(objective))) <= constraint[1], f"Constraint_{i+1}"
    
    # Add constraints based on the generated columns
    for i, column in enumerate(sub_columns):
        post_prob += pulp.lpSum(column[j] * post_vars[j] for j in range(len(objective))) <= master_solution[f'x_{i}'] * sum(column), f"Column_{i}"
    
    # Solve the post-processing problem
    post_prob.solve()
    
    # Update the original solution with the post-processed results
    post_processed_solution = {f'x_{i}': pulp.value(post_vars[i]) for i in range(len(objective))}
    
    print("Original Dantzig-Wolfe solution:", original_solution)
    print("Post-processed solution:", post_processed_solution)
    print("Post-processed objective value:", pulp.value(post_prob.objective))
    
    return post_processed_solution

def recover_original_solution(objective, constraints, sub_columns, master_solution):
    original_solution = [0] * len(objective)
    for i, val in master_solution.items():
        col_index = int(i.split('_')[1])
        for j, coeff in enumerate(sub_columns[col_index]):
            original_solution[j] += coeff * val
    return {f'x_{i}': round(val, 8) for i, val in enumerate(original_solution)}


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

solution_1 = LP_solve(objective_coeffs, constraints_)
solution_2 = dantzig_wolfe_decomposition(objective_coeffs, constraints)
solution_3, objective_3 = simple_column_generation(objective_coeffs, constraints)
print("Simple column generation solution:", solution_3)
print("Simple column generation objective:", objective_3)
print(solution_1)
print(solution_2)

# Compute the final objective value of solution_2
final_obj_value = sum(objective_coeffs[i] * solution_2[f'x_{i}'] for i in range(len(objective_coeffs)))

# Print the final objective value
print(f"Final objective value of solution_2: {final_obj_value}")


