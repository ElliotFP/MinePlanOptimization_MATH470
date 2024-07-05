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
def dantzig_wolfe_decomposition(objective, constraints):
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
    max_iterations = 100  # Set a maximum number of iterations to prevent infinite loops
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
        master_prob, master_solution = solve_master(objective, sub_columns)
        iteration += 1
    
    if iteration == max_iterations:
        print("Maximum iterations reached. Terminating.")
    
    print(f"Final master problem objective value: {pulp.value(master_prob.objective)}")
    print(f"Final master problem status: {pulp.LpStatus[master_prob.status]}")
    
    return master_prob, master_solution

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
master_prob, solution_2 = dantzig_wolfe_decomposition(objective_coeffs, constraints)
print(solution_1)
print(solution_2)
print(f"Final master problem objective function: {master_prob.objective}")
# Print the computed final number
final_number = pulp.value(master_prob.objective)
print(f"Computed final number: {final_number}")

