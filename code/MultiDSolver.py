# This is a solver for problems that take the form of block angular matrices.
# This specific solver assumes that each block has a single constraint, and that
# each block's constraint has the same number of variables.

import numpy as np
import pulp

def block_angular_solve(objective, constraints):
    """
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

##################################
## Decomposition Implementation ##
##################################


def solve_subproblem(subproblem_def, x_vals):
    """
    Solves a subproblem given the subproblem definition and master problem variable values. In this case, we know the number of variables in the subproblem, and we know there is only one constraint.

    Input:
        subproblem_def: A tuple containing:
            - A list of coefficients for the objective function.
            - A list of coefficients for the constraint.
            - A right-hand side value for the constraint.
        x_vals: A list of values for the variables in the master problem.

    Output:
        new_cut: A list of values for the new cut.
        subproblem_obj: The objective value of the subproblem.
    """
    sub_prob = pulp.LpProblem("Subproblem", pulp.LpMaximize)
    sub_vars = [pulp.LpVariable(f"y_{i}", lowBound=0) for i in range(len(subproblem_def[0]))]
    sub_prob += pulp.lpSum([subproblem_def[0][i] * sub_vars[i] for i in range(len(subproblem_def[0]))]), "Objective"
    
    for coeffs, rhs in zip(subproblem_def[1], subproblem_def[2]):
        sub_prob += (pulp.lpSum([coeffs[i] * sub_vars[i] for i in range(len(coeffs))]) <= rhs)

    sub_prob.solve()
    
    if pulp.LpStatus[sub_prob.status] == "Optimal":
        new_cut = [sub_vars[i].varValue for i in range(len(sub_vars))]
        return new_cut, sub_prob.objective.value()
    else:
        return None, None

def solve_multidivisional_problem_decomposition(objective_coeffs, constraints, subproblem_defs):
    """
    Input:
        objective: list of coefficients for the objective function.
        constraints: List of tuples, where each tuple contains:
            - A list of coefficients for the constraint.
            - A right-hand side value for the constraint.
            - A string indicating the type of constraint ('<=' or '==').: 
    Output:
        results: dictionary with status, objective value, and variable values.
    """
    # Initial Master Problem
    master_prob = pulp.LpProblem("Master_Problem", pulp.LpMaximize)

    # Define initial decision variables for the master problem
    num_vars = len(objective_coeffs)
    x = pulp.LpVariable.dicts("x", range(num_vars), lowBound=0)

    # Objective function for master problem
    master_prob += pulp.lpSum([objective_coeffs[i] * x[i] for i in range(num_vars)]), "Objective"

    # Add master constraints
    for idx, (coeffs, rhs, sense) in enumerate(constraints):
        if sense == '<=':
            master_prob += (pulp.lpSum([coeffs[i] * x[i] for i in range(num_vars)]) <= rhs), f"Constraint_{idx+1}"
        elif sense == '==':
            master_prob += (pulp.lpSum([coeffs[i] * x[i] for i in range(num_vars)]) == rhs), f"Constraint_{idx+1}"
        else:
            raise ValueError("Unsupported constraint type. Use '<=' or '=='.")

    # Benders Decomposition Loop
    benders_cuts = []
    iteration = 0

    while True:
        # Solve the master problem
        master_prob.solve()
        
        if pulp.LpStatus[master_prob.status] != "Optimal":
            break

        x_vals = [x[i].varValue for i in range(num_vars)]

        # Solve subproblems and generate Benders cuts
        new_cuts = []
        for subproblem_def in subproblem_defs:
            new_cut, subproblem_obj = solve_subproblem(subproblem_def, x_vals)
            if new_cut is not None:
                new_cuts.append((new_cut, subproblem_obj))
        
        if not new_cuts:
            break

        # Add new Benders cuts to the master problem
        for new_cut, subproblem_obj in new_cuts:
            benders_cuts.append(new_cut)
            master_prob += (pulp.lpSum(new_cut[i] * x[i] for i in range(num_vars)) <= subproblem_obj), f"Benders_Cut_{iteration}"
            iteration += 1

    # Get the results
    result = {
        "Status": pulp.LpStatus[master_prob.status],
        "Objective Value": pulp.value(master_prob.objective),
        "Variable Values": {v.name: v.varValue for v in master_prob.variables()}
    }

    return result


def main():
    # Example usage:
    objective_coeffs = [8, 5, 6, 9, 7, 9, 6, 5]
    constraints = [
        ([5, 3, 4, 2, 7, 3, 4, 6], 30, '<='),
        ([7, 3, 6, 0, 0, 0, 0, 0], 15, '<='),
        ([0, 0, 0, 3, 1, 2, 0, 0], 7, '<='),
        ([0, 0, 0, 0, 0, 0, 8, 5], 25, '<=')
    ]

    solution = block_angular_solve(objective_coeffs, constraints)
    print(solution)

    return 

if __name__ == "__main__":
    main()