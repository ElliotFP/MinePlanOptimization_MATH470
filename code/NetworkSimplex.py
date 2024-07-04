## Code for Network Simplex
## Implementation just for reference

import numpy as np
import pulp


# Define the network simplex algorithm
def network_simplex(A, b, c):
    """
    Network Simplex algorithm

    INPUT:
        A: numpy array, the constraint matrix
        b: numpy array, the constraint vector
        c: numpy array, the objective vector

    OUTPUT:
        x: numpy array, the solution vector
    """

    # Initialize the problem
    num_nodes = A.shape[0]
    num_arcs = A.shape[1]

    # Create the problem
    prob = pulp.LpProblem("Network_Simplex", pulp.LpMinimize)

    # Define decision variables
    x = pulp.LpVariable.dicts("x", range(num_arcs), lowBound=0)

    # Define the objective function
    prob += pulp.lpSum([c[i] * x[i] for i in range(num_arcs)]), "Objective"

    # Define the constraints
    for i in range(num_nodes):
        prob += (pulp.lpSum([A[i, j] * x[j] for j in range(num_arcs)]) == b[i]), f"Constraint_{i+1}"

    # Solve the problem
    prob.solve()

    # Get the results
    x_values = np.array([x[i].varValue for i in range(num_arcs)])

    return x_values

# main method for testing
def main():
    # Define the problem
    A = np.array([[1, 1, 1, 0], [1, 0, 0, 1]])
    b = np.array([3, 2])
    c = np.array([1, 2, 3, 4])

    # Solve the problem
    x = network_simplex(A, b, c)

    # code to verify correctness
    # Verify the solution
    # Check if the solution satisfies the constraints
    constraints_satisfied = np.allclose(np.dot(A, x), b)
    
    # Calculate the objective value
    objective_value = np.dot(c, x)
    
    # Print verification results
    print("Constraints satisfied:", constraints_satisfied)
    print("Objective value:", objective_value)
    print(x)

if __name__ == "__main__":
    main()