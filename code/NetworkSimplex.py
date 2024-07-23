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
    num_nodes = A.shape[0]  # Number of nodes in the network
    num_arcs = A.shape[1]   # Number of arcs in the network

    # Create the problem
    prob = pulp.LpProblem("Network_Simplex", pulp.LpMinimize)  # Define the LP problem with a minimization objective

    # Define decision variables
    x = pulp.LpVariable.dicts("x", range(num_arcs), lowBound=0)  # Create decision variables for each arc with a lower bound of 0

    # Define the objective function
    prob += pulp.lpSum([c[i] * x[i] for i in range(num_arcs)]), "Objective"  # Objective is to minimize the cost

    # Define the constraints
    for i in range(num_nodes):
        prob += (pulp.lpSum([A[i, j] * x[j] for j in range(num_arcs)]) == b[i]), f"Constraint_{i+1}"  # Add flow conservation constraints for each node

    # Solve the problem
    prob.solve()  # Use the default solver to solve the LP problem

    # Get the results
    x_values = np.array([x[i].varValue for i in range(num_arcs)])  # Extract the values of the decision variables

    return x_values  # Return the solution vector

# main method for testing
def main():
    # Define the problem
    A = np.array([[1, 1, 1, 0], [1, 0, 0, 1]])  # Constraint matrix
    b = np.array([3, 2])  # Constraint vector
    c = np.array([1, 2, 3, 4])  # Objective vector

    # Solve the problem
    x = network_simplex(A, b, c)  # Call the network simplex function to solve the problem

    # code to verify correctness
    # Verify the solution
    # Check if the solution satisfies the constraints
    constraints_satisfied = np.allclose(np.dot(A, x), b)  # Verify if the solution satisfies the constraints
    
    # Calculate the objective value
    objective_value = np.dot(c, x)  # Calculate the objective value using the solution
    
    # Print verification results
    print("Constraints satisfied:", constraints_satisfied)  # Print whether the constraints are satisfied
    print("Objective value:", objective_value)  # Print the objective value
    print(x)  # Print the solution vector

if __name__ == "__main__":
    main()  # Call the main function to execute the test
