import pyomo.environ as pyo

# Testing script to make sure GLPK and pyomo is installed and working
def test_with_glpk():
    model = pyo.ConcreteModel()

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2], sense=pyo.minimize)
    model.constraint = pyo.Constraint(expr=model.x[1] + model.x[2] == 1)

    model.pprint()

    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True, logfile='glpk_test_log.txt')

    print("Solver Results:")
    print(results)
    print(f"x[1] = {pyo.value(model.x[1])}")
    print(f"x[2] = {pyo.value(model.x[2])}")

    with open('glpk_test_log.txt', 'r') as log_file:
        solver_log = log_file.read()
    print("Solver log:")
    print(solver_log)

if __name__ == "__main__":
    test_with_glpk()
