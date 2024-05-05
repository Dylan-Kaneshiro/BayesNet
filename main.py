import sys
from functools import reduce

class BayesNet:
    def __init__(self):
        self.nodes = {}
        self.vars = []

    def add_node(self, var_name, parents, cpt):
        """Add a node to the Bayesian Network."""
        self.nodes[var_name] = (parents, cpt)
        if var_name not in self.vars:
            self.vars.append(var_name)
        for parent in parents:
            if parent not in self.vars:
                self.vars.append(parent)

    def get_parents(self, var):
        """Return the parents of a variable."""
        return self.nodes[var][0]

    def get_cpt(self, var):
        """Return the CPT of a variable."""
        return self.nodes[var][1]
    
    def topological_sort(self):
        """Return a list of the variables ordered topologically and alphabetically."""
        visited = {var: False for var in self.vars}
        sorted_vars = []

        while len(sorted_vars) < len(self.vars):
            # Find the set of all unvisited nodes with no unvisited parents
            candidates = [var for var in self.vars if not visited[var] and all(visited[parent] for parent in self.get_parents(var))]

            # Sort the candidates alphabetically and visit the first one
            candidates.sort()
            var = candidates[0]
            visited[var] = True

            # Add the visited node to the sorted list
            sorted_vars.append(var)

        return sorted_vars
    
    def get_children(self, var):
        """Return the children of a variable."""
        return [node for node in self.nodes if var in self.get_parents(node)]

    def topological_sort_elim(self, evidence):
        """Return a list of the variables ordered based on elimination ordering."""
        visited = {var: False for var in self.vars}
        sorted_vars = []

        while len(sorted_vars) < len(self.vars):
            # Find the set of all unvisited nodes with no unvisited children
            candidates = [var for var in self.vars if not visited[var] and all(visited[child] for child in self.get_children(var))]

            # Choose the node with the least amount of parents that are not evidence variables
            candidates.sort(key=lambda var: (len([parent for parent in self.get_parents(var) if parent not in evidence]), var))
            var = candidates[0]
            visited[var] = True

            # Add the visited node to the sorted list
            sorted_vars.append(var)

        return sorted_vars

def parse_bayes_net(filename):
    """Parse the Bayesian network from the specified file."""
    net = BayesNet()
    with open(filename, 'r') as file:
        content = file.read().strip().split('\n\n')
        for block in content:
            lines = block.split('\n')
            if '|' in lines[0]:
                parts = lines[0].split('|')
                child = parts[1].strip()
                parents = parts[0].strip().split()
                cpt = {}
                for line in lines[2:]:
                    values, prob = line.split('|')
                    key = tuple(values.split())
                    cpt[key] = float(prob)
                net.add_node(child, parents, cpt)
            else:
                var = lines[0].split('(')[1].split(')')[0].strip()
                prob = float(lines[0].split('=')[1].strip())
                net.add_node(var, [], {(): prob})
    return net

def enumeration_ask(X, e, bn):
    """The ENUMERATION-ASK function as per the provided algorithm."""
    # Q(X) to hold the distribution over X, initially empty
    Q = {}

    # Iterate over each possible value xi of X (f, t for Boolean variables)
    for xi in ['f', 't']:
        # Extend e with value xi for X
        ex = e.copy()
        ex[X] = xi
        # Calculate the probability distribution for xi
        Q[xi] = enumerate_all(bn.topological_sort(), ex, bn)
    # Normalize the distribution over X
    result = normalize(Q)

    # Print results
    e_string = ', '.join(f"{var} = {value}" for var, value in e.items())
    print()
    print("RESULT:")
    print(f"P({X} = f | {e_string}) = {result['f']}")
    print(f"P({X} = t | {e_string}) = {result['t']}")

def enumerate_all(vars, e, bn):
    """Recursively enumerate over all variables."""
    if not vars:
        return 1.0
    Y = vars[0]
    if Y in e:
        # If Y has a value in e, use it
        result = probability(Y, e[Y], e, bn) * enumerate_all(vars[1:], e, bn)
        print_formatted_output(vars, e, result)
        return result
    else:
        # Sum over all possible values of Y
        result = sum(probability(Y, y, e, bn) * enumerate_all(vars[1:], dict(e, **{Y: y}), bn) for y in ["f", "t"])
        print_formatted_output(vars, e, result)
        return result

def probability(var, value, e, bn):
    """Compute the probability of a variable with its parents' values in e."""
    parents = bn.get_parents(var)
    if parents:
        # If the variable has parents, we need to index into the CPT with the values of the parents
        parent_values = tuple(e[parent] for parent in parents)
        value_true = bn.get_cpt(var)[parent_values]
        if value=="t":
          return value_true
        else:
          return 1 - value_true
    else:
        # If no parents, it's just the probability of the variable's value
        return bn.get_cpt(var)[()] if value=="t" else 1 - bn.get_cpt(var)[()]

def normalize(Q):
    """Normalize a dictionary of probabilities."""
    total = sum(Q.values())
    return {k: v / total for k, v in Q.items()}

def print_formatted_output(vars, e, result):
    # Sort the evidence dictionary for consistent ordering
    sorted_evidence = sorted(e.items())
    # Create the variable part
    var_part = ' '.join(vars)
    # Create the evidence part
    evidence_part = ' '.join(f"{var}={value}" for var, value in sorted_evidence)
    # Format the result with fixed decimal places
    result_part = f"{result:.8f}"
    # Print formatted output
    print(f"{var_part:15} | {evidence_part:25} = {result_part}")

def parse_query(query):
    # Remove the 'P(' and ')' from the query
    query = query[2:-1]

    # Split the query into variable and evidence parts
    parts = query.split('|')

    # The variable 'X' is always the first part
    X = parts[0]

    # The evidence 'e' is an empty dictionary if there's no second part
    e = {}
    if len(parts) > 1:
        # Split the evidence part into individual pieces of evidence
        evidence_parts = parts[1].split(',')

        # For each piece of evidence, split it into variable and value and add it to 'e'
        for evidence in evidence_parts:
            variable, value = evidence.split('=')
            e[variable] = value

    return X, e


# ELIMINATION

import itertools

class Factor:
    def __init__(self, bn, evidence, target="Dylan14"):
        self.bn = bn
        self.target = target
        self.evidence = evidence

        
        if target != "Dylan14":
          # Initialize dependencies to include the target variable and its parents, excluding evidence variables
          self.dependencies = [var for var in [self.target] + self.bn.get_parents(self.target) if var not in self.evidence]

          # Pre-calculate all possible probabilities
          self.probabilities = {}
          for values in itertools.product(['t', 'f'], repeat=len(self.dependencies)):
              values_dict = dict(zip(self.dependencies, values))
              combined_values = {**self.evidence, **values_dict}
              self.probabilities[tuple(sorted(values_dict.items()))] = probability(self.target, combined_values[self.target], combined_values, self.bn)


    def query(self, values):
        # Check that all dependencies have defined values
        if set(values.keys()) != set(self.dependencies):
            raise ValueError("All dependencies must have defined values")

        # Retrieve the pre-calculated probability
        return self.probabilities[tuple(sorted(values.items()))]

    def group_by(self, var):
        # Create a new factor with the same dependencies excluding the grouped variable and the same evidence
        dependencies = [dependency for dependency in self.dependencies if dependency != var]
        new_factor = Factor(self.bn, self.evidence)
        new_factor.dependencies = dependencies

        # Pre-calculate all possible probabilities
        new_factor.probabilities = {}
        for values in itertools.product(['t', 'f'], repeat=len(dependencies)):
            values_dict = dict(zip(dependencies, values))
            combined_values = {**self.evidence, **values_dict}

            # The probability of the new factor is the sum of the probabilities of the original factor for all values of the grouped variable
            prob_sum = 0
            for var_value in ['t', 'f']:
                combined_values[var] = var_value
                prob_sum += self.query({**values_dict, var: var_value})

            new_factor.probabilities[tuple(sorted(values_dict.items()))] = prob_sum

        return new_factor

    @classmethod
    def join(cls, factor1, factor2):
        # Create a new factor with the combined dependencies and evidence
        dependencies = list(set(factor1.dependencies + factor2.dependencies))
        evidence = {**factor1.evidence, **factor2.evidence}

        new_factor = cls(factor1.bn, evidence)
        new_factor.dependencies = dependencies

        # Pre-calculate all possible probabilities
        new_factor.probabilities = {}
        for values in itertools.product(['t', 'f'], repeat=len(dependencies)):
            values_dict = dict(zip(dependencies, values))
            combined_values = {**evidence, **values_dict}

            # The probability of the new factor is the product of the probabilities of the two factors
            prob1 = factor1.query({var: combined_values[var] for var in factor1.dependencies})
            prob2 = factor2.query({var: combined_values[var] for var in factor2.dependencies})
            new_factor.probabilities[tuple(sorted(values_dict.items()))] = prob1 * prob2

        return new_factor
    
    @staticmethod
    def join_all(factors):
        if not factors:
            return None

        result = factors[0]
        for factor in factors[1:]:
            # Find a shared variable to join on
            shared_var = next((var for var in result.dependencies if var in factor.dependencies), None)
            if shared_var is None:
                raise ValueError("No shared variable to join on")

            result = Factor.join(result, factor)

        return result

def variable_elimination(X, e, bn):
    # Step 1: Call topological_sort_elim and save that as the variable order
    var_order = bn.topological_sort_elim(e.keys())

    # Step 2: Create a set of variables that are neither the query variable nor in the evidence and call this set sum_over
    sum_over = set(var_order) - set([X]) - set(e.keys())

    # Initialize the factors list with a factor for each variable in the variable order
    factors = []

    for var in var_order:
        factors.append(Factor(bn, e, var))

        if var in sum_over:
            # Find all factors with the current variable as a dependency
            dependent_factors = [factor for factor in factors if var in factor.dependencies]

            # If the variable is in sum_over, replace the dependent factors with a join_all of them, then group by the target variable
            new_factor = Factor.join_all(dependent_factors).group_by(var)


            # Replace the dependent factors in the factors list with the new factor
            factors = [factor for factor in factors if factor not in dependent_factors] + [new_factor]

        # Print the probabilities to the console
        print(f"----- Variable: {var} -----")
        print("Factors:")
        for fr in factors:
            for values, prob in fr.probabilities.items():
                print(f"{dict(values)}: {prob}")
            print()



    # The final factor is the answer to the query
    final_factor = Factor.join_all(factors)
    print(f"RESULT:")
    result = {values[0][1]:prob for values, prob in final_factor.probabilities.items()}
    result = normalize(result)
    for key, value in result.items():
        print(f"P({X} = {key} | {dict(e)}) = {value}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <path_to_bayesnet_file> <elim|enum> <query>")
        sys.exit(1)

    path_to_bayesnet_file, method, query = sys.argv[1:]
    bayes_net = parse_bayes_net(path_to_bayesnet_file)
    X, e = parse_query(query)

    if method == "enum":
        enumeration_ask(X, e, bayes_net)
    elif method == "elim":
        # Call to variable elimination function
        variable_elimination(X, e, bayes_net)
    else:
        print("Invalid method. Choose 'elim' or 'enum'.")

if __name__ == "__main__":
    main()
