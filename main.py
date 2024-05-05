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

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <path_to_bayesnet_file> <elim|enum> <query>")
        sys.exit(1)

    path_to_bayesnet_file, method, query = sys.argv[1:]
    bayes_net = parse_bayes_net(path_to_bayesnet_file)

    if method == "enum":
        X, e = parse_query(query)
        enumeration_ask(X, e, bayes_net)
    elif method == "elim":
        # Call to variable elimination function (not implemented here)
        pass
    else:
        print("Invalid method. Choose 'elim' or 'enum'.")

if __name__ == "__main__":
    main()
