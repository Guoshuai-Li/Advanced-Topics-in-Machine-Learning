import numpy as np
import requests as rq

# Retrieve answer to challenge for a given query
def query(challenge_id, query_vector, submit=False):
    # Only alphanumeric challenge_id and vector entries in {-1,+1} are allowed:
    assert(challenge_id.isalnum())
    assert(np.max(np.minimum(np.abs(query_vector-1),np.abs(query_vector+1)))==0)

    # if query array is 1d, make it 2d
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1,-1)

    payload = { 'challengeid': challenge_id, 'submit': submit,
                'query': str(query_vector.tolist()) }
    response = rq.post("https://rasmuspagh.pythonanywhere.com/query", data = payload).json()
    if submit == False:
        return np.array(eval(response['result']))
    else:
        return response['result']


challenge_id = 'PISCES1' # identifier for hidden dataset
n = 256 # number of entries in hidden dataset
num_queries = 512 # number of queries to be asked

queries = np.random.choice([-1,+1], size=(num_queries,n)) # Set of random queries
query_results = query(challenge_id, queries)

print(query_results)


from scipy.optimize import linprog

def reconstruction_attack(challenge_id, query_results, queries, submit=True):
    """
    Dinur-Nissim reconstruction attack using linear programming
    to minimize L1 distance to observed noisy query results
    """
    
    n = 256  # database size
    m = len(queries)  # number of queries
    
    # Linear programming formulation for L1 minimization
    # Variables: [x1, x2, ..., x256, s1, s2, ..., sm]
    # where x_i are database entries and s_j are slack variables for |error_j|
    
    # Objective: minimize sum of slack variables (L1 norm of errors)
    c = np.concatenate([np.zeros(n), np.ones(m)])
    
    # Inequality constraints for L1 norm: -s_j <= error_j <= s_j
    A_ub = []
    b_ub = []
    
    for j in range(m):
        # Constraint: sum(query[j] * x) - result[j] <= s[j]
        row1 = np.zeros(n + m)
        row1[:n] = queries[j]
        row1[n + j] = -1
        A_ub.append(row1)
        b_ub.append(query_results[j])
        
        # Constraint: -(sum(query[j] * x) - result[j]) <= s[j]  
        row2 = np.zeros(n + m)
        row2[:n] = -queries[j]
        row2[n + j] = -1
        A_ub.append(row2)
        b_ub.append(-query_results[j])
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: x_i in [-1, 1], s_j >= 0
    bounds = [(-1, 1) for _ in range(n)] + [(0, None) for _ in range(m)]
    
    # Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Extract database variables and round to {-1, 1}
    x_optimal = result.x[:n]
    reconstructed = np.sign(x_optimal)
    reconstructed[reconstructed == 0] = 1  # handle any zeros
    
    return reconstructed

# Execute reconstruction attack
reconstructed = reconstruction_attack(challenge_id, query_results, queries, submit=False)
best_query_result = query(challenge_id, reconstructed, submit=True)
print(f"\nReconstruction attack achieves fraction {(1 + best_query_result / n) / 2} correct values")
