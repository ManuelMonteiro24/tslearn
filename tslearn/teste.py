from tslearn.metrics import     min_dist_matrix, build_dist_table
from scipy.stats import norm

print(norm.ppf([float(a) / 3 for a in range(1, 3)], scale=1))
