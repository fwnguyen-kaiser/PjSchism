"""
Hungarian label alignment across refits. Cost matrix Cij = ||mu_old_i - mu_new_j||^2. Raises RegimeAlignmentWarning if cost > delta_align.
"""
