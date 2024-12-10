"""
Designed to test memory usages of MASH

Run with following command:
python -m memory_profiler /yunity/arusty/Graph-Manifold-Alignment/mashspud/mashspud/MemoryTests.py
"""

print{"Process running..."}

#Imports 
from Main.test_manifold_algorithms import test_manifold_algorithms as tma
from mashspud.MASH import MASH

#Create Data class to fit the data later
dc = tma("SML2010.csv", split = "random", verbose = 1)

#Fit the data
optimized = MASH(DTM = "kl_optimized")
natural = MASH(DTM = "kl")

optimized.fit(dc.split_A, dc.split_B, dc.anchors[:40])
print("\nFinished fitting the optimized...")
print(f"Optimized scores {optimized.get_scores(n_jobs = 5, n_init = 4)} \n\n")

natural.fit(dc.split_A, dc.split_B, dc.anchors[:40])
print("\nFinished fitting the natural kl...")
print(f"Optimized scores {natural.get_scores(n_jobs = 5, n_init = 4)} \n\n")