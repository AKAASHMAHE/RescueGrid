RESCUEGRID allocator starter

Files:
- allocator.py: transparent greedy allocation baseline
- data/demo_incidents.csv: synthetic incident candidates and priority scores
- data/resource_inventory.csv: synthetic available inventory

Run from this folder:
    python allocator.py

Expected output files:
- data/allocation_results.csv
- data/unmet_needs.csv

Important:
All incident scores, needs, and inventory values in this starter are synthetic demo inputs.
This is decision support only, not a real dispatch system. Human approval is required.
The greedy algorithm prioritizes incidents by score; it is not guaranteed to find a globally optimal allocation.
