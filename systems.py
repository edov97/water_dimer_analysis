TEST_SYSTEMS = {
    'Li+..H2O': """
symmetry c1
no_reorient
no_com
1 1
Li       0.00000000      0.00000000       {R}
--
0 1
O        0.00000000      0.00000000       0.00000000
H        0.75668992      0.00000000       0.00000000
H       -0.75668992      0.00000000       0.00000000
""",
    'He..Li+': """
units bohr
symmetry c1
no_reorient
no_com
0 1
He 0.0 0.0 0.0
--
1 1
Li 0.0 0.0 {R}
                    
""",
    'H2O..H2O': """
units bohr
symmetry c1
no_reorient
no_com
0 1
O    0.000000    0.000000    0.000000
H    0.758000    0.000000    0.504000
H   -0.758000    0.000000    0.504000
--
0 1
O    0.000000    0.000000    {R}
H    0.758000    0.000000    {R_plus}
H   -0.758000    0.000000    {R_plus}
""",
    'F-..H2O':"""
units bohr
symmetry c1
no_reorient
no_com
-1 1
F       -1.23638900       0.01223900       0.00000000  
--
0 1
O        1.19756600      -0.10808700       0.00000000
H        1.41539700       0.82701400       0.00000000
H        0.13483000      -0.08437800       0.00000000

""", # Add {R} to modify distance in this system
    # Add other systems as needed
}
