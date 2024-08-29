import numpy as np

# A = np.array([[-0.023759296131765542, -0.003708258529909953, 0.024642405904054976, 1],
#               [-0.037834450380501705, -0.006940789717537817, 0.03250796172555894, 1],
#               [-0.05564077024197142, -0.008512550052807735, 0.043645471673180246, 1]])

# B = np.array([[0.008703884097511496, -0.005602503708830586, -0.08137724136636117, 1],
#               [0.0099941096383411, -0.005009100052734907, -0.06672447387654747, 1],
#               [0.008845608517011966, -0.0026530401540408384, -0.04668284143278133, 1]])

# A = A.T
# B = B.T

# [-0.05564077024197142, -0.008512550052807735, 0.043645471673180246]

A = np.array([[0.7120501819289544, 0.04392954406135609, -0.7007529781290959, -0.05564077024197142], 
              [-0.04168725338573012, 0.9989252271481464, 0.02026236616406952, -0.008512550052807735], 
              [0.7008899443588767, 0.014784645277255462, 0.713116191748319, 0.043645471673180246],
              [0, 0, 0, 1]])

B = np.array([[0.9910191384191831, 0.07220684752340058, 0.1125488226243483, 0.00983743891400903], 
              [-0.090103633136492, 0.9824939970729138, 0.1630548346237852, -0.0008402512140499717], 
              [-0.09880486627185474, -0.1717315200227316, 0.9801764541452638, -0.04150584673824161],
              [0, 0, 0, 1]])

test = np.array([[0.9334799979696337, 0.06583527997456279, 0.352534836571793, 0.008703884097511496], 
                 [-0.12275105976258015, 0.98228480592006, 0.14159356566336834, -0.005602503708830586], 
                 [-0.3369677616823208, -0.1754487857489303, 0.9250245689045001, -0.08137724136636117],
                 [0, 0, 0, 1]])

# 计算C*B=A
C = np.dot(A, np.linalg.pinv(B))
print(C)
# print(np.dot(C, test)[:, 3])