import numpy as np
from gurobipy import *
from matplotlib import pyplot as plt
rnd = np.random
rnd.seed(4)

n=10
xc = rnd.rand(n+1)*200
yc = rnd.rand(n+1)*200
N = np.arange(1,n+1)
V = np.arange(0,n+1)
Q=24
q = {i:rnd.randint(1,10) for i in N}
distance = np.array([[np.hypot(xc[i]-xc[j],yc[i]-yc[j]) for i in V] for j in V ])

md1 = Model('cvrp')
v_A = [(i,j) for i in V for j in V if i!=j]
x = md1.addVars(v_A,vtype=GRB.BINARY)
u = md1.addVars(N, vtype=GRB.CONTINUOUS)

md1.modelSense = GRB.MINIMIZE
md1.setObjective(quicksum(distance[i,j]*x[i,j] for i,j in v_A))
md1.addConstrs(quicksum(x[i,j] for j in V if i!=j)==1 for i in N)
md1.addConstrs(quicksum(x[i,j] for i in V if i!=j)==1 for j in N)
md1.addConstrs((x[i,j]==1)>>(u[i]+q[j]==u[j]) for i,j in v_A if i!=0 and j!=0)
md1.addConstrs(u[i]>=q[i] for i in N)
md1.addConstrs(u[i]<=Q for i in N)
md1.optimize()
solutions =[a for a in v_A if x[a].x>0.9]
print(solutions)


plt.figure(figsize=(10,10))
plt.scatter(xc,yc)
plt.plot(xc[0],yc[0],c='r',marker='s')
for i,j in solutions:
    plt.plot([xc[i],xc[j]],[yc[i],yc[j]],c='g')
plt.show()