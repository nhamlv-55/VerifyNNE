from maraboupy import MarabouCore, MarabouUtils, Marabou
from utils import set_default_bound
ipq: MarabouCore.InputQuery = MarabouCore.InputQuery()
ipq.setNumberOfVariables(3)

#v0>=v1
c1 = MarabouUtils.Equation(MarabouCore.Equation.GE)
c1.addAddend(1, 0)
c1.addAddend(-1, 1)
c1.setScalar(0)
print("c1", c1)
ipq.addEquation(c1.toCoreEquation())

#v2<=v1
c2 = MarabouUtils.Equation(MarabouCore.Equation.LE)
c2.addAddend(1, 2)
c2.addAddend(-1, 1)
c2.setScalar(0)
print("c2", c2)
ipq.addEquation(c2.toCoreEquation())

#set v0 = 1
set_default_bound(ipq, [0], 1, 1)
#set v2 = -1
set_default_bound(ipq, [2], -1, -1)
#compute v1
set_default_bound(ipq, [1], -2, 2)

Marabou.saveQuery(ipq, "threeNumbers")

