from prettytable import PrettyTable
from typing import List, Tuple
class Var:
    def __init__(self, name:int, value:float = float('-inf'), 
                bounds:List[float] = [float('-inf'), float('inf')], 
                is_basic:bool = False, is_slack:bool = False):
        self.name:int = name
        self.bounds:List[float] = bounds
        self.value:float = value
        self.is_basic:bool = is_basic
        self.is_slack:bool = is_slack

    def is_violated(self)->bool:
        return self.is_basic and (self.value > self.bounds[1] or self.value < self.bounds[0])

class Constraint:
    def __init__(self, terms: List[Tuple[float, int]], constant: float):
        self.terms = terms
        self.constant = constant
        self.max_var_idx = max([v for v,_ in terms])
        print(self.max_var_idx)
        self.lhs: List[Tuple[float, int]]
        self.rhs: List[Tuple[float, int]]

    def to_eq_str(self)->str:
        lhs_str:str = "v{}".format(self.lhs[0][1])
        addends:List[str] = []
        for t in self.rhs:
            addends.append("{}*v{} ".format(t[0], t[1]))
        rhs_str=" + ".join(addends)
        return "{} = {}".format(lhs_str, rhs_str)

    def compute_eq(self, vars: List[Var])->None:
        lhs = []
        rhs = []
        for c, v in self.terms:
            if  vars[v].is_basic:
                lhs.append((c,v))
                break
        for c, v in self.terms:
            if not vars[v].is_basic:
                rhs.append((-c/lhs[0][0],v))
        self.lhs = lhs
        self.rhs = rhs



class LPQuery:
    def __init__(self):
        self.vars: List[Var] = []
        self.n_vars: int = 0
        self.n_constraints:int = 0
        self.bounds:List[Tuple[float, float]]

        self.constraints: List[Constraint] = []
        self.simplex_rows:List[List[float]] = []

    def add_constraint(self, terms: List[Tuple[int, float]], constant: float):
        self.constraints.append(Constraint(terms, constant))
        self.n_constraints+=1

    def __str__(self):
        res = ""
        res+="Eqs:\n"
        for c in self.constraints:
            res+=str(c)+"\n"
        
        return  res

    def dump_simplex(self)->str:
        res = ""
        res+="Eqs:\n"
        for r in self.simplex_rows:
            res+=str(r)+"\n"
        res+="Bounds:\n"
        
        for v in self.vars:
            res+="{}:{} {}\n".format("v"+str(v.name), v.value, v.bounds)

        return res

    def to_simplex(self):
        self.n_vars = max([c.max_var_idx for c in self.constraints ])+1
        #add non-slack vars
        for i in range(self.n_vars):
            self.vars.append(Var(name=i, value = 0, is_basic = False, is_slack = False))
        for i, c in enumerate(self.constraints):
            self.vars.append(Var(name = self.n_vars+i, bounds = [c.constant, float('inf')], is_basic = True, is_slack=True))
            new_row = [0]*(self.n_vars + self.n_constraints)
            for c, v in c.terms:
                new_row[v] = c
                new_row[self.n_vars + i] = -1
            self.simplex_rows.append(new_row)

    def is_satisfied(self)->bool:
        """
            check if the current self.assignments satisfy all the constraints
            Only need to check bounds, since all eqs are satisfied
        """
        for v in self.vars:
            if v.value > v.bounds[1] or v.value < v.bounds[0]:
                return False
        return True

    def compute_assignment(self):
        """
        compute the full assignment given a partial assignment
        """
        for r in self.simplex_rows:
            for vi, c in enumerate(r):
                if self.vars[vi].is_basic:
                    assert c==-1 or c==0
                    if c==-1:
                        new_value = 0
                        for vj, coeff in enumerate(r):
                            if not self.vars[vj].is_basic:
                                new_value +=coeff*self.vars[vj].value
                        self.vars[vi].value = new_value

    def get_assignment(self):
        return [v.value for v in self.vars]        

    def pivot(self, vidx:int, vjdx:int):
        print("pivoting {} and {}".format(vidx, vjdx))
        vidx_row = 0
        for ri, r in enumerate(self.simplex_rows):
            if r[vidx]==-1:
                break
        
        #rewrite the vidx_row first
        vjdx_coeff = r[vjdx]
        for cid in range(len(r)):
            r[cid] = r[cid] / (-vjdx_coeff)
        #enter vjdx to the basic set and demote vidx to the non-basic set
        self.vars[vjdx].is_basic = True
        self.vars[vidx].is_basic = False

        #update all other rows that has vj
        for ridx, row in enumerate(self.simplex_rows):
            if ri==ridx: continue
            vjdx_coeff = row[vjdx]*1.0
            for cid in range(len(row)):
                row[cid] =row[cid]+r[cid]*vjdx_coeff

        print(self.dump_simplex())
                
    def solve(self):
        self.compute_assignment()
        print(self.get_assignment())
        while True:
            if self.is_satisfied():
                return self.get_assignment()
            
            #find the first violated bound
            for vidx, vi in enumerate(self.vars):
                if vi.is_violated():
                    print("{}'s bound is violated".format(vidx))
                    #find the corresponding row
                    for ri, r in enumerate(self.simplex_rows):
                        if r[vidx]==-1:
                            break
                    break
            
            if vi.value < vi.bounds[0]:
                #find the first adjustable var
                found_adjustable_var:bool = False
                for vjdx, vj in enumerate(self.vars):
                    if (vj.value < vj.bounds[1] and self.simplex_rows[ri][vjdx]>0) or \
                        (vj.value > vj.bounds[0] and self.simplex_rows[ri][vjdx]<0):
                        found_adjustable_var = True
                        break
                if not found_adjustable_var:
                    return None
                #update vj
                vj.value = vj.value + (vi.bounds[0] - vi.value)/self.simplex_rows[ri][vjdx]

                self.compute_assignment()
                print(self.get_assignment())
            elif vi.value > vi.bounds[1]:
                #find the first adjustable var
                found_adjustable_var:bool = False
                for vjdx, vj in enumerate(self.vars):
                    if (vj.value > vj.bounds[1] and self.simplex_rows[ri][vjdx]>0) or \
                        (vj.value < vj.bounds[0] and self.simplex_rows[ri][vjdx]<0):
                        found_adjustable_var = True
                        break
                if not found_adjustable_var:
                    return None
                #update vj
                vj.value = vj.value + (vi.bounds[1] - vi.value)/self.simplex_rows[ri][vjdx]

                self.compute_assignment()
                print(self.get_assignment())
            self.pivot(vidx, vjdx)
if __name__=="__main__":
    q = LPQuery()
    q.add_constraint([(1,0),(1,1)], 0)
    q.add_constraint([(-2,0), (1,1)], 2)
    q.add_constraint([(-10,0), (1,1)], -5)
    q.to_simplex()
    print(q.dump_simplex())
    res = q.solve()
    target = [-2/3, 2/3, 0, 2, 22/3]
    for i in range(len(res)):
        assert abs(res[i] - target[i]) <=10**-6, print(res[i], "!=", target[i])

    q2 = LPQuery()
    q2.add_constraint([(1,0), (1,1)], 0)
    q2.add_constraint([(-1,0), (-2,1)], 2)
    q2.add_constraint([(-1,0), (1,1)], 1)
    q2.to_simplex()
    print(q2.solve())
    assert q2.solve() is None