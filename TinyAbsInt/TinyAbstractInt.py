from typing import List, Tuple
class Interval:
    def __init__(self, lower = float('-inf'), upper = float('inf')):
        self.lower = lower
        self.upper = upper
    def __str__(self):
        return "[{}, {}]".format(self.lower, self.upper)
    def contains(self, val:float)->bool:
        return self.lower<=val and val <=self.upper
    def is_inf(self):
        return self.lower == float('-inf') and self.upper == float('inf')

class Zonotope:
    def __init__(self):
        self.is_inited = False
        self.cs: List[float] = [] #coeffs
        pass
    def __str__(self):
        return str(self.cs)

    def compute_interval(self)->Interval:
        res = Interval()
        lower = self.cs[0] - sum(abs(coeff) for coeff in self.cs[1:])
        upper = self.cs[0] + sum(abs(coeff) for coeff in self.cs[1:])
        res.lower = lower
        res.upper = upper
        return res

    def is_empty(self)->bool:
        return len(self.cs)==0

class Node:
    def __init__(self):
        self._val:float
        self.interval = Interval()
        self.zonotope = Zonotope()
        pass

    def compute_interval(self)->Interval:
        raise NotImplementedError

    def compute_zonotope(self)->Zonotope:
        raise NotImplementedError

    def val(self)->float:
        raise NotImplementedError

class Input(Node):
    def __init__(self, val:float = float('inf')):
        super().__init__()
        self._val = val
    def compute_interval(self)->Interval:
        assert not self.interval.is_inf(), "The input interval is not set!"
        return self.interval
    def compute_zonotope(self)->Zonotope:
        assert not self.zonotope.is_empty(), "The input zonotope is empty!"
        return self.zonotope
    def val(self)->float:
        return self._val

class Linear(Node):
    def __init__(self):
        super().__init__()
        self.addends: List[Tuple[Node, float]] = []
        self.bias:float = 0
    def add_addends(self, addends: List[Tuple[Node, float]]):
        for input, coeff in addends:
            self.addends.append((input, coeff))
    def add_bias(self, bias:float)->None:
        self.bias = bias
    def val(self)->float:
        res:float = 0
        for input, coeff in self.addends:
            res+= input.val()*coeff
        res+=self.bias
        self._val = res
        return self._val
    def compute_interval(self)->Interval:
        addend_mins:List[float] = []
        addend_maxs:List[float] = []
        for input, coeff in self.addends:
            input_interval = input.compute_interval()
            addend_mins.append(min(coeff*input_interval.lower, coeff*input_interval.upper))
            addend_maxs.append(max(coeff*input_interval.lower, coeff*input_interval.upper))
        res = Interval(sum(addend_mins)+self.bias, sum(addend_maxs)+self.bias)
        self.interval = res
        return res
    def compute_zonotope(self)->Zonotope:
        res = Zonotope()
        child_zonotopes = [input.compute_zonotope() for input, coeff in self.addends]
        n_gens = len(child_zonotopes[0].cs)
        new_cs:List[float] = [0]*n_gens
        for gen_id in range(n_gens):
            new_coeff = 0
            for i in range(len(self.addends)):
                new_coeff+=self.addends[i][1]*child_zonotopes[i].cs[gen_id]
            new_cs[gen_id] = new_coeff
        #shift the bias
        new_cs[0]+=self.bias
        res.cs = new_cs
        return res
    
class Relu(Node):
    def __init__(self, input: Node):
        super().__init__()
        self.input = input
    def val(self):
        self._val = max(self.input.val(), 0)
        return self._val
    def compute_interval(self)->Interval:
        input_interval = self.input.compute_interval()
        res = Interval(max(0, input_interval.lower), max(0, input_interval.upper))
        self.interval = res
        return res
    def compute_zonotope(self) -> Zonotope:
        input_zonotope = self.input.compute_zonotope()
        input_interval:Interval = input_zonotope.compute_interval()
        input_lower = input_interval.lower
        input_upper = input_interval.upper
        if input_upper <=0:
            res = Zonotope()
            res.cs = [0]*len(input_zonotope.cs)
            return res
        elif input_lower>0:
            return input_zonotope
        else:
            slope = input_upper/(input_upper - input_lower)
            center = input_upper*(1-slope)/2
            n_gens = len(input_zonotope.cs)
            new_cs:List[float] = [0]*(n_gens+1)
            new_cs[0] = slope*input_zonotope.cs[0] + center
            for i in range(1, len(new_cs)-1):
                new_cs[i] = slope*input_zonotope.cs[i]
            new_cs[-1] = center
            res = Zonotope()
            res.cs = new_cs
            return res


        


def main():
    x0 = Input(); x1 = Input()
    x1 = Input()

    z0 = Linear()
    z0.add_addends([(x0, 4.3744),(x1, -4.8)])
    z1 = Linear()
    z1.add_addends([(x0, -3.8103), (x1, 3.4598)])

    h0 = Relu(z0); h1 = Relu(z1)
    y0 = Linear()
    y0.add_addends([(h0, -3.2355), (h1, -4.8071)])
    y0.add_bias(4.7557)

    y1 = Linear()
    y1.add_addends([(h0 ,3.8986), (h1,4.3353)])
    y1.add_bias(- 5.7399)

    output = Linear()
    output.add_addends([(y0, 1),(y1, -1)])

    #--------------verify using interval abstraction
    x0.interval = Interval(0, 0.3)
    x1.interval = Interval(0, 0.3)
    print(output.compute_interval())
    x0.interval = Interval(0, 0.3)
    x1.interval = Interval(0.7, 1)
    print(output.compute_interval())

    x0.interval = Interval(0.7, 1)
    x1.interval = Interval(0, 0.3)
    print(output.compute_interval())

    x0.interval = Interval(0.7, 1)
    x1.interval = Interval(0.7, 1)
    print(output.compute_interval())

    #-------------verify using zonotope abstraction
    x0.zonotope = Zonotope(); x0.zonotope.cs=[0.15, 0.15, 0]
    x1.zonotope = Zonotope(); x1.zonotope.cs=[0.15, 0, 0.15]
    print(output.compute_zonotope().compute_interval())

    x0.zonotope = Zonotope(); x0.zonotope.cs=[0.15, 0.15, 0]
    x1.zonotope = Zonotope(); x1.zonotope.cs=[0.85, 0, 0.15]
    print(output.compute_zonotope().compute_interval())

    x0.zonotope = Zonotope(); x0.zonotope.cs=[0.85, 0.15, 0]
    x1.zonotope = Zonotope(); x1.zonotope.cs=[0.15, 0, 0.15]
    print(output.compute_zonotope().compute_interval())

    x0.zonotope = Zonotope(); x0.zonotope.cs=[0.85, 0.15, 0]
    x1.zonotope = Zonotope(); x1.zonotope.cs=[0.85, 0, 0.15]
    print(output.compute_zonotope().compute_interval())

if __name__=="__main__":
    main()