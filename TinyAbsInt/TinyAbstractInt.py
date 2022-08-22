from typing import List, Tuple
class Zonotope:
    def __init__(self):
        pass
    def visualize(self):
        pass
class Interval:
    def __init__(self, lower = float('-inf'), upper = float('inf')):
        self.lower = lower
        self.upper = upper
    def __str__(self):
        return "[{}, {}]".format(self.lower, self.upper)
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

    def val(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self, val:float):
        super().__init__()
        self._val = val
    def compute_interval(self)->Interval:
        return self.interval
    def compute_zonotope(self)->Zonotope:
        return self.zonotope
    def val(self):
        return self._val

class Linear(Node):
    def __init__(self):
        super().__init__()
        self.addends: List[Tuple[Node, float]] = []
        self.bias:float = 0
    def add_addends(self, addends: List[Tuple[Node, float]]):
        for input, coeff in addends:
            self.addends.append((input, coeff))
    def add_bias(self, bias):
        self.bias = bias
    def val(self):
        res = 0
        for input, coeff in self.addends:
            res+= input.val()*coeff
        res+=self.bias
        self._val = res
        return self._val
    def compute_interval(self)->Interval:
        addend_mins = []
        addend_maxs = []
        for input, coeff in self.addends:
            input_interval = input.compute_interval()
            addend_mins.append(min(coeff*input_interval.lower, coeff*input_interval.upper))
            addend_maxs.append(max(coeff*input_interval.lower, coeff*input_interval.upper))
        res = Interval(sum(addend_mins)+self.bias, sum(addend_maxs)+self.bias)
        self.interval = res
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



def main():
    x0 = Input(0.034415998613296694)
    x0.interval = Interval(0, 0.3)
    x1 = Input(0.9091358480144816)
    x1.interval = Interval(0.7, 1)

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

    print(output.val())
    print(output.compute_interval())

if __name__=="__main__":
    main()