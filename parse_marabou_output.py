import string
from lark import Lark, Transformer, v_args
INPUT='Marabou_output.txt'

value_grammar = '''
    start: varname "(index:" index ")" "-->" value "[" vartype "]." "Range:" "[" lower "," upper "]" -> assign_var
    varname: CNAME
    index: SIGNED_NUMBER 
    vartype: CNAME
    value: SIGNED_NUMBER 
    lower: SIGNED_NUMBER 
    upper: SIGNED_NUMBER 

    %import common.CNAME
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS           // Disregard spaces in text
'''
@v_args(inline=True)
class VariableAssignment(Transformer):
    def __init__(self):
        self.vars = {}
    varname = str
    vartype = str
    index = int
    value = float
    lower = float
    upper = float
    def assign_var(self, varname, index, value, vartype, lower, upper):
        return varname, {"index": index, "value": value, "vartype": vartype, "bounds": [lower, upper]}
parser = Lark(value_grammar, parser='lalr', transformer = VariableAssignment())

with open(INPUT, "r") as f:
    raw = f.readlines()
data_lines = []
start_values_idx = -1
end_values_idx = -1
start_eq_idx = -1
end_eq_idx = -1


for idx, l in enumerate(raw):
    if l.startswith("Done with computeAssignment"):
        start_values_idx = idx+2
    if l.startswith("Dumping tableau equations") and idx > start_values_idx:
        end_values_idx = idx
        start_eq_idx = idx+1
    if l.startswith("Engine::solve: unsat query"):
        end_eq_idx = idx
values_lines=raw[start_values_idx:end_values_idx]
eq_lines = raw[start_eq_idx:end_eq_idx]
values = {}
for l in values_lines:
    # print(l)
    varname, attributes = parser.parse(l.strip())
    values[varname] = attributes

print(values)

for l in eq_lines:
    print(l)