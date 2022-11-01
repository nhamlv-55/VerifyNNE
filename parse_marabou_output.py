import string
from lark import Lark, Transformer, v_args
INPUT='Marabou_output.txt'

"""
Grammar and parser for parsing the assignments
"""
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
    varname = str
    vartype = str
    index = int
    value = float
    lower = float
    upper = float
    def assign_var(self, varname, index, value, vartype, lower, upper):
        return varname, {"index": index, "value": value, "vartype": vartype, "bounds": [lower, upper]}
var_parser = Lark(value_grammar, parser='lalr', transformer = VariableAssignment())

"""
Grammar and parser for parsing the equations
"""
eq_grammar = '''
    start: equation+ ->assign_all_eqs

    equation: linear scalar lhs ->assign_equation

    linear: varname "=" term* ->assign_linear
    term: coeff "*" varname "," ->assign_term
    
    scalar: "scalar = " value ->assign_scalar

    lhs: "lhs = " varname ->assign_lhs 
    coeff: SIGNED_NUMBER
    varname: CNAME
    value: SIGNED_NUMBER
    %import common.CNAME
    %import common.NEWLINE
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS           // Disregard spaces in text
'''
@v_args(inline=True)
class EqAssignment(Transformer):
    varname = str
    coeff = float
    value = float 
    def assign_all_eqs(self, *equation):
        return equation

    def assign_equation(self, linear, scalar, lhs):
        return linear, scalar, lhs

    def assign_scalar(self, value:float):
        return value

    def assign_term(self, coeff, varname):
        return coeff, varname

    def assign_linear(self, varname, *terms):
        all_terms = []        
        for t in terms:
            all_terms.append(t)
        return all_terms
    def assign_lhs(self, varname):
        return varname
eq_parser = Lark(eq_grammar, parser='lalr', transformer = EqAssignment())

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
    if l.startswith("Engine: Linear constraints") or l.startswith("Engine::solve: unsat") and idx>start_eq_idx:
        end_eq_idx = idx
values_lines=raw[start_values_idx:end_values_idx]
eq_lines = raw[start_eq_idx:end_eq_idx]
values = {}
for l in values_lines:
    print(l)
    varname, attributes = var_parser.parse(l.strip())
    values[varname] = attributes

for v in values:
    print(v, "=", values[v]["value"])

eq_raws = "".join(eq_lines)
print(eq_raws)

all_eqs = eq_parser.parse(eq_raws)

#compute eqs using vals
for eq in all_eqs:
    rhs, bias, lhs = eq
    print("computing", rhs, bias)
    rhs_val = 0
    for term in rhs:
        rhs_val += term[0]*values[term[1]]["value"]
    rhs_val+=bias
    print(rhs_val, lhs, values[lhs]["value"]) 