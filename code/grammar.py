def extract(state):
    """ "
    input : "1(g)F/[+(2B)]-(2B)"
    output : [g,2B,2B]

    Beware! inside () must be 1 number(optional) + 1 character only
    """
    start = 0
    re = []
    while state.find("(", start) != -1:
        start = state.find("(", start)
        end = state.find(")", start)
        start += 1
        re.append(state[start:end])
    return re


class LSystem:
    def __init__(self, axiom, rules, extra=True):
        # self.alphabet
        self.axiom = axiom
        self.rules = rules
        self.state = self.axiom
        self.extra = extra

    def reset(self):
        self.state = self.axiom

    def step(self):
        if self.extra:
            ex = extract(self.state)
            idx = 0
            newState = ""
            # print(self.state, ex)
            i = 0
            while i < len(self.state):
                if self.state[i] == "(":
                    try:
                        newState += self.rules.get(ex[idx])
                    except:
                        pass
                    i += len(ex[idx]) + 2
                    idx += 1
                else:
                    newState += self.state[i]
                    i += 1
            self.state = newState
        else:
            self.state = "".join(self.rules.get(x, x) for x in self.state)


def gen_rule(init_rules, num):
    """
    for each rule, increment every numeric
    numeric value must in range(1,10) --> 1 digit
    """
    rules = {}
    for _ in range(num):
        new_init_rules = []
        for init_rule in init_rules:
            for key, value in init_rule.items():
                rules[key] = value
                rule = ""
                for c in value:
                    if c.isnumeric():
                        rule += str(int(c) + 1)
                    else:
                        rule += c
                new_key = str(int(key[0]) + 1) + key[1]
                new_init_rules.append({new_key: rule})
            init_rules = new_init_rules
    return rules


# testing

# print(
#     gen_rule(
#         [
#             {
#                 "1X": "1(g)F[&+(2X)](2b)[&-(2X)][^(2X)]",
#                 "1b": "1(g)F[&+(2X)](2b)F[&-(2X)][^(2X)]",
#             }
#         ],
#         5,
#     )
# )
