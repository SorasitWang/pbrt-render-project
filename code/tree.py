from grammar import *
from plot import *
from pbrtGen import *


# 3D plots
debug = True


def treeB3d():
    step = 4
    idx_rule = gen_rule(
        [
            {
                "1X": "1(g)(2X)F[+(2X)F][-(2X)F]<2(b)FF[&(2X)F][^(2X)F]&2(b)F",
                "1b": "1(b)(2X)F[+(2X)F][-(2X)F]<2(b)FF[&(2X)F][^(2X)F]&2(b)F",
            }
        ],
        step,
    )
    rule = {}
    rule.update(idx_rule)
    system = LSystem(
        "(1X)",
        rule,
        extra=True,
    )
    for _ in range(step):
        system.step()
        if debug:
            print(system.state)
            print()
    allPos = plot3d(
        system.state, 30, 10, np.array([0, 0, 1.0]), {"g": "g", "b": "#654321"}
    )
    if not debug:
        genPbrtBamboo(allPos, step, True)


def tree3d():

    system = LSystem(
        "(1B)",
        {
            "1B": "1(g)F/[+(2B)][-(2B)][&(2B)]^(2B)*",
            "2B": "2(g)F/[+(3B)][-(3B)][&(3B)]^(3B)*",
            "3B": "3(g)F/[+(4B)][-(4B)][&(4B)]^(4B)*",
            "4B": "4(g)F/[+(5B)][-(5B)][&(5B)]^(5B)*",
            "5B": "5(g)F/[+(6B)][-(6B)][&(6B)]^(6B)*",
            "6B": "6(g)F/[+(7B)][-(7B)][&(7B)]^(7B)*",
            "7B": "7(g)F/[+(8B)][-(8B)][&(8B)]^(8B)*",
            "8B": "8(g)F/[+(9B)][-(9B)][&(9B)]^(9B)*",
            "g": "b",
        },
        extra=True,
    )

    step = 4
    idx_rule = gen_rule(
        [
            {
                "1X": "1(g)F[&+(2X)](2b)[&-(2X)][^(2X)]",
                "1b": "1(b)FF[&+(2X)](2b)FF[&-(2X)][^(2X)]",
            }
        ],
        step,
    )
    rule = {"0X": "bFF(1X)"}
    rule.update(idx_rule)
    system = LSystem(
        "(0X)",
        rule,
        extra=True,
    )
    for _ in range(step):
        system.step()
        if debug:
            print(system.state)
            print()

    allPos = plotA3d(
        system.state,
        step,
        17.5,
        8,
        np.array([0, 0, 1.0]),
        {"g": "g", "b": "#654321"},
        gen=False,
    )
    if not debug:
        genPbrt(allPos, step, True)


if __name__ == "__main__":
    tree3d()
