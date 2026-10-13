"""회로도 그리기 (schemdraw). 함수는 슬라이더 값 dict를 받아 Drawing을 돌려준다."""
import schemdraw
import schemdraw.elements as elm

from circuitlab import eng

schemdraw.use("svg")


def rc_lowpass(p):
    """V1 --R1-- out --+-- (C1 to GND)"""
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=11)
        d += elm.SourceSin().up().label("V1", loc="left")
        d += elm.Line().right().length(0.5)
        d += elm.Resistor().right().label(f"R1\n{eng(p['R'], 'Ω')}")
        d += elm.Dot().label("out", loc="top")
        d.push()
        d += elm.Line().right().length(1)
        d += elm.Dot(open=True).label("V(out)", loc="right")
        d.pop()
        d += elm.Capacitor().down().label(f"C1\n{eng(p['C'], 'F')}", loc="bottom")
        d += elm.Ground()
    return d


def rlc_series(p):
    """V1 --L1--C1-- out --R1-- GND"""
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=11)
        d += elm.SourceSin().up().label("V1", loc="left")
        d += elm.Inductor2().right().label(f"L1\n{eng(p['L'], 'H')}")
        d += elm.Capacitor().right().label(f"C1\n{eng(p['C'], 'F')}")
        d += elm.Dot().label("out", loc="top")
        d.push()
        d += elm.Line().right().length(1)
        d += elm.Dot(open=True).label("V(out)", loc="right")
        d.pop()
        d += elm.Resistor().down().label(f"R1\n{eng(p['R'], 'Ω')}", loc="bottom")
        d += elm.Ground()
    return d
