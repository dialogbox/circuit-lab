"""SPICE(ngspice/LTspice) + Jupyter 인터랙티브 대시보드 헬퍼.

흐름: 슬라이더 값 -> .param 줄 생성 -> SPICE 배치 실행 -> .raw 읽기 -> Plotly 그래프 갱신
"""
import html
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import ipywidgets as w
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots
from spicelib import RawRead
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

# 시뮬레이션 엔진: "ngspice"(빠름, 기본) 또는 "ltspice"(macOS에서는 실행당 약 4초)
ENGINE = "ngspice"

_SI = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")]


def eng(x, unit=""):
    """1.59e3 -> '1.59 kΩ' 처럼 사람이 읽기 쉬운 표기."""
    for scale, prefix in _SI:
        if abs(x) >= scale:
            return f"{x / scale:.3g} {prefix}{unit}"
    return f"{x:.3g} {unit}"


def _wait_until_done(raw_path, log_path, timeout=30.0):
    """macOS LTspice는 프로세스가 끝난 뒤에도 .raw/.log를 쓰고 있을 수 있어,
    로그에 종료 표시가 나오고 .raw 크기가 안정될 때까지 기다린다."""
    deadline, last_size, stable = time.time() + timeout, -1, 0
    while time.time() < deadline:
        try:  # LTspice가 파일을 지웠다 다시 만드는 순간이 있어 FileNotFoundError를 무시
            done = "Total elapsed time" in log_path.read_text(errors="ignore")
            size = raw_path.stat().st_size
        except FileNotFoundError:
            done, size = False, -1
        stable = stable + 1 if size == last_size and size > 0 else 0
        last_size = size
        if done and stable >= 2:
            return
        time.sleep(0.05)


def simulate(netlist, **params):
    """넷리스트의 $PARAMS$ 를 `.param` 줄로 치환해 SPICE 엔진을 돌리고 RawRead를 반환."""
    text = Path(netlist).read_text()
    param_line = ".param " + " ".join(f"{k}={v:.6g}" for k, v in params.items())
    with tempfile.TemporaryDirectory() as tmp:
        cir = Path(tmp) / "run.cir"
        cir.write_text(text.replace("$PARAMS$", param_line))
        if ENGINE == "ltspice":
            LTspice.run(cir, timeout=60)
        else:
            NGspiceSimulator.run(cir, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw_path, log = cir.with_suffix(".raw"), cir.with_suffix(".log")
        if ENGINE == "ltspice":
            _wait_until_done(raw_path, log)
        log_text = log.read_text(errors="ignore") if log.exists() else "(로그 없음)"
        files = ", ".join(f"{p.name}({p.stat().st_size}B)" for p in Path(tmp).iterdir())
        if not raw_path.exists():
            raise RuntimeError(f"{ENGINE}가 .raw를 만들지 못함. 생성된 파일: {files}\n{log_text}")
        try:
            result = Result(RawRead(str(raw_path), verbose=False))
        except Exception as e:
            shutil.rmtree("debug_last", ignore_errors=True)
            shutil.copytree(tmp, "debug_last")  # 실패한 실행 결과를 분석용으로 보존
            raise RuntimeError(f"raw 읽기 실패({e}). 생성된 파일: {files}\n--- {ENGINE} 로그 ---\n{log_text}")
        return result


class Result:
    """시뮬레이션 결과(메모리에 완전히 적재됨). 이름은 대소문자 구분 없이 조회."""

    def __init__(self, raw):
        names = raw.get_trace_names()
        # spicelib는 지연 로딩이라, 임시 폴더가 지워지기 전에 모든 트레이스를 읽어 둬야 한다.
        self.waves = {n.lower(): np.asarray(raw.get_trace(n).get_wave(0)) for n in names}
        self.axis = self.waves[names[0].lower()]  # 0번 트레이스가 축(time/frequency)

    def __getitem__(self, name):
        """.op 결과처럼 값이 하나뿐인 트레이스를 실수 스칼라로 반환: res["v(out)"]"""
        return float(np.real(self.waves[name.lower()][0]))


def _trace(res, name):
    return res.waves[name.lower()]


def _axis(res):
    return np.abs(res.axis)


# y 값 변환: 보드 선도용
TRANSFORMS = {
    "raw": lambda y: np.real(y),
    "mag_db": lambda y: 20 * np.log10(np.abs(y) + 1e-30),
    "phase_deg": lambda y: np.degrees(np.angle(y)),
}


def dashboard(netlist, sliders, views, info=None, title="", schematic=None):
    """인터랙티브 대시보드를 화면에 표시.

    sliders: {"R": (최소, 최대, 초기값, "Ω"), ...}  (로그 스케일 슬라이더)
    views:   [{"title": "..", "traces": ["V(out)"], "transform": "mag_db",
               "xlog": True, "xlabel": "Hz", "ylabel": "dB"}, ...]
    info:    함수(params 딕셔너리) -> 설명 HTML 문자열 (예: 차단주파수 계산)
    schematic: 함수(params 딕셔너리) -> schemdraw Drawing (슬라이더 값이 라벨에 반영됨)
    """
    widgets, labels = {}, {}
    for name, (lo, hi, init, unit) in sliders.items():
        widgets[name] = w.FloatLogSlider(
            value=init, base=10, min=np.log10(lo), max=np.log10(hi), step=0.02,
            description=name, readout=False, continuous_update=True,
            layout=w.Layout(width="420px"),
        )
        labels[name] = w.HTML()

    fig = make_subplots(rows=len(views), cols=1, subplot_titles=[v["title"] for v in views],
                        vertical_spacing=0.14)
    for i, v in enumerate(views, start=1):
        for t in v["traces"]:
            fig.add_scatter(x=[], y=[], name=t, row=i, col=1, mode="lines")
        fig.update_xaxes(type="log" if v.get("xlog") else "linear", title_text=v.get("xlabel", ""), row=i, col=1)
        fig.update_yaxes(title_text=v.get("ylabel", ""), row=i, col=1)
    fig.update_layout(height=330 * len(views), margin=dict(t=50, b=40), template="plotly_white")
    fw = go.FigureWidget(fig)

    info_box, status, sch_box = w.HTML(), w.HTML(), w.HTML()
    last = {}

    def refresh(_=None):
        params = {n: s.value for n, s in widgets.items()}
        if params == last.get("p"):   # 이미 그린 값이면 건너뜀 (슬라이더를 빠르게 움직일 때 중복 실행 방지)
            return
        last["p"] = params
        for n, s in widgets.items():
            labels[n].value = f"<b>{eng(s.value, sliders[n][3])}</b>"
        try:
            raw = simulate(netlist, **params)
            x = _axis(raw)
            with fw.batch_update():
                k = 0
                for v in views:
                    fn = TRANSFORMS[v.get("transform", "raw")]
                    for t in v["traces"]:
                        fw.data[k].x, fw.data[k].y = x, fn(_trace(raw, t))
                        k += 1
            status.value = ""
        except Exception as e:  # 초보자가 원인을 볼 수 있게 화면에 표시
            status.value = f"<pre style='color:#c00'>{e}</pre>"
        if info:
            info_box.value = info(params)
        if schematic:
            sch_box.value = schematic(params).get_imagedata("svg").decode()

    for s in widgets.values():
        s.observe(refresh, names="value")

    rows = [w.HBox([widgets[n], labels[n]]) for n in widgets]
    display(w.VBox([w.HTML(f"<h3>{title}</h3>"), sch_box, *rows, info_box, status, fw]))
    refresh()


_ERR = re.compile(r"error|fatal|singular|no such|unknown|can't|cannot|not found|too small|failed|abort|undefined|invalid|unrecognized", re.I)
_BANNER = re.compile(r"^\*{2,}|^Batch mode|^Simulation output|^Comments and warnings|^$")


def check(netlist, **params):
    """수동으로 편집한 .cir 파일을 ngspice로 실행해 로그와 결과 그래프를 바로 보여준다.

    문법/연결 오류는 ngspice 자신이 판단하므로 로그의 빨간 줄(에러)과 주황 줄(경고)을 본다.
    netlist에 $PARAMS$ 자리가 있으면 params(R=1e3, ...)로 채운다.
    """
    text = Path(netlist).read_text()
    if "$PARAMS$" in text:
        text = text.replace("$PARAMS$", ".param " + " ".join(f"{k}={v:.6g}" for k, v in params.items()))
    with tempfile.TemporaryDirectory() as tmp:
        cir, log, raw_path = (Path(tmp) / f"run{s}" for s in (".cir", ".log", ".raw"))
        cir.write_text(text)
        t0 = time.time()
        p = subprocess.run(["ngspice", "-b", "-o", str(log), "-r", str(raw_path), str(cir)],
                           capture_output=True, text=True, timeout=60, cwd=tmp)
        dt = time.time() - t0
        log_text = log.read_text(errors="ignore") if log.exists() else ""
        extra = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not _BANNER.match(l))
        result, plotname = None, ""
        if raw_path.exists():
            try:
                rr = RawRead(str(raw_path), verbose=False)
                plotname = rr.get_raw_property("Plotname") or ""
                result = Result(rr)
            except Exception as e:
                extra += f"\nraw 읽기 실패: {e}"

    lines = [l for l in (log_text + "\n" + extra).splitlines() if l.strip()]
    warns = [l for l in lines if l.lower().lstrip().startswith("warning")]
    errors = [l for l in lines if _ERR.search(l) and l not in warns]
    if result is None:
        head = "<b style='color:#c00'>❌ 실패</b> — 결과(.raw)가 만들어지지 않음"
    elif errors or warns:
        head = f"<b style='color:#c60'>⚠️ 결과는 나왔지만 {'에러' if errors else '경고'} 메시지가 있음</b> ({plotname})"
    else:
        head = f"<b style='color:#080'>✅ 성공</b> — {plotname}, {len(next(iter(result.waves.values())))}포인트, {dt*1000:.0f} ms"

    def fmt(l):
        c = "#c00" if l in errors else "#c60" if l in warns else "inherit"
        return f"<span style='color:{c}'>{html.escape(l)}</span>"

    shown, counts = [], {}
    for l in lines:  # 같은 줄이 반복되면 하나로 접고 횟수 표시
        counts[l] = counts.get(l, 0) + 1
        if counts[l] == 1:
            shown.append(l)
    body = "\n".join(fmt(l) + (f"  <b>×{counts[l]}</b>" if counts[l] > 1 else "") for l in shown)
    display(w.HTML(f"<div>{head}</div><details {'open' if errors or warns or result is None else ''}><summary>ngspice 로그</summary>"
                   f"<pre style='font-size:12px'>{body}</pre></details>"))
    if result is not None:
        _plot_all(result, plotname, Path(netlist).name)
    return None


def _plot_all(res, plotname, title):
    names = list(res.waves)
    xname, rest = names[0], names[1:]
    pn = plotname.lower()
    if "operating point" in pn:
        rows = "".join(f"<tr><td>{n}</td><td style='text-align:right'>{eng(float(np.real(res.waves[n][0])))}</td></tr>" for n in rest)
        display(w.HTML(f"<table>{rows}</table>"))
        return
    volts = [n for n in rest if n.startswith("v(")]
    amps = [n for n in rest if not n.startswith("v(")]
    ac = "ac" in pn
    x = _axis(res)
    if ac:
        fig = make_subplots(rows=2, cols=1, subplot_titles=["크기 (dB)", "위상 (°)"], vertical_spacing=0.14)
        for n in volts:
            y = res.waves[n]
            fig.add_scatter(x=x, y=TRANSFORMS["mag_db"](y), name=n, row=1, col=1)
            fig.add_scatter(x=x, y=TRANSFORMS["phase_deg"](y), name=n, row=2, col=1, showlegend=False)
        fig.update_xaxes(type="log", title_text=xname)
        height = 650
    else:
        rows = 2 if amps else 1
        fig = make_subplots(rows=rows, cols=1, subplot_titles=["전압 (V)", "전류 (A)"][:rows], vertical_spacing=0.14, shared_xaxes=True)
        for n in volts:
            fig.add_scatter(x=x, y=np.real(res.waves[n]), name=n, row=1, col=1)
        for n in amps:
            fig.add_scatter(x=x, y=np.real(res.waves[n]), name=n, row=2, col=1)
        fig.update_xaxes(title_text=xname, row=rows, col=1)
        height = 330 * rows + 60
    fig.update_layout(height=height, margin=dict(t=50, b=40), template="plotly_white", title=title)
    display(go.FigureWidget(fig))


def sweep(netlist, name, values, workers=8, **fixed):
    """파라미터 `name`을 values로 바꿔가며 ngspice를 병렬 실행해 Result 리스트를 반환."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda v: simulate(netlist, **{**fixed, name: v}), values))


def op_explorer(netlist, sliders, outputs, curves, title=""):
    """.op(직류 동작점) 회로를 슬라이더로 탐색: 현재 값 표시 + 파라미터 스윕 곡선.

    sliders: {"V": (최소, 최대, 초기값, "V"), "R": (...)}  (넷리스트의 {V}, {R}에 대응)
    outputs: {"I_D": (함수(res)->값, "A")}  res["v(out)"] 처럼 결과 조회
    curves:  [dict(x="R", y="I_D", lo=10, hi=500e3, log=True, ylog=True), ...]
             x 파라미터만 lo~hi로 훑고, 나머지는 슬라이더 값 고정. 현재 값은 점으로 표시.
    """
    widgets, labels = {}, {}
    for name, (lo, hi, init, unit) in sliders.items():
        widgets[name] = w.FloatLogSlider(value=init, base=10, min=np.log10(lo), max=np.log10(hi), step=0.02,
                                         description=name, readout=False, continuous_update=True,
                                         layout=w.Layout(width="420px"))
        labels[name] = w.HTML()

    fig = make_subplots(rows=len(curves), cols=1, vertical_spacing=0.16,
                        subplot_titles=[f"{c['y']} vs {c['x']}" for c in curves])
    for i, c in enumerate(curves, start=1):
        fig.add_scatter(x=[], y=[], mode="lines", name=c["y"], showlegend=False, row=i, col=1)
        fig.add_scatter(x=[], y=[], mode="markers", marker=dict(size=11, color="#d62728"),
                        name="현재 값", showlegend=False, row=i, col=1)
        fig.update_xaxes(type="log" if c.get("log") else "linear", title_text=f"{c['x']} ({sliders[c['x']][3]})",
                         row=i, col=1)
        fig.update_yaxes(type="log" if c.get("ylog") else "linear", tickformat="~s",
                         title_text=f"{c['y']} ({outputs[c['y']][1]})", row=i, col=1)
    fig.update_layout(height=340 * len(curves), margin=dict(t=50, b=40), template="plotly_white")
    fw = go.FigureWidget(fig)
    table, status = w.HTML(), w.HTML()
    last = {}

    def refresh(_=None):
        params = {n: s.value for n, s in widgets.items()}
        if params == last.get("p"):
            return
        last["p"] = params
        for n, s in widgets.items():
            labels[n].value = f"<b>{eng(s.value, sliders[n][3])}</b>"
        try:
            now = simulate(netlist, **params)
            vals = {k: fn(now) for k, (fn, _) in outputs.items()}
            table.value = "<table>" + "".join(
                f"<tr><td style='padding-right:16px'>{k}</td><td style='text-align:right'><b>{eng(v, outputs[k][1])}</b></td></tr>"
                for k, v in vals.items()) + "</table>"
            with fw.batch_update():
                for i, c in enumerate(curves):
                    xs = np.geomspace(c["lo"], c["hi"], c.get("n", 40)) if c.get("log") else np.linspace(c["lo"], c["hi"], c.get("n", 40))
                    others = {k: v for k, v in params.items() if k != c["x"]}
                    ys = [outputs[c["y"]][0](r) for r in sweep(netlist, c["x"], xs, **others)]
                    fw.data[2 * i].x, fw.data[2 * i].y = xs, ys
                    fw.data[2 * i + 1].x, fw.data[2 * i + 1].y = [params[c["x"]]], [vals[c["y"]]]
            status.value = ""
        except Exception as e:
            status.value = f"<pre style='color:#c00'>{e}</pre>"

    for s in widgets.values():
        s.observe(refresh, names="value")
    display(w.VBox([w.HTML(f"<h3>{title}</h3>"), *[w.HBox([widgets[n], labels[n]]) for n in widgets], table, status, fw]))
    refresh()
