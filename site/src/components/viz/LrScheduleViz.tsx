import { useMemo, useState } from 'react';
import { Axes, C, Choice, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

type Kind = 'const' | 'step' | 'exp' | 'cos' | 'warm';
const T = 100;
const W = 560, H = 220;

function lr(kind: Kind, t: number, p: { max: number; min: number; gamma: number; s: number; warm: number }) {
	switch (kind) {
		case 'const':
			return p.max;
		case 'step':
			return p.max * p.gamma ** Math.floor(t / p.s);
		case 'exp':
			return p.max * p.gamma ** (t / p.s);
		case 'cos':
			return p.min + 0.5 * (p.max - p.min) * (1 + Math.cos((Math.PI * t) / T));
		case 'warm':
			if (t < p.warm) return (p.max * t) / p.warm;
			return p.min + 0.5 * (p.max - p.min) * (1 + Math.cos((Math.PI * (t - p.warm)) / (T - p.warm)));
	}
}

/** Hình 5.6: đồ thị ηₜ và tác động lên SGD có nhiễu. */
export default function LrScheduleViz() {
	const [kind, setKind] = useState<Kind>('cos');
	const [max, setMax] = useState(0.5);
	const [min, setMin] = useState(0);
	const [gamma, setGamma] = useState(0.5);
	const [s, setS] = useState(25);
	const [warm, setWarm] = useState(10);
	const p = { max, min, gamma, s, warm };
	const etas = Array.from({ length: T + 1 }, (_, t) => lr(kind, t, p));
	const consts = Array.from({ length: T + 1 }, () => max);
	const sim = useMemo(() => {
		const go = (e: number[]) => {
			const r = rng(11);
			let x = 4;
			const out = [0.5 * x * x];
			for (let t = 0; t < T; t++) {
				x -= e[t] * (x + 1.5 * gauss(r));
				out.push(0.5 * x * x);
			}
			return out;
		};
		return { sched: go(etas), base: go(consts) };
	}, [kind, max, min, gamma, s, warm]);
	const X = scale(0, T, 40, W - 10);
	const Ye = scale(0, max * 1.1, H - 22, 10);
	const Lmax = 8;
	const Yl = scale(0, Lmax, H - 22, 10);
	const line = (arr: number[], Y: (v: number) => number) => arr.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(Math.min(v, Lmax * 1.05)).toFixed(1)}`).join('');
	const tail = (arr: number[]) => arr.slice(-20).reduce((a, b) => a + b, 0) / 20;
	return (
		<Figure
			title="Lịch learning rate và tác động lên SGD có nhiễu"
			caption={
				<>
					Trên: ηₜ theo bước t (đường xám nhạt: η không đổi). Dưới: mất mát của SGD trên f(x) = x²/2 với gradient có nhiễu (x + nhiễu).
					η không đổi: mất mát xuống nhanh rồi <b>dao động mãi</b> ở một “sàn nhiễu” tỷ lệ với η. Lịch giảm dần: lúc đầu đi nhanh như η lớn, cuối cùng η nhỏ nên đáp sát đáy. Warmup làm vài bước đầu rất nhẹ.
				</>
			}
		>
			<Controls>
				<Choice
					value={kind}
					onChange={setKind}
					options={[
						['const', 'không đổi'],
						['step', 'step decay'],
						['exp', 'exponential'],
						['cos', 'cosine'],
						['warm', 'warmup + cosine'],
					]}
				/>
			</Controls>
			<Controls>
				<Slider label="η_max" value={max} min={0.05} max={0.9} step={0.05} onChange={setMax} />
				{(kind === 'cos' || kind === 'warm') && <Slider label="η_min" value={min} min={0} max={max} step={0.01} onChange={setMin} />}
				{(kind === 'step' || kind === 'exp') && <Slider label="γ" value={gamma} min={0.1} max={0.95} step={0.05} onChange={setGamma} />}
				{(kind === 'step' || kind === 'exp') && <Slider label="s (bước/bậc)" value={s} min={5} max={50} step={5} onChange={setS} show={String(s)} />}
				{kind === 'warm' && <Slider label="W (bước warmup)" value={warm} min={1} max={40} step={1} onChange={setWarm} show={String(warm)} />}
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đồ thị learning rate theo bước">
				<Axes x={X} y={Ye} xr={[0, T]} yr={[0, max * 1.1]} xticks={[25, 50, 75, 100]} yticks={[max / 2, max]} xlabel="t" ylabel="ηₜ" />
				<path d={line(consts, Ye)} stroke={C.muted} strokeWidth={1.5} strokeDasharray="5 4" fill="none" />
				<path d={line(etas, Ye)} stroke={C.accent} strokeWidth={2.5} fill="none" />
			</svg>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mất mát của SGD theo bước" style={{ marginTop: '0.5rem' }}>
				<Axes x={X} y={Yl} xr={[0, T]} yr={[0, Lmax]} xticks={[25, 50, 75, 100]} yticks={[2, 4, 6, 8]} xlabel="t" ylabel="mất mát" />
				<path d={line(sim.base, Yl)} stroke={C.muted} strokeWidth={1.5} fill="none" />
				<path d={line(sim.sched, Yl)} stroke={C.red} strokeWidth={2} fill="none" />
			</svg>
			<div className="viz-legend">
				<span style={{ ['--c' as string]: C.muted }}>η không đổi = η_max</span>
				<span style={{ ['--c' as string]: C.red }}>lịch đang chọn</span>
			</div>
			<div className="viz-kv">
				<span>mất mát trung bình 20 bước cuối — η không đổi: <b>{fmt(tail(sim.base), 3)}</b></span>
				<span>lịch đang chọn: <b>{fmt(tail(sim.sched), 3)}</b></span>
			</div>
		</Figure>
	);
}
