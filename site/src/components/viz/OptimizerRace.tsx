import { useMemo, useState } from 'react';
import { C, Choice, Contours, Controls, Figure, Slider, fmt, scale, useStepper } from './common';

type Fn = 'valley' | 'rosen' | 'scale';
const FN: Record<Fn, { label: string; f: (x: number, y: number) => number; g: (x: number, y: number) => [number, number]; xr: [number, number]; yr: [number, number]; start: [number, number]; opt: [number, number]; levels: number[]; lr: number; alr: number }> = {
	valley: {
		label: 'thung lũng hẹp',
		f: (x, y) => 0.5 * (x * x + 20 * y * y),
		g: (x, y) => [x, 20 * y],
		xr: [-5, 5], yr: [-3, 3], start: [-4.5, 2], opt: [0, 0],
		levels: [0.5, 2, 5, 10, 20, 35, 55, 80], lr: 0.09, alr: 0.3,
	},
	rosen: {
		label: 'Rosenbrock (lòng chảo cong)',
		f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
		g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
		xr: [-2, 2], yr: [-1, 3], start: [-1.5, 2.5], opt: [1, 1],
		levels: [0.2, 0.8, 2, 4, 8, 15, 30, 60], lr: 0.008, alr: 0.08,
	},
	scale: {
		label: 'hai tham số lệch thang đo',
		f: (x, y) => 0.5 * (0.04 * x * x + 4 * y * y),
		g: (x, y) => [0.04 * x, 4 * y],
		xr: [-10, 10], yr: [-4, 4], start: [-9, 3], opt: [0, 0],
		levels: [0.2, 0.8, 2, 4, 8, 13, 18], lr: 0.1, alr: 0.4,
	},
};

const OPTS = [
	['gd', 'GD', C.blue],
	['mom', 'Momentum (β = 0,9)', C.red],
	['rms', 'RMSProp (ρ = 0,9)', C.green],
	['adam', 'Adam', C.purple],
] as const;
type Opt = (typeof OPTS)[number][0];
const MAX = 120;

function run(fn: (typeof FN)[Fn], opt: Opt, lr: number, alr: number) {
	let [x, y] = fn.start;
	const pts: [number, number][] = [[x, y]];
	let vx = 0, vy = 0, sx = 0, sy = 0;
	for (let t = 1; t <= MAX; t++) {
		const [gx, gy] = fn.g(x, y);
		if (opt === 'gd') {
			x -= lr * gx;
			y -= lr * gy;
		} else if (opt === 'mom') {
			vx = 0.9 * vx + gx;
			vy = 0.9 * vy + gy;
			x -= lr * vx;
			y -= lr * vy;
		} else if (opt === 'rms') {
			sx = 0.9 * sx + 0.1 * gx * gx;
			sy = 0.9 * sy + 0.1 * gy * gy;
			x -= (alr * gx) / (Math.sqrt(sx) + 1e-8);
			y -= (alr * gy) / (Math.sqrt(sy) + 1e-8);
		} else {
			vx = 0.9 * vx + 0.1 * gx;
			vy = 0.9 * vy + 0.1 * gy;
			sx = 0.999 * sx + 0.001 * gx * gx;
			sy = 0.999 * sy + 0.001 * gy * gy;
			const mx = vx / (1 - 0.9 ** t), my = vy / (1 - 0.9 ** t);
			const ux = sx / (1 - 0.999 ** t), uy = sy / (1 - 0.999 ** t);
			x -= (alr * mx) / (Math.sqrt(ux) + 1e-8);
			y -= (alr * my) / (Math.sqrt(uy) + 1e-8);
		}
		if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6) break;
		pts.push([x, y]);
	}
	return pts;
}

/** Hình 5.5: bốn optimizer cùng xuất phát trên một mặt mất mát. */
export default function OptimizerRace() {
	const [fk, setFk] = useState<Fn>('valley');
	const F = FN[fk];
	const [lr, setLr] = useState(F.lr);
	const [alr, setAlr] = useState(F.alr);
	const [on, setOn] = useState<Record<Opt, boolean>>({ gd: true, mom: true, rms: true, adam: true });
	const { step, setStep, controls } = useStepper(MAX, 80);
	const paths = useMemo(() => Object.fromEntries(OPTS.map(([k]) => [k, run(F, k, lr, alr)])) as Record<Opt, [number, number][]>, [F, lr, alr]);
	const Wd = 560;
	const Ht = Math.round((Wd * (F.yr[1] - F.yr[0])) / (F.xr[1] - F.xr[0]));
	const X = scale(F.xr[0], F.xr[1], 0, Wd);
	const Y = scale(F.yr[0], F.yr[1], Ht, 0);
	const pick = (k: Fn) => {
		setFk(k);
		setLr(FN[k].lr);
		setAlr(FN[k].alr);
		setStep(0);
	};
	return (
		<Figure
			title="Cuộc đua của các optimizer"
			caption={
				<>
					Cùng điểm xuất phát (chấm đen), cùng số bước. <b>Thung lũng hẹp</b>: GD phải dùng η nhỏ (bị giới hạn bởi hướng dốc đứng) nên đi rất chậm dọc đáy; momentum tích đà dọc đáy; RMSProp/Adam chia bước theo độ lớn gradient của từng tọa độ nên hai hướng đi đều nhau.
					<b> Lệch thang đo</b>: một tham số có gradient nhỏ gấp trăm lần tham số kia — GD gần như đứng yên theo hướng đó, còn Adam thì không. Dấu × là cực tiểu.
				</>
			}
		>
			<Controls>
				<Choice value={fk} onChange={pick} options={(Object.keys(FN) as Fn[]).map((k) => [k, FN[k].label])} />
			</Controls>
			<Controls>
				<Slider label="η cho GD và Momentum" value={lr} min={F.lr / 10} max={F.lr * 2.4} step={F.lr / 20} onChange={setLr} />
				<Slider label="η cho RMSProp và Adam" value={alr} min={F.alr / 10} max={F.alr * 3} step={F.alr / 20} onChange={setAlr} />
			</Controls>
			<div className="viz-choice">
				{OPTS.map(([k, l, c]) => (
					<label key={k} style={{ color: c, fontWeight: 600, marginRight: '0.8rem' }}>
						<input type="checkbox" checked={on[k]} onChange={(e) => setOn({ ...on, [k]: e.target.checked })} /> {l}
					</label>
				))}
			</div>
			{controls}
			<svg viewBox={`0 0 ${Wd} ${Ht}`} role="img" aria-label="Quỹ đạo của bốn optimizer">
				<Contours f={F.f} xr={F.xr} yr={F.yr} levels={F.levels} X={X} Y={Y} n={80} />
				{OPTS.map(([k, , c]) => {
					if (!on[k]) return null;
					const pts = paths[k].slice(0, step + 1);
					const last = pts[pts.length - 1];
					return (
						<g key={k}>
							<polyline points={pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ')} fill="none" stroke={c} strokeWidth={2} strokeLinejoin="round" />
							<circle cx={X(last[0])} cy={Y(last[1])} r={5} fill={c} />
						</g>
					);
				})}
				<text x={X(F.opt[0])} y={Y(F.opt[1]) + 5} textAnchor="middle" fontSize={16} fill={C.text}>×</text>
				<circle cx={X(F.start[0])} cy={Y(F.start[1])} r={5} fill={C.text} />
			</svg>
			<table className="viz-table">
				<thead>
					<tr><th>optimizer</th><th>f sau {step} bước</th><th>khoảng cách tới cực tiểu</th></tr>
				</thead>
				<tbody>
					{OPTS.map(([k, l, c]) => {
						const pts = paths[k];
						const p = pts[Math.min(step, pts.length - 1)];
						const diverged = pts.length - 1 < step;
						return (
							<tr key={k}>
								<td style={{ color: c, fontWeight: 600 }}>{l}</td>
								<td>{diverged ? 'phân kỳ' : fmt(F.f(p[0], p[1]), 4)}</td>
								<td>{diverged ? '—' : fmt(Math.hypot(p[0] - F.opt[0], p[1] - F.opt[1]), 3)}</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</Figure>
	);
}
