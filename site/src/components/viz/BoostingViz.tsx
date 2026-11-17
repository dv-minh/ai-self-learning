import { useMemo, useState } from 'react';
import { Axes, C, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

const N = 30, MMAX = 40;
function data() {
	const r = rng(61);
	const xs = Array.from({ length: N }, (_, i) => 0.2 + (9.6 * (i + r() * 0.8)) / N);
	const ys = xs.map((x) => 2 + 1.6 * Math.sin(x * 0.9) + 0.08 * x * x - 0.5 + 0.35 * gauss(r));
	return { xs, ys };
}
const D = data();
type Stump = { t: number; left: number; right: number };
const apply = (s: Stump, x: number) => (x <= s.t ? s.left : s.right);

function fitStump(r: number[]): Stump {
	let best: Stump = { t: D.xs[0], left: 0, right: 0 };
	let bs = Infinity;
	for (let i = 0; i + 1 < N; i++) {
		const t = (D.xs[i] + D.xs[i + 1]) / 2;
		const L = r.filter((_, k) => D.xs[k] <= t), R = r.filter((_, k) => D.xs[k] > t);
		const ml = L.reduce((a, b) => a + b, 0) / L.length, mr = R.reduce((a, b) => a + b, 0) / R.length;
		const sse = L.reduce((s, v) => s + (v - ml) ** 2, 0) + R.reduce((s, v) => s + (v - mr) ** 2, 0);
		if (sse < bs) { bs = sse; best = { t, left: ml, right: mr }; }
	}
	return best;
}

/** Hình 6.8: mỗi cây nông khớp phần dư của tổng các cây trước; cộng một phần ν của nó. */
export default function BoostingViz() {
	const [m, setM] = useState(1);
	const [nu, setNu] = useState(0.5);
	const run = useMemo(() => {
		const F0 = D.ys.reduce((a, b) => a + b, 0) / N;
		const stumps: Stump[] = [];
		let F = D.xs.map(() => F0);
		const mses = [D.ys.reduce((s, y, i) => s + (y - F[i]) ** 2, 0) / N];
		for (let k = 0; k < MMAX; k++) {
			const r = D.ys.map((y, i) => y - F[i]);
			const s = fitStump(r);
			stumps.push(s);
			F = F.map((f, i) => f + nu * apply(s, D.xs[i]));
			mses.push(D.ys.reduce((acc, y, i) => acc + (y - F[i]) ** 2, 0) / N);
		}
		return { F0, stumps, mses };
	}, [nu]);
	const Fm = (x: number, mm: number) => run.F0 + run.stumps.slice(0, mm).reduce((s, st) => s + nu * apply(st, x), 0);
	const W = 520, H = 250;
	const X = scale(0, 10, 30, W - 10);
	const Y = scale(-0.5, 6.5, H - 22, 10);
	const resid = D.xs.map((x, i) => D.ys[i] - Fm(x, m));
	const next = run.stumps[m];
	const H2 = 190;
	const rmax = Math.max(1, ...resid.map(Math.abs)) * 1.15;
	const Y2 = scale(-rmax, rmax, H2 - 10, 10);
	const stepPath = (f: (x: number) => number) => {
		let d = '';
		for (let i = 0; i <= 400; i++) {
			const x = (10 * i) / 400;
			d += `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(f(x)).toFixed(1)}`;
		}
		return d;
	};
	return (
		<Figure
			title="Gradient boosting: sửa lỗi từng chút một"
			caption={
				<>
					Trên: dữ liệu và mô hình hiện tại F_m (bậc thang, vì là tổng các stump). Đoạn đỏ là phần dư rᵢ = yᵢ − F_m(xᵢ). Dưới: chính các phần dư đó, và stump tiếp theo h_(m+1) khớp vào chúng (màu cam: mỗi lá là trung bình phần dư trong lá).
					Bước tiếp theo cộng ν·h_(m+1) vào F. m = 0 chỉ là đường hằng ȳ. ν nhỏ: mỗi bước sửa ít, cần nhiều cây hơn nhưng thường tổng quát hóa tốt hơn.
				</>
			}
		>
			<Controls>
				<Slider label="số cây m" value={m} min={0} max={MMAX - 1} step={1} onChange={setM} show={String(m)} />
				<Slider label="shrinkage ν" value={nu} min={0.1} max={1} step={0.05} onChange={setNu} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dữ liệu và mô hình boosting">
				<Axes x={X} y={Y} xr={[0, 10]} yr={[-0.5, 6.5]} xticks={[2, 4, 6, 8]} yticks={[2, 4, 6]} xlabel="x" ylabel="y" />
				{D.xs.map((x, i) => <line key={`r${i}`} x1={X(x)} x2={X(x)} y1={Y(D.ys[i])} y2={Y(Fm(x, m))} stroke={C.red} strokeWidth={1.5} />)}
				<path d={stepPath((x) => Fm(x, m))} stroke={C.accent} strokeWidth={2.5} fill="none" />
				{D.xs.map((x, i) => <circle key={i} cx={X(x)} cy={Y(D.ys[i])} r={4} fill={C.text} />)}
			</svg>
			<svg viewBox={`0 0 ${W} ${H2}`} role="img" aria-label="Phần dư và stump tiếp theo" style={{ marginTop: '0.4rem' }}>
				<line x1={X(0)} x2={X(10)} y1={Y2(0)} y2={Y2(0)} stroke={C.axis} />
				<text x={X(0) + 4} y={16} fontSize={12} fill={C.text}>phần dư rᵢ (sau {m} cây) và stump h_{m + 1}</text>
				{D.xs.map((x, i) => (
					<g key={i}>
						<line x1={X(x)} x2={X(x)} y1={Y2(0)} y2={Y2(resid[i])} stroke={C.red} strokeWidth={1.5} />
						<circle cx={X(x)} cy={Y2(resid[i])} r={3.5} fill={C.red} />
					</g>
				))}
				{next && (
					<>
						<line x1={X(0)} x2={X(next.t)} y1={Y2(next.left)} y2={Y2(next.left)} stroke={C.orange} strokeWidth={3} />
						<line x1={X(next.t)} x2={X(10)} y1={Y2(next.right)} y2={Y2(next.right)} stroke={C.orange} strokeWidth={3} />
						<line x1={X(next.t)} x2={X(next.t)} y1={10} y2={H2 - 10} stroke={C.orange} strokeDasharray="4 3" />
					</>
				)}
			</svg>
			<div className="viz-kv">
				<span>MSE huấn luyện sau {m} cây: <b>{fmt(run.mses[m], 4)}</b></span>
				<span>sau {m + 1} cây: <b>{fmt(run.mses[m + 1], 4)}</b></span>
			</div>
		</Figure>
	);
}
