import { useMemo, useState } from 'react';
import { C, Controls, Figure, Slider, fmt, scale, useSvgDrag } from './common';

type P = [number, number, 1 | -1];
const BASE: P[] = [
	[-3.2, 1.8, -1], [-2.4, -0.6, -1], [-1.6, 2.6, -1], [-3.6, -1.8, -1], [-1.0, 0.6, -1], [-2.2, 3.6, -1], [-0.6, -2.6, -1],
	[1.8, -1.2, 1], [2.8, 1.4, 1], [3.6, -2.2, 1], [1.2, -3.2, 1], [3.2, 2.8, 1], [2.0, 0.6, 1],
];
const BIAS = 10; // cột hằng để học b (b = BIAS·w₃, phạt b rất nhẹ)

/** SVM lề mềm bằng dual coordinate descent: 0 ≤ αᵢ ≤ C, w = Σ αᵢyᵢx̃ᵢ. */
function train(pts: P[], Cc: number) {
	const xt = pts.map(([a, b]) => [a, b, BIAS]);
	const alpha = new Array(pts.length).fill(0);
	const w = [0, 0, 0];
	for (let ep = 0; ep < 3000; ep++) {
		let change = 0;
		for (let i = 0; i < pts.length; i++) {
			const y = pts[i][2];
			const x = xt[i];
			const G = y * (w[0] * x[0] + w[1] * x[1] + w[2] * x[2]) - 1;
			const q = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
			const na = Math.min(Cc, Math.max(0, alpha[i] - G / q));
			const d = na - alpha[i];
			if (d !== 0) {
				alpha[i] = na;
				for (let k = 0; k < 3; k++) w[k] += d * y * x[k];
				change = Math.max(change, Math.abs(d));
			}
		}
		if (change < 1e-9) break;
	}
	return { w: [w[0], w[1]] as [number, number], b: w[2] * BIAS, alpha };
}

/** Hình 6.9: đường phân tách có lề lớn nhất; chỉ support vector quyết định đường. */
export default function SvmViz() {
	const [logC, setLogC] = useState(1);
	const [extra, setExtra] = useState<[number, number]>([0.4, 1.2]);
	const Cc = 10 ** logC;
	const pts: P[] = useMemo(() => [...BASE, [extra[0], extra[1], 1]], [extra]);
	const { w, b, alpha } = useMemo(() => train(pts, Cc), [pts, Cc]);
	const S = 400, R = 5;
	const X = scale(-R, R, 0, S);
	const Y = scale(-R, R, S, 0);
	const { svgRef, handlers } = useSvgDrag((sx, sy) => setExtra([Math.max(-4.7, Math.min(4.7, -R + (sx / S) * 2 * R)), Math.max(-4.7, Math.min(4.7, R - (sy / S) * 2 * R))]));
	const nw = Math.hypot(w[0], w[1]);
	// đường w·x + b = c
	const line = (c: number) => {
		if (Math.abs(w[1]) > Math.abs(w[0])) {
			const y = (x: number) => (c - b - w[0] * x) / w[1];
			return [X(-R), Y(y(-R)), X(R), Y(y(R))];
		}
		const x = (y: number) => (c - b - w[1] * y) / w[0];
		return [X(x(-R)), Y(-R), X(x(R)), Y(R)];
	};
	const [a0, a1, a2, a3] = line(0);
	const [p0, p1, p2, p3] = line(1);
	const [m0, m1, m2, m3] = line(-1);
	const sv = alpha.map((a) => a > 1e-6);
	const viol = pts.filter(([x, y, c]) => c * (w[0] * x + w[1] * y + b) < 1 - 1e-6).length;
	return (
		<Figure
			title="SVM: lề lớn nhất và support vector"
			caption={
				<>
					Đường đậm: siêu phẳng wᵀx + b = 0. Hai đường nét đứt: hai lề wᵀx + b = ±1, cách nhau 2/‖w‖. Điểm có vòng đen là <b>support vector</b> (αᵢ &gt; 0): nằm trên lề hoặc vi phạm lề. Các điểm khác có thể dời tùy ý (miễn không vào lề) mà đường không đổi.
					<b>Kéo điểm đỏ có viền</b> vào giữa hai lớp: với C lớn, SVM cố xếp đúng nó nên lề co hẹp lại; với C nhỏ, SVM chấp nhận cho nó vi phạm lề để giữ lề rộng.
				</>
			}
		>
			<Controls>
				<Slider label="C" value={logC} min={-1.5} max={2} step={0.05} onChange={setLogC} show={fmt(Cc, Cc < 1 ? 3 : 1)} />
			</Controls>
			<svg ref={svgRef} viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 440 }} role="img" aria-label="Dữ liệu hai lớp, siêu phẳng và lề SVM">
				<polygon
					points={`${p0},${p1} ${p2},${p3} ${m2},${m3} ${m0},${m1}`}
					fill={C.accent}
					opacity={0.1}
				/>
				<line x1={a0} y1={a1} x2={a2} y2={a3} stroke={C.text} strokeWidth={2.5} />
				<line x1={p0} y1={p1} x2={p2} y2={p3} stroke={C.red} strokeDasharray="6 4" strokeWidth={1.5} />
				<line x1={m0} y1={m1} x2={m2} y2={m3} stroke={C.blue} strokeDasharray="6 4" strokeWidth={1.5} />
				{pts.map(([x, y, c], i) => {
					const last = i === pts.length - 1;
					return (
						<g key={i}>
							{sv[i] && <circle cx={X(x)} cy={Y(y)} r={11} fill="none" stroke={C.text} strokeWidth={2} />}
							<circle cx={X(x)} cy={Y(y)} r={last ? 8 : 6} fill={c === 1 ? C.red : C.blue} stroke={last ? C.orange : 'none'} strokeWidth={3} style={last ? { cursor: 'grab' } : undefined} {...(last ? handlers : {})} />
						</g>
					);
				})}
			</svg>
			<div className="viz-kv">
				<span>w = (<b>{fmt(w[0], 3)}</b>; <b>{fmt(w[1], 3)}</b>), b = <b>{fmt(b, 3)}</b></span>
				<span>độ rộng lề 2/‖w‖ = <b>{fmt(2 / nw, 3)}</b></span>
				<span>support vector: <b>{sv.filter(Boolean).length}</b>, điểm vi phạm lề: <b>{viol}</b></span>
			</div>
		</Figure>
	);
}
