import { useMemo, useState } from 'react';
import { Axes, C, Controls, Figure, SERIES, Slider, curvePath, fmt, scale } from './common';

const target = (x: number) => Math.sin(1.6 * x) + 0.4 * x;
const XR: [number, number] = [-3, 3];

function solve(A: number[][], y: number[]): number[] {
	const m = A[0].length;
	const M = Array.from({ length: m }, () => new Array(m + 1).fill(0));
	A.forEach((row, r) => {
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < m; j++) M[i][j] += row[i] * row[j];
			M[i][m] += row[i] * y[r];
		}
	});
	for (let i = 0; i < m; i++) M[i][i] += 1e-8;
	for (let c = 0; c < m; c++) {
		let p = c;
		for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
		[M[c], M[p]] = [M[p], M[c]];
		for (let r = 0; r < m; r++) {
			if (r === c) continue;
			const f = M[r][c] / M[c][c];
			for (let k = c; k <= m; k++) M[r][k] -= f * M[c][k];
		}
	}
	return M.map((row, i) => row[m] / row[i]);
}

/** Hình 7.1: cộng các “bản lề” ReLU để xấp xỉ một đường cong (định lý xấp xỉ vạn năng, bằng hình). */
export default function ReluSumViz() {
	const [H, setH] = useState(3);
	const [parts, setParts] = useState(true);
	const model = useMemo(() => {
		const knots = Array.from({ length: H }, (_, k) => XR[0] + ((k + 1) * (XR[1] - XR[0])) / (H + 1));
		const xs = Array.from({ length: 121 }, (_, i) => XR[0] + (i * (XR[1] - XR[0])) / 120);
		const feats = (x: number) => [1, x, ...knots.map((c) => Math.max(0, x - c))];
		const co = solve(xs.map(feats), xs.map(target));
		const f = (x: number) => feats(x).reduce((s, v, i) => s + v * co[i], 0);
		const err = Math.sqrt(xs.reduce((s, x) => s + (f(x) - target(x)) ** 2, 0) / xs.length);
		return { knots, co, f, err };
	}, [H]);
	const W = 540, Ht = 320;
	const X = scale(XR[0], XR[1], 30, W - 10);
	const Y = scale(-3, 3, Ht - 20, 10);
	return (
		<Figure
			title="Mỗi nơ-ron ReLU là một “bản lề”; cộng nhiều bản lề thành đường cong"
			caption={
				<>
					Mạng một lớp ẩn H nơ-ron ReLU, một đầu vào, một đầu ra tuyến tính: ŷ = Σₖ vₖ·ReLU(x − cₖ) + ax + d. Mỗi số hạng (nét đứt màu) là một đường gãy đúng một lần tại cₖ; tổng của chúng (đường xanh đậm) là đường gãy khúc với H chỗ gãy.
					Ở đây các chỗ gãy đặt cách đều và các hệ số được chọn bằng bình phương nhỏ nhất để bám hàm mục tiêu (nét đứt xám). Càng nhiều nơ-ron, đường gãy càng bám sát: đó là ý tưởng của <b>định lý xấp xỉ vạn năng</b>.
				</>
			}
		>
			<Controls>
				<Slider label="số nơ-ron ẩn H" value={H} min={1} max={16} step={1} onChange={setH} show={String(H)} />
				<label className="viz-slider">
					<span><input type="checkbox" checked={parts} onChange={(e) => setParts(e.target.checked)} /> vẽ từng số hạng vₖ·ReLU(x − cₖ)</span>
				</label>
			</Controls>
			<svg viewBox={`0 0 ${W} ${Ht}`} role="img" aria-label="Tổng các hàm ReLU xấp xỉ đường cong">
				<Axes x={X} y={Y} xr={XR} yr={[-3, 3]} xticks={[-2, -1, 1, 2]} yticks={[-2, 2]} xlabel="x" />
				<path d={curvePath(target, XR[0], XR[1], X, Y)} stroke={C.muted} strokeWidth={3} strokeDasharray="7 5" fill="none" />
				{parts &&
					model.knots.map((c, k) => (
						<path key={k} d={curvePath((x) => Math.max(-3.2, Math.min(3.2, model.co[k + 2] * Math.max(0, x - c))), XR[0], XR[1], X, Y, 120)} stroke={SERIES[k % SERIES.length]} strokeWidth={1.4} strokeDasharray="4 3" fill="none" />
					))}
				{model.knots.map((c, k) => <line key={`k${k}`} x1={X(c)} x2={X(c)} y1={Y(-3)} y2={Y(-2.8)} stroke={C.text} strokeWidth={2} />)}
				<path d={curvePath(model.f, XR[0], XR[1], X, Y, 400)} stroke={C.blue} strokeWidth={3} fill="none" />
			</svg>
			<div className="viz-kv">
				<span>sai số trung bình (RMS) so với hàm mục tiêu: <b>{fmt(model.err, 4)}</b></span>
				<span>số chỗ gãy = H = {H}</span>
			</div>
		</Figure>
	);
}
