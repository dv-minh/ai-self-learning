import { useState } from 'react';
import { Axes, C, Controls, Figure, Slider, curvePath, fmt, scale } from './common';

const W = 540, H = 340;
const XR: [number, number] = [-5, 5];
// dữ liệu một chiều: (x, nhãn)
const PTS: [number, 0 | 1][] = [
	[-4.2, 0], [-3.1, 0], [-2.5, 0], [-1.6, 0], [-0.9, 0], [0.6, 0],
	[-0.4, 1], [0.9, 1], [1.5, 1], [2.2, 1], [3.0, 1], [4.1, 1],
];
const sig = (z: number) => 1 / (1 + Math.exp(-z));

function fit(): [number, number] {
	let w = 0, b = 0;
	for (let t = 0; t < 4000; t++) {
		let gw = 0, gb = 0;
		for (const [x, y] of PTS) {
			const p = sig(w * x + b);
			gw += (p - y) * x;
			gb += p - y;
		}
		w -= (0.5 * gw) / PTS.length;
		b -= (0.5 * gb) / PTS.length;
	}
	return [w, b];
}
const BEST = fit();

/** Hình 6.2: sigmoid biến điểm số z thành xác suất; log-loss phạt nặng điểm bị đoán sai một cách tự tin. */
export default function LogisticViz() {
	const [w, setW] = useState(0.6);
	const [b, setB] = useState(-1);
	const X = scale(XR[0], XR[1], 30, W - 10);
	const Y = scale(-0.55, 1.55, H - 25, 10);
	const losses = PTS.map(([x, y]) => {
		const p = sig(w * x + b);
		return -Math.log(y === 1 ? p : 1 - p);
	});
	const J = losses.reduce((a, c) => a + c, 0) / PTS.length;
	const boundary = w !== 0 ? -b / w : NaN;
	const LMAX = 3.5;
	return (
		<Figure
			title="Sigmoid, biên quyết định và log-loss từng điểm"
			caption={
				<>
					Điểm lớp 1 nằm ở hàng trên (y = 1), lớp 0 ở hàng dưới (y = 0). Đường cong là p = σ(wx + b): xác suất mô hình gán cho lớp 1. Đường nét đứt đứng là <b>biên quyết định</b> x = −b/w, nơi p = ½.
					Thanh màu cam ở phía ngoài mỗi điểm dài bằng log-loss của điểm đó, −ln(xác suất gán cho nhãn đúng): gần 0 khi đoán đúng chắc chắn, rất dài khi đoán sai mà tự tin. Tăng w làm sigmoid dốc hơn (mô hình “tự tin” hơn).
				</>
			}
		>
			<Controls>
				<Slider label="w" value={w} min={-1} max={4} step={0.05} onChange={setW} />
				<Slider label="b" value={b} min={-5} max={5} step={0.05} onChange={setB} />
			</Controls>
			<div className="viz-buttons">
				<button type="button" className="primary" onClick={() => { setW(BEST[0]); setB(BEST[1]); }}>Khớp (cực tiểu log-loss)</button>
			</div>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đồ thị sigmoid và dữ liệu">
				<Axes x={X} y={Y} xr={XR} yr={[-0.55, 1.55]} xticks={[-4, -2, 2, 4]} yticks={[0.5, 1]} xlabel="x" ylabel="p" />
				<line x1={X(XR[0])} x2={X(XR[1])} y1={Y(0.5)} y2={Y(0.5)} stroke={C.grid} strokeDasharray="3 3" />
				{Number.isFinite(boundary) && boundary > XR[0] && boundary < XR[1] && (
					<line x1={X(boundary)} x2={X(boundary)} y1={Y(-0.55)} y2={Y(1.55)} stroke={C.text} strokeDasharray="6 4" />
				)}
				<path d={curvePath((x) => sig(w * x + b), XR[0], XR[1], X, Y)} stroke={C.accent} strokeWidth={2.5} fill="none" />
				{PTS.map(([x, y], i) => {
					const p = sig(w * x + b);
					const len = Math.min(losses[i], LMAX) * 12;
					const yy = Y(y);
					const dir = y === 1 ? -1 : 1;
					return (
						<g key={i}>
							<line x1={X(x)} x2={X(x)} y1={Y(p)} y2={yy} stroke={C.muted} strokeDasharray="2 2" />
							<rect x={X(x) - 4} y={dir > 0 ? yy + 7 : yy - 7 - len} width={8} height={len} fill={C.orange} opacity={0.8} />
							<circle cx={X(x)} cy={yy} r={6} fill={y === 1 ? C.red : C.blue} stroke={(p >= 0.5 ? 1 : 0) === y ? 'none' : C.text} strokeWidth={2} />
						</g>
					);
				})}
			</svg>
			<div className="viz-kv">
				<span>biên quyết định x = <b>{fmt(boundary, 3)}</b></span>
				<span>log-loss trung bình J = <b>{fmt(J, 4)}</b></span>
				<span>số điểm phân loại sai (viền đen): <b>{PTS.filter(([x, y]) => (sig(w * x + b) >= 0.5 ? 1 : 0) !== y).length}</b></span>
			</div>
		</Figure>
	);
}
