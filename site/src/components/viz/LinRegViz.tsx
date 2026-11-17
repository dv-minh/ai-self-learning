import { useMemo, useState } from 'react';
import { Axes, C, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

const W = 520, H = 340;
const XR: [number, number] = [0, 10];
const YR: [number, number] = [0, 10];

function makeData() {
	const r = rng(21);
	const xs: number[] = [], ys: number[] = [];
	for (let i = 0; i < 9; i++) {
		const x = 0.8 + i * 1.05 + 0.3 * r();
		xs.push(x);
		ys.push(1.5 + 0.7 * x + 0.9 * gauss(r));
	}
	return { xs, ys };
}
const D = makeData();
const n = D.xs.length;
const mx = D.xs.reduce((a, b) => a + b, 0) / n;
const my = D.ys.reduce((a, b) => a + b, 0) / n;
const wOLS = D.xs.reduce((s, x, i) => s + (x - mx) * (D.ys[i] - my), 0) / D.xs.reduce((s, x) => s + (x - mx) ** 2, 0);
const bOLS = my - wOLS * mx;

/** Hình 6.1: đường thẳng, phần dư và “bình phương” của phần dư được vẽ thành ô vuông thật. */
export default function LinRegViz() {
	const [w, setW] = useState(0.2);
	const [b, setB] = useState(4);
	const [squares, setSquares] = useState(true);
	const X = scale(XR[0], XR[1], 30, W - 10);
	const Y = scale(YR[0], YR[1], H - 25, 10);
	const res = useMemo(() => D.xs.map((x, i) => w * x + b - D.ys[i]), [w, b]);
	const sse = res.reduce((s, e) => s + e * e, 0);
	const sseOLS = D.xs.reduce((s, x, i) => s + (wOLS * x + bOLS - D.ys[i]) ** 2, 0);
	return (
		<Figure
			title="Bình phương nhỏ nhất: làm tổng diện tích các ô vuông nhỏ nhất"
			caption={
				<>
					Mỗi đoạn thẳng đứng là một phần dư eᵢ = ŷᵢ − yᵢ (đỏ: dự đoán cao quá, xanh: thấp quá). Ô vuông dựng trên đoạn đó có diện tích đúng bằng eᵢ². Bình phương nhỏ nhất (least squares) tìm w, b làm <b>tổng diện tích</b> các ô vuông nhỏ nhất.
					Vì bình phương, một phần dư gấp đôi bị phạt gấp bốn: điểm xa đường thẳng có ảnh hưởng rất lớn.
				</>
			}
		>
			<Controls>
				<Slider label="w (độ dốc)" value={w} min={-0.5} max={1.8} step={0.01} onChange={setW} />
				<Slider label="b (hệ số chặn)" value={b} min={-2} max={8} step={0.05} onChange={setB} />
			</Controls>
			<div className="viz-buttons">
				<button type="button" className="primary" onClick={() => { setW(wOLS); setB(bOLS); }}>Khớp bình phương nhỏ nhất</button>
				<label>
					<input type="checkbox" checked={squares} onChange={(e) => setSquares(e.target.checked)} /> vẽ ô vuông eᵢ²
				</label>
			</div>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dữ liệu, đường hồi quy và phần dư">
				<Axes x={X} y={Y} xr={XR} yr={YR} xticks={[2, 4, 6, 8, 10]} yticks={[2, 4, 6, 8, 10]} xlabel="x" ylabel="y" />
				{squares &&
					D.xs.map((x, i) => {
						const e = res[i];
						const side = Math.abs(Y(D.ys[i]) - Y(D.ys[i] + e));
						const top = Math.min(Y(D.ys[i]), Y(D.ys[i] + e));
						return <rect key={`s${i}`} x={X(x)} y={top} width={side} height={side} fill={e > 0 ? C.red : C.blue} fillOpacity={0.15} stroke={e > 0 ? C.red : C.blue} strokeOpacity={0.5} />;
					})}
				<line x1={X(XR[0])} y1={Y(w * XR[0] + b)} x2={X(XR[1])} y2={Y(w * XR[1] + b)} stroke={C.accent} strokeWidth={2.5} />
				{D.xs.map((x, i) => (
					<line key={`r${i}`} x1={X(x)} x2={X(x)} y1={Y(D.ys[i])} y2={Y(w * x + b)} stroke={res[i] > 0 ? C.red : C.blue} strokeWidth={2} />
				))}
				{D.xs.map((x, i) => (
					<circle key={i} cx={X(x)} cy={Y(D.ys[i])} r={5} fill={C.text} />
				))}
			</svg>
			<div className="viz-kv">
				<span>tổng diện tích Σeᵢ² = <b>{fmt(sse, 3)}</b></span>
				<span>MSE = <b>{fmt(sse / n, 3)}</b></span>
				<span>nhỏ nhất có thể (OLS): <b>{fmt(sseOLS, 3)}</b> tại w = {fmt(wOLS, 3)}, b = {fmt(bOLS, 3)}</span>
			</div>
		</Figure>
	);
}
