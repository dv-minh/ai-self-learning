import { useMemo, useState } from 'react';
import { Arrow, C, Contours, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

const N = 40;
const S = 400;
const WR: [number, number] = [-2.5, 4.5];
const BR: [number, number] = [-2.5, 4.5];

function makeData() {
	const r = rng(7);
	const xs: number[] = [], ys: number[] = [];
	for (let i = 0; i < N; i++) {
		const x = -1 + 2 * r();
		xs.push(x);
		ys.push(2 * x + 1 + 0.6 * gauss(r));
	}
	return { xs, ys };
}
const DATA = makeData();
const grad = (w: number, b: number, idx: number[]) => {
	let gw = 0, gb = 0;
	for (const i of idx) {
		const e = w * DATA.xs[i] + b - DATA.ys[i];
		gw += e * DATA.xs[i];
		gb += e;
	}
	return [gw / idx.length, gb / idx.length] as const;
};
const ALL = Array.from({ length: N }, (_, i) => i);
const loss = (w: number, b: number) => {
	let s = 0;
	for (let i = 0; i < N; i++) s += (w * DATA.xs[i] + b - DATA.ys[i]) ** 2;
	return s / (2 * N);
};
const BATCHES = [1, 2, 4, 5, 8, 10, 20, 40];

/** Hình 5.3: full-batch GD đi êm, mini-batch SGD đi “lảo đảo” nhưng đúng hướng trung bình. */
export default function SgdViz() {
	const [bi, setBi] = useState(2);
	const [eta, setEta] = useState(0.3);
	const [steps, setSteps] = useState(30);
	const [seed, setSeed] = useState(1);
	const [cloud, setCloud] = useState(true);
	const B = BATCHES[bi];
	const start: [number, number] = [-2, 3.8];
	const paths = useMemo(() => {
		const full: [number, number][] = [start];
		const sgd: [number, number][] = [start];
		const r = rng(seed * 101 + B);
		let order: number[] = [];
		let p = 0;
		for (let t = 0; t < 60; t++) {
			const [w, b] = full[t];
			const [gw, gb] = grad(w, b, ALL);
			full.push([w - eta * gw, b - eta * gb]);
			if (p + B > order.length) {
				order = [...ALL];
				for (let i = N - 1; i > 0; i--) {
					const j = Math.floor(r() * (i + 1));
					[order[i], order[j]] = [order[j], order[i]];
				}
				p = 0;
			}
			const batch = order.slice(p, p + B);
			p += B;
			const [sw, sb] = sgd[t];
			const [hw, hb] = grad(sw, sb, batch);
			sgd.push([sw - eta * hw, sb - eta * hb]);
		}
		return { full, sgd };
	}, [B, eta, seed]);
	const X = scale(WR[0], WR[1], 0, S);
	const Y = scale(BR[0], BR[1], S, 0);
	const lv = [0.3, 0.6, 1, 1.6, 2.5, 4, 6, 9];
	const sp = paths.sgd.slice(0, steps + 1);
	const fp = paths.full.slice(0, steps + 1);
	// mây gradient mini-batch tại điểm bắt đầu
	const cl = useMemo(() => {
		const r = rng(99 + B);
		const out: [number, number][] = [];
		for (let k = 0; k < 40; k++) {
			const idx = [...ALL];
			for (let i = N - 1; i > 0; i--) {
				const j = Math.floor(r() * (i + 1));
				[idx[i], idx[j]] = [idx[j], idx[i]];
			}
			out.push(grad(start[0], start[1], idx.slice(0, B)) as [number, number]);
		}
		return out;
	}, [B]);
	const g0 = grad(start[0], start[1], ALL);
	const k = 0.12;
	const pl = (pts: [number, number][]) => pts.map(([w, b]) => `${X(w).toFixed(1)},${Y(b).toFixed(1)}`).join(' ');
	return (
		<Figure
			title="Full-batch và mini-batch trên cùng mặt mất mát"
			caption={
				<>
					Mặt mất mát L(w, b) của hồi quy tuyến tính trên {N} điểm (các vòng là đường đồng mức). Đường <span style={{ color: C.blue }}>xanh</span>: full-batch, mỗi bước dùng cả {N} điểm.
					Đường <span style={{ color: C.red }}>đỏ</span>: mini-batch B = {B}, mỗi bước chỉ dùng {B} điểm nên rẻ hơn {N / B} lần nhưng lảo đảo. Các mũi tên mảnh màu cam ở điểm bắt đầu là −η·g của 40 mini-batch khác nhau: tản ra quanh mũi tên đen −η·g của full-batch — trung bình thì trùng nhau (không chệch).
					Batch càng nhỏ, mây càng rộng.
				</>
			}
		>
			<Controls>
				<Slider label="cỡ batch B" value={bi} min={0} max={BATCHES.length - 1} step={1} onChange={setBi} show={String(B)} />
				<Slider label="η" value={eta} min={0.05} max={0.9} step={0.05} onChange={setEta} />
				<Slider label="số lần cập nhật" value={steps} min={1} max={60} step={1} onChange={setSteps} show={String(steps)} />
			</Controls>
			<div className="viz-buttons">
				<button type="button" onClick={() => setSeed((s) => s + 1)}>🎲 Xáo dữ liệu lại</button>
				<label>
					<input type="checkbox" checked={cloud} onChange={(e) => setCloud(e.target.checked)} /> hiện mây gradient mini-batch
				</label>
			</div>
			<svg viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 440 }} role="img" aria-label="Quỹ đạo GD và SGD">
				<Contours f={loss} xr={WR} yr={BR} levels={lv} X={X} Y={Y} />
				<text x={S - 6} y={Y(0) - 6} fontSize={12} textAnchor="end" fill={C.muted}>w</text>
				<text x={X(0) + 6} y={14} fontSize={12} fill={C.muted}>b</text>
				<line x1={0} x2={S} y1={Y(0)} y2={Y(0)} stroke={C.grid} />
				<line y1={0} y2={S} x1={X(0)} x2={X(0)} stroke={C.grid} />
				{cloud && cl.map(([gw, gb], i) => <Arrow key={i} x1={X(start[0])} y1={Y(start[1])} x2={X(start[0] - eta * gw * k * 8)} y2={Y(start[1] - eta * gb * k * 8)} color={C.orange} width={1} />)}
				{cloud && <Arrow x1={X(start[0])} y1={Y(start[1])} x2={X(start[0] - eta * g0[0] * k * 8)} y2={Y(start[1] - eta * g0[1] * k * 8)} color={C.text} width={3} />}
				<polyline points={pl(fp)} fill="none" stroke={C.blue} strokeWidth={2.5} />
				<polyline points={pl(sp)} fill="none" stroke={C.red} strokeWidth={1.8} />
				{sp.map(([w, b], i) => <circle key={i} cx={X(w)} cy={Y(b)} r={2.2} fill={C.red} />)}
				<circle cx={X(start[0])} cy={Y(start[1])} r={5} fill={C.text} />
			</svg>
			<div className="viz-kv">
				<span>L sau {steps} bước — full-batch: <b>{fmt(loss(...fp[fp.length - 1]), 4)}</b></span>
				<span>mini-batch: <b>{fmt(loss(...sp[sp.length - 1]), 4)}</b></span>
				<span>số mẫu đã xem — full: <b>{steps * N}</b>, mini-batch: <b>{steps * B}</b></span>
			</div>
		</Figure>
	);
}
