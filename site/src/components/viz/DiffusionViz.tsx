import { useMemo, useState } from 'react';
import { Axes, C, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

const T = 1000;
const NPTS = 400;

function data() {
	const r = rng(91);
	const pts: [number, number][] = [];
	for (let i = 0; i < NPTS; i++) {
		// xoắn ốc
		const t = 0.5 + 3.2 * r();
		const rad = 0.35 * t;
		pts.push([rad * Math.cos(2.2 * t) + 0.04 * gauss(r), rad * Math.sin(2.2 * t) + 0.04 * gauss(r)]);
	}
	const eps = pts.map(() => [gauss(r), gauss(r)] as [number, number]);
	return { pts, eps };
}
const { pts: X0, eps: EPS } = data();
const abar = (() => {
	const out = [1];
	let a = 1;
	for (let t = 1; t <= T; t++) {
		const beta = 1e-4 + ((0.02 - 1e-4) * (t - 1)) / (T - 1);
		a *= 1 - beta;
		out.push(a);
	}
	return out;
})();

/** Hình 8.9: khuếch tán thuận x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε trên một tập điểm 2D hình xoắn ốc. */
export default function DiffusionViz() {
	const [t, setT] = useState(0);
	const a = abar[t];
	const sa = Math.sqrt(a), sn = Math.sqrt(1 - a);
	const xt = useMemo(() => X0.map(([x, y], i) => [sa * x + sn * EPS[i][0], sa * y + sn * EPS[i][1]]), [t]);
	const S = 300, R = 3;
	const X = scale(-R, R, 0, S);
	const Y = scale(-R, R, S, 0);
	const W2 = 300, H2 = 300;
	const X2 = scale(0, T, 35, W2 - 10);
	const Y2 = scale(0, 1, H2 - 25, 15);
	const curve = (f: (s: number) => number) => abar.map((_, s) => `${s ? 'L' : 'M'}${X2(s).toFixed(1)},${Y2(f(s)).toFixed(1)}`).filter((_, s) => s % 5 === 0).join('');
	return (
		<Figure
			title="Khuếch tán thuận: tín hiệu nhạt dần, nhiễu đậm dần"
			caption={
				<>
					Trái: {NPTS} điểm dữ liệu (một “ảnh” 2 chiều hình xoắn ốc) ở bước t. Mỗi điểm có một vector nhiễu ε cố định; x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε. Phải: hai hệ số theo t với lịch β tuyến tính (DDPM, T = {T}).
					t nhỏ: gần như dữ liệu gốc; t lớn: chỉ còn đám mây Gauss tròn, mất hết cấu trúc. Mô hình khuếch tán học <b>đi ngược</b> quá trình này: từ nhiễu thuần dự đoán ε để khử dần.
				</>
			}
		>
			<Controls>
				<Slider label="bước t" value={t} min={0} max={T} step={10} onChange={setT} show={String(t)} />
			</Controls>
			<div className="viz-row">
				<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Điểm dữ liệu sau khi thêm nhiễu">
					<line x1={0} x2={S} y1={Y(0)} y2={Y(0)} stroke={C.grid} />
					<line y1={0} y2={S} x1={X(0)} x2={X(0)} stroke={C.grid} />
					{xt.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={2.2} fill={C.accent} opacity={0.75} />)}
				</svg>
				<svg viewBox={`0 0 ${W2} ${H2}`} role="img" aria-label="Hệ số tín hiệu và nhiễu theo bước">
					<Axes x={X2} y={Y2} xr={[0, T]} yr={[0, 1]} xticks={[250, 500, 750, 1000]} yticks={[0.5, 1]} xlabel="t" grid={false} />
					<path d={curve((s) => Math.sqrt(abar[s]))} stroke={C.blue} strokeWidth={2.5} fill="none" />
					<path d={curve((s) => Math.sqrt(1 - abar[s]))} stroke={C.red} strokeWidth={2.5} fill="none" />
					<line x1={X2(t)} x2={X2(t)} y1={Y2(0)} y2={Y2(1)} stroke={C.text} strokeDasharray="4 3" />
					<circle cx={X2(t)} cy={Y2(sa)} r={4} fill={C.blue} />
					<circle cx={X2(t)} cy={Y2(sn)} r={4} fill={C.red} />
				</svg>
			</div>
			<div className="viz-legend">
				<span style={{ ['--c' as string]: C.blue }}>√ᾱ_t = {fmt(sa, 3)} (hệ số tín hiệu)</span>
				<span style={{ ['--c' as string]: C.red }}>√(1−ᾱ_t) = {fmt(sn, 3)} (hệ số nhiễu)</span>
			</div>
			<div className="viz-kv"><span>kiểm tra: (√ᾱ)² + (√(1−ᾱ))² = <b>{fmt(sa * sa + sn * sn, 3)}</b> — tổng phương sai giữ nguyên nếu x₀ có phương sai 1</span></div>
		</Figure>
	);
}
