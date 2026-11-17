import { useMemo, useState } from 'react';
import { Axes, C, Choice, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

type Seq = 'const' | 'alt' | 'noisy';
const T = 30;
const W = 560, H = 290;

function gradients(seq: Seq): number[] {
	const r = rng(3);
	return Array.from({ length: T }, (_, i) => {
		const t = i + 1;
		if (seq === 'const') return 1;
		if (seq === 'alt') return 0.25 + (t % 2 ? 1 : -1);
		return 1 + 1.2 * gauss(r);
	});
}

/**
 * Hình 5.4 / 5.5: vận tốc momentum (mode="momentum": v = βv + g)
 * hoặc trung bình động mũ của Adam (mode="adam": m = βm + (1−β)g và m̂ = m/(1−βᵗ)).
 */
export default function EmaViz({ mode = 'momentum' }: { mode?: 'momentum' | 'adam' }) {
	const [seq, setSeq] = useState<Seq>(mode === 'momentum' ? 'alt' : 'const');
	const [beta, setBeta] = useState(0.9);
	const g = useMemo(() => gradients(seq), [seq]);
	const { a, b } = useMemo(() => {
		const a: number[] = [], b: number[] = [];
		let v = 0;
		for (let i = 0; i < T; i++) {
			if (mode === 'momentum') {
				v = beta * v + g[i];
				a.push(v);
			} else {
				v = beta * v + (1 - beta) * g[i];
				a.push(v);
				b.push(v / (1 - beta ** (i + 1)));
			}
		}
		return { a, b };
	}, [g, beta, mode]);
	const mean = g.reduce((s, x) => s + x, 0) / T;
	const vals = [...g, ...a, ...b, 0];
	const ymax = Math.max(...vals) * 1.1 + 0.1;
	const ymin = Math.min(...vals) * 1.1 - 0.1;
	const X = scale(0, T + 1, 34, W - 10);
	const Y = scale(ymin, ymax, H - 22, 10);
	const bw = ((W - 44) / (T + 1)) * 0.55;
	const line = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${X(i + 1).toFixed(1)},${Y(v).toFixed(1)}`).join('');
	const yt = [Math.ceil(ymin), 0, Math.floor(ymax)].filter((v, i, s) => s.indexOf(v) === i);
	const asym = mode === 'momentum' ? mean / (1 - beta) : mean;
	return (
		<Figure
			title={mode === 'momentum' ? 'Vận tốc v tích lũy gradient như thế nào' : 'Trung bình động mũ và hiệu chỉnh độ chệch'}
			caption={
				mode === 'momentum' ? (
					<>
						Cột xám: gradient gₜ từng bước. Đường <span style={{ color: C.red }}>đỏ</span>: vận tốc vₜ = βvₜ₋₁ + gₜ (v₀ = 0). Đường nét đứt: mức gₜ trung bình chia (1 − β) — nơi v ổn định nếu gradient giữ nguyên trung bình.
						Gradient <b>đổi dấu liên tục</b>: phần dao động triệt tiêu, v chỉ giữ lại phần “trôi” chung. Gradient <b>cùng dấu</b>: v lớn dần tới g/(1−β), tức bước dài gấp 1/(1−β) lần.
					</>
				) : (
					<>
						Cột xám: gradient gₜ. Đường <span style={{ color: C.orange }}>cam</span>: mₜ = βmₜ₋₁ + (1−β)gₜ với m₀ = 0 — bị kéo về 0 ở các bước đầu.
						Đường <span style={{ color: C.green }}>xanh lá</span>: m̂ₜ = mₜ/(1−βᵗ) sau hiệu chỉnh — đúng mức ngay từ bước đầu. Nét đứt: trung bình thật của gₜ. β càng gần 1 thì độ chệch ban đầu càng kéo dài.
					</>
				)
			}
		>
			<Controls>
				<Choice
					label="Chuỗi gradient:"
					value={seq}
					onChange={setSeq}
					options={[
						['const', 'cùng dấu, không đổi'],
						['alt', 'đổi dấu liên tục'],
						['noisy', 'nhiễu quanh 1'],
					]}
				/>
				<Slider label="β" value={beta} min={0} max={0.98} step={0.01} onChange={setBeta} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Biểu đồ gradient và trung bình động">
				<Axes x={X} y={Y} xr={[0, T + 1]} yr={[ymin, ymax]} xticks={[5, 10, 15, 20, 25, 30]} yticks={yt} xlabel="t" />
				{g.map((v, i) => (
					<rect key={i} x={X(i + 1) - bw / 2} y={Math.min(Y(v), Y(0))} width={bw} height={Math.abs(Y(v) - Y(0))} fill={C.muted} opacity={0.45} />
				))}
				<line x1={X(0)} x2={X(T + 1)} y1={Y(asym)} y2={Y(asym)} stroke={C.text} strokeDasharray="6 4" strokeWidth={1.2} />
				<path d={line(a)} stroke={mode === 'momentum' ? C.red : C.orange} strokeWidth={2.5} fill="none" />
				{mode === 'adam' && <path d={line(b)} stroke={C.green} strokeWidth={2.5} fill="none" />}
			</svg>
			<div className="viz-kv">
				<span>trung bình gₜ = <b>{fmt(mean, 3)}</b></span>
				{mode === 'momentum' ? (
					<>
						<span>1/(1−β) = <b>{fmt(1 / (1 - beta), 2)}</b></span>
						<span>v₁ = <b>{fmt(a[0], 3)}</b>, v₃₀ = <b>{fmt(a[T - 1], 3)}</b></span>
					</>
				) : (
					<>
						<span>m₁ = <b>{fmt(a[0], 3)}</b> nhưng m̂₁ = <b>{fmt(b[0], 3)}</b></span>
						<span>hệ số 1−β¹ = <b>{fmt(1 - beta, 3)}</b></span>
					</>
				)}
			</div>
		</Figure>
	);
}
