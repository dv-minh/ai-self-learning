import { useMemo, useState } from 'react';
import { C, Choice, Contours, Controls, Figure, Slider, fmt, scale, useSvgDrag } from './common';

const S = 400;
const R = 3.5;
// A = Rot · diag(1, 4) · Rotᵀ (cố định)
const th = 0.5;
const ca = Math.cos(th), sa = Math.sin(th);
const a11 = 1 * ca * ca + 4 * sa * sa, a22 = 1 * sa * sa + 4 * ca * ca, a12 = (1 - 4) * ca * sa;
const LMAX = 4;

function solve(kind: 'l1' | 'l2', c: [number, number], lam: number): [number, number] {
	if (kind === 'l2') {
		// (A + λI) w = A c
		const b1 = a11 * c[0] + a12 * c[1], b2 = a12 * c[0] + a22 * c[1];
		const m11 = a11 + lam, m22 = a22 + lam;
		const det = m11 * m22 - a12 * a12;
		return [(m22 * b1 - a12 * b2) / det, (m11 * b2 - a12 * b1) / det];
	}
	// ISTA: w ← S_{ηλ}(w − η A (w − c))
	let w: [number, number] = [c[0], c[1]];
	const eta = 1 / LMAX;
	const st = (z: number, t: number) => Math.sign(z) * Math.max(Math.abs(z) - t, 0);
	for (let i = 0; i < 600; i++) {
		const d0 = w[0] - c[0], d1 = w[1] - c[1];
		const g0 = a11 * d0 + a12 * d1, g1 = a12 * d0 + a22 * d1;
		w = [st(w[0] - eta * g0, eta * lam), st(w[1] - eta * g1, eta * lam)];
	}
	return w;
}

/** Hình 5.7: hình học của L1 và L2 — vì sao L1 cho nghiệm thưa. */
export default function RegGeometryViz() {
	const [kind, setKind] = useState<'l1' | 'l2'>('l1');
	const [lam, setLam] = useState(1.2);
	const [c, setC] = useState<[number, number]>([2.2, 1.0]);
	const X = scale(-R, R, 0, S);
	const Y = scale(-R, R, S, 0);
	const { svgRef, handlers } = useSvgDrag((sx, sy) => {
		const x = -R + (sx / S) * 2 * R;
		const y = R - (sy / S) * 2 * R;
		setC([Math.max(-R + 0.2, Math.min(R - 0.2, x)), Math.max(-R + 0.2, Math.min(R - 0.2, y))]);
	});
	const L = (x: number, y: number) => {
		const d0 = x - c[0], d1 = y - c[1];
		return 0.5 * (a11 * d0 * d0 + 2 * a12 * d0 * d1 + a22 * d1 * d1);
	};
	const w = useMemo(() => solve(kind, c, lam), [kind, c, lam]);
	const path = useMemo(() => {
		const out: [number, number][] = [];
		for (let i = 0; i <= 60; i++) out.push(solve(kind, c, (8 * i) / 60));
		return out;
	}, [kind, c]);
	const lw = L(w[0], w[1]);
	const r1 = Math.abs(w[0]) + Math.abs(w[1]);
	const r2 = Math.hypot(w[0], w[1]);
	const zero0 = Math.abs(w[0]) < 1e-6, zero1 = Math.abs(w[1]) < 1e-6;
	return (
		<Figure
			title="Vì sao L1 ép trọng số về đúng 0 còn L2 thì không"
			caption={
				<>
					Các elip xám: đường đồng mức của mất mát L(w), tâm là nghiệm không regularization (chấm đen — <b>kéo thả được</b>). Hình {kind === 'l1' ? 'thoi' : 'tròn'} màu cam: tập các w có cùng {kind === 'l1' ? '‖w‖₁' : '‖w‖₂'} với nghiệm.
					Nghiệm có regularization (chấm đỏ) là chỗ elip <b>vừa chạm</b> hình {kind === 'l1' ? 'thoi' : 'tròn'}. Hình thoi có góc nhọn nằm trên trục, nên elip thường chạm đúng góc → một tọa độ bằng 0. Hình tròn trơn, không có góc, nên chỗ chạm hầu như không nằm trên trục.
					Đường tím: nghiệm khi λ tăng dần từ 0 (đường regularization).
				</>
			}
		>
			<Controls>
				<Choice value={kind} onChange={setKind} options={[['l1', 'L1 (lasso): λ‖w‖₁'], ['l2', 'L2 (ridge): (λ/2)‖w‖₂²']]} />
				<Slider label="λ" value={lam} min={0} max={8} step={0.1} onChange={setLam} />
			</Controls>
			<svg ref={svgRef} viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 440 }} role="img" aria-label="Đồng mức mất mát và vùng ràng buộc">
				<line x1={0} x2={S} y1={Y(0)} y2={Y(0)} stroke={C.axis} />
				<line y1={0} y2={S} x1={X(0)} x2={X(0)} stroke={C.axis} />
				<text x={S - 6} y={Y(0) - 6} textAnchor="end" fontSize={13} fill={C.text}>w₁</text>
				<text x={X(0) + 6} y={14} fontSize={13} fill={C.text}>w₂</text>
				<Contours f={L} xr={[-R, R]} yr={[-R, R]} levels={[0.1, 0.4, 1, 2, 3.5, 5.5, 8, 11]} X={X} Y={Y} />
				<g opacity={0.95}>
					{lw > 1e-9 && <Contours f={L} xr={[-R, R]} yr={[-R, R]} levels={[lw]} X={X} Y={Y} n={90} />}
				</g>
				{kind === 'l1' ? (
					<polygon points={`${X(r1)},${Y(0)} ${X(0)},${Y(r1)} ${X(-r1)},${Y(0)} ${X(0)},${Y(-r1)}`} fill={C.orange} fillOpacity={0.15} stroke={C.orange} strokeWidth={2} />
				) : (
					<circle cx={X(0)} cy={Y(0)} r={X(r2) - X(0)} fill={C.orange} fillOpacity={0.15} stroke={C.orange} strokeWidth={2} />
				)}
				<polyline points={path.map(([a, b]) => `${X(a).toFixed(1)},${Y(b).toFixed(1)}`).join(' ')} fill="none" stroke={C.purple} strokeWidth={2} strokeDasharray="5 3" />
				<line x1={X(c[0])} y1={Y(c[1])} x2={X(w[0])} y2={Y(w[1])} stroke={C.red} strokeDasharray="2 3" />
				<circle cx={X(w[0])} cy={Y(w[1])} r={6} fill={C.red} />
				<circle cx={X(c[0])} cy={Y(c[1])} r={8} fill={C.text} style={{ cursor: 'grab' }} {...handlers} />
			</svg>
			<div className="viz-kv">
				<span>không regularization: w = (<b>{fmt(c[0])}</b>; <b>{fmt(c[1])}</b>)</span>
				<span>có regularization: w = (<b style={{ color: zero0 ? C.green : undefined }}>{fmt(w[0], 3)}</b>; <b style={{ color: zero1 ? C.green : undefined }}>{fmt(w[1], 3)}</b>)</span>
			</div>
			<div className="viz-note">
				{zero0 && zero1
					? 'λ đủ lớn: cả hai trọng số đều bằng 0.'
					: zero0 || zero1
						? `Nghiệm thưa: w${zero0 ? '₁' : '₂'} bằng đúng 0 — đặc trưng đó bị loại khỏi mô hình.`
						: kind === 'l1'
							? 'Chưa có tọa độ nào bằng 0; tăng λ hoặc kéo tâm lại gần một trục.'
							: 'L2 chỉ co cả hai trọng số về phía 0 (theo tỷ lệ), gần như không bao giờ làm một trọng số bằng đúng 0.'}
			</div>
		</Figure>
	);
}
