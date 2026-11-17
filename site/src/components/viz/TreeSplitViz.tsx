import { useMemo, useState } from 'react';
import { Axes, C, Choice, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

type P = [number, number, 0 | 1];
function data(): P[] {
	const r = rng(51);
	const out: P[] = [];
	for (let i = 0; i < 20; i++) {
		const x = 1 + 8 * r(), y = 1 + 8 * r();
		const c = x + 0.35 * gauss(r) > 5.5 || (y > 7.2 && x > 3.5) ? 1 : 0;
		out.push([x, y, c as 0 | 1]);
	}
	return out;
}
const D = data();
const gini = (pts: P[]) => {
	if (!pts.length) return 0;
	const p = pts.filter((q) => q[2] === 1).length / pts.length;
	return 1 - p * p - (1 - p) * (1 - p);
};
const COL = [C.blue, C.red];

/** Hình 6.7: chọn câu hỏi “xⱼ ≤ t?” làm hai nhóm con thuần nhất nhất (Gini sau tách nhỏ nhất). */
export default function TreeSplitViz() {
	const [feat, setFeat] = useState<'0' | '1'>('0');
	const [t, setT] = useState(3);
	const f = Number(feat);
	const split = (th: number) => {
		const L = D.filter((p) => p[f] <= th), R = D.filter((p) => p[f] > th);
		return { L, R, G: (L.length / D.length) * gini(L) + (R.length / D.length) * gini(R) };
	};
	const s = split(t);
	const Gp = gini(D);
	const curve = useMemo(() => {
		const out: [number, number][] = [];
		for (let i = 0; i <= 200; i++) {
			const th = 0.5 + (9 * i) / 200;
			out.push([th, split(th).G]);
		}
		return out;
	}, [f]);
	const best = useMemo(() => {
		const vals = [...new Set(D.map((p) => p[f]))].sort((a, b) => a - b);
		let bt = vals[0], bg = Infinity;
		for (let i = 0; i + 1 < vals.length; i++) {
			const th = (vals[i] + vals[i + 1]) / 2;
			const g = split(th).G;
			if (g < bg - 1e-12) { bg = g; bt = th; }
		}
		return [bt, bg];
	}, [f]);
	const S = 340;
	const X = scale(0, 10, 0, S);
	const Y = scale(0, 10, S, 0);
	const W2 = 340, H2 = 200;
	const X2 = scale(0.5, 9.5, 35, W2 - 10);
	const Y2 = scale(0, 0.55, H2 - 25, 10);
	const cnt = (pts: P[]) => `${pts.filter((p) => p[2] === 0).length} xanh, ${pts.filter((p) => p[2] === 1).length} đỏ`;
	return (
		<Figure
			title="Tìm câu hỏi tách tốt nhất cho một nút"
			caption={
				<>
					Chọn đặc trưng (x₁ hoặc x₂) và kéo ngưỡng t: đường cắt chia các điểm thành nhánh trái (xⱼ ≤ t) và phải. Bên phải là G_split theo t — cây quyết định thử mọi ngưỡng ứng viên (trung điểm giữa hai giá trị liền kề) rồi chọn chỗ <b>thấp nhất</b>.
					G_split chỉ đổi khi đường cắt vượt qua một điểm, nên đồ thị là bậc thang.
				</>
			}
		>
			<Controls>
				<Choice label="Đặc trưng:" value={feat} onChange={(v) => setFeat(v)} options={[['0', 'x₁ (ngang)'], ['1', 'x₂ (dọc)']]} />
				<Slider label="ngưỡng t" value={t} min={0.5} max={9.5} step={0.05} onChange={setT} />
			</Controls>
			<div className="viz-buttons">
				<button type="button" className="primary" onClick={() => setT(best[0])}>Nhảy tới ngưỡng tốt nhất</button>
			</div>
			<div className="viz-row">
				<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Dữ liệu và đường tách">
					{f === 0 ? (
						<>
							<rect x={0} y={0} width={X(t)} height={S} fill={C.green} opacity={0.06} />
							<line x1={X(t)} x2={X(t)} y1={0} y2={S} stroke={C.text} strokeWidth={2} />
						</>
					) : (
						<>
							<rect x={0} y={Y(t)} width={S} height={S - Y(t)} fill={C.green} opacity={0.06} />
							<line y1={Y(t)} y2={Y(t)} x1={0} x2={S} stroke={C.text} strokeWidth={2} />
						</>
					)}
					{D.map(([x, y, c], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={6} fill={COL[c]} />)}
					<text x={6} y={S - 6} fontSize={12} fill={C.muted}>x₁ →</text>
					<text x={6} y={16} fontSize={12} fill={C.muted}>↑ x₂</text>
				</svg>
				<svg viewBox={`0 0 ${W2} ${H2}`} role="img" aria-label="Gini sau tách theo ngưỡng">
					<Axes x={X2} y={Y2} xr={[0.5, 9.5]} yr={[0, 0.55]} xticks={[2, 4, 6, 8]} yticks={[0.25, 0.5]} xlabel="t" ylabel="G_split" grid={false} />
					<line x1={X2(0.5)} x2={X2(9.5)} y1={Y2(Gp)} y2={Y2(Gp)} stroke={C.muted} strokeDasharray="4 3" />
					<polyline points={curve.map(([a, b]) => `${X2(a).toFixed(1)},${Y2(b).toFixed(1)}`).join(' ')} fill="none" stroke={C.accent} strokeWidth={2} />
					<circle cx={X2(best[0])} cy={Y2(best[1])} r={5} fill="none" stroke={C.green} strokeWidth={2.5} />
					<circle cx={X2(t)} cy={Y2(s.G)} r={5} fill={C.red} />
				</svg>
			</div>
			<table className="viz-table">
				<thead><tr><th></th><th>số mẫu</th><th>thành phần</th><th>Gini</th></tr></thead>
				<tbody>
					<tr><th>nút cha</th><td>{D.length}</td><td>{cnt(D)}</td><td>{fmt(Gp, 3)}</td></tr>
					<tr><th>trái (≤ t)</th><td>{s.L.length}</td><td>{cnt(s.L)}</td><td>{fmt(gini(s.L), 3)}</td></tr>
					<tr><th>phải (&gt; t)</th><td>{s.R.length}</td><td>{cnt(s.R)}</td><td>{fmt(gini(s.R), 3)}</td></tr>
					<tr><th>sau tách</th><td colSpan={2}>trung bình có trọng số theo số mẫu</td><td className="hl">{fmt(s.G, 3)}</td></tr>
				</tbody>
			</table>
			<div className="viz-kv"><span>gain = G_cha − G_split = <b>{fmt(Gp - s.G, 3)}</b></span><span>(nét đứt ngang: G của nút cha; vòng xanh lá: ngưỡng tốt nhất)</span></div>
		</Figure>
	);
}
