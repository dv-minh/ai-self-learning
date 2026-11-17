import { useMemo, useState } from 'react';
import { Axes, C, Controls, Figure, Slider, fmt, gauss, rng, scale } from './common';

function scores(sep: number) {
	const r = rng(13);
	const pos: number[] = [], neg: number[] = [];
	for (let i = 0; i < 30; i++) pos.push(Math.min(0.99, Math.max(0.01, 0.5 + sep / 2 + 0.16 * gauss(r))));
	for (let i = 0; i < 50; i++) neg.push(Math.min(0.99, Math.max(0.01, 0.5 - sep / 2 + 0.16 * gauss(r))));
	return { pos, neg };
}

/** Hình 6.4: kéo ngưỡng → ma trận nhầm lẫn, precision/recall và điểm trên đường ROC thay đổi theo. */
export default function ThresholdViz() {
	const [tau, setTau] = useState(0.5);
	const [sep, setSep] = useState(0.3);
	const { pos, neg } = useMemo(() => scores(sep), [sep]);
	const TP = pos.filter((s) => s >= tau).length, FN = pos.length - TP;
	const FP = neg.filter((s) => s >= tau).length, TN = neg.length - FP;
	const P = TP + FP ? TP / (TP + FP) : NaN;
	const R = TP / pos.length;
	const F1 = 2 * TP + FP + FN ? (2 * TP) / (2 * TP + FP + FN) : NaN;
	const FPR = FP / neg.length;
	const roc = useMemo(() => {
		const all = [...new Set([...pos, ...neg])].sort((a, b) => b - a);
		const pts: [number, number][] = [[0, 0]];
		for (const t of all) pts.push([neg.filter((s) => s >= t).length / neg.length, pos.filter((s) => s >= t).length / pos.length]);
		pts.push([1, 1]);
		let auc = 0;
		for (const p of pos) for (const n of neg) auc += p > n ? 1 : p === n ? 0.5 : 0;
		return { pts, auc: auc / (pos.length * neg.length) };
	}, [pos, neg]);
	const W = 520, H = 190;
	const X = scale(0, 1, 20, W - 20);
	const rowY = { pos: 55, neg: 135 };
	const r2 = rng(4);
	const jit = useMemo(() => [...pos, ...neg].map(() => (r2() - 0.5) * 34), [pos, neg]);
	const S = 230;
	const RX = scale(0, 1, 30, S - 10);
	const RY = scale(0, 1, S - 25, 10);
	return (
		<Figure
			title="Một ngưỡng, bốn ô của ma trận nhầm lẫn"
			caption={
				<>
					Mỗi chấm là một mẫu, đặt theo điểm số (score) mô hình cho. Hàng trên: mẫu thật sự dương; hàng dưới: mẫu âm. Mọi chấm bên phải vạch ngưỡng τ được dự đoán dương.
					Kéo τ sang phải: ít báo nhầm hơn (FP giảm, precision tăng) nhưng bỏ sót nhiều hơn (FN tăng, recall giảm) — không có ngưỡng nào tốt cả hai. Đường ROC vẽ (FPR, TPR) của <b>mọi</b> ngưỡng; chấm đỏ là ngưỡng hiện tại. Tăng “độ tách” để thấy ROC phồng về góc trên trái và AUC tăng.
				</>
			}
		>
			<Controls>
				<Slider label="ngưỡng τ" value={tau} min={0} max={1} step={0.01} onChange={setTau} />
				<Slider label="độ tách hai lớp" value={sep} min={0} max={0.7} step={0.05} onChange={setSep} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Điểm số của mẫu dương và âm">
				<rect x={X(tau)} y={20} width={X(1) - X(tau)} height={H - 50} fill={C.accent} opacity={0.1} />
				<text x={X(1) - 4} y={34} fontSize={12} textAnchor="end" fill={C.text}>dự đoán dương</text>
				<text x={X(0) + 4} y={34} fontSize={12} fill={C.text}>dự đoán âm</text>
				<text x={X(0)} y={rowY.pos - 22} fontSize={12} fill={C.red}>thật dương</text>
				<text x={X(0)} y={rowY.neg + 30} fontSize={12} fill={C.blue}>thật âm</text>
				{pos.map((s, i) => <circle key={`p${i}`} cx={X(s)} cy={rowY.pos + jit[i] * 0.5} r={5} fill={C.red} opacity={s >= tau ? 1 : 0.35} />)}
				{neg.map((s, i) => <circle key={`n${i}`} cx={X(s)} cy={rowY.neg + jit[pos.length + i] * 0.5} r={5} fill={C.blue} opacity={s >= tau ? 1 : 0.35} />)}
				<line x1={X(tau)} x2={X(tau)} y1={14} y2={H - 26} stroke={C.text} strokeWidth={2} />
				<line x1={X(0)} x2={X(1)} y1={H - 22} y2={H - 22} stroke={C.axis} />
				{[0, 0.25, 0.5, 0.75, 1].map((t) => <text key={t} x={X(t)} y={H - 6} fontSize={11} textAnchor="middle" fill={C.muted}>{fmt(t)}</text>)}
			</svg>
			<div className="viz-row" style={{ marginTop: '0.6rem' }}>
				<div>
					<table className="viz-table">
						<thead><tr><th></th><th>dự đoán 1</th><th>dự đoán 0</th></tr></thead>
						<tbody>
							<tr><th>thật 1</th><td className="hl">TP = {TP}</td><td>FN = {FN}</td></tr>
							<tr><th>thật 0</th><td>FP = {FP}</td><td className="hl">TN = {TN}</td></tr>
						</tbody>
					</table>
					<div className="viz-kv" style={{ flexDirection: 'column', alignItems: 'center', gap: '0.15rem' }}>
						<span>precision = TP/(TP+FP) = <b>{fmt(P, 3)}</b></span>
						<span>recall = TP/(TP+FN) = <b>{fmt(R, 3)}</b></span>
						<span>F₁ = <b>{fmt(F1, 3)}</b>, accuracy = <b>{fmt((TP + TN) / (pos.length + neg.length), 3)}</b></span>
					</div>
				</div>
				<svg viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 260 }} role="img" aria-label="Đường ROC">
					<Axes x={RX} y={RY} xr={[0, 1]} yr={[0, 1]} xticks={[0.5, 1]} yticks={[0.5, 1]} xlabel="FPR" ylabel="TPR" />
					<line x1={RX(0)} y1={RY(0)} x2={RX(1)} y2={RY(1)} stroke={C.grid} strokeDasharray="4 4" />
					<polyline points={roc.pts.map(([a, b]) => `${RX(a)},${RY(b)}`).join(' ')} fill="none" stroke={C.accent} strokeWidth={2.2} />
					<circle cx={RX(FPR)} cy={RY(R)} r={6} fill={C.red} />
					<text x={RX(0.97)} y={RY(0.08)} fontSize={12} textAnchor="end" fill={C.text}>AUC = {fmt(roc.auc, 3)}</text>
				</svg>
			</div>
		</Figure>
	);
}
