import { useMemo, useState } from 'react';
import { C, Figure, fmt, gauss, heatFill, rng, useStepper } from './common';

const TOK = ['Con', 'mèo', 'ngồi', 'trên', 'thảm', 'vì', 'nó', 'mệt'];
const D = 8;

function build() {
	const r = rng(23);
	// key trực giao (Gram–Schmidt), độ dài √D, để điểm chỉ phản ánh “query hướng về key nào”
	const K: number[][] = [];
	for (let i = 0; i < TOK.length; i++) {
		let v = Array.from({ length: D }, () => gauss(r));
		for (const u of K) {
			const c = v.reduce((s, x, j) => s + x * u[j], 0) / D;
			v = v.map((x, j) => x - c * u[j]);
		}
		const nv = Math.hypot(...v);
		K.push(v.map((x) => (x / nv) * Math.sqrt(D)));
	}
	const V = TOK.map(() => Array.from({ length: D }, () => gauss(r)));
	// query mỗi token: hướng nhẹ về chính nó, riêng “nó” hướng mạnh về “mèo”, “ngồi” hướng về “mèo” và “thảm”
	const Q = TOK.map((_, i) => K[i].map((k) => 0.4 * k + 0.4 * gauss(r)));
	Q[6] = K[1].map((k, j) => 1.2 * k + 0.25 * K[6][j]);
	Q[2] = K[1].map((k, j) => 0.7 * k + 0.7 * K[4][j]);
	Q[7] = K[1].map((k, j) => 0.8 * k + 0.5 * K[7][j]);
	return { Q, K, V };
}
const { Q, K, V } = build();
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

/** Hình 7.6: self-attention từng bước cho một token — điểm, chia √d, softmax, trộn value — và cả ma trận trọng số. */
export default function AttentionViz() {
	const [qi, setQi] = useState(6);
	const [causal, setCausal] = useState(false);
	const { step, controls } = useStepper(3, 1300);
	const n = TOK.length;
	const A = useMemo(() => {
		return TOK.map((_, i) => {
			const s = TOK.map((_, j) => (causal && j > i ? -Infinity : dot(Q[i], K[j]) / Math.sqrt(D)));
			const m = Math.max(...s);
			const e = s.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
			const z = e.reduce((a, b) => a + b, 0);
			return e.map((v) => v / z);
		});
	}, [causal]);
	const raw = TOK.map((_, j) => (causal && j > qi ? -Infinity : dot(Q[qi], K[j])));
	const scaled = raw.map((v) => v / Math.sqrt(D));
	const w = A[qi];
	const out = Array.from({ length: D }, (_, k) => w.reduce((s, wj, j) => s + wj * V[j][k], 0));
	const row = [raw, scaled, w, w][step];
	const rowMax = Math.max(...row.filter(Number.isFinite).map(Math.abs), 1e-9);
	const W = 560, Ht = 150;
	const tx = (j: number) => 40 + (j * (W - 80)) / (n - 1);
	const title = ['1. Điểm sⱼ = q·kⱼ', '2. Chia √d để điểm không quá lớn', '3. Softmax: trọng số dương, cộng bằng 1', '4. Đầu ra = Σ trọng số × vⱼ'][step];
	const cell = 30;
	return (
		<Figure
			title="Self-attention: một token “hỏi” mọi token khác"
			caption={
				<>
					Bấm vào một từ để chọn query. Các bước: (1) tích vô hướng của query với key của từng token, (2) chia √d (d = {D}), (3) softmax thành trọng số — độ dày cung nối thể hiện trọng số, (4) trộn các vector value theo trọng số.
					Bảng dưới là ma trận trọng số attention đầy đủ: hàng = query, cột = key, mỗi hàng cộng bằng 1. Bật mặt nạ nhân quả để mỗi token chỉ nhìn được các token đứng trước nó (tam giác dưới) — như trong GPT. Vector ở đây là ngẫu nhiên có chủ đích (để “nó” chú ý tới “mèo”), không phải từ mô hình thật.
				</>
			}
		>
			{controls}
			<div className="viz-buttons">
				<label>
					<input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} /> mặt nạ nhân quả (causal mask)
				</label>
			</div>
			<div className="viz-title" style={{ fontSize: '0.95rem' }}>{title}</div>
			<svg viewBox={`0 0 ${W} ${Ht}`} role="img" aria-label="Trọng số attention của token được chọn">
				{step >= 2 &&
					TOK.map((_, j) =>
						w[j] > 0.005 ? (
							<path key={`a${j}`} d={`M${tx(qi)},${Ht - 44} Q${(tx(qi) + tx(j)) / 2},${Ht - 44 - 30 - Math.abs(tx(qi) - tx(j)) * 0.25} ${tx(j)},${Ht - 44}`} fill="none" stroke={C.accent} strokeWidth={1 + 14 * w[j]} strokeOpacity={0.35 + 0.6 * w[j]} />
						) : null,
					)}
				{TOK.map((t, j) => {
					const v = row[j];
					const fin = Number.isFinite(v);
					return (
						<g key={t} onClick={() => setQi(j)} style={{ cursor: 'pointer' }}>
							<rect x={tx(j) - 30} y={Ht - 40} width={60} height={26} rx={6} fill={j === qi ? C.accent : C.grid} fillOpacity={j === qi ? 0.9 : 0.4} />
							<text x={tx(j)} y={Ht - 22} fontSize={14} textAnchor="middle" fill={j === qi ? 'var(--sl-color-black)' : C.text} fontWeight={j === qi ? 700 : 400}>{t}</text>
							<text x={tx(j)} y={Ht - 2} fontSize={11} textAnchor="middle" fill={fin ? (v >= 0 ? C.red : C.blue) : C.muted} opacity={fin ? 0.4 + 0.6 * Math.abs(v) / rowMax : 1}>
								{fin ? fmt(v, 2) : '−∞'}
							</text>
						</g>
					);
				})}
			</svg>
			{step === 3 && (
				<div className="viz-kv">
					<span>đầu ra cho “{TOK[qi]}” = Σⱼ wⱼvⱼ ≈ [{out.map((v) => fmt(v, 2)).join('; ')}] — chủ yếu là value của: <b>{TOK.map((t, j) => [t, w[j]] as const).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([t, v]) => `${t} (${fmt(v * 100, 0)}%)`).join(', ')}</b></span>
				</div>
			)}
			<svg viewBox={`0 0 ${70 + cell * n} ${40 + cell * n}`} style={{ maxWidth: 380, marginTop: '0.6rem' }} role="img" aria-label="Ma trận trọng số attention">
				{TOK.map((t, j) => <text key={`c${t}`} x={70 + j * cell + cell / 2} y={30} fontSize={10} textAnchor="middle" fill={C.text}>{t}</text>)}
				{TOK.map((t, i) => (
					<g key={`r${t}`}>
						<text x={64} y={40 + i * cell + cell / 2 + 4} fontSize={11} textAnchor="end" fill={i === qi ? C.accent : C.text} fontWeight={i === qi ? 700 : 400}>{t}</text>
						{A[i].map((v, j) => (
							<g key={j} onClick={() => setQi(i)} style={{ cursor: 'pointer' }}>
								<rect x={70 + j * cell} y={40 + i * cell} width={cell - 1} height={cell - 1} {...heatFill(v)} stroke={i === qi ? C.accent : 'none'} strokeWidth={1.5} />
								{v >= 0.1 && <text x={70 + j * cell + cell / 2} y={40 + i * cell + cell / 2 + 4} fontSize={9} textAnchor="middle" fill={C.text}>{fmt(v, 2)}</text>}
							</g>
						))}
					</g>
				))}
			</svg>
		</Figure>
	);
}
