import { useMemo, useState } from 'react';
import { C, Controls, Figure, Slider, fmt, gauss, heatFill, rng } from './common';

const PAIRS = ['mèo', 'chó', 'xe đạp', 'bãi biển', 'pizza'];
const D = 16;
const norm = (v: number[]) => {
	const n = Math.hypot(...v);
	return v.map((x) => x / n);
};

/** Hình 8.8: ma trận tương đồng ảnh–chú thích của CLIP; huấn luyện đẩy đường chéo lên, phần còn lại xuống. */
export default function ClipViz() {
	const [align, setAlign] = useState(0.5);
	const [tau, setTau] = useState(0.1);
	const { I, T } = useMemo(() => {
		const r = rng(5);
		const base = PAIRS.map(() => Array.from({ length: D }, () => gauss(r)));
		const noiseI = PAIRS.map(() => Array.from({ length: D }, () => gauss(r)));
		const noiseT = PAIRS.map(() => Array.from({ length: D }, () => gauss(r)));
		return {
			I: base.map((b, k) => norm(b.map((x, j) => align * x + (1 - align) * noiseI[k][j]))),
			T: base.map((b, k) => norm(b.map((x, j) => align * x + (1 - align) * noiseT[k][j]))),
		};
	}, [align]);
	const n = PAIRS.length;
	const S = I.map((a) => T.map((b) => a.reduce((s, x, j) => s + x * b[j], 0)));
	const rowsP = S.map((row) => {
		const m = Math.max(...row.map((v) => v / tau));
		const e = row.map((v) => Math.exp(v / tau - m));
		const z = e.reduce((a, b) => a + b, 0);
		return e.map((v) => v / z);
	});
	const loss = -rowsP.reduce((s, row, i) => s + Math.log(row[i]), 0) / n;
	const top1 = S.filter((row, i) => row.indexOf(Math.max(...row)) === i).length;
	const cell = 54;
	return (
		<Figure
			title="CLIP: mỗi ảnh phải tìm đúng chú thích của mình trong batch"
			caption={
				<>
					Hàng: ảnh; cột: chú thích; ô (i, j) là cosine Sᵢⱼ giữa embedding ảnh i và chú thích j (đỏ: dương, xanh: âm). Đường chéo là các cặp đúng. Mỗi hàng là một bài toán phân loại {n} lớp: softmax của Sᵢⱼ/τ, nhãn đúng là cột i; số nhỏ dưới mỗi ô là xác suất đó.
					Kéo “mức khớp” để mô phỏng mô hình đã học tốt hơn: đường chéo đỏ dần, mất mát giảm. τ nhỏ làm softmax sắc hơn.
				</>
			}
		>
			<Controls>
				<Slider label="mức khớp embedding (giả lập huấn luyện)" value={align} min={0} max={1} step={0.05} onChange={setAlign} />
				<Slider label="nhiệt độ τ" value={tau} min={0.02} max={1} step={0.01} onChange={setTau} />
			</Controls>
			<svg viewBox={`0 0 ${110 + cell * n} ${40 + cell * n}`} style={{ maxWidth: 440 }} role="img" aria-label="Ma trận tương đồng ảnh và chú thích">
				{PAIRS.map((p, j) => <text key={`c${j}`} x={110 + j * cell + cell / 2} y={28} fontSize={11} textAnchor="middle" fill={C.text}>“{p}”</text>)}
				{PAIRS.map((p, i) => (
					<g key={`r${i}`}>
						<text x={104} y={40 + i * cell + cell / 2 + 4} fontSize={11} textAnchor="end" fill={C.text}>ảnh {p}</text>
						{S[i].map((v, j) => (
							<g key={j}>
								<rect x={110 + j * cell} y={40 + i * cell} width={cell - 2} height={cell - 2} {...heatFill(v)} stroke={i === j ? C.text : 'none'} strokeWidth={2} />
								<text x={110 + j * cell + cell / 2} y={40 + i * cell + cell / 2} fontSize={12} textAnchor="middle" fill={C.text} fontWeight={i === j ? 700 : 400}>{fmt(v, 2)}</text>
								<text x={110 + j * cell + cell / 2} y={40 + i * cell + cell / 2 + 15} fontSize={9} textAnchor="middle" fill={C.text} opacity={0.75}>p={fmt(rowsP[i][j], 2)}</text>
							</g>
						))}
					</g>
				))}
			</svg>
			<div className="viz-kv">
				<span>mất mát ảnh → chú thích (trung bình −ln pᵢᵢ) = <b>{fmt(loss, 3)}</b></span>
				<span>truy hồi top-1 đúng: <b>{top1}/{n}</b></span>
			</div>
		</Figure>
	);
}
