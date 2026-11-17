import { useState } from 'react';
import { C, Choice, Controls, Figure, SERIES, Slider, fmt } from './common';

const NAMES = ['mèo', 'chó', 'chim', 'cá'];

/** Hình 7.2: logit → exp → chuẩn hóa → xác suất; cross-entropy và gradient p − y. */
export default function SoftmaxViz() {
	const [z, setZ] = useState([2, 1, 0.1, -1]);
	const [T, setT] = useState(1);
	const [t, setTrue] = useState<'0' | '1' | '2' | '3'>('0');
	const ti = Number(t);
	const m = Math.max(...z.map((v) => v / T));
	const ex = z.map((v) => Math.exp(v / T - m));
	const sum = ex.reduce((a, b) => a + b, 0);
	const p = ex.map((e) => e / sum);
	const loss = -Math.log(p[ti]);
	const grad = p.map((v, i) => v - (i === ti ? 1 : 0));
	const W = 540, rowH = 34;
	const cols = [
		{ title: 'logit zᵢ', vals: z, max: 4, signed: true },
		{ title: 'eᶻⁱᐟᵀ (tỉ lệ)', vals: ex.map((e) => e / Math.max(...ex)), max: 1, signed: false },
		{ title: 'pᵢ = softmax', vals: p, max: 1, signed: false },
		{ title: 'pᵢ − yᵢ', vals: grad, max: 1, signed: true },
	];
	const cw = (W - 70) / cols.length;
	return (
		<Figure
			title="Softmax và cross-entropy, từng cột một"
			caption={
				<>
					Đọc từ trái sang phải. (1) Logit z là điểm thô, có thể âm. (2) Lấy mũ: mọi giá trị thành dương, chênh lệch được phóng đại. (3) Chia cho tổng: xác suất cộng bằng 1. (4) Với T = 1, gradient của ℓ theo logit là p − y (với T khác 1 thì là (p − y)/T): lớp đúng bị kéo lên (cột âm, vì ta trừ gradient), các lớp sai bị đẩy xuống đúng bằng xác suất của chúng.
					Nhiệt độ T: chia logit cho T trước softmax — T nhỏ làm phân phối “nhọn” (gần one-hot), T lớn làm phân phối “phẳng”.
				</>
			}
		>
			<Controls>
				{z.map((v, i) => (
					<Slider key={i} label={`z (${NAMES[i]})`} value={v} min={-4} max={4} step={0.1} onChange={(nv) => setZ(z.map((o, k) => (k === i ? nv : o)))} />
				))}
				<Slider label="nhiệt độ T" value={T} min={0.2} max={5} step={0.1} onChange={setT} />
			</Controls>
			<Controls>
				<Choice label="Lớp thật:" value={t} onChange={setTrue} options={NAMES.map((n, i) => [String(i) as '0', n])} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${rowH * 4 + 40}`} role="img" aria-label="Các bước tính softmax">
				{cols.map((c, ci) => (
					<g key={ci} transform={`translate(${70 + ci * cw},0)`}>
						<text x={cw / 2} y={16} fontSize={12} textAnchor="middle" fill={C.text} fontWeight={600}>{c.title}</text>
						{c.signed && <line x1={cw / 2} x2={cw / 2} y1={24} y2={24 + rowH * 4} stroke={C.axis} />}
						{c.vals.map((v, i) => {
							const y = 28 + i * rowH;
							const len = (Math.min(Math.abs(v), c.max) / c.max) * (c.signed ? cw / 2 - 8 : cw - 16);
							const x0 = c.signed ? (v >= 0 ? cw / 2 : cw / 2 - len) : 4;
							return (
								<g key={i}>
									<rect x={x0} y={y} width={len} height={rowH - 12} fill={SERIES[i]} opacity={i === ti && ci >= 2 ? 1 : 0.65} />
									<text x={c.signed ? cw / 2 + (v >= 0 ? -4 : 4) : len + 8} y={y + rowH / 2 - 2} fontSize={11} textAnchor={c.signed ? (v >= 0 ? 'end' : 'start') : 'start'} fill={C.text}>{fmt(v, 2)}</text>
								</g>
							);
						})}
					</g>
				))}
				{NAMES.map((n, i) => (
					<text key={n} x={4} y={28 + i * rowH + rowH / 2 - 2} fontSize={13} fill={SERIES[i]} fontWeight={i === ti ? 700 : 400}>{n}{i === ti ? ' ✓' : ''}</text>
				))}
			</svg>
			<div className="viz-kv">
				<span>Σpᵢ = <b>{fmt(p.reduce((a, b) => a + b, 0), 3)}</b></span>
				<span>xác suất lớp thật p_t = <b>{fmt(p[ti], 4)}</b></span>
				<span>cross-entropy ℓ = −ln p_t = <b>{fmt(loss, 4)}</b></span>
			</div>
		</Figure>
	);
}
