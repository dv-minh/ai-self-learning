import { useState } from 'react';
import { C, Choice, Controls, Figure, Slider, fmt, scale } from './common';

type Task = 'and' | 'or' | 'xor';
const PTS: [number, number][] = [[0, 0], [0, 1], [1, 0], [1, 1]];
const LABEL: Record<Task, number[]> = { and: [0, 0, 0, 1], or: [0, 1, 1, 1], xor: [0, 1, 1, 0] };
const relu = (z: number) => Math.max(0, z);

/**
 * Hình 7.1: một nơ-ron = một đường thẳng cắt mặt phẳng. AND/OR tách được, XOR thì không;
 * thêm một lớp ẩn 2 nơ-ron ReLU “bẻ” mặt phẳng để XOR tách được.
 */
export default function NeuronViz() {
	const [task, setTask] = useState<Task>('xor');
	const [w1, setW1] = useState(1);
	const [w2, setW2] = useState(1);
	const [b, setB] = useState(-0.5);
	const [hidden, setHidden] = useState(false);
	const S = 300;
	const X = scale(-0.5, 1.5, 0, S);
	const Y = scale(-0.5, 1.5, S, 0);
	// mạng có lớp ẩn: h1 = ReLU(x1 + x2), h2 = ReLU(x1 + x2 − 1), y = h1 − 2h2 − 0,5
	const h = (x1: number, x2: number) => [relu(x1 + x2), relu(x1 + x2 - 1)];
	const out = (x1: number, x2: number) => {
		if (!hidden) return w1 * x1 + w2 * x2 + b;
		const [h1, h2] = h(x1, x2);
		return h1 - 2 * h2 - 0.5;
	};
	const G = 50;
	const cells = [];
	for (let i = 0; i < G; i++)
		for (let j = 0; j < G; j++) {
			const x1 = -0.5 + ((i + 0.5) * 2) / G, x2 = -0.5 + ((j + 0.5) * 2) / G;
			cells.push(<rect key={`${i},${j}`} x={(i * S) / G} y={S - ((j + 1) * S) / G} width={S / G + 0.5} height={S / G + 0.5} fill={out(x1, x2) > 0 ? C.red : C.blue} opacity={0.14} />);
		}
	const lab = LABEL[task];
	const correct = PTS.filter(([a, c], i) => (out(a, c) > 0 ? 1 : 0) === lab[i]).length;
	// mặt phẳng ẩn
	const S2 = 300;
	const HX = scale(-0.3, 2.3, 0, S2);
	const HY = scale(-0.3, 1.3, S2, 0);
	return (
		<Figure
			title="Một nơ-ron là một đường thẳng — và vì sao cần lớp ẩn"
			caption={
				<>
					Nơ-ron tính z = w₁x₁ + w₂x₂ + b và dự đoán lớp đỏ khi z &gt; 0: vùng đỏ và xanh luôn bị chia bởi <b>một đường thẳng</b>. Với AND hoặc OR, chỉnh w, b sẽ tìm được đường đúng cả 4 điểm. Với XOR (hai điểm đỏ ở hai góc đối nhau) thì không đường thẳng nào làm được.
					Bật “thêm lớp ẩn”: hai nơ-ron ReLU biến 4 điểm thành tọa độ mới (h₁, h₂) — hình bên phải. Trong không gian mới, hai điểm đỏ chụm vào nhau và một đường thẳng tách được. Nhìn từ không gian gốc, biên quyết định đã bị “bẻ gập”.
				</>
			}
		>
			<Controls>
				<Choice label="Bài toán:" value={task} onChange={setTask} options={[['and', 'AND'], ['or', 'OR'], ['xor', 'XOR']]} />
				<label className="viz-slider">
					<span><input type="checkbox" checked={hidden} onChange={(e) => setHidden(e.target.checked)} /> thêm lớp ẩn 2 nơ-ron ReLU (trọng số đã chọn sẵn cho XOR)</span>
				</label>
			</Controls>
			{!hidden && (
				<Controls>
					<Slider label="w₁" value={w1} min={-3} max={3} step={0.1} onChange={setW1} />
					<Slider label="w₂" value={w2} min={-3} max={3} step={0.1} onChange={setW2} />
					<Slider label="b" value={b} min={-3} max={3} step={0.05} onChange={setB} />
				</Controls>
			)}
			<div className="viz-row">
				<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Biên quyết định trong mặt phẳng đầu vào">
					{cells}
					<line x1={X(0)} x2={X(0)} y1={0} y2={S} stroke={C.grid} />
					<line y1={Y(0)} y2={Y(0)} x1={0} x2={S} stroke={C.grid} />
					{PTS.map(([a, c], i) => (
						<g key={i}>
							<circle cx={X(a)} cy={Y(c)} r={11} fill={lab[i] ? C.red : C.blue} stroke={(out(a, c) > 0 ? 1 : 0) === lab[i] ? 'none' : C.text} strokeWidth={3} />
							<text x={X(a)} y={Y(c) + 26} fontSize={12} textAnchor="middle" fill={C.text}>({a}, {c})</text>
						</g>
					))}
					<text x={S - 6} y={Y(0) - 6} fontSize={13} textAnchor="end" fill={C.text}>x₁</text>
					<text x={X(0) + 6} y={14} fontSize={13} fill={C.text}>x₂</text>
				</svg>
				{hidden && (
					<svg viewBox={`0 0 ${S2} ${S2}`} role="img" aria-label="Các điểm trong không gian lớp ẩn">
						<rect x={0} y={0} width={S2} height={S2} fill={C.blue} opacity={0.1} />
						<polygon points={`${HX(-0.1)},${HY(-0.3)} ${HX(2.3)},${HY(-0.3)} ${HX(2.3)},${HY(0.9)}`} fill={C.red} opacity={0.18} />
						<line x1={HX(-0.1)} y1={HY(-0.3)} x2={HX(2.3)} y2={HY(0.9)} stroke={C.text} strokeWidth={2} />
						<line x1={HX(0)} x2={HX(0)} y1={0} y2={S2} stroke={C.grid} />
						<line y1={HY(0)} y2={HY(0)} x1={0} x2={S2} stroke={C.grid} />
						{PTS.map(([a, c], i) => {
							const [h1, h2] = h(a, c);
							return (
								<g key={i}>
									<circle cx={HX(h1)} cy={HY(h2)} r={i === 1 ? 14 : 10} fill={LABEL.xor[i] ? C.red : C.blue} opacity={i === 1 ? 0.6 : 1} />
									<text x={HX(h1)} y={HY(h2) - 16 - (i === 2 ? 14 : 0)} fontSize={11} textAnchor="middle" fill={C.text}>từ ({a}, {c})</text>
								</g>
							);
						})}
						<text x={S2 - 6} y={HY(0) - 6} fontSize={13} textAnchor="end" fill={C.text}>h₁</text>
						<text x={HX(0) + 6} y={14} fontSize={13} fill={C.text}>h₂</text>
					</svg>
				)}
			</div>
			<div className="viz-kv">
				{hidden ? (
					<span>h₁ = ReLU(x₁ + x₂), h₂ = ReLU(x₁ + x₂ − 1), đầu ra z = h₁ − 2h₂ − 0,5</span>
				) : (
					<span>z = {fmt(w1)}·x₁ + {fmt(w2)}·x₂ + {fmt(b)}</span>
				)}
				<span>đúng <b>{correct}/4</b> điểm {correct === 4 ? '✓' : '(viền đen: sai)'}</span>
			</div>
		</Figure>
	);
}
