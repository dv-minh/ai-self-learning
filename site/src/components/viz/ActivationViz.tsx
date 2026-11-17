import { useState } from 'react';
import { Axes, C, Choice, Controls, Figure, curvePath, scale } from './common';

type Act = 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'gelu';
const erf = (x: number) => {
	// xấp xỉ Abramowitz–Stegun 7.1.26 (đủ cho hình vẽ)
	const t = 1 / (1 + 0.3275911 * Math.abs(x));
	const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
	return x >= 0 ? y : -y;
};
const sig = (z: number) => 1 / (1 + Math.exp(-z));
const ACT: Record<Act, { label: string; f: (z: number) => number; d: (z: number) => number; note: string }> = {
	sigmoid: { label: 'sigmoid', f: sig, d: (z) => sig(z) * (1 - sig(z)), note: 'Đạo hàm lớn nhất chỉ 1/4 và gần 0 khi |z| lớn (bão hòa): qua nhiều lớp, gradient bị nhân với các số ≤ 1/4 nên biến mất.' },
	tanh: { label: 'tanh', f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2, note: 'Giống sigmoid nhưng đối xứng qua gốc (đầu ra trung bình 0), đạo hàm lớn nhất bằng 1; vẫn bão hòa ở hai đầu.' },
	relu: { label: 'ReLU', f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0), note: 'Đạo hàm đúng bằng 1 khi z > 0: gradient đi qua không bị co. Nhưng bằng 0 khi z < 0: nơ-ron luôn nhận z âm sẽ “chết”, không học nữa.' },
	leaky: { label: 'LeakyReLU (α = 0,1)', f: (z) => (z > 0 ? z : 0.1 * z), d: (z) => (z > 0 ? 1 : 0.1), note: 'Giữ một độ dốc nhỏ α ở phía âm nên nơ-ron không bao giờ chết hẳn.' },
	gelu: { label: 'GELU', f: (z) => 0.5 * z * (1 + erf(z / Math.SQRT2)), d: (z) => 0.5 * (1 + erf(z / Math.SQRT2)) + (z * Math.exp(-z * z / 2)) / Math.sqrt(2 * Math.PI), note: 'GELU(z) = z·Φ(z): bản “trơn” của ReLU, dùng trong Transformer; hơi âm ở gần −1.' },
};

/** Hình 7.2: đồ thị hàm kích hoạt và đạo hàm của nó. */
export default function ActivationViz() {
	const [a, setA] = useState<Act>('sigmoid');
	const F = ACT[a];
	const W = 540, H = 280;
	const X = scale(-5, 5, 30, W - 10);
	const Y = scale(-1.5, 3, H - 20, 10);
	return (
		<Figure
			title="Hàm kích hoạt φ(z) và đạo hàm φ′(z)"
			caption={<>Đường đậm: φ(z). Đường đỏ: φ′(z) — chính là thừa số nhân vào gradient khi lan truyền ngược qua hàm này (bài 7.3). Vùng nào φ′ gần 0 thì gradient đi qua vùng đó gần như mất hết.</>}
		>
			<Controls>
				<Choice value={a} onChange={setA} options={(Object.keys(ACT) as Act[]).map((k) => [k, ACT[k].label])} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đồ thị hàm kích hoạt và đạo hàm">
				<Axes x={X} y={Y} xr={[-5, 5]} yr={[-1.5, 3]} xticks={[-4, -2, 2, 4]} yticks={[-1, 1, 2]} xlabel="z" />
				<path d={curvePath(F.f, -5, 5, X, Y)} stroke={C.blue} strokeWidth={3} fill="none" />
				<path d={curvePath(F.d, -5, 5, X, Y, 400)} stroke={C.red} strokeWidth={2} strokeDasharray="6 3" fill="none" />
			</svg>
			<div className="viz-legend">
				<span style={{ ['--c' as string]: C.blue }}>φ(z)</span>
				<span style={{ ['--c' as string]: C.red }}>φ′(z)</span>
			</div>
			<div className="viz-note">{F.note}</div>
		</Figure>
	);
}
