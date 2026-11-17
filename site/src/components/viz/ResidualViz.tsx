import { useMemo, useState } from 'react';
import { C, Controls, Figure, Slider, fmt, rng, scale } from './common';

const N = 40;

/** Hình 7.5: độ lớn gradient đi ngược qua N khối, có và không có đường tắt residual. */
export default function ResidualViz() {
	const [m, setM] = useState(0.6);
	const fp = useMemo(() => {
		const r = rng(9);
		return Array.from({ length: N }, () => m * (0.6 + 0.8 * r()) * (r() < 0.5 ? -1 : 1) * 0.5);
	}, [m]);
	// gradient tại khối k (đếm từ đầu ra): tích các đạo hàm phía sau
	const plain: number[] = [], res: number[] = [];
	let a = 1, b = 1;
	for (let k = 0; k < N; k++) {
		a *= Math.abs(fp[k]);
		b *= Math.abs(1 + fp[k]);
		plain.push(a);
		res.push(b);
	}
	const W = 540, H = 260;
	const X = scale(0.5, N + 0.5, 50, W - 10);
	const lo = -16, hi = 4;
	const Y = scale(lo, hi, H - 22, 10);
	const ly = (v: number) => Y(Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-30)))));
	return (
		<Figure
			title="Đường tắt residual giữ cho gradient không tắt"
			caption={
				<>
					Mỗi khối có đạo hàm cục bộ F′ nhỏ (độ lớn quanh {fmt(m / 2, 2)}, dấu ngẫu nhiên). Không có residual, gradient về tới khối thứ k là tích k số F′ — nhỏ dần theo cấp số nhân (đường đỏ, thang log).
					Có residual y = x + F(x), mỗi thừa số thành 1 + F′, quanh 1 — gradient giữ được độ lớn (đường xanh) dù mạng rất sâu.
				</>
			}
		>
			<Controls>
				<Slider label="|F′| điển hình" value={m} min={0.1} max={1.6} step={0.05} onChange={setM} show={fmt(m / 2, 3)} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Độ lớn gradient theo độ sâu">
				{[-16, -12, -8, -4, 0, 4].map((e) => (
					<g key={e}>
						<line x1={X(0.5)} x2={X(N + 0.5)} y1={Y(e)} y2={Y(e)} stroke={C.grid} strokeWidth={0.6} />
						<text x={X(0.5) - 4} y={Y(e) + 4} fontSize={11} textAnchor="end" fill={C.muted}>10^{e}</text>
					</g>
				))}
				<line x1={X(0.5)} x2={X(N + 0.5)} y1={Y(0)} y2={Y(0)} stroke={C.axis} />
				<polyline points={plain.map((v, i) => `${X(i + 1)},${ly(v)}`).join(' ')} fill="none" stroke={C.red} strokeWidth={2.5} />
				<polyline points={res.map((v, i) => `${X(i + 1)},${ly(v)}`).join(' ')} fill="none" stroke={C.blue} strokeWidth={2.5} />
				<text x={X(N)} y={H - 4} fontSize={12} textAnchor="end" fill={C.text}>số khối đi ngược qua →</text>
			</svg>
			<div className="viz-legend">
				<span style={{ ['--c' as string]: C.red }}>không residual: Π|F′ₖ|</span>
				<span style={{ ['--c' as string]: C.blue }}>có residual: Π|1 + F′ₖ|</span>
			</div>
			<div className="viz-kv">
				<span>sau {N} khối — không residual: <b>{plain[N - 1].toExponential(1).replace('.', ',')}</b></span>
				<span>có residual: <b>{fmt(res[N - 1], 3)}</b></span>
			</div>
		</Figure>
	);
}
