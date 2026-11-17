import { useState, type ReactElement } from 'react';
import { Arrow, C, Contours, Controls, Figure, Slider, fmt, scale } from './common';

const S = 380;
const R = 3;

/**
 * Hình 5.1: hàm bậc hai f(x) = ½ xᵀHx với H = R·diag(λ₁, λ₂)·Rᵀ.
 * Dấu hai trị riêng quyết định hình dạng: cái bát, thung lũng, yên ngựa, cái mũ.
 */
export default function QuadFormViz() {
	const [l1, setL1] = useState(2);
	const [l2, setL2] = useState(0.5);
	const [deg, setDeg] = useState(30);
	const [field, setField] = useState(true);
	const t = (deg * Math.PI) / 180;
	const c = Math.cos(t), s = Math.sin(t);
	// H = R diag R^T
	const h11 = l1 * c * c + l2 * s * s;
	const h22 = l1 * s * s + l2 * c * c;
	const h12 = (l1 - l2) * c * s;
	const f = (x: number, y: number) => 0.5 * (h11 * x * x + 2 * h12 * x * y + h22 * y * y);
	const X = scale(-R, R, 0, S);
	const Y = scale(-R, R, S, 0);
	const fmax = 0.5 * Math.max(Math.abs(l1), Math.abs(l2)) * R * R;
	const levels: number[] = [];
	for (let k = -6; k <= 6; k++) if (k !== 0) levels.push((fmax * k) / 7);
	let kind: string;
	if (l1 > 0 && l2 > 0) kind = 'Xác định dương → cái bát: lồi chặt, cực tiểu duy nhất tại gốc.';
	else if (l1 < 0 && l2 < 0) kind = 'Xác định âm → cái mũ úp: lõm, gốc là cực đại.';
	else if (l1 * l2 < 0) kind = 'Trị riêng trái dấu → yên ngựa: không lồi, không lõm; gốc là điểm yên ngựa.';
	else if (l1 === 0 && l2 === 0) kind = 'H = 0: hàm hằng.';
	else kind = 'Một trị riêng bằng 0 → máng (nửa xác định): lồi nhưng không chặt nếu trị còn lại ≥ 0; có cả một đường cực tiểu.';
	const arrows: ReactElement[] = [];
	if (field)
		for (let i = -2; i <= 2; i++)
			for (let j = -2; j <= 2; j++) {
				if (i === 0 && j === 0) continue;
				const x = i * 1.2, y = j * 1.2;
				const gx = h11 * x + h12 * y, gy = h12 * x + h22 * y;
				const g = Math.hypot(gx, gy) || 1;
				const len = Math.min(0.55, 0.18 * g);
				arrows.push(<Arrow key={`${i},${j}`} x1={X(x)} y1={Y(y)} x2={X(x - (gx / g) * len)} y2={Y(y - (gy / g) * len)} color={C.green} width={1.5} />);
			}
	return (
		<Figure
			title="Trị riêng của Hessian quyết định hình dạng hàm bậc hai"
			caption={
				<>
					Đường đồng mức của f(x) = ½xᵀHx. Hai đường nét đứt là hai <b>vector riêng</b> của H: dọc vector riêng thứ i, f là parabol ½λᵢu². Mũi tên xanh lá là −∇f (hướng gradient descent sẽ đi).
					Đồng mức đậm hơn ứng với giá trị lớn hơn.
				</>
			}
		>
			<Controls>
				<Slider label="λ₁" value={l1} min={-3} max={3} step={0.25} onChange={setL1} />
				<Slider label="λ₂" value={l2} min={-3} max={3} step={0.25} onChange={setL2} />
				<Slider label="góc xoay" value={deg} min={0} max={180} step={5} onChange={setDeg} show={`${deg}°`} />
				<label className="viz-slider">
					<span>
						<input type="checkbox" checked={field} onChange={(e) => setField(e.target.checked)} /> hiện −∇f
					</span>
				</label>
			</Controls>
			<svg viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 420 }} role="img" aria-label="Đường đồng mức hàm bậc hai">
				<Contours f={f} xr={[-R, R]} yr={[-R, R]} levels={levels.filter((l) => l > 0)} X={X} Y={Y} />
				<g opacity={0.9}>
					<Contours f={(x, y) => -f(x, y)} xr={[-R, R]} yr={[-R, R]} levels={levels.filter((l) => l > 0)} X={X} Y={Y} />
				</g>
				<line x1={X(-R * c)} y1={Y(-R * s)} x2={X(R * c)} y2={Y(R * s)} stroke={C.blue} strokeDasharray="6 4" strokeWidth={2} />
				<line x1={X(R * s)} y1={Y(-R * c)} x2={X(-R * s)} y2={Y(R * c)} stroke={C.orange} strokeDasharray="6 4" strokeWidth={2} />
				<text x={X(2.6 * c) + 4} y={Y(2.6 * s) - 4} fontSize={13} fill={C.blue}>λ₁</text>
				<text x={X(-2.6 * s) + 4} y={Y(2.6 * c) - 4} fontSize={13} fill={C.orange}>λ₂</text>
				{arrows}
				<circle cx={X(0)} cy={Y(0)} r={4} fill={C.text} />
			</svg>
			<div className="viz-kv">
				<span>
					H ≈ [[{fmt(h11)}; {fmt(h12)}], [{fmt(h12)}; {fmt(h22)}]]
				</span>
				<span>det H = λ₁λ₂ = <b>{fmt(l1 * l2, 3)}</b></span>
				<span>tr H = λ₁ + λ₂ = <b>{fmt(l1 + l2, 3)}</b></span>
			</div>
			<div className="viz-note">{kind}</div>
		</Figure>
	);
}
