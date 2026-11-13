import { useMemo, useState } from 'react';

const SIZE = 420;
const R = 5; // vẽ trong [-R, R]
const S = SIZE / (2 * R);
const START: [number, number] = [4, 3];

const fmt = (x: number, d = 2) => Number(x.toFixed(d)).toString().replace('.', ',');

/**
 * Hình minh họa chỉ để xem (docs/spec.md FR10): đường đồng mức của f(x, y) = (x² + k·y²)/2 và quỹ đạo gradient descent.
 * Dùng số thực, không dùng để chấm bài.
 */
export default function GdPath() {
	const [k, setK] = useState(6);
	const [eta, setEta] = useState(0.15);
	const [steps, setSteps] = useState(20);
	const [momentum, setMomentum] = useState(0);

	const path = useMemo(() => {
		const pts: [number, number][] = [START];
		let [x, y] = START;
		let vx = 0;
		let vy = 0;
		for (let i = 0; i < steps; i++) {
			const gx = x;
			const gy = k * y;
			vx = momentum * vx + gx;
			vy = momentum * vy + gy;
			x -= eta * vx;
			y -= eta * vy;
			pts.push([x, y]);
		}
		return pts;
	}, [k, eta, steps, momentum]);

	const lmax = Math.max(1, k);
	const rho = Math.max(Math.abs(1 - eta), Math.abs(1 - eta * k));
	const status = momentum > 0 ? '' : eta < 2 / lmax ? 'hội tụ' : eta === 2 / lmax ? 'dao động đều' : 'phân kỳ';
	const px = (x: number) => SIZE / 2 + x * S;
	const py = (y: number) => SIZE / 2 - y * S;
	const inView = (p: [number, number]) => Math.abs(p[0]) <= R * 3 && Math.abs(p[1]) <= R * 3;
	const levels = [0.5, 2, 4.5, 8, 12.5, 18];

	return (
		<figure className="grid-transform gd-path">
			<div className="gt-controls">
				<label>
					Độ méo k = {k}
					<input type="range" min={1} max={10} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
				</label>
				<label>
					η = {fmt(eta, 3)}
					<input type="range" min={0.01} max={0.5} step={0.01} value={eta} onChange={(e) => setEta(Number(e.target.value))} />
				</label>
				<label>
					Số bước = {steps}
					<input type="range" min={1} max={40} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
				</label>
				<label>
					β (momentum) = {fmt(momentum, 2)}
					<input type="range" min={0} max={0.95} step={0.05} value={momentum} onChange={(e) => setMomentum(Number(e.target.value))} />
				</label>
			</div>
			<svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Quỹ đạo gradient descent trên các đường đồng mức" style={{ maxWidth: SIZE, width: '100%' }}>
				<line x1={0} y1={SIZE / 2} x2={SIZE} y2={SIZE / 2} stroke="var(--sl-color-gray-4)" strokeWidth={1} />
				<line x1={SIZE / 2} y1={0} x2={SIZE / 2} y2={SIZE} stroke="var(--sl-color-gray-4)" strokeWidth={1} />
				{levels.map((c) => (
					<ellipse key={c} cx={SIZE / 2} cy={SIZE / 2} rx={Math.sqrt(2 * c) * S} ry={Math.sqrt((2 * c) / k) * S} fill="none" stroke="var(--sl-color-gray-3)" strokeWidth={1} />
				))}
				<polyline fill="none" stroke="var(--sl-color-accent)" strokeWidth={2} points={path.filter(inView).map((p) => `${px(p[0])},${py(p[1])}`).join(' ')} />
				{path.filter(inView).map((p, i) => (
					<circle key={i} cx={px(p[0])} cy={py(p[1])} r={i === 0 ? 5 : 3} fill={i === 0 ? 'var(--sl-color-text)' : 'var(--sl-color-accent)'} />
				))}
				<circle cx={SIZE / 2} cy={SIZE / 2} r={4} fill="none" stroke="var(--sl-color-text)" strokeWidth={2} />
			</svg>
			<figcaption>
				Hàm f(x, y) = (x² + {k}y²)/2, bắt đầu từ (4; 3) (chấm đậm); vòng tròn là cực tiểu (0; 0). Trị riêng của Hessian là 1 và {k}.
				{momentum === 0 ? ` Hệ số co mỗi bước ρ = ${fmt(rho, 3)}; với η = ${fmt(eta, 3)} gradient descent ${status} (ngưỡng η < ${fmt(2 / lmax, 3)}).` : ' Có momentum: thử tăng β để thấy quỹ đạo dao động và vượt qua rồi quay lại.'} Cuối quỹ đạo: ({fmt(path[path.length - 1][0], 3)}; {fmt(path[path.length - 1][1], 3)}).
			</figcaption>
		</figure>
	);
}
