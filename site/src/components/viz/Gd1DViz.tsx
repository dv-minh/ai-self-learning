import { useMemo, useState } from 'react';
import { Arrow, Axes, C, Choice, Controls, Figure, Slider, curvePath, fmt, scale, useStepper } from './common';

type Key = 'sq' | 'well' | 'flat';
const FUNCS: Record<Key, { label: string; f: (x: number) => number; df: (x: number) => number; xr: [number, number]; yr: [number, number]; x0: number; eta: number }> = {
	sq: { label: 'f(x) = x²', f: (x) => x * x, df: (x) => 2 * x, xr: [-3.2, 3.2], yr: [-1, 10], x0: 2.8, eta: 0.2 },
	well: { label: 'f(x) = x⁴/4 − x² + x/2 (hai giếng)', f: (x) => x ** 4 / 4 - x * x + x / 2, df: (x) => x ** 3 - 2 * x + 0.5, xr: [-2.4, 2.4], yr: [-2.5, 3], x0: 2.1, eta: 0.1 },
	flat: { label: 'f(x) = x⁴/8 (đáy phẳng)', f: (x) => x ** 4 / 8, df: (x) => x ** 3 / 2, xr: [-2.6, 2.6], yr: [-0.5, 5.5], x0: 2.4, eta: 0.3 },
};
const W = 560, H = 330;
const MAXSTEP = 15;

/** Hình 5.2: từng bước gradient descent một biến — tiếp tuyến, độ dốc, bước nhảy. */
export default function Gd1DViz() {
	const [key, setKey] = useState<Key>('sq');
	const F = FUNCS[key];
	const [eta, setEta] = useState(F.eta);
	const [x0, setX0] = useState(F.x0);
	const { step, setStep, controls } = useStepper(MAXSTEP, 900);
	const choose = (k: Key) => {
		setKey(k);
		setEta(FUNCS[k].eta);
		setX0(FUNCS[k].x0);
		setStep(0);
	};
	const xs = useMemo(() => {
		const out = [x0];
		for (let i = 0; i < MAXSTEP; i++) {
			const x = out[i];
			out.push(x - eta * F.df(x));
		}
		return out;
	}, [x0, eta, F]);
	const X = scale(F.xr[0], F.xr[1], 30, W - 10);
	const Y = scale(F.yr[0], F.yr[1], H - 28, 10);
	const inside = (x: number) => Number.isFinite(x) && x >= F.xr[0] - 0.5 && x <= F.xr[1] + 0.5;
	const cur = xs[step];
	const nxt = xs[step + 1];
	const g = F.df(cur);
	const tangent = (x: number) => F.f(cur) + g * (x - cur);
	const ty = Y(F.yr[0]) + 0;
	return (
		<Figure
			title="Một bước gradient descent nhìn gần"
			caption={
				<>
					Đường cam là <b>tiếp tuyến</b> tại điểm hiện tại, độ dốc bằng f′(xₜ). Mũi tên dưới trục hoành là bước đi −η·f′(xₜ): dốc lên bên phải (f′ &gt; 0) thì lùi sang trái, dốc xuống thì tiến sang phải.
					Gần đáy độ dốc nhỏ dần nên bước tự ngắn lại dù η không đổi. Bấm “Bước tiếp” để đi từng bước.
				</>
			}
		>
			<Controls>
				<Choice value={key} onChange={choose} options={(Object.keys(FUNCS) as Key[]).map((k) => [k, FUNCS[k].label])} />
			</Controls>
			<Controls>
				<Slider label="learning rate η" value={eta} min={0.01} max={1.1} step={0.01} onChange={(v) => { setEta(v); setStep(0); }} />
				<Slider label="điểm bắt đầu x₀" value={x0} min={F.xr[0] + 0.3} max={F.xr[1] - 0.3} step={0.05} onChange={(v) => { setX0(v); setStep(0); }} />
			</Controls>
			{controls}
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đồ thị hàm và các bước gradient descent">
				<Axes x={X} y={Y} xr={F.xr} yr={F.yr} xticks={[-2, -1, 1, 2]} yticks={[]} xlabel="x" ylabel="f(x)" />
				<path d={curvePath(F.f, F.xr[0], F.xr[1], X, Y)} stroke={C.blue} strokeWidth={2.5} fill="none" />
				{xs.slice(0, step + 1).map((x, i) =>
					inside(x) && i < step && inside(xs[i + 1]) ? (
						<line key={`p${i}`} x1={X(x)} y1={Y(F.f(x))} x2={X(xs[i + 1])} y2={Y(F.f(xs[i + 1]))} stroke={C.purple} strokeOpacity={0.5} strokeWidth={1.5} />
					) : null,
				)}
				{xs.slice(0, step + 1).map((x, i) => (inside(x) ? <circle key={i} cx={X(x)} cy={Y(F.f(x))} r={i === step ? 6 : 3.5} fill={i === step ? C.purple : C.purple} opacity={i === step ? 1 : 0.45} /> : null))}
				{inside(cur) && (
					<>
						<path d={curvePath(tangent, cur - 0.9, cur + 0.9, X, Y, 2)} stroke={C.orange} strokeWidth={2} />
						<line x1={X(cur)} y1={Y(F.f(cur))} x2={X(cur)} y2={ty} stroke={C.purple} strokeDasharray="3 3" />
						{inside(nxt) && (
							<>
								<Arrow x1={X(cur)} y1={ty - 8} x2={X(nxt)} y2={ty - 8} color={C.red} width={2.5} />
								<line x1={X(nxt)} y1={ty} x2={X(nxt)} y2={Y(F.f(nxt))} stroke={C.red} strokeDasharray="3 3" />
								<circle cx={X(nxt)} cy={Y(F.f(nxt))} r={5} fill="none" stroke={C.red} strokeWidth={2} />
							</>
						)}
					</>
				)}
			</svg>
			<table className="viz-table">
				<thead>
					<tr><th>t</th><th>xₜ</th><th>f′(xₜ)</th><th>bước −η·f′(xₜ)</th><th>xₜ₊₁</th><th>f(xₜ₊₁)</th></tr>
				</thead>
				<tbody>
					<tr>
						<td>{step}</td>
						<td>{fmt(cur, 4)}</td>
						<td>{fmt(g, 4)}</td>
						<td className="hl">{fmt(-eta * g, 4)}</td>
						<td>{fmt(nxt, 4)}</td>
						<td>{fmt(F.f(nxt), 4)}</td>
					</tr>
				</tbody>
			</table>
			<div className="viz-note">
				{!inside(nxt) || Math.abs(nxt) > 1e3
					? 'Bước nhảy ra khỏi khung: η quá lớn làm gradient descent phân kỳ.'
					: Math.abs(-eta * g) < 1e-3
						? 'Bước gần như bằng 0: đã đến gần điểm có f′ = 0 (điểm dừng).'
						: Math.sign(nxt - (F.xr[0] + F.xr[1]) / 2) !== Math.sign(cur - (F.xr[0] + F.xr[1]) / 2) && key === 'sq'
							? 'Bước vượt qua đáy sang phía bên kia: η đang lớn hơn 1/f″, quỹ đạo dao động hai bên.'
							: key === 'well'
								? 'Hàm hai giếng: điểm bắt đầu khác nhau có thể rơi vào giếng khác nhau (cực tiểu địa phương). Thử x₀ âm và x₀ dương.'
								: key === 'flat'
									? 'Đáy phẳng: f′ rất nhỏ gần 0 nên các bước cuối rất ngắn, hội tụ chậm.'
									: 'Mỗi bước: tính độ dốc tại điểm hiện tại → nhân với η → đi ngược dấu độ dốc.'}
			</div>
		</Figure>
	);
}
