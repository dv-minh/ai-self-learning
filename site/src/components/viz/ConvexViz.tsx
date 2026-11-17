import { useState } from 'react';
import { Axes, C, Choice, Controls, Figure, Slider, curvePath, fmt, scale } from './common';

type Key = 'sq' | 'exp' | 'abs' | 'well' | 'sin';
const FUNCS: Record<Key, { label: string; f: (x: number) => number; d2: (x: number) => number; yr: [number, number]; convex: boolean }> = {
	sq: { label: 'x²', f: (x) => x * x, d2: () => 2, yr: [-1, 9], convex: true },
	exp: { label: 'eˣ', f: (x) => Math.exp(x), d2: (x) => Math.exp(x), yr: [-1, 9], convex: true },
	abs: { label: '|x|', f: (x) => Math.abs(x), d2: () => 0, yr: [-1, 4], convex: true },
	well: { label: 'x⁴ − 3x² (hai giếng)', f: (x) => x ** 4 - 3 * x * x, d2: (x) => 12 * x * x - 6, yr: [-3, 6], convex: false },
	sin: { label: 'sin(2x) + x²/4', f: (x) => Math.sin(2 * x) + (x * x) / 4, d2: (x) => -4 * Math.sin(2 * x) + 0.5, yr: [-1.5, 3.5], convex: false },
};

const W = 520, H = 320;

/** Hình 5.1: dây cung (chord) nằm trên hay dưới đồ thị — định nghĩa hàm lồi bằng hình. */
export default function ConvexViz() {
	const [key, setKey] = useState<Key>('well');
	const [a, setA] = useState(-1.6);
	const [b, setB] = useState(1.2);
	const [theta, setTheta] = useState(0.5);
	const F = FUNCS[key];
	const xr: [number, number] = [-2.2, 2.2];
	const X = scale(xr[0], xr[1], 30, W - 10);
	const Y = scale(F.yr[0], F.yr[1], H - 25, 10);
	const m = theta * a + (1 - theta) * b;
	const onCurve = F.f(m);
	const onChord = theta * F.f(a) + (1 - theta) * F.f(b);
	const ok = onCurve <= onChord + 1e-9;
	// vùng cong lên / cong xuống theo dấu f''
	const bands: [number, number, boolean][] = [];
	const n = 220;
	let start = xr[0];
	let sign = F.d2(xr[0]) >= 0;
	for (let i = 1; i <= n; i++) {
		const x = xr[0] + ((xr[1] - xr[0]) * i) / n;
		const s = F.d2(x) >= 0;
		if (s !== sign || i === n) {
			bands.push([start, x, sign]);
			start = x;
			sign = s;
		}
	}
	return (
		<Figure
			title="Dây cung nằm trên đồ thị ⇔ hàm lồi"
			caption={
				<>
					Hai điểm đỏ là (a, f(a)) và (b, f(b)); đoạn thẳng nối chúng là <b>dây cung</b>. Điểm tím nằm trên đồ thị tại x = θa + (1−θ)b; điểm cam nằm trên dây cung ở cùng hoành độ.
					Hàm lồi khi điểm tím <b>không bao giờ</b> cao hơn điểm cam, với mọi a, b, θ. Nền xanh: f″ ≥ 0 (cong lên); nền đỏ: f″ &lt; 0 (cong xuống) — chỉ cần một đoạn cong xuống là tìm được dây cung nằm dưới đồ thị.
				</>
			}
		>
			<Controls>
				<Choice value={key} onChange={setKey} options={(Object.keys(FUNCS) as Key[]).map((k) => [k, FUNCS[k].label])} />
			</Controls>
			<Controls>
				<Slider label="a" value={a} min={-2} max={2} step={0.05} onChange={setA} />
				<Slider label="b" value={b} min={-2} max={2} step={0.05} onChange={setB} />
				<Slider label="θ" value={theta} min={0} max={1} step={0.01} onChange={setTheta} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đồ thị hàm số và dây cung">
				{bands.map(([s, e, up], i) => (
					<rect key={i} x={X(s)} y={10} width={X(e) - X(s)} height={H - 35} fill={up ? C.green : C.red} opacity={0.07} />
				))}
				<Axes x={X} y={Y} xr={xr} yr={F.yr} xticks={[-2, -1, 1, 2]} yticks={[]} xlabel="x" />
				<path d={curvePath(F.f, xr[0], xr[1], X, Y)} stroke={C.blue} strokeWidth={2.5} fill="none" />
				<line x1={X(a)} y1={Y(F.f(a))} x2={X(b)} y2={Y(F.f(b))} stroke={C.orange} strokeWidth={2} />
				<line x1={X(m)} y1={Y(onCurve)} x2={X(m)} y2={Y(onChord)} stroke={ok ? C.green : C.red} strokeWidth={2} strokeDasharray="4 3" />
				<circle cx={X(a)} cy={Y(F.f(a))} r={5} fill={C.red} />
				<circle cx={X(b)} cy={Y(F.f(b))} r={5} fill={C.red} />
				<circle cx={X(m)} cy={Y(onChord)} r={5} fill={C.orange} />
				<circle cx={X(m)} cy={Y(onCurve)} r={5} fill={C.purple} />
				<text x={X(a)} y={Y(F.f(a)) - 9} fontSize={12} textAnchor="middle" fill={C.text}>a</text>
				<text x={X(b)} y={Y(F.f(b)) - 9} fontSize={12} textAnchor="middle" fill={C.text}>b</text>
			</svg>
			<div className="viz-kv">
				<span>f(θa + (1−θ)b) = <b>{fmt(onCurve, 3)}</b></span>
				<span>θf(a) + (1−θ)f(b) = <b>{fmt(onChord, 3)}</b></span>
				<span style={{ color: ok ? C.green : C.red }}><b>{ok ? '✓ đồ thị nằm dưới dây cung' : '✗ đồ thị vượt lên trên dây cung → không lồi'}</b></span>
			</div>
			<div className="viz-note">
				Hàm đang chọn: <b>{F.label}</b> — {F.convex ? 'lồi (thử mọi a, b, θ đều thấy ✓).' : 'không lồi: hãy đặt a, b ở hai bên một vùng nền đỏ để thấy ✗.'}
			</div>
		</Figure>
	);
}
