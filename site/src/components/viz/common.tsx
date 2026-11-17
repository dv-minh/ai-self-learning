import { useEffect, useRef, useState, type ReactNode } from 'react';

/**
 * Tiện ích dùng chung cho các hình minh họa tương tác (docs/spec.md FR10).
 * Hình chỉ để xem: tính bằng số thực của trình duyệt, KHÔNG dùng để chấm bài.
 */

export const fmt = (x: number, d = 2) => {
	if (!Number.isFinite(x)) return x > 0 ? '∞' : x < 0 ? '−∞' : '—';
	const s = Number(x.toFixed(d)).toString();
	return s.replace('-', '−').replace('.', ',');
};

/** Thang tuyến tính [d0, d1] → [r0, r1]. */
export const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Bộ sinh số giả ngẫu nhiên có hạt giống (mulberry32) để hình giống nhau mỗi lần mở. */
export function rng(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Phân phối chuẩn từ một bộ sinh đều (Box–Muller). */
export const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

/** Bảng màu dùng chung; đều đọc được ở cả nền sáng và tối. */
export const C = {
	accent: 'var(--sl-color-accent)',
	text: 'var(--sl-color-text)',
	grid: 'var(--sl-color-gray-5)',
	axis: 'var(--sl-color-gray-3)',
	muted: 'var(--sl-color-gray-3)',
	red: 'var(--viz-red)',
	blue: 'var(--viz-blue)',
	green: 'var(--viz-green)',
	orange: 'var(--viz-orange)',
	purple: 'var(--viz-purple)',
};

export const SERIES = [C.blue, C.orange, C.green, C.purple, C.red];

export function Figure({ title, children, caption, className = '' }: { title?: string; children: ReactNode; caption?: ReactNode; className?: string }) {
	return (
		<figure className={`viz not-content ${className}`}>
			{title && <div className="viz-title">{title}</div>}
			{children}
			{caption && <figcaption>{caption}</figcaption>}
		</figure>
	);
}

export function Controls({ children }: { children: ReactNode }) {
	return <div className="viz-controls">{children}</div>;
}

export function Slider(props: { label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; show?: string }) {
	const { label, value, min, max, step, onChange, show } = props;
	return (
		<label className="viz-slider">
			<span>
				{label} = <b>{show ?? fmt(value, 3)}</b>
			</span>
			<input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
		</label>
	);
}

export function Choice<T extends string>(props: { label?: ReactNode; value: T; options: [T, ReactNode][]; onChange: (v: T) => void }) {
	return (
		<div className="viz-choice" role="group">
			{props.label && <span className="viz-choice-label">{props.label}</span>}
			{props.options.map(([v, l]) => (
				<button key={v} type="button" className={v === props.value ? 'on' : ''} aria-pressed={v === props.value} onClick={() => props.onChange(v)}>
					{l}
				</button>
			))}
		</div>
	);
}

export function Buttons({ children }: { children: ReactNode }) {
	return <div className="viz-buttons">{children}</div>;
}

/** Bộ đếm bước có nút Lùi / Tiếp / Chạy / Về đầu. */
export function useStepper(max: number, intervalMs = 700) {
	const [step, setStep] = useState(0);
	const [playing, setPlaying] = useState(false);
	const ref = useRef<number | null>(null);
	useEffect(() => {
		if (!playing) return;
		ref.current = window.setInterval(() => {
			setStep((s) => {
				if (s >= max) {
					setPlaying(false);
					return s;
				}
				return s + 1;
			});
		}, intervalMs);
		return () => {
			if (ref.current) window.clearInterval(ref.current);
		};
	}, [playing, max, intervalMs]);
	useEffect(() => {
		if (step > max) setStep(max);
	}, [max, step]);
	const controls = (
		<Buttons>
			<button type="button" onClick={() => { setPlaying(false); setStep(0); }} disabled={step === 0}>⏮ Về đầu</button>
			<button type="button" onClick={() => { setPlaying(false); setStep((s) => Math.max(0, s - 1)); }} disabled={step === 0}>◀ Lùi</button>
			<button type="button" className="primary" onClick={() => { setPlaying(false); setStep((s) => Math.min(max, s + 1)); }} disabled={step >= max}>Bước tiếp ▶</button>
			<button type="button" onClick={() => { if (step >= max) setStep(0); setPlaying((p) => !p); }}>{playing ? '⏸ Dừng' : '⏵ Tự chạy'}</button>
			<span className="viz-step-count">bước {step}/{max}</span>
		</Buttons>
	);
	return { step, setStep, controls };
}

/** Khung trục tọa độ cho đồ thị 2D trong SVG. */
export function Axes(props: {
	x: (v: number) => number;
	y: (v: number) => number;
	xr: [number, number];
	yr: [number, number];
	xticks?: number[];
	yticks?: number[];
	xlabel?: string;
	ylabel?: string;
	grid?: boolean;
}) {
	const { x, y, xr, yr, xticks = [], yticks = [], xlabel, ylabel, grid = true } = props;
	const x0 = clamp(0, xr[0], xr[1]);
	const y0 = clamp(0, yr[0], yr[1]);
	return (
		<g className="viz-axes">
			{grid && xticks.map((t) => <line key={`gx${t}`} x1={x(t)} x2={x(t)} y1={y(yr[0])} y2={y(yr[1])} stroke={C.grid} strokeWidth={0.6} />)}
			{grid && yticks.map((t) => <line key={`gy${t}`} y1={y(t)} y2={y(t)} x1={x(xr[0])} x2={x(xr[1])} stroke={C.grid} strokeWidth={0.6} />)}
			<line x1={x(xr[0])} x2={x(xr[1])} y1={y(y0)} y2={y(y0)} stroke={C.axis} strokeWidth={1.2} />
			<line y1={y(yr[0])} y2={y(yr[1])} x1={x(x0)} x2={x(x0)} stroke={C.axis} strokeWidth={1.2} />
			{xticks.map((t) => (
				<text key={`tx${t}`} x={x(t)} y={y(y0) + 14} fontSize={11} textAnchor="middle" fill={C.muted}>{fmt(t, 2)}</text>
			))}
			{yticks.map((t) => (
				<text key={`ty${t}`} x={x(x0) - 5} y={y(t) + 4} fontSize={11} textAnchor="end" fill={C.muted}>{fmt(t, 2)}</text>
			))}
			{xlabel && <text x={x(xr[1]) - 2} y={y(y0) - 6} fontSize={13} textAnchor="end" fill={C.text} fontStyle="italic">{xlabel}</text>}
			{ylabel && <text x={x(x0) + 6} y={y(yr[1]) + 12} fontSize={13} fill={C.text} fontStyle="italic">{ylabel}</text>}
		</g>
	);
}

/** Đường cong của hàm f trên [a, b]. */
export function curvePath(f: (v: number) => number, a: number, b: number, X: (v: number) => number, Y: (v: number) => number, n = 200) {
	const pts: string[] = [];
	for (let i = 0; i <= n; i++) {
		const t = a + ((b - a) * i) / n;
		const v = f(t);
		if (!Number.isFinite(v)) continue;
		pts.push(`${pts.length ? 'L' : 'M'}${X(t).toFixed(2)},${Y(v).toFixed(2)}`);
	}
	return pts.join(' ');
}

/** Mũi tên từ (x1,y1) đến (x2,y2) (tọa độ pixel). */
export function Arrow({ x1, y1, x2, y2, color = C.accent, width = 2, dash }: { x1: number; y1: number; x2: number; y2: number; color?: string; width?: number; dash?: string }) {
	const dx = x2 - x1;
	const dy = y2 - y1;
	const len = Math.hypot(dx, dy);
	if (len < 1e-6) return null;
	const ux = dx / len;
	const uy = dy / len;
	const h = Math.min(9, len * 0.45);
	const bx = x2 - ux * h;
	const by = y2 - uy * h;
	return (
		<g>
			<line x1={x1} y1={y1} x2={bx} y2={by} stroke={color} strokeWidth={width} strokeDasharray={dash} />
			<polygon points={`${x2},${y2} ${bx - uy * h * 0.5},${by + ux * h * 0.5} ${bx + uy * h * 0.5},${by - ux * h * 0.5}`} fill={color} />
		</g>
	);
}

/** Đường đồng mức bằng marching squares (đủ cho hình minh họa). */
export function contourSegments(f: (x: number, y: number) => number, xr: [number, number], yr: [number, number], level: number, n = 60) {
	const segs: [number, number, number, number][] = [];
	const dx = (xr[1] - xr[0]) / n;
	const dy = (yr[1] - yr[0]) / n;
	const val: number[][] = [];
	for (let i = 0; i <= n; i++) {
		val.push([]);
		for (let j = 0; j <= n; j++) val[i].push(f(xr[0] + i * dx, yr[0] + j * dy) - level);
	}
	const lerp = (a: number, b: number) => a / (a - b);
	for (let i = 0; i < n; i++)
		for (let j = 0; j < n; j++) {
			const x = xr[0] + i * dx;
			const y = yr[0] + j * dy;
			const a = val[i][j], b = val[i + 1][j], c = val[i + 1][j + 1], d = val[i][j + 1];
			const pts: [number, number][] = [];
			if (a * b < 0) pts.push([x + lerp(a, b) * dx, y]);
			if (b * c < 0) pts.push([x + dx, y + lerp(b, c) * dy]);
			if (c * d < 0) pts.push([x + (1 - lerp(c, d)) * dx, y + dy]);
			if (d * a < 0) pts.push([x, y + (1 - lerp(d, a)) * dy]);
			if (pts.length >= 2) segs.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
			if (pts.length === 4) segs.push([pts[2][0], pts[2][1], pts[3][0], pts[3][1]]);
		}
	return segs;
}

export function Contours(props: { f: (x: number, y: number) => number; xr: [number, number]; yr: [number, number]; levels: number[]; X: (v: number) => number; Y: (v: number) => number; n?: number }) {
	const { f, xr, yr, levels, X, Y, n } = props;
	return (
		<g>
			{levels.map((lv, k) => {
				const d = contourSegments(f, xr, yr, lv, n)
					.map(([a, b, c, e]) => `M${X(a).toFixed(1)},${Y(b).toFixed(1)}L${X(c).toFixed(1)},${Y(e).toFixed(1)}`)
					.join('');
				return <path key={k} d={d} stroke={C.muted} strokeOpacity={0.35 + (0.5 * k) / Math.max(1, levels.length)} strokeWidth={1} fill="none" />;
			})}
		</g>
	);
}

/** Thang màu cho ô ma trận/heatmap: giá trị trong [-1, 1] → xanh–trắng–đỏ (dùng opacity để hợp cả hai chế độ màu). */
export function heatFill(v: number) {
	const a = clamp(Math.abs(v), 0, 1);
	return { fill: v >= 0 ? 'var(--viz-red)' : 'var(--viz-blue)', fillOpacity: 0.08 + 0.75 * a };
}

/** Cho phép kéo một điểm trong SVG; trả về tọa độ SVG (viewBox). */
export function useSvgDrag(onMove: (sx: number, sy: number) => void) {
	const svgRef = useRef<SVGSVGElement | null>(null);
	const dragging = useRef(false);
	const toSvg = (e: { clientX: number; clientY: number }) => {
		const svg = svgRef.current;
		if (!svg) return null;
		const pt = svg.createSVGPoint();
		pt.x = e.clientX;
		pt.y = e.clientY;
		const m = svg.getScreenCTM();
		if (!m) return null;
		const p = pt.matrixTransform(m.inverse());
		return [p.x, p.y] as const;
	};
	const handlers = {
		onPointerDown: (e: React.PointerEvent) => {
			dragging.current = true;
			(e.target as Element).setPointerCapture?.(e.pointerId);
			const p = toSvg(e);
			if (p) onMove(p[0], p[1]);
		},
		onPointerMove: (e: React.PointerEvent) => {
			if (!dragging.current) return;
			const p = toSvg(e);
			if (p) onMove(p[0], p[1]);
		},
		onPointerUp: () => {
			dragging.current = false;
		},
	};
	return { svgRef, handlers, toSvg };
}
