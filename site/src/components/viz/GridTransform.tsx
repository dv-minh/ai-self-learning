import { useMemo, useState, type ReactElement } from 'react';

type Mat = [number, number, number, number]; // [a, b, c, d] = [[a, b], [c, d]]

const PRESETS: { name: string; m: Mat }[] = [
	{ name: 'Đơn vị', m: [1, 0, 0, 1] },
	{ name: 'Quay 90°', m: [0, -1, 1, 0] },
	{ name: 'Co giãn', m: [2, 0, 0, 0.5] },
	{ name: 'Phản xạ qua Ox', m: [1, 0, 0, -1] },
	{ name: 'Trượt', m: [1, 1, 0, 1] },
	{ name: 'Chiếu lên Ox', m: [1, 0, 0, 0] },
	{ name: 'Đối xứng', m: [2, 1, 1, 2] },
];

const R = 5; // vẽ trong [-R, R]
const SIZE = 420;
const S = SIZE / (2 * R);

const fmt = (x: number) => (Math.abs(x) < 1e-9 ? '0' : Number(x.toFixed(3)).toString().replace('.', ','));

/** Trị riêng và hướng riêng thực của ma trận 2×2 (chỉ để minh họa, dùng số thực). */
function eigen(m: Mat): { lambda: number; v: [number, number] }[] {
	const [a, b, c, d] = m;
	const tr = a + d;
	const det = a * d - b * c;
	const disc = tr * tr - 4 * det;
	if (disc < -1e-12) return [];
	const s = Math.sqrt(Math.max(disc, 0));
	const lams = s < 1e-9 ? [tr / 2] : [(tr + s) / 2, (tr - s) / 2];
	return lams.map((lambda) => {
		let v: [number, number];
		if (Math.abs(b) > 1e-9) v = [b, lambda - a];
		else if (Math.abs(c) > 1e-9) v = [lambda - d, c];
		else v = Math.abs(lambda - a) < 1e-9 ? [1, 0] : [0, 1];
		const n = Math.hypot(v[0], v[1]) || 1;
		return { lambda, v: [v[0] / n, v[1] / n] as [number, number] };
	});
}

interface Props {
	initial?: Mat;
	showEigen?: boolean;
}

/** Hình minh họa chỉ để xem: tấm lưới 2D bị biến đổi bởi ma trận 2×2 (docs/spec.md FR10). */
export default function GridTransform({ initial = [1, 1, 0, 1], showEigen = false }: Props) {
	const [cells, setCells] = useState<string[]>(initial.map(String));
	const [t, setT] = useState(1);
	// Giá trị số suy ra từ chuỗi người dùng nhập; ô trống hoặc sai định dạng tạm tính là 0.
	const target = useMemo(
		() => cells.map((c) => { const n = Number(c.trim().replace(',', '.')); return Number.isFinite(n) ? n : 0; }) as Mat,
		[cells],
	);
	const setTarget = (m: Mat) => setCells(m.map(String));

	// M(t) = (1 - t)I + t A: kéo thanh trượt để thấy phép biến đổi diễn ra dần
	const m: Mat = useMemo(() => [1 - t + t * target[0], t * target[1], t * target[2], 1 - t + t * target[3]], [target, t]);
	const apply = (x: number, y: number): [number, number] => [m[0] * x + m[1] * y, m[2] * x + m[3] * y];
	const px = (x: number) => SIZE / 2 + x * S;
	const py = (y: number) => SIZE / 2 - y * S;
	const det = target[0] * target[3] - target[1] * target[2];
	const eig = showEigen ? eigen(target) : [];

	const lines: ReactElement[] = [];
	for (let k = -R; k <= R; k++) {
		const [x1, y1] = apply(k, -R);
		const [x2, y2] = apply(k, R);
		const [x3, y3] = apply(-R, k);
		const [x4, y4] = apply(R, k);
		lines.push(
			<line key={`v${k}`} x1={px(x1)} y1={py(y1)} x2={px(x2)} y2={py(y2)} className="gt-grid" />,
			<line key={`h${k}`} x1={px(x3)} y1={py(y3)} x2={px(x4)} y2={py(y4)} className="gt-grid" />,
		);
	}
	const sq = [apply(0, 0), apply(1, 0), apply(1, 1), apply(0, 1)].map(([x, y]) => `${px(x)},${py(y)}`).join(' ');
	const e1 = apply(1, 0);
	const e2 = apply(0, 1);

	const setCell = (i: number, v: string) => setCells((prev) => prev.map((x, j) => (j === i ? v : x)));

	return (
		<figure className="grid-transform">
			<div className="gt-controls">
				<div className="gt-presets" role="group" aria-label="Phép biến đổi mẫu">
					{PRESETS.map((p) => (
						<button key={p.name} type="button" className={p.m.every((x, i) => x === target[i]) ? 'chosen' : ''} onClick={() => setTarget(p.m)}>
							{p.name}
						</button>
					))}
				</div>
				<div className="gt-matrix" role="group" aria-label="Ma trận A">
					<span className="gt-bracket">A =</span>
					<div className="gt-cells">
						{cells.map((v, i) => (
							<input key={i} aria-label={`Phần tử ${i < 2 ? 1 : 2},${(i % 2) + 1}`} value={v}
								onChange={(e) => setCell(i, e.target.value)} inputMode="decimal" />
						))}
					</div>
				</div>
				<label className="gt-slider">
					Mức biến đổi
					<input type="range" min={0} max={1} step={0.02} value={t} onChange={(e) => setT(Number(e.target.value))} aria-label="Mức biến đổi từ đơn vị đến A" />
				</label>
			</div>
			<svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Lưới 2D bị biến đổi bởi ma trận A" className="gt-svg">
				{/* lưới gốc */}
				{Array.from({ length: 2 * R + 1 }, (_, i) => i - R).flatMap((k) => [
					<line key={`ov${k}`} x1={px(k)} y1={py(-R)} x2={px(k)} y2={py(R)} className="gt-orig" />,
					<line key={`oh${k}`} x1={px(-R)} y1={py(k)} x2={px(R)} y2={py(k)} className="gt-orig" />,
				])}
				{lines}
				<polygon points={sq} className="gt-square" />
				{eig.map((e, i) => (
					<line key={`e${i}`} x1={px(-R * 1.5 * e.v[0])} y1={py(-R * 1.5 * e.v[1])} x2={px(R * 1.5 * e.v[0])} y2={py(R * 1.5 * e.v[1])} className="gt-eigen" />
				))}
				<defs>
					<marker id="gt-arrow-1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" className="gt-e1-fill" /></marker>
					<marker id="gt-arrow-2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" className="gt-e2-fill" /></marker>
				</defs>
				<line x1={px(0)} y1={py(0)} x2={px(e1[0])} y2={py(e1[1])} className="gt-e1" markerEnd="url(#gt-arrow-1)" />
				<line x1={px(0)} y1={py(0)} x2={px(e2[0])} y2={py(e2[1])} className="gt-e2" markerEnd="url(#gt-arrow-2)" />
			</svg>
			<figcaption>
				<span className="gt-key gt-key-e1">ảnh của e₁</span> <span className="gt-key gt-key-e2">ảnh của e₂</span> · hình bình hành là ảnh của hình vuông đơn vị ·
				det(A) = <strong>{fmt(det)}</strong> (diện tích nhân <strong>{fmt(Math.abs(det))}</strong>{det < 0 ? ', có lật hướng' : ''})
				{showEigen && (
					<>
						<br />
						{eig.length === 0
							? 'Ma trận này không có hướng riêng thực (không có đường nào được giữ nguyên hướng).'
							: eig.map((e, i) => `Trị riêng λ${eig.length > 1 ? i + 1 : ''} = ${fmt(e.lambda)}, hướng (${fmt(e.v[0])}; ${fmt(e.v[1])})`).join(' · ')}
						{eig.length > 0 && ' — đường nét đứt màu cam là các hướng riêng.'}
					</>
				)}
			</figcaption>
		</figure>
	);
}
