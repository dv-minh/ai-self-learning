import { useState } from 'react';
import { C, Figure, fmt, rng, useStepper } from './common';

type Vals = { x1: number; x2: number; w1: number; w2: number; b: number; v: number; c: number; y: number };
const DEFAULT: Vals = { x1: 2, x2: -1, w1: 0.5, w2: 1, b: 0.5, v: 2, c: -0.5, y: 2 };

type NodeId = 'x1' | 'x2' | 'w1' | 'w2' | 'b' | 'z' | 'h' | 'v' | 'c' | 'yh' | 'y' | 'L';
const POS: Record<NodeId, [number, number]> = {
	x1: [50, 40], w1: [50, 100], x2: [50, 170], w2: [50, 230], b: [50, 290],
	z: [210, 165], h: [340, 165], v: [340, 60], c: [340, 270], yh: [470, 165], y: [470, 60], L: [590, 165],
};
const LABEL: Record<NodeId, string> = { x1: 'x₁', x2: 'x₂', w1: 'w₁', w2: 'w₂', b: 'b', z: 'z', h: 'h = ReLU(z)', v: 'v', c: 'c', yh: 'ŷ', y: 'y', L: 'L' };
const EDGES: [NodeId, NodeId][] = [['x1', 'z'], ['w1', 'z'], ['x2', 'z'], ['w2', 'z'], ['b', 'z'], ['z', 'h'], ['h', 'yh'], ['v', 'yh'], ['c', 'yh'], ['yh', 'L'], ['y', 'L']];

/** Hình 7.3: đồ thị tính toán của một mạng tí hon; đi forward rồi backward từng bước. */
export default function BackpropViz() {
	const [p, setP] = useState<Vals>(DEFAULT);
	const [seed, setSeed] = useState(1);
	const z = p.w1 * p.x1 + p.w2 * p.x2 + p.b;
	const h = Math.max(0, z);
	const yh = p.v * h + p.c;
	const L = 0.5 * (yh - p.y) ** 2;
	const gyh = yh - p.y;
	const gv = gyh * h, gc = gyh;
	const gh = gyh * p.v;
	const gz = gh * (z > 0 ? 1 : 0);
	const gw1 = gz * p.x1, gw2 = gz * p.x2, gb = gz;
	type S = { title: string; text: string; fwd: NodeId[]; bwd: NodeId[] };
	const steps: S[] = [
		{ title: 'Bắt đầu', text: 'Các ô bên trái là đầu vào và tham số, đã biết giá trị. Bấm “Bước tiếp” để chạy lượt forward.', fwd: [], bwd: [] },
		{ title: 'Forward 1: tổng có trọng số', text: `z = w₁x₁ + w₂x₂ + b = ${fmt(p.w1)}·${fmt(p.x1)} + ${fmt(p.w2)}·${fmt(p.x2)} + ${fmt(p.b)} = ${fmt(z, 3)}`, fwd: ['z'], bwd: [] },
		{ title: 'Forward 2: kích hoạt', text: `h = ReLU(z) = max(0; ${fmt(z, 3)}) = ${fmt(h, 3)}`, fwd: ['z', 'h'], bwd: [] },
		{ title: 'Forward 3: đầu ra', text: `ŷ = v·h + c = ${fmt(p.v)}·${fmt(h, 3)} + ${fmt(p.c)} = ${fmt(yh, 3)}`, fwd: ['z', 'h', 'yh'], bwd: [] },
		{ title: 'Forward 4: mất mát', text: `L = ½(ŷ − y)² = ½(${fmt(yh, 3)} − ${fmt(p.y)})² = ${fmt(L, 4)}. Mọi giá trị trung gian được LƯU lại để dùng cho lượt ngược.`, fwd: ['z', 'h', 'yh', 'L'], bwd: [] },
		{ title: 'Backward 1: bắt đầu từ L', text: `∂L/∂L = 1, và ∂L/∂ŷ = ŷ − y = ${fmt(gyh, 3)} (đạo hàm cục bộ của ½(ŷ − y)²).`, fwd: ['z', 'h', 'yh', 'L'], bwd: ['L', 'yh'] },
		{ title: 'Backward 2: qua phép ŷ = v·h + c', text: `Đạo hàm cục bộ: ∂ŷ/∂v = h, ∂ŷ/∂c = 1, ∂ŷ/∂h = v. Nhân với gradient đến: ∂L/∂v = ${fmt(gyh, 3)}·${fmt(h, 3)} = ${fmt(gv, 3)}; ∂L/∂c = ${fmt(gc, 3)}; ∂L/∂h = ${fmt(gyh, 3)}·${fmt(p.v)} = ${fmt(gh, 3)}.`, fwd: ['z', 'h', 'yh', 'L'], bwd: ['L', 'yh', 'v', 'c', 'h'] },
		{ title: 'Backward 3: qua ReLU', text: `∂h/∂z = 1 nếu z > 0, ngược lại 0. Ở đây z = ${fmt(z, 3)} nên ∂L/∂z = ${fmt(gh, 3)}·${z > 0 ? 1 : 0} = ${fmt(gz, 3)}.${z > 0 ? '' : ' ReLU đang tắt: gradient bị chặn hoàn toàn, các trọng số phía trước không học được ở mẫu này!'}`, fwd: ['z', 'h', 'yh', 'L'], bwd: ['L', 'yh', 'v', 'c', 'h', 'z'] },
		{ title: 'Backward 4: tới các trọng số đầu', text: `∂z/∂w₁ = x₁, ∂z/∂w₂ = x₂, ∂z/∂b = 1. Vậy ∂L/∂w₁ = ${fmt(gz, 3)}·${fmt(p.x1)} = ${fmt(gw1, 3)}; ∂L/∂w₂ = ${fmt(gz, 3)}·${fmt(p.x2)} = ${fmt(gw2, 3)}; ∂L/∂b = ${fmt(gb, 3)}. Xong: có gradient của mọi tham số chỉ sau MỘT lượt ngược.`, fwd: ['z', 'h', 'yh', 'L'], bwd: ['L', 'yh', 'v', 'c', 'h', 'z', 'w1', 'w2', 'b'] },
	];
	const { step, setStep, controls } = useStepper(steps.length - 1, 1600);
	const st = steps[step];
	const val: Record<NodeId, number> = { x1: p.x1, x2: p.x2, w1: p.w1, w2: p.w2, b: p.b, z, h, v: p.v, c: p.c, yh, y: p.y, L };
	const grad: Partial<Record<NodeId, number>> = { L: 1, yh: gyh, v: gv, c: gc, h: gh, z: gz, w1: gw1, w2: gw2, b: gb };
	const leaf = new Set<NodeId>(['x1', 'x2', 'w1', 'w2', 'b', 'v', 'c', 'y']);
	const shown = (n: NodeId) => leaf.has(n) || st.fwd.includes(n);
	const randomize = () => {
		const r = rng(seed * 13 + 5);
		const q = (lo: number, hi: number) => Math.round((lo + (hi - lo) * r()) * 2) / 2;
		setP({ x1: q(-2, 2), x2: q(-2, 2), w1: q(-1.5, 1.5), w2: q(-1.5, 1.5), b: q(-1, 1), v: q(-2, 2), c: q(-1, 1), y: q(-2, 2) });
		setSeed(seed + 1);
		setStep(0);
	};
	const W = 650, H = 330;
	return (
		<Figure
			title="Lan truyền ngược trên đồ thị tính toán"
			caption={
				<>
					Mạng: z = w₁x₁ + w₂x₂ + b, h = ReLU(z), ŷ = v·h + c, L = ½(ŷ − y)². Số màu <span style={{ color: C.blue }}>xanh</span> là giá trị (forward), số màu <span style={{ color: C.red }}>đỏ</span> là gradient ∂L/∂(ô đó) (backward).
					Quy tắc ở mỗi ô: <b>gradient đi ra = gradient đi vào × đạo hàm cục bộ</b>. Bấm “Đổi số” nhiều lần đến khi z âm để thấy ReLU chặn gradient.
				</>
			}
		>
			{controls}
			<div className="viz-buttons">
				<button type="button" onClick={randomize}>🎲 Đổi số</button>
				<button type="button" onClick={() => { setP(DEFAULT); setStep(0); }}>Số ban đầu</button>
			</div>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Đồ thị tính toán">
				{EDGES.map(([a, b2]) => {
					const back = st.bwd.includes(a) && st.bwd.includes(b2);
					return <line key={a + b2} x1={POS[a][0] + 30} y1={POS[a][1]} x2={POS[b2][0] - 30} y2={POS[b2][1]} stroke={back ? C.red : C.muted} strokeWidth={back ? 2.5 : 1.3} strokeDasharray={back ? '6 3' : undefined} />;
				})}
				{(Object.keys(POS) as NodeId[]).map((n) => {
					const [x, y] = POS[n];
					const g = grad[n];
					const isFwdNew = st.fwd[st.fwd.length - 1] === n && st.bwd.length === 0;
					const isBwd = st.bwd.includes(n);
					const wide = n === 'h' ? 46 : 30;
					return (
						<g key={n}>
							<rect x={x - wide} y={y - 22} width={wide * 2} height={44} rx={8} fill={isFwdNew ? C.blue : isBwd ? C.red : C.grid} fillOpacity={isFwdNew || isBwd ? 0.18 : 0.35} stroke={leaf.has(n) ? C.muted : C.text} strokeWidth={leaf.has(n) ? 1 : 1.8} />
							<text x={x} y={y - 5} fontSize={13} textAnchor="middle" fill={C.text} fontWeight={600}>{LABEL[n]}</text>
							<text x={x} y={y + 13} fontSize={12} textAnchor="middle" fill={C.blue}>{shown(n) ? fmt(val[n], 3) : '?'}</text>
							{isBwd && g !== undefined && (
								<text x={x} y={y + 38} fontSize={12} textAnchor="middle" fill={C.red} fontWeight={700}>∂: {fmt(g, 3)}</text>
							)}
						</g>
					);
				})}
			</svg>
			<div className="viz-note">
				<b>{st.title}.</b> {st.text}
			</div>
		</Figure>
	);
}
