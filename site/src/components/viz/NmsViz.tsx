import { useMemo, useState } from 'react';
import { C, Controls, Figure, Slider, fmt, useStepper } from './common';
import { iou } from './IouViz';

type Box = [number, number, number, number];
const BOXES: { b: Box; s: number }[] = [
	{ b: [60, 60, 190, 200], s: 0.95 },
	{ b: [72, 50, 200, 190], s: 0.88 },
	{ b: [50, 75, 175, 215], s: 0.8 },
	{ b: [95, 90, 215, 230], s: 0.55 },
	{ b: [250, 70, 380, 210], s: 0.9 },
	{ b: [262, 84, 392, 222], s: 0.75 },
	{ b: [170, 110, 290, 230], s: 0.5 },
];

/** Hình 8.6: NMS từng bước — giữ hộp điểm cao nhất, loại các hộp trùng nhiều với nó. */
export default function NmsViz() {
	const [tau, setTau] = useState(0.5);
	const trace = useMemo(() => {
		const order = BOXES.map((_, i) => i).sort((a, b) => BOXES[b].s - BOXES[a].s);
		const status: ('cand' | 'keep' | 'drop')[] = BOXES.map(() => 'cand');
		const frames: { status: typeof status; cur: number | null; text: string }[] = [{ status: [...status], cur: null, text: 'Sắp các hộp theo điểm giảm dần. Mọi hộp đang là ứng viên (nét mảnh).' }];
		for (const i of order) {
			if (status[i] !== 'cand') continue;
			status[i] = 'keep';
			const dropped: string[] = [];
			for (const j of order) {
				if (status[j] !== 'cand') continue;
				const v = iou(BOXES[i].b, BOXES[j].b).iou;
				if (v > tau) {
					status[j] = 'drop';
					dropped.push(`#${j + 1} (IoU ${fmt(v, 2)})`);
				}
			}
			frames.push({ status: [...status], cur: i, text: `Giữ hộp #${i + 1} (điểm ${fmt(BOXES[i].s, 2)}) — cao nhất trong các ứng viên còn lại. ${dropped.length ? `Loại vì IoU > τ: ${dropped.join(', ')}.` : 'Không hộp nào trùng quá τ với nó.'}` });
		}
		frames.push({ status: [...status], cur: null, text: `Hết ứng viên. Còn lại ${status.filter((s) => s === 'keep').length} hộp.` });
		return frames;
	}, [tau]);
	const { step, controls } = useStepper(trace.length - 1, 1300);
	const fr = trace[Math.min(step, trace.length - 1)];
	return (
		<Figure
			title="Non-maximum suppression từng bước"
			caption={
				<>
					Bộ dò thường bắn ra nhiều hộp chồng lên cùng một vật. NMS: lấy hộp điểm cao nhất, giữ nó (nét đậm xanh lá), xóa mọi hộp còn lại trùng với nó quá ngưỡng τ (gạch mờ đỏ), rồi lặp lại. τ thấp: xóa mạnh tay, có thể xóa nhầm vật thể nằm sát nhau (hộp #7); τ cao: còn sót hộp trùng.
				</>
			}
		>
			<Controls>
				<Slider label="ngưỡng τ" value={tau} min={0.1} max={0.9} step={0.05} onChange={setTau} />
			</Controls>
			{controls}
			<svg viewBox="0 0 430 260" style={{ maxWidth: 480 }} role="img" aria-label="Các hộp dự đoán và kết quả NMS">
				{BOXES.map(({ b, s }, i) => {
					const st = fr.status[i];
					const col = st === 'keep' ? C.green : st === 'drop' ? C.red : C.orange;
					return (
						<g key={i} opacity={st === 'drop' ? 0.35 : 1}>
							<rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill={i === fr.cur ? C.green : 'none'} fillOpacity={0.12} stroke={col} strokeWidth={st === 'keep' ? 3.5 : 1.5} strokeDasharray={st === 'drop' ? '5 4' : undefined} />
							<text x={b[0] + 3} y={b[1] + 13} fontSize={11} fill={col} fontWeight={700}>#{i + 1}: {fmt(s, 2)}</text>
						</g>
					);
				})}
			</svg>
			<div className="viz-note">{fr.text}</div>
		</Figure>
	);
}
