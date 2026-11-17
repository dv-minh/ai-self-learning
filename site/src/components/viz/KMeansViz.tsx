import { useMemo, useState } from 'react';
import { C, Controls, Figure, SERIES, Slider, fmt, gauss, rng, scale, useStepper } from './common';

const S = 400;
type Pt = [number, number];

function data(): Pt[] {
	const r = rng(41);
	const centers: Pt[] = [[2.5, 7.2], [7.3, 7.6], [5, 2.6], [7.6, 3.4]];
	const out: Pt[] = [];
	centers.forEach(([cx, cy], k) => {
		for (let i = 0; i < (k === 3 ? 10 : 18); i++) out.push([cx + 0.9 * gauss(r), cy + 0.9 * gauss(r)]);
	});
	return out;
}
const D = data();
const d2 = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

type State = { mu: Pt[]; assign: number[] | null; kind: 'init' | 'assign' | 'update'; J: number | null };

function runKMeans(K: number, seed: number): State[] {
	const r = rng(seed * 7 + K);
	const idx = new Set<number>();
	while (idx.size < K) idx.add(Math.floor(r() * D.length));
	let mu: Pt[] = [...idx].map((i) => [D[i][0], D[i][1]]);
	const states: State[] = [{ mu, assign: null, kind: 'init', J: null }];
	let prev: number[] | null = null;
	for (let it = 0; it < 15; it++) {
		const assign = D.map((p) => {
			let best = 0;
			for (let j = 1; j < K; j++) if (d2(p, mu[j]) < d2(p, mu[best])) best = j;
			return best;
		});
		const J1 = D.reduce((s, p, i) => s + d2(p, mu[assign[i]]), 0);
		states.push({ mu, assign, kind: 'assign', J: J1 });
		if (prev && prev.every((a, i) => a === assign[i])) break;
		prev = assign;
		mu = mu.map((m, j) => {
			const pts = D.filter((_, i) => assign[i] === j);
			if (!pts.length) return m;
			return [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
		});
		const J2 = D.reduce((s, p, i) => s + d2(p, mu[assign[i]]), 0);
		states.push({ mu, assign, kind: 'update', J: J2 });
	}
	return states;
}

/** Hình 6.6: K-means chạy từng nửa bước — gán điểm, rồi dời tâm — và inertia J không bao giờ tăng. */
export default function KMeansViz() {
	const [K, setK] = useState(3);
	const [seed, setSeed] = useState(1);
	const states = useMemo(() => runKMeans(K, seed), [K, seed]);
	const { step, setStep, controls } = useStepper(states.length - 1, 900);
	const st = states[Math.min(step, states.length - 1)];
	const prevMu = step > 0 ? states[step - 1].mu : null;
	const X = scale(-0.5, 10.5, 0, S);
	const Y = scale(-0.5, 10.5, S, 0);
	const desc =
		st.kind === 'init'
			? `Khởi tạo: chọn ngẫu nhiên ${K} điểm dữ liệu làm tâm (dấu ×). Chưa điểm nào được gán.`
			: st.kind === 'assign'
				? 'Bước GÁN: mỗi điểm tô màu theo tâm gần nó nhất. Tâm đứng yên, J giảm (hoặc giữ nguyên).'
				: 'Bước CẬP NHẬT: mỗi tâm dời về trung bình các điểm cùng màu (mũi tên mờ là đường dời). Gán đứng yên, J giảm (hoặc giữ nguyên).';
	const done = step === states.length - 1 && st.kind === 'assign' && step > 1;
	return (
		<Figure
			title="K-means từng bước"
			caption={
				<>
					Mỗi lần bấm “Bước tiếp” là <b>một nửa vòng lặp</b>: gán, rồi cập nhật. Theo dõi J (inertia, tổng bình phương khoảng cách tới tâm của cụm mình): nó chỉ giảm hoặc đứng yên, nên thuật toán chắc chắn dừng.
					Nhưng nghiệm cuối phụ thuộc khởi tạo: bấm “Khởi tạo khác” vài lần với K = 4 để gặp trường hợp hai tâm chia nhau một cụm trong khi hai cụm khác bị gộp — một cực tiểu địa phương.
				</>
			}
		>
			<Controls>
				<Slider label="số cụm K" value={K} min={2} max={5} step={1} onChange={(v) => { setK(v); setStep(0); }} show={String(K)} />
				<div className="viz-buttons">
					<button type="button" onClick={() => { setSeed((s) => s + 1); setStep(0); }}>🎲 Khởi tạo khác</button>
				</div>
			</Controls>
			{controls}
			<svg viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 440 }} role="img" aria-label="Các điểm dữ liệu và tâm cụm">
				{st.assign && st.mu.map((m, j) => D.map((p, i) => (st.assign![i] === j ? <line key={`${j}-${i}`} x1={X(p[0])} y1={Y(p[1])} x2={X(m[0])} y2={Y(m[1])} stroke={SERIES[j]} strokeOpacity={0.18} /> : null)))}
				{D.map((p, i) => (
					<circle key={i} cx={X(p[0])} cy={Y(p[1])} r={5} fill={st.assign ? SERIES[st.assign[i]] : C.muted} opacity={0.85} />
				))}
				{st.kind === 'update' && prevMu && prevMu.map((m, j) => <line key={`mv${j}`} x1={X(m[0])} y1={Y(m[1])} x2={X(st.mu[j][0])} y2={Y(st.mu[j][1])} stroke={C.text} strokeDasharray="3 3" />)}
				{st.mu.map((m, j) => (
					<g key={`c${j}`}>
						<circle cx={X(m[0])} cy={Y(m[1])} r={11} fill={C.text} opacity={0.9} />
						<text x={X(m[0])} y={Y(m[1]) + 6} textAnchor="middle" fontSize={17} fontWeight={700} fill={SERIES[j]}>×</text>
					</g>
				))}
			</svg>
			<div className="viz-note">{done ? 'Hội tụ: bước gán không đổi màu điểm nào nữa, nên tâm cũng không dời nữa.' : desc}</div>
			<div className="viz-kv">
				<span>J = <b>{st.J === null ? '—' : fmt(st.J, 2)}</b></span>
				<span>lịch sử J: {states.filter((s, i) => i <= step && s.J !== null).map((s) => fmt(s.J!, 1)).join(' → ') || '—'}</span>
			</div>
		</Figure>
	);
}
