import { useMemo, useState } from 'react';
import { Axes, C, Controls, Figure, Slider, curvePath, fmt, gauss, rng, scale } from './common';

const truth = (x: number) => Math.sin(2.6 * x) + 0.3 * x;
const NTR = 12, NTE = 60, MAXD = 11;

function dataset(seed: number) {
	const r = rng(seed);
	const tr: [number, number][] = [];
	for (let i = 0; i < NTR; i++) {
		const x = -1 + (2 * (i + 0.2 + 0.6 * r())) / NTR;
		tr.push([x, truth(x) + 0.28 * gauss(r)]);
	}
	const te: [number, number][] = [];
	for (let i = 0; i < NTE; i++) {
		const x = -1 + 2 * r();
		te.push([x, truth(x) + 0.28 * gauss(r)]);
	}
	return { tr, te };
}

/** Bình phương nhỏ nhất với đa thức bậc d (thêm ridge rất nhỏ cho ổn định số). */
function polyfit(pts: [number, number][], d: number): number[] {
	const m = d + 1;
	const A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
	for (const [x, y] of pts) {
		const p = Array.from({ length: m }, (_, k) => x ** k);
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < m; j++) A[i][j] += p[i] * p[j];
			A[i][m] += p[i] * y;
		}
	}
	for (let i = 0; i < m; i++) A[i][i] += 1e-9;
	for (let c = 0; c < m; c++) {
		let piv = c;
		for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
		[A[c], A[piv]] = [A[piv], A[c]];
		for (let r = 0; r < m; r++) {
			if (r === c) continue;
			const f = A[r][c] / A[c][c];
			for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k];
		}
	}
	return A.map((row, i) => row[m] / row[i]);
}
const ev = (co: number[], x: number) => co.reduce((s, c, k) => s + c * x ** k, 0);
const mse = (co: number[], pts: [number, number][]) => pts.reduce((s, [x, y]) => s + (ev(co, x) - y) ** 2, 0) / pts.length;

/** Hình 6.3: underfit → vừa → overfit khi tăng bậc đa thức; lỗi train giảm mãi, lỗi test hình chữ U. */
export default function PolyFitViz() {
	const [deg, setDeg] = useState(1);
	const [seed, setSeed] = useState(5);
	const { tr, te } = useMemo(() => dataset(seed), [seed]);
	const fits = useMemo(() => Array.from({ length: MAXD + 1 }, (_, d) => polyfit(tr, d)), [tr]);
	const errs = fits.map((co) => [mse(co, tr), mse(co, te)]);
	const W = 520, H = 300;
	const X = scale(-1.05, 1.05, 10, W - 10);
	const Y = scale(-2.2, 2.2, H - 10, 10);
	const W2 = 520, H2 = 220;
	const X2 = scale(-0.5, MAXD + 0.5, 40, W2 - 10);
	const EMAX = 1.2;
	const Y2 = scale(0, EMAX, H2 - 25, 10);
	const best = errs.reduce((bi, e, i) => (e[1] < errs[bi][1] ? i : bi), 0);
	const lineE = (k: 0 | 1) => errs.map((e, i) => `${i ? 'L' : 'M'}${X2(i).toFixed(1)},${Y2(Math.min(e[k], EMAX)).toFixed(1)}`).join('');
	const verdict = deg < Math.max(1, best - 1) ? 'Underfit (bias cao): đường quá cứng, sai cả trên dữ liệu huấn luyện.' : deg > best + 2 ? 'Overfit (variance cao): đường uốn để đi qua từng điểm huấn luyện, kể cả nhiễu; sai lớn trên dữ liệu mới.' : 'Vùng vừa phải: lỗi test gần nhỏ nhất.';
	return (
		<Figure
			title="Độ phức tạp mô hình: từ underfit đến overfit"
			caption={
				<>
					Trên: {NTR} điểm huấn luyện (đen), dữ liệu test (chấm nhạt), hàm thật (nét đứt) và đa thức bậc d khớp bằng bình phương nhỏ nhất (màu). Dưới: MSE trên tập huấn luyện và tập test theo bậc d.
					Lỗi huấn luyện <b>chỉ giảm</b> khi d tăng (bậc {NTR - 1} đi qua đúng mọi điểm); lỗi test giảm rồi <b>tăng lại</b> — hình chữ U. Bấm “Bộ dữ liệu khác” với d lớn để thấy đường cong thay đổi dữ dội: đó là variance.
				</>
			}
		>
			<Controls>
				<Slider label="bậc đa thức d" value={deg} min={0} max={MAXD} step={1} onChange={setDeg} show={String(deg)} />
				<div className="viz-buttons">
					<button type="button" onClick={() => setSeed((s) => s + 1)}>🎲 Bộ dữ liệu khác</button>
				</div>
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dữ liệu và đường khớp đa thức">
				<line x1={10} x2={W - 10} y1={Y(0)} y2={Y(0)} stroke={C.grid} />
				<path d={curvePath(truth, -1.05, 1.05, X, Y)} stroke={C.muted} strokeDasharray="6 4" strokeWidth={1.5} fill="none" />
				{te.map(([x, y], i) => <circle key={`t${i}`} cx={X(x)} cy={Y(y)} r={2.5} fill={C.orange} opacity={0.45} />)}
				<clipPath id="pf-clip"><rect x={0} y={0} width={W} height={H} /></clipPath>
				<path clipPath="url(#pf-clip)" d={curvePath((x) => Math.max(-5, Math.min(5, ev(fits[deg], x))), -1.05, 1.05, X, Y, 300)} stroke={C.accent} strokeWidth={2.5} fill="none" />
				{tr.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(y)} r={5} fill={C.text} />)}
			</svg>
			<svg viewBox={`0 0 ${W2} ${H2}`} role="img" aria-label="Lỗi train và test theo bậc" style={{ marginTop: '0.5rem' }}>
				<Axes x={X2} y={Y2} xr={[-0.5, MAXD + 0.5]} yr={[0, EMAX]} xticks={[0, 2, 4, 6, 8, 10]} yticks={[0.4, 0.8, 1.2]} xlabel="bậc d" ylabel="MSE" grid={false} />
				<rect x={X2(deg) - 9} y={10} width={18} height={H2 - 35} fill={C.accent} opacity={0.15} />
				<path d={lineE(0)} stroke={C.blue} strokeWidth={2.2} fill="none" />
				<path d={lineE(1)} stroke={C.red} strokeWidth={2.2} fill="none" />
				{errs.map((e, i) => (
					<g key={i}>
						<circle cx={X2(i)} cy={Y2(Math.min(e[0], EMAX))} r={3} fill={C.blue} />
						<circle cx={X2(i)} cy={Y2(Math.min(e[1], EMAX))} r={3} fill={C.red} />
					</g>
				))}
			</svg>
			<div className="viz-legend">
				<span style={{ ['--c' as string]: C.blue }}>MSE huấn luyện = {fmt(errs[deg][0], 3)}</span>
				<span style={{ ['--c' as string]: C.red }}>MSE test = {fmt(errs[deg][1], 3)}</span>
			</div>
			<div className="viz-note">{verdict}</div>
		</Figure>
	);
}
