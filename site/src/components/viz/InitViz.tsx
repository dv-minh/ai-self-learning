import { useMemo, useState } from 'react';
import { Axes, C, Choice, Controls, Figure, fmt, gauss, rng, scale } from './common';

type Act = 'relu' | 'tanh';
type Init = 'small' | 'xavier' | 'he' | 'large';
const L = 12, N = 64, B = 128;

function simulate(act: Act, init: Init) {
	const r = rng(77);
	let a: number[][] = Array.from({ length: B }, () => Array.from({ length: N }, () => gauss(r)));
	const stds: number[] = [];
	const hists: number[][] = [];
	const std = init === 'small' ? 0.01 : init === 'large' ? 0.5 : init === 'xavier' ? Math.sqrt(2 / (N + N)) : Math.sqrt(2 / N);
	for (let l = 0; l < L; l++) {
		const W = Array.from({ length: N }, () => Array.from({ length: N }, () => std * gauss(r)));
		a = a.map((row) =>
			W.map((wr) => {
				let s = 0;
				for (let k = 0; k < N; k++) s += wr[k] * row[k];
				return act === 'relu' ? Math.max(0, s) : Math.tanh(s);
			}),
		);
		const flat = a.flat();
		const m = flat.reduce((x, y) => x + y, 0) / flat.length;
		stds.push(Math.sqrt(flat.reduce((x, y) => x + (y - m) ** 2, 0) / flat.length));
		hists.push(flat);
	}
	return { stds, hists };
}

/** Hình 7.4: độ lệch chuẩn của kích hoạt qua 12 lớp với các cách khởi tạo khác nhau. */
export default function InitViz() {
	const [act, setAct] = useState<Act>('relu');
	const [init, setInit] = useState<Init>('he');
	const sim = useMemo(() => simulate(act, init), [act, init]);
	const W = 540, H = 240;
	const X = scale(0.5, L + 0.5, 50, W - 10);
	const lo = -6, hi = 4; // log10
	const Y = scale(lo, hi, H - 22, 10);
	const ly = (v: number) => Y(Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-12)))));
	const layers = [1, 4, 8, 12];
	const hw = 120, hh = 70;
	return (
		<Figure
			title="Tín hiệu đi qua mạng sâu: tắt dần, giữ nguyên hay bùng nổ"
			caption={
				<>
					Mạng {L} lớp, mỗi lớp {N} nơ-ron, đầu vào chuẩn N(0, 1), trọng số rút từ N(0, σ²). Trên: độ lệch chuẩn của kích hoạt ở từng lớp (thang log). Dưới: histogram kích hoạt ở vài lớp.
					σ quá nhỏ: tín hiệu co về 0 theo cấp số nhân; quá lớn: phình ra (ReLU) hoặc bão hòa ở ±1 (tanh). He giữ ổn định cho ReLU, Xavier cho tanh — vì chúng chọn σ² để phương sai được nhân với đúng 1 ở mỗi lớp.
				</>
			}
		>
			<Controls>
				<Choice label="Kích hoạt:" value={act} onChange={setAct} options={[['relu', 'ReLU'], ['tanh', 'tanh']]} />
				<Choice
					label="Khởi tạo:"
					value={init}
					onChange={setInit}
					options={[
						['small', 'σ = 0,01'],
						['xavier', 'Xavier σ² = 2/(n_in+n_out)'],
						['he', 'He σ² = 2/n_in'],
						['large', 'σ = 0,5'],
					]}
				/>
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Độ lệch chuẩn kích hoạt theo lớp">
				<Axes x={X} y={Y} xr={[0.5, L + 0.5]} yr={[lo, hi]} xticks={[2, 4, 6, 8, 10, 12]} yticks={[]} xlabel="lớp" grid={false} />
				{[-6, -4, -2, 0, 2, 4].map((e) => (
					<g key={e}>
						<line x1={X(0.5)} x2={X(L + 0.5)} y1={Y(e)} y2={Y(e)} stroke={C.grid} strokeWidth={0.6} />
						<text x={X(0.5) - 4} y={Y(e) + 4} fontSize={11} textAnchor="end" fill={C.muted}>10^{e}</text>
					</g>
				))}
				<line x1={X(0.5)} x2={X(L + 0.5)} y1={Y(0)} y2={Y(0)} stroke={C.green} strokeDasharray="5 4" />
				<polyline points={sim.stds.map((s, i) => `${X(i + 1)},${ly(s)}`).join(' ')} fill="none" stroke={C.accent} strokeWidth={2.5} />
				{sim.stds.map((s, i) => <circle key={i} cx={X(i + 1)} cy={ly(s)} r={3.5} fill={C.accent} />)}
			</svg>
			<svg viewBox={`0 0 ${W} ${hh + 30}`} role="img" aria-label="Histogram kích hoạt" style={{ marginTop: '0.4rem' }}>
				{layers.map((l, k) => {
					const vals = sim.hists[l - 1];
					const s = sim.stds[0] || 1;
					const range = act === 'tanh' ? 1.1 : Math.max(3 * s, 1e-9);
					const lo2 = act === 'tanh' ? -range : 0;
					const bins = new Array(24).fill(0);
					for (const v of act === 'relu' ? vals.filter((u) => u > 0) : vals) {
						const t = Math.floor(((v - lo2) / (range - lo2)) * 24);
						if (t >= 0 && t < 24) bins[t]++;
						else if (t >= 24) bins[23]++;
					}
					const bm = Math.max(...bins, 1);
					const x0 = 10 + k * (hw + 12);
					return (
						<g key={l}>
							{bins.map((c, i) => <rect key={i} x={x0 + (i * hw) / 24} y={hh - (c / bm) * (hh - 10)} width={hw / 24 - 0.5} height={(c / bm) * (hh - 10)} fill={C.accent} opacity={0.7} />)}
							<line x1={x0} x2={x0 + hw} y1={hh} y2={hh} stroke={C.axis} />
							<text x={x0 + hw / 2} y={hh + 16} fontSize={11} textAnchor="middle" fill={C.text}>lớp {l}: std = {fmt(sim.stds[l - 1], 4)}</text>
						</g>
					);
				})}
			</svg>
			<div className="viz-note">Histogram cùng một thang ngang (theo lớp 1; với ReLU chỉ vẽ các giá trị dương): nếu cột dồn hết vào một vạch ở 0 là tín hiệu đã tắt; nếu tràn ra mép phải là đã bùng nổ.</div>
		</Figure>
	);
}
