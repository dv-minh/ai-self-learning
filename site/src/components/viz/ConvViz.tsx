import { useMemo, useState } from 'react';
import { C, Choice, Controls, Figure, fmt, useStepper } from './common';

const IMG = [
	[0, 0, 0, 0, 0, 0, 0],
	[0, 9, 9, 9, 9, 9, 0],
	[0, 0, 0, 0, 0, 9, 0],
	[0, 0, 0, 0, 9, 0, 0],
	[0, 0, 0, 9, 0, 0, 0],
	[0, 0, 9, 0, 0, 0, 0],
	[0, 0, 9, 0, 0, 0, 0],
];
type KName = 'vert' | 'horiz' | 'blur' | 'sharp';
const KERNELS: Record<KName, { label: string; k: number[][] }> = {
	vert: { label: 'cạnh dọc', k: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
	horiz: { label: 'cạnh ngang', k: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
	blur: { label: 'làm mờ (trung bình)', k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
	sharp: { label: 'làm nét', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
};

/** Hình 8.1: kernel 3×3 trượt trên ảnh; mỗi ô đầu ra là tổng các tích trong cửa sổ. */
export default function ConvViz() {
	const [kn, setKn] = useState<KName>('vert');
	const [pad, setPad] = useState<'0' | '1'>('0');
	const [stride, setStride] = useState<'1' | '2'>('1');
	const p = Number(pad), s = Number(stride), k = 3, n = IMG.length;
	const K = KERNELS[kn].k;
	const nOut = Math.floor((n + 2 * p - k) / s) + 1;
	const X = (i: number, j: number) => (i < p || j < p || i >= n + p || j >= n + p ? 0 : IMG[i - p][j - p]);
	const out = useMemo(() => {
		const o: number[][] = [];
		for (let i = 0; i < nOut; i++) {
			o.push([]);
			for (let j = 0; j < nOut; j++) {
				let sum = 0;
				for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) sum += K[a][b] * X(i * s + a, j * s + b);
				o[i].push(sum);
			}
		}
		return o;
	}, [kn, p, s]);
	const total = nOut * nOut;
	const { step, controls } = useStepper(total - 1, 500);
	const ci = Math.floor(step / nOut), cj = step % nOut;
	const cs = 30, N = n + 2 * p;
	const oMax = Math.max(...out.flat().map(Math.abs), 1e-9);
	const terms: string[] = [];
	for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
		const kv = K[a][b], xv = X(ci * s + a, cj * s + b);
		if (kv !== 0 && xv !== 0) terms.push(`${kv < 0 ? `(${fmt(kv, 2)})` : fmt(kv, 2)}·${xv}`);
	}
	const gw = N * cs, ow = nOut * cs;
	const W = gw + ow + 3 * cs + 60;
	return (
		<Figure
			title="Tích chập: trượt cửa sổ, nhân từng cặp, cộng lại"
			caption={
				<>
					Bên trái: ảnh {n}×{n} (số càng lớn càng sáng){p ? ', viền xám là padding số 0' : ''}. Khung cam là cửa sổ đang xét; kernel {k}×{k} ở giữa. Bên phải: bản đồ đặc trưng đầu ra {nOut}×{nOut} — đỏ là dương, xanh là âm, đậm là lớn.
					Kernel “cạnh dọc” cho giá trị lớn đúng ở chỗ ảnh đổi từ tối sang sáng theo chiều ngang, và gần 0 ở vùng phẳng: kernel là một <b>bộ dò mẫu</b>. Đổi padding/stride để thấy kích thước đầu ra thay đổi theo công thức.
				</>
			}
		>
			<Controls>
				<Choice label="Kernel:" value={kn} onChange={setKn} options={(Object.keys(KERNELS) as KName[]).map((x) => [x, KERNELS[x].label])} />
			</Controls>
			<Controls>
				<Choice label="Padding p:" value={pad} onChange={setPad} options={[['0', '0'], ['1', '1']]} />
				<Choice label="Stride s:" value={stride} onChange={setStride} options={[['1', '1'], ['2', '2']]} />
			</Controls>
			{controls}
			<svg viewBox={`0 0 ${W} ${Math.max(gw, 3 * cs) + 30}`} role="img" aria-label="Ảnh đầu vào, kernel và bản đồ đầu ra">
				<g transform="translate(0,20)">
					{Array.from({ length: N }, (_, i) =>
						Array.from({ length: N }, (_, j) => {
							const isPad = i < p || j < p || i >= n + p || j >= n + p;
							const v = X(i, j);
							return (
								<g key={`${i},${j}`}>
									<rect x={j * cs} y={i * cs} width={cs - 1} height={cs - 1} fill={isPad ? C.muted : C.text} fillOpacity={isPad ? 0.15 : 0.04 + (v / 9) * 0.6} />
									<text x={j * cs + cs / 2} y={i * cs + cs / 2 + 4} fontSize={11} textAnchor="middle" fill={v > 4 ? 'var(--sl-color-bg)' : C.text}>{v}</text>
								</g>
							);
						}),
					)}
					<rect x={cj * s * cs - 1} y={ci * s * cs - 1} width={k * cs + 1} height={k * cs + 1} fill="none" stroke={C.orange} strokeWidth={3} />
				</g>
				<g transform={`translate(${gw + 20},${20 + cs})`}>
					<text x={(k * cs) / 2} y={-8} fontSize={12} textAnchor="middle" fill={C.text}>kernel</text>
					{K.map((row, a) =>
						row.map((v, b) => (
							<g key={`${a},${b}`}>
								<rect x={b * cs} y={a * cs} width={cs - 1} height={cs - 1} fill={v > 0 ? C.red : v < 0 ? C.blue : C.grid} fillOpacity={v === 0 ? 0.3 : 0.25} stroke={C.orange} />
								<text x={b * cs + cs / 2} y={a * cs + cs / 2 + 4} fontSize={10} textAnchor="middle" fill={C.text}>{Math.abs(v) < 1 && v !== 0 ? '1/9' : v}</text>
							</g>
						)),
					)}
				</g>
				<g transform={`translate(${gw + k * cs + 40},20)`}>
					{out.map((row, i) =>
						row.map((v, j) => {
							const idx = i * nOut + j;
							const done = idx <= step;
							return (
								<g key={`${i},${j}`}>
									<rect x={j * cs} y={i * cs} width={cs - 1} height={cs - 1} fill={!done ? C.grid : v >= 0 ? C.red : C.blue} fillOpacity={!done ? 0.25 : 0.1 + 0.7 * (Math.abs(v) / oMax)} stroke={idx === step ? C.orange : 'none'} strokeWidth={3} />
									{done && <text x={j * cs + cs / 2} y={i * cs + cs / 2 + 4} fontSize={10} textAnchor="middle" fill={C.text}>{fmt(v, 1)}</text>}
								</g>
							);
						}),
					)}
					<text x={ow / 2} y={-6} fontSize={12} textAnchor="middle" fill={C.text}>đầu ra {nOut}×{nOut}</text>
				</g>
			</svg>
			<div className="viz-note">
				Ô đầu ra ({ci}, {cj}) = tổng 9 tích kernel × ô ảnh trong khung cam = {terms.length ? terms.join(' + ') : '0 (mọi tích đều bằng 0)'} = <b>{fmt(out[ci][cj], 3)}</b>
			</div>
			<div className="viz-kv">
				<span>n_out = ⌊(n + 2p − k)/s⌋ + 1 = ⌊({n} + {2 * p} − {k})/{s}⌋ + 1 = <b>{nOut}</b></span>
			</div>
		</Figure>
	);
}
