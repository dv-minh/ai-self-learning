import { useState } from 'react';
import { C, Choice, Controls, Figure } from './common';

type Layer = { name: string; k: number; s: number; d: number };
const PRESETS: Record<string, { label: string; layers: Layer[] }> = {
	conv: { label: '3 lớp conv 3, stride 1', layers: [{ name: 'conv3', k: 3, s: 1, d: 1 }, { name: 'conv3', k: 3, s: 1, d: 1 }, { name: 'conv3', k: 3, s: 1, d: 1 }] },
	pool: { label: 'conv3 → pool2 → conv3 → pool2', layers: [{ name: 'conv3', k: 3, s: 1, d: 1 }, { name: 'pool2 s2', k: 2, s: 2, d: 1 }, { name: 'conv3', k: 3, s: 1, d: 1 }, { name: 'pool2 s2', k: 2, s: 2, d: 1 }] },
	dil: { label: 'conv3 dilation 1, 2, 4', layers: [{ name: 'conv3 d1', k: 3, s: 1, d: 1 }, { name: 'conv3 d2', k: 3, s: 1, d: 2 }, { name: 'conv3 d4', k: 3, s: 1, d: 4 }] },
};
const NIN = 32;

/** Hình 8.2: một nơ-ron ở lớp sâu “nhìn thấy” bao nhiêu ô của ảnh đầu vào. */
export default function ReceptiveFieldViz() {
	const [key, setKey] = useState('pool');
	const layers = PRESETS[key].layers;
	// kích thước mỗi lớp (không padding)
	const sizes = [NIN];
	for (const L of layers) sizes.push(Math.floor((sizes[sizes.length - 1] - L.d * (L.k - 1) - 1) / L.s) + 1);
	const top = sizes[sizes.length - 1];
	const [pick, setPick] = useState(-1);
	const sel = pick < 0 || pick >= top ? Math.floor((top - 1) / 2) : pick;
	// đi ngược: khoảng chỉ số [lo, hi] ở mỗi lớp
	const ranges: [number, number][] = new Array(layers.length + 1);
	ranges[layers.length] = [sel, sel];
	for (let l = layers.length - 1; l >= 0; l--) {
		const L = layers[l];
		const [lo, hi] = ranges[l + 1];
		ranges[l] = [lo * L.s, hi * L.s + L.d * (L.k - 1)];
	}
	// công thức r, j
	const rj: [number, number][] = [[1, 1]];
	for (const L of layers) {
		const [r, j] = rj[rj.length - 1];
		rj.push([r + L.d * (L.k - 1) * j, j * L.s]);
	}
	const cw = 16, rowH = 46;
	const W = NIN * cw + 150, H = (layers.length + 1) * rowH + 20;
	const xOf = (l: number, i: number) => {
		// căn giữa mỗi lớp theo vị trí trung tâm của trường tiếp nhận
		const [r, j] = rj[l];
		return 10 + ((i * j + (r - 1) / 2) * cw);
	};
	return (
		<Figure
			title="Trường tiếp nhận lớn dần qua các lớp"
			caption={
				<>
					Mỗi hàng là một lớp (dưới cùng là ảnh đầu vào, một chiều cho dễ nhìn). Bấm một ô ở hàng trên cùng: các ô tô màu ở hàng dưới là mọi ô có ảnh hưởng tới nó. Mỗi conv 3 chỉ nới thêm 2 ô của lớp ngay dưới, nhưng sau pooling stride 2 thì “2 ô” đó tương ứng nhiều ô ảnh hơn (bước nhảy j nhân đôi). Dilation đạt hiệu ứng tương tự mà không giảm kích thước.
				</>
			}
		>
			<Controls>
				<Choice value={key} onChange={(v) => { setKey(v); setPick(-1); }} options={Object.entries(PRESETS).map(([k2, v]) => [k2, v.label])} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Trường tiếp nhận qua các lớp">
				{sizes.map((sz, l) => {
					const y = H - 20 - (l + 1) * rowH + 10;
					const [lo, hi] = ranges[l];
					return (
						<g key={l}>
							<text x={W - 140} y={y + 14} fontSize={11} fill={C.text}>
								{l === 0 ? 'ảnh vào' : layers[l - 1].name} · r = {rj[l][0]}, j = {rj[l][1]}
							</text>
							{Array.from({ length: sz }, (_, i) => {
								const inR = i >= lo && i <= hi && (l === layers.length || true);
								const isTop = l === layers.length;
								return (
									<rect
										key={i}
										x={xOf(l, i) - cw / 2 + 1}
										y={y}
										width={cw - 2}
										height={20}
										rx={3}
										fill={inR ? (isTop ? C.orange : C.accent) : C.grid}
										fillOpacity={inR ? 0.75 : 0.35}
										onClick={isTop ? () => setPick(i) : undefined}
										style={isTop ? { cursor: 'pointer' } : undefined}
									/>
								);
							})}
						</g>
					);
				})}
			</svg>
			<div className="viz-kv">
				<span>một nơ-ron ở lớp trên cùng nhìn thấy <b>{ranges[0][1] - ranges[0][0] + 1}</b> ô của ảnh vào; công thức cho r = <b>{rj[rj.length - 1][0]}</b></span>
			</div>
		</Figure>
	);
}
