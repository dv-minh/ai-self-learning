import { useMemo, useState } from 'react';
import { C, Controls, Figure, Slider, gauss, rng, scale, useSvgDrag } from './common';

const S = 400;
const R: [number, number] = [0, 10];
type P = [number, number, 0 | 1];

function data(): P[] {
	const r = rng(31);
	const out: P[] = [];
	for (let i = 0; i < 22; i++) out.push([3.4 + 1.5 * gauss(r), 6 + 1.5 * gauss(r), 0]);
	for (let i = 0; i < 22; i++) out.push([6.6 + 1.5 * gauss(r), 4 + 1.5 * gauss(r), 1]);
	out.push([7.5, 7.5, 0], [2.4, 2.6, 1]);
	return out.map(([x, y, c]) => [Math.min(9.7, Math.max(0.3, x)), Math.min(9.7, Math.max(0.3, y)), c]);
}
const D = data();
const COL = [C.blue, C.red];

function neighbours(q: [number, number], k: number) {
	return D.map((p, i) => [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, i] as [number, number])
		.sort((a, b) => a[0] - b[0])
		.slice(0, k);
}
function vote(q: [number, number], k: number) {
	const nb = neighbours(q, k);
	const c1 = nb.filter(([, i]) => D[i][2] === 1).length;
	return { c1, c0: k - c1, nb };
}

/** Hình 6.5: k láng giềng gần nhất bỏ phiếu; k nhỏ → biên gồ ghề, k lớn → biên mượt. */
export default function KnnViz() {
	const [k, setK] = useState(5);
	const [q, setQ] = useState<[number, number]>([5, 5.2]);
	const X = scale(R[0], R[1], 0, S);
	const Y = scale(R[0], R[1], S, 0);
	const { svgRef, handlers } = useSvgDrag((sx, sy) => setQ([Math.min(9.8, Math.max(0.2, (sx / S) * 10)), Math.min(9.8, Math.max(0.2, 10 - (sy / S) * 10))]));
	const G = 40;
	const cells = useMemo(() => {
		const out: { x: number; y: number; c: number }[] = [];
		for (let i = 0; i < G; i++)
			for (let j = 0; j < G; j++) {
				const x = ((i + 0.5) * 10) / G, y = ((j + 0.5) * 10) / G;
				const v = vote([x, y], k);
				out.push({ x: i, y: j, c: v.c1 > v.c0 ? 1 : 0 });
			}
		return out;
	}, [k]);
	const v = vote(q, k);
	const rad = Math.sqrt(v.nb[v.nb.length - 1][0]);
	const pred = v.c1 > v.c0 ? 1 : 0;
	const cs = S / G;
	return (
		<Figure
			title="KNN: hỏi k hàng xóm gần nhất"
			caption={
				<>
					Kéo điểm hỏi (ngôi sao) đi khắp nơi. Vòng tròn đi qua láng giềng thứ k; các đoạn nối tới k láng giềng đang bỏ phiếu. Màu nền là lớp KNN sẽ dự đoán tại mỗi vị trí — đó là <b>biên quyết định</b>.
					k = 1: biên bám theo từng điểm, kể cả hai điểm “lạc loài” (tạo ra các ốc đảo màu) — variance cao. Tăng k: biên mượt dần, các ốc đảo biến mất — nhưng k quá lớn thì lớp đông hơn lấn át (bias cao). Dùng k lẻ để tránh hòa phiếu với hai lớp.
				</>
			}
		>
			<Controls>
				<Slider label="k" value={k} min={1} max={25} step={1} onChange={setK} show={String(k)} />
			</Controls>
			<svg ref={svgRef} viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 440 }} role="img" aria-label="Dữ liệu hai lớp và vùng quyết định KNN" {...handlers}>
				{cells.map((c) => (
					<rect key={`${c.x},${c.y}`} x={c.x * cs} y={S - (c.y + 1) * cs} width={cs + 0.5} height={cs + 0.5} fill={COL[c.c]} opacity={0.13} />
				))}
				<circle cx={X(q[0])} cy={Y(q[1])} r={X(rad) - X(0)} fill="none" stroke={C.text} strokeDasharray="5 4" />
				{v.nb.map(([, i]) => <line key={`l${i}`} x1={X(q[0])} y1={Y(q[1])} x2={X(D[i][0])} y2={Y(D[i][1])} stroke={COL[D[i][2]]} strokeWidth={1.5} />)}
				{D.map(([x, y, c], i) => (
					<circle key={i} cx={X(x)} cy={Y(y)} r={5.5} fill={COL[c]} stroke={v.nb.some(([, j]) => j === i) ? C.text : 'none'} strokeWidth={2} />
				))}
				<text x={X(q[0])} y={Y(q[1]) + 9} fontSize={28} textAnchor="middle" fill={COL[pred]} stroke={C.text} strokeWidth={1} style={{ cursor: 'grab' }}>★</text>
			</svg>
			<div className="viz-kv">
				<span>phiếu: <b style={{ color: C.blue }}>{v.c0} xanh</b> – <b style={{ color: C.red }}>{v.c1} đỏ</b></span>
				<span>dự đoán: <b style={{ color: COL[pred] }}>{pred ? 'đỏ' : 'xanh'}</b>{v.c0 === v.c1 ? ' (hòa phiếu, ở đây chọn xanh)' : ''}</span>
			</div>
		</Figure>
	);
}
