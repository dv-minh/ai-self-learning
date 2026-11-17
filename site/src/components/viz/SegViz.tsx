import { useState } from 'react';
import { C, Controls, Figure, Slider, fmt, useSvgDrag } from './common';

const G = 20;
const inGT = (i: number, j: number) => {
	// vật thể thật: hình chữ L ghép từ hai hình chữ nhật
	return (i >= 4 && i <= 15 && j >= 4 && j <= 8) || (i >= 11 && i <= 15 && j >= 4 && j <= 14);
};

/** Hình 8.7: phân đoạn là phân loại từng pixel; IoU, Dice và pixel accuracy đếm trên pixel. */
export default function SegViz() {
	const [c, setC] = useState<[number, number]>([10, 9]);
	const [rad, setRad] = useState(5);
	const cs = 18, S = G * cs;
	const { svgRef, handlers } = useSvgDrag((x, y) => setC([Math.max(0, Math.min(G, y / cs)), Math.max(0, Math.min(G, x / cs))]));
	const inP = (i: number, j: number) => (i + 0.5 - c[0]) ** 2 + (j + 0.5 - c[1]) ** 2 <= rad * rad;
	let tp = 0, fp = 0, fn = 0, tn = 0;
	const cells = [];
	for (let i = 0; i < G; i++)
		for (let j = 0; j < G; j++) {
			const g = inGT(i, j), p = inP(i, j);
			if (g && p) tp++;
			else if (p) fp++;
			else if (g) fn++;
			else tn++;
			const fill = g && p ? C.green : p ? C.red : g ? C.blue : 'none';
			cells.push(<rect key={`${i},${j}`} x={j * cs} y={i * cs} width={cs - 1} height={cs - 1} fill={fill === 'none' ? C.grid : fill} fillOpacity={fill === 'none' ? 0.18 : 0.6} />);
		}
	const iouV = tp / (tp + fp + fn || 1);
	const dice = (2 * tp) / (2 * tp + fp + fn || 1);
	return (
		<Figure
			title="Đo chất lượng mặt nạ phân đoạn bằng cách đếm pixel"
			caption={
				<>
					Mỗi ô là một pixel. Vật thể thật là hình chữ L; mặt nạ dự đoán là hình tròn — <b>kéo</b> nó đi và đổi bán kính. Xanh lá: đúng (thuộc cả hai, TP). Đỏ: báo thừa (FP). Xanh dương: bỏ sót (FN).
					Pixel accuracy bị nền chiếm áp đảo nên luôn cao; IoU và Dice chỉ quan tâm tới vật thể nên phản ánh đúng chất lượng hơn. Dice luôn ≥ IoU.
				</>
			}
		>
			<Controls>
				<Slider label="bán kính mặt nạ" value={rad} min={1} max={11} step={0.25} onChange={setRad} />
			</Controls>
			<svg ref={svgRef} viewBox={`0 0 ${S} ${S}`} style={{ maxWidth: 400 }} role="img" aria-label="Lưới pixel với mặt nạ thật và mặt nạ dự đoán" {...handlers}>
				{cells}
				<circle cx={c[1] * cs} cy={c[0] * cs} r={rad * cs} fill="none" stroke={C.red} strokeWidth={2} strokeDasharray="5 4" style={{ cursor: 'grab' }} />
			</svg>
			<div className="viz-kv">
				<span>TP = <b>{tp}</b>, FP = <b>{fp}</b>, FN = <b>{fn}</b>, TN = <b>{tn}</b></span>
			</div>
			<div className="viz-kv">
				<span>pixel acc = (TP+TN)/{G * G} = <b>{fmt((tp + tn) / (G * G), 3)}</b></span>
				<span>IoU = TP/(TP+FP+FN) = <b>{fmt(iouV, 3)}</b></span>
				<span>Dice = 2TP/(2TP+FP+FN) = <b>{fmt(dice, 3)}</b></span>
			</div>
		</Figure>
	);
}
