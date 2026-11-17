import { useState } from 'react';
import { C, Controls, Figure, Slider, fmt, useSvgDrag } from './common';

type Box = [number, number, number, number];
const area = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export function iou(a: Box, b: Box) {
	const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
	const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
	const inter = w * h;
	return { inter, union: area(a) + area(b) - inter, iou: inter / (area(a) + area(b) - inter) };
}

/** Hình 8.6: IoU và GIoU của hộp dự đoán (kéo được) so với hộp thật. */
export default function IouViz() {
	const W = 420, H = 300;
	const gt: Box = [120, 80, 260, 220];
	const [c, setC] = useState<[number, number]>([230, 170]);
	const [size, setSize] = useState(130);
	const pred: Box = [c[0] - size / 2, c[1] - size / 2 * 0.9, c[0] + size / 2, c[1] + size / 2 * 0.9];
	const { svgRef, handlers } = useSvgDrag((x, y) => setC([Math.max(20, Math.min(W - 20, x)), Math.max(20, Math.min(H - 20, y))]));
	const r = iou(gt, pred);
	const enc: Box = [Math.min(gt[0], pred[0]), Math.min(gt[1], pred[1]), Math.max(gt[2], pred[2]), Math.max(gt[3], pred[3])];
	const giou = r.iou - (area(enc) - r.union) / area(enc);
	const ix: Box = [Math.max(gt[0], pred[0]), Math.max(gt[1], pred[1]), Math.min(gt[2], pred[2]), Math.min(gt[3], pred[3])];
	const rect = (b: Box, props: Record<string, unknown>) => <rect x={b[0]} y={b[1]} width={Math.max(0, b[2] - b[0])} height={Math.max(0, b[3] - b[1])} {...props} />;
	return (
		<Figure
			title="IoU: phần giao chia phần hợp"
			caption={
				<>
					Hộp xanh là hộp thật, hộp đỏ là dự đoán — <b>kéo</b> hộp đỏ đi và đổi cỡ. Vùng tím là phần giao A ∩ B. IoU = diện tích giao / diện tích hợp, bằng 1 khi trùng khít và 0 khi không chạm nhau. Khung nét đứt là hộp bao nhỏ nhất C dùng cho GIoU: khi hai hộp rời nhau, IoU luôn bằng 0 (không biết xa hay gần), còn GIoU vẫn giảm dần khi hai hộp xa nhau hơn.
				</>
			}
		>
			<Controls>
				<Slider label="cỡ hộp dự đoán" value={size} min={40} max={240} step={2} onChange={setSize} show={String(size)} />
			</Controls>
			<svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: 460 }} role="img" aria-label="Hai hộp và phần giao" {...handlers}>
				{rect(enc, { fill: 'none', stroke: C.muted, strokeDasharray: '6 4' })}
				{rect(gt, { fill: C.blue, fillOpacity: 0.15, stroke: C.blue, strokeWidth: 2.5 })}
				{rect(pred, { fill: C.red, fillOpacity: 0.12, stroke: C.red, strokeWidth: 2.5, style: { cursor: 'grab' } })}
				{r.inter > 0 && rect(ix, { fill: C.purple, fillOpacity: 0.45 })}
				<text x={gt[0] + 4} y={gt[1] + 16} fontSize={12} fill={C.blue}>thật</text>
				<text x={pred[0] + 4} y={pred[3] - 6} fontSize={12} fill={C.red}>dự đoán</text>
			</svg>
			<div className="viz-kv">
				<span>giao = <b>{fmt(r.inter, 0)}</b></span>
				<span>hợp = <b>{fmt(r.union, 0)}</b></span>
				<span>IoU = <b>{fmt(r.iou, 3)}</b></span>
				<span>GIoU = <b>{fmt(giou, 3)}</b></span>
			</div>
		</Figure>
	);
}
