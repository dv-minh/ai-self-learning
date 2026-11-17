import { useState } from 'react';
import { C, Controls, Figure, Slider, fmt } from './common';

/** Hình 7.7: kích thước tương đối của W (d_out × d_in) so với B (d_out × r) và A (r × d_in). */
export default function LoraViz() {
	const [lg, setLg] = useState(10); // d = 2^lg
	const [r, setR] = useState(8);
	const d = 2 ** lg;
	const full = d * d;
	const lora = r * (d + d);
	const S = 220; // cạnh hình vuông W
	const rw = Math.max(2, (r / d) * S);
	const W = 2 * S + rw + 140, H = S + 70;
	const n = (x: number) => x.toLocaleString('vi-VN');
	return (
		<Figure
			title="LoRA: thay vì cập nhật cả W, học hai ma trận mỏng B và A"
			caption={
				<>
					Hình vẽ đúng tỉ lệ. W (xám) đóng băng, không huấn luyện. Phần cập nhật ΔW = BA có cùng kích thước d × d với W nhưng hạng không quá r, và chỉ cần lưu B (d × r) và A (r × d) — hai dải rất mỏng.
					Số tham số huấn luyện giảm từ d² xuống r(d + d). Khi bắt đầu, B = 0 nên ΔW = 0: mô hình ban đầu y hệt mô hình gốc.
				</>
			}
		>
			<Controls>
				<Slider label="d = d_in = d_out" value={lg} min={6} max={13} step={1} onChange={setLg} show={n(d)} />
				<Slider label="hạng r" value={r} min={1} max={64} step={1} onChange={setR} show={String(r)} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="So sánh kích thước W và hai ma trận LoRA">
				<rect x={20} y={30} width={S} height={S} fill={C.muted} opacity={0.35} stroke={C.text} />
				<text x={20 + S / 2} y={30 + S / 2} fontSize={16} textAnchor="middle" fill={C.text}>W (đóng băng)</text>
				<text x={20 + S / 2} y={22} fontSize={12} textAnchor="middle" fill={C.text}>{n(d)} × {n(d)}</text>
				<text x={S + 45} y={30 + S / 2 + 6} fontSize={22} textAnchor="middle" fill={C.text}>+</text>
				<rect x={S + 75} y={30} width={rw} height={S} fill={C.orange} opacity={0.8} />
				<text x={S + 75 + rw / 2} y={22} fontSize={12} textAnchor="middle" fill={C.orange}>B: {n(d)} × {r}</text>
				<text x={S + 75 + rw + 18} y={30 + S / 2 + 6} fontSize={18} textAnchor="middle" fill={C.text}>·</text>
				<rect x={S + 75 + rw + 36} y={30} width={S} height={rw} fill={C.blue} opacity={0.8} />
				<text x={S + 75 + rw + 36 + S / 2} y={30 + rw + 16} fontSize={12} textAnchor="middle" fill={C.blue}>A: {r} × {n(d)}</text>
				<text x={W / 2} y={H - 10} fontSize={13} textAnchor="middle" fill={C.text}>W′ = W + (α/r)·B·A</text>
			</svg>
			<div className="viz-kv">
				<span>cập nhật toàn bộ: <b>{n(full)}</b> tham số</span>
				<span>LoRA: r(d + d) = <b>{n(lora)}</b> tham số</span>
				<span>tỉ lệ: <b>{fmt((100 * lora) / full, 3)}%</b></span>
			</div>
		</Figure>
	);
}
