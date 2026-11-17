import { useState } from 'react';
import { C, Figure, rng } from './common';
import { SceneArt } from './PatchViz';

type Aug = { hflip: boolean; vflip: boolean; rot: number; crop: [number, number, number] | null; bright: number; label: string };
const NONE: Aug = { hflip: false, vflip: false, rot: 0, crop: null, bright: 1, label: 'ảnh gốc' };
const S = 224;

/** Hình 8.5: cùng một ảnh, nhiều phiên bản biến đổi — nhãn không đổi. */
export default function AugmentViz() {
	const [aug, setAug] = useState<Aug[]>([
		{ ...NONE, hflip: true, label: 'lật ngang' },
		{ ...NONE, rot: 90, label: 'quay 90°' },
		{ ...NONE, crop: [40, 60, 130], label: 'cắt + phóng to' },
		{ ...NONE, bright: 1.45, label: 'tăng độ sáng' },
	]);
	const [seed, setSeed] = useState(1);
	const random = () => {
		const r = rng(seed * 17 + 3);
		setSeed(seed + 1);
		setAug(
			Array.from({ length: 4 }, () => {
				const a: Aug = { ...NONE, label: '' };
				const parts: string[] = [];
				if (r() < 0.5) { a.hflip = true; parts.push('lật ngang'); }
				if (r() < 0.5) {
					const sz = 120 + Math.floor(r() * 80);
					a.crop = [Math.floor(r() * (S - sz)), Math.floor(r() * (S - sz)), sz];
					parts.push('cắt');
				}
				if (r() < 0.4) { a.bright = 0.6 + r() * 0.9; parts.push('độ sáng'); }
				if (r() < 0.2) { a.rot = 90 * (1 + Math.floor(r() * 3)); parts.push(`quay ${a.rot}°`); }
				a.label = parts.join(' + ') || 'giữ nguyên';
				return a;
			}),
		);
	};
	const tile = 150;
	const render = (a: Aug, key: string) => {
		const vb = a.crop ? `${a.crop[0]} ${a.crop[1]} ${a.crop[2]} ${a.crop[2]}` : `0 0 ${S} ${S}`;
		const tf = `rotate(${a.rot} ${S / 2} ${S / 2}) translate(${a.hflip ? S : 0} ${a.vflip ? S : 0}) scale(${a.hflip ? -1 : 1} ${a.vflip ? -1 : 1})`;
		return (
			<figure key={key} style={{ margin: 0, textAlign: 'center' }}>
				<svg viewBox={vb} style={{ width: tile, height: tile, borderRadius: 6, border: `1px solid ${C.grid}` }} role="img" aria-label={a.label}>
					<g transform={tf} style={{ filter: `brightness(${a.bright})` }}>
						<SceneArt />
					</g>
				</svg>
				<div style={{ fontSize: '0.85rem' }}>{a.label}</div>
			</figure>
		);
	};
	return (
		<Figure
			title="Data augmentation: một ảnh, nhiều phiên bản, cùng một nhãn “ngôi nhà”"
			caption={
				<>
					Mỗi epoch, mô hình thấy một phiên bản biến đổi ngẫu nhiên khác của cùng ảnh. Nhãn giữ nguyên, nên mô hình buộc phải học “ngôi nhà” không phụ thuộc vị trí, hướng trái/phải hay độ sáng — tương đương có thêm nhiều dữ liệu miễn phí.
					Lưu ý chọn phép biến đổi hợp lý: lật dọc hoặc quay 180° ảnh phong cảnh tạo ra ảnh không có thật (trời ở dưới); với chữ số, lật ngang có thể đổi nghĩa.
				</>
			}
		>
			<div className="viz-buttons">
				<button type="button" className="primary" onClick={random}>🎲 Lấy 4 phiên bản ngẫu nhiên</button>
			</div>
			<div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem', justifyContent: 'center' }}>
				{render(NONE, 'orig')}
				{aug.map((a, i) => render(a, String(i)))}
			</div>
		</Figure>
	);
}
