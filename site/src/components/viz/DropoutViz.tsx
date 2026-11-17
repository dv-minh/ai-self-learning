import { useMemo, useState } from 'react';
import { C, Controls, Figure, Slider, fmt, rng } from './common';

const LAYERS = [4, 7, 7, 3];

/** Hình 7.5: mỗi bước huấn luyện, dropout tắt ngẫu nhiên một phần nơ-ron và phóng to phần còn lại 1/(1−p). */
export default function DropoutViz() {
	const [p, setP] = useState(0.5);
	const [seed, setSeed] = useState(1);
	const [train, setTrain] = useState(true);
	const mask = useMemo(() => {
		const r = rng(seed * 31 + 7);
		return LAYERS.map((n, l) => Array.from({ length: n }, () => (l === 0 || l === LAYERS.length - 1 ? true : r() >= p)));
	}, [seed, p]);
	const W = 520, H = 300;
	const xs = LAYERS.map((_, l) => 60 + (l * (W - 120)) / (LAYERS.length - 1));
	const ys = (n: number) => Array.from({ length: n }, (_, i) => H / 2 + (i - (n - 1) / 2) * 38);
	const on = (l: number, i: number) => !train || mask[l][i];
	const kept = mask.slice(1, -1).flat().filter(Boolean).length;
	const total = LAYERS.slice(1, -1).reduce((a, b) => a + b, 0);
	return (
		<Figure
			title="Dropout: mỗi bước huấn luyện là một mạng con khác"
			caption={
				<>
					Mỗi lần bấm “Bước huấn luyện mới”, mỗi nơ-ron ẩn bị tắt độc lập với xác suất p (ô gạch chéo) — các cạnh nối với nó biến mất trong bước đó. Nơ-ron còn lại nhân đầu ra với 1/(1 − p) = <b>{fmt(1 / (1 - p), 3)}</b> để kỳ vọng tổng tín hiệu không đổi (inverted dropout).
					Khi suy luận, mọi nơ-ron đều bật và không nhân gì thêm. Vì không thể dựa vào một nơ-ron cụ thể nào, mạng học các đặc trưng dư thừa, bền hơn.
				</>
			}
		>
			<Controls>
				<Slider label="xác suất bỏ p" value={p} min={0} max={0.9} step={0.05} onChange={setP} />
			</Controls>
			<div className="viz-buttons">
				<button type="button" className="primary" onClick={() => { setTrain(true); setSeed((s) => s + 1); }}>🎲 Bước huấn luyện mới</button>
				<button type="button" onClick={() => setTrain(false)}>Chế độ suy luận</button>
			</div>
			<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mạng với các nơ-ron bị dropout">
				{LAYERS.slice(0, -1).map((n, l) =>
					ys(n).map((y1, i) =>
						ys(LAYERS[l + 1]).map((y2, j) => {
							const active = on(l, i) && on(l + 1, j);
							return <line key={`${l}-${i}-${j}`} x1={xs[l]} y1={y1} x2={xs[l + 1]} y2={y2} stroke={active ? C.accent : C.grid} strokeOpacity={active ? 0.55 : 0.25} strokeWidth={active ? 1.3 : 0.6} />;
						}),
					),
				)}
				{LAYERS.map((n, l) =>
					ys(n).map((y, i) => {
						const a = on(l, i);
						return (
							<g key={`${l}-${i}`}>
								<circle cx={xs[l]} cy={y} r={13} fill={a ? C.accent : 'var(--sl-color-bg)'} fillOpacity={a ? 0.85 : 1} stroke={a ? C.accent : C.muted} strokeWidth={1.5} />
								{!a && <path d={`M${xs[l] - 8},${y - 8}L${xs[l] + 8},${y + 8}M${xs[l] + 8},${y - 8}L${xs[l] - 8},${y + 8}`} stroke={C.red} strokeWidth={2} />}
								{a && train && l > 0 && l < LAYERS.length - 1 && p > 0 && <text x={xs[l] + 16} y={y + 4} fontSize={10} fill={C.text}>×{fmt(1 / (1 - p), 2)}</text>}
							</g>
						);
					}),
				)}
				<text x={xs[0]} y={22} fontSize={12} textAnchor="middle" fill={C.text}>đầu vào</text>
				<text x={xs[1]} y={22} fontSize={12} textAnchor="middle" fill={C.text}>ẩn 1</text>
				<text x={xs[2]} y={22} fontSize={12} textAnchor="middle" fill={C.text}>ẩn 2</text>
				<text x={xs[3]} y={22} fontSize={12} textAnchor="middle" fill={C.text}>đầu ra</text>
			</svg>
			<div className="viz-kv">
				<span>{train ? <>huấn luyện: còn <b>{kept}/{total}</b> nơ-ron ẩn (kỳ vọng {fmt((1 - p) * total, 1)})</> : <b>suy luận: bật tất cả, không nhân hệ số</b>}</span>
			</div>
		</Figure>
	);
}
