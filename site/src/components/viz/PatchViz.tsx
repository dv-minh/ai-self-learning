import { useState, type ReactNode } from 'react';
import { C, Choice, Controls, Figure } from './common';

const IMG = 224;

export function SceneArt(): ReactNode {
	return (
		<g>
			<rect x={0} y={0} width={IMG} height={IMG} fill="#9fd3ff" />
			<rect x={0} y={150} width={IMG} height={74} fill="#5fb85a" />
			<circle cx={175} cy={48} r={26} fill="#ffd23f" />
			<polygon points="40,150 40,100 90,62 140,100 140,150" fill="#e76f51" />
			<rect x={52} y={100} width={76} height={50} fill="#f4e1c1" />
			<rect x={78} y={118} width={22} height={32} fill="#8d5524" />
			<rect x={58} y={108} width={14} height={14} fill="#6ec1e4" />
			<rect x={106} y={108} width={14} height={14} fill="#6ec1e4" />
			<rect x={165} y={110} width={10} height={44} fill="#7a4a24" />
			<circle cx={170} cy={100} r={22} fill="#2d7d32" />
			<ellipse cx={60} cy={40} rx={28} ry={11} fill="#ffffff" opacity={0.9} />
		</g>
	);
}

/** Hình 8.4: ViT cắt ảnh thành các patch P×P, duỗi thành chuỗi token, thêm [CLS]. */
export default function PatchViz() {
	const [P, setP] = useState<'16' | '32' | '56'>('56');
	const p = Number(P);
	const g = IMG / p;
	const N = g * g;
	const show = Math.min(N, 17);
	const tile = 44;
	const PER = 9; // số ô mỗi hàng của chuỗi
	const rows = Math.ceil((show + 1) / PER);
	const W = PER * (tile + 8) + 10;
	const H = IMG + 40 + rows * (tile + 22);
	const ox = (W - IMG) / 2;
	const id = 'vit-scene';
	return (
		<Figure
			title="Ảnh thành chuỗi token: cắt patch, duỗi, thêm [CLS]"
			caption={
				<>
					Ảnh {IMG}×{IMG} được lát bằng các ô {p}×{p}: có {g}·{g} = <b>{N}</b> patch, đánh số theo hàng từ trái sang phải. Mỗi patch được duỗi thành {3 * p * p} số (3 kênh màu) rồi chiếu tuyến tính thành một vector D chiều — một “từ” của câu.
					Bên dưới là chuỗi đưa vào Transformer: token [CLS] rồi các patch theo thứ tự (chỉ vẽ tối đa {show} patch đầu). Patch nhỏ hơn: chuỗi dài hơn, attention (bình phương độ dài) đắt hơn nhiều.
				</>
			}
		>
			<Controls>
				<Choice label="Patch P:" value={P} onChange={setP} options={[['56', '56'], ['32', '32'], ['16', '16']]} />
			</Controls>
			<svg viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: 560 }} role="img" aria-label="Ảnh được cắt thành patch và chuỗi token">
				<defs>
					<symbol id={id} viewBox={`0 0 ${IMG} ${IMG}`}>
						<SceneArt />
					</symbol>
				</defs>
				<g transform={`translate(${ox},6)`}>
					<use href={`#${id}`} x={0} y={0} width={IMG} height={IMG} />
					{Array.from({ length: g + 1 }, (_, i) => (
						<g key={i}>
							<line x1={i * p} x2={i * p} y1={0} y2={IMG} stroke="#fff" strokeWidth={1.5} />
							<line y1={i * p} y2={i * p} x1={0} x2={IMG} stroke="#fff" strokeWidth={1.5} />
						</g>
					))}
					{p >= 32 &&
						Array.from({ length: N }, (_, k) => (
							<text key={k} x={(k % g) * p + 4} y={Math.floor(k / g) * p + 13} fontSize={11} fill="#000" fontWeight={700}>{k + 1}</text>
						))}
				</g>
				<text x={W / 2} y={IMG + 30} fontSize={18} textAnchor="middle" fill={C.text}>↓ duỗi theo hàng, thêm [CLS] ở đầu</text>
				<g transform={`translate(10,${IMG + 42})`}>
					{Array.from({ length: show + 1 }, (_, idx) => {
						const x = (idx % PER) * (tile + 8);
						const y = Math.floor(idx / PER) * (tile + 22);
						if (idx === 0)
							return (
								<g key="cls">
									<rect x={x} y={y} width={tile} height={tile} rx={4} fill={C.purple} opacity={0.85} />
									<text x={x + tile / 2} y={y + tile / 2 + 4} fontSize={11} textAnchor="middle" fill="#fff">CLS</text>
								</g>
							);
						const k = idx - 1;
						const r = Math.floor(k / g), c = k % g;
						const last = idx === show && N > show;
						return (
							<g key={k}>
								<svg x={x} y={y} width={tile} height={tile} viewBox={`${c * p} ${r * p} ${p} ${p}`}>
									<use href={`#${id}`} x={0} y={0} width={IMG} height={IMG} />
								</svg>
								<rect x={x} y={y} width={tile} height={tile} fill="none" stroke={C.text} strokeWidth={0.8} />
								<text x={x + tile / 2} y={y + tile + 13} fontSize={10} textAnchor="middle" fill={C.text}>{last ? `${k + 1} … ${N}` : k + 1}</text>
							</g>
						);
					})}
				</g>
			</svg>
			<div className="viz-kv">
				<span>số patch N = (H/P)·(W/P) = <b>{N}</b></span>
				<span>độ dài chuỗi N + 1 = <b>{N + 1}</b></span>
				<span>kích thước ma trận attention mỗi đầu: <b>{(N + 1).toLocaleString('vi-VN')} × {(N + 1).toLocaleString('vi-VN')}</b></span>
			</div>
		</Figure>
	);
}
