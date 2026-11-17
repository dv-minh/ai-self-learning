import { useMemo, useState } from 'react';
import { C, Choice, Controls, Figure, fmt, gauss, rng } from './common';

const NB = 4, NF = 5;

/** Hình 7.4: BatchNorm chuẩn hóa theo cột (mỗi đặc trưng, qua các mẫu); LayerNorm theo hàng (mỗi mẫu, qua các đặc trưng). */
export default function NormViz() {
	const [kind, setKind] = useState<'bn' | 'ln'>('bn');
	const [after, setAfter] = useState(false);
	const [focus, setFocus] = useState(0);
	const X = useMemo(() => {
		const r = rng(19);
		return Array.from({ length: NB }, () => Array.from({ length: NF }, (_, j) => Math.round((j * 2 - 3 + 1.5 * gauss(r)) * 10) / 10));
	}, []);
	const norm = useMemo(() => {
		const out = X.map((row) => row.slice());
		if (kind === 'bn') {
			for (let j = 0; j < NF; j++) {
				const col = X.map((r) => r[j]);
				const m = col.reduce((a, b) => a + b, 0) / NB;
				const s = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / NB) || 1;
				for (let i = 0; i < NB; i++) out[i][j] = (X[i][j] - m) / s;
			}
		} else {
			for (let i = 0; i < NB; i++) {
				const m = X[i].reduce((a, b) => a + b, 0) / NF;
				const s = Math.sqrt(X[i].reduce((a, b) => a + (b - m) ** 2, 0) / NF) || 1;
				for (let j = 0; j < NF; j++) out[i][j] = (X[i][j] - m) / s;
			}
		}
		return out;
	}, [X, kind]);
	const M = after ? norm : X;
	const inGroup = (i: number, j: number) => (kind === 'bn' ? j === focus : i === focus);
	const grp = kind === 'bn' ? X.map((r) => r[focus]) : X[focus];
	const mu = grp.reduce((a, b) => a + b, 0) / grp.length;
	const sd = Math.sqrt(grp.reduce((a, b) => a + (b - mu) ** 2, 0) / grp.length);
	return (
		<Figure
			title="BatchNorm và LayerNorm khác nhau ở hướng lấy trung bình"
			caption={
				<>
					Bảng là một batch: mỗi hàng là một mẫu, mỗi cột là một đặc trưng. Ô viền đậm là <b>một nhóm</b> được chuẩn hóa cùng nhau (bấm vào bảng để chọn nhóm khác). BatchNorm: mỗi <b>cột</b> (một đặc trưng qua mọi mẫu trong batch) — nên phụ thuộc vào batch. LayerNorm: mỗi <b>hàng</b> (mọi đặc trưng của một mẫu) — không phụ thuộc mẫu khác, nên dùng được với batch cỡ 1 và trong Transformer.
					Bật “sau chuẩn hóa” để thấy mỗi nhóm có trung bình 0 và độ lệch chuẩn 1 (chưa nhân γ, cộng β).
				</>
			}
		>
			<Controls>
				<Choice value={kind} onChange={(v) => { setKind(v); setFocus(0); }} options={[['bn', 'BatchNorm (theo cột)'], ['ln', 'LayerNorm (theo hàng)']]} />
				<Choice value={after ? 'a' : 'b'} onChange={(v) => setAfter(v === 'a')} options={[['b', 'trước chuẩn hóa'], ['a', 'sau chuẩn hóa']]} />
			</Controls>
			<table className="viz-table" style={{ fontSize: '0.95rem' }}>
				<thead>
					<tr>
						<th></th>
						{Array.from({ length: NF }, (_, j) => <th key={j}>đặc trưng {j + 1}</th>)}
					</tr>
				</thead>
				<tbody>
					{M.map((row, i) => (
						<tr key={i}>
							<th>mẫu {i + 1}</th>
							{row.map((v, j) => (
								<td
									key={j}
									onClick={() => setFocus(kind === 'bn' ? j : i)}
									style={{
										cursor: 'pointer',
										background: inGroup(i, j) ? 'var(--sl-color-accent-low)' : undefined,
										outline: inGroup(i, j) ? `2px solid ${C.accent}` : undefined,
										fontWeight: inGroup(i, j) ? 700 : 400,
									}}
								>
									{fmt(v, 2)}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>
			<div className="viz-kv">
				<span>nhóm đang chọn ({kind === 'bn' ? `cột ${focus + 1}` : `hàng ${focus + 1}`}): μ = <b>{fmt(mu, 3)}</b>, σ = <b>{fmt(sd, 3)}</b></span>
				<span>mỗi ô: x̂ = (x − μ)/σ</span>
			</div>
		</Figure>
	);
}
