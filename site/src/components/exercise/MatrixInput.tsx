import { useRef } from 'react';
import { splitMatrixText } from '../../lib/math/parse';

interface Props {
	rows: number;
	cols: number;
	cells: string[][];
	onChange: (cells: string[][]) => void;
	wrong?: boolean[][];
	disabled?: boolean;
	label?: string;
}

/** Lưới ô nhập: Tab/Enter/phím mũi tên để di chuyển; dán được văn bản "1 2 3; 4 5 6". */
export default function MatrixInput({ rows, cols, cells, onChange, wrong, disabled, label }: Props) {
	const refs = useRef<(HTMLInputElement | null)[][]>([]);

	const set = (i: number, j: number, v: string) => {
		const next = cells.map((r) => [...r]);
		next[i][j] = v;
		onChange(next);
	};
	const focus = (i: number, j: number) => refs.current[i]?.[j]?.focus();

	const onKey = (e: React.KeyboardEvent<HTMLInputElement>, i: number, j: number) => {
		const input = e.currentTarget;
		const atStart = input.selectionStart === 0 && input.selectionEnd === 0;
		const atEnd = input.selectionStart === input.value.length;
		if (e.key === 'Enter') {
			e.preventDefault();
			const k = i * cols + j + 1;
			if (k < rows * cols) focus(Math.floor(k / cols), k % cols);
		} else if (e.key === 'ArrowDown' && i + 1 < rows) {
			e.preventDefault();
			focus(i + 1, j);
		} else if (e.key === 'ArrowUp' && i > 0) {
			e.preventDefault();
			focus(i - 1, j);
		} else if (e.key === 'ArrowRight' && atEnd && j + 1 < cols) {
			e.preventDefault();
			focus(i, j + 1);
		} else if (e.key === 'ArrowLeft' && atStart && j > 0) {
			e.preventDefault();
			focus(i, j - 1);
		}
	};

	const onPaste = (e: React.ClipboardEvent<HTMLInputElement>, i: number, j: number) => {
		const text = e.clipboardData.getData('text');
		if (!/[;\n\t ]/.test(text.trim())) return; // dán một giá trị: để mặc định
		e.preventDefault();
		const grid = splitMatrixText(text);
		const next = cells.map((r) => [...r]);
		grid.forEach((row, di) =>
			row.forEach((v, dj) => {
				if (i + di < rows && j + dj < cols) next[i + di][j + dj] = v;
			}),
		);
		onChange(next);
	};

	return (
		<div className="matrix-input" role="group" aria-label={label ?? `Ma trận ${rows} nhân ${cols}`}>
			<div className="matrix-bracket" style={{ gridTemplateColumns: `repeat(${cols}, 4.2rem)` }}>
				{Array.from({ length: rows }, (_, i) =>
					Array.from({ length: cols }, (_, j) => (
						<input
							key={`${i}-${j}`}
							ref={(el) => {
								(refs.current[i] ??= [])[j] = el;
							}}
							className={wrong?.[i]?.[j] ? 'cell wrong' : wrong ? 'cell right' : 'cell'}
							value={cells[i]?.[j] ?? ''}
							disabled={disabled}
							inputMode="text"
							autoComplete="off"
							aria-label={`Hàng ${i + 1}, cột ${j + 1}`}
							onChange={(e) => set(i, j, e.target.value)}
							onKeyDown={(e) => onKey(e, i, j)}
							onPaste={(e) => onPaste(e, i, j)}
						/>
					)),
				)}
			</div>
		</div>
	);
}
