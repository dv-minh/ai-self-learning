import { Fraction } from './fraction';

export type FMatrix = Fraction[][];

/** Hạng của ma trận bằng khử Gauss với phân số chính xác. */
export function rank(m: FMatrix): number {
	const a = m.map((r) => [...r]);
	const rows = a.length;
	const cols = rows ? a[0].length : 0;
	let r = 0;
	for (let c = 0; c < cols && r < rows; c++) {
		const p = a.findIndex((row, i) => i >= r && !row[c].isZero());
		if (p === -1) continue;
		[a[r], a[p]] = [a[p], a[r]];
		for (let i = r + 1; i < rows; i++) {
			if (a[i][c].isZero()) continue;
			const f = a[i][c].div(a[r][c]);
			for (let j = c; j < cols; j++) a[i][j] = a[i][j].sub(f.mul(a[r][j]));
		}
		r++;
	}
	return r;
}

/** Tích ma trận–vector. */
export function matVec(a: FMatrix, x: Fraction[]): Fraction[] {
	return a.map((row) => row.reduce((s, v, j) => s.add(v.mul(x[j])), Fraction.ZERO));
}

export const vecEquals = (u: Fraction[], v: Fraction[]) => u.length === v.length && u.every((x, i) => x.equals(v[i]));
