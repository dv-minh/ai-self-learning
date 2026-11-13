import type { Answer, Check } from '../../types/exercise';

/** Dạng rút gọn của Answer/Check: chỉ cần các trường liên quan (kho bài luôn có đủ trường). */
export type AnswerLike = Pick<Answer, 'kind'> & Partial<Answer>;
export type CheckLike = Pick<Check, 'mode'> & Partial<Check>;
import { Fraction, fromCanonical } from './fraction';
import { matVec, rank, vecEquals } from './linalg';
import { parseExpr, sameFunction } from './expr';
import { parseNumber } from './parse';

export type UserAnswer =
	| { kind: 'number'; value: string }
	| { kind: 'grid'; cells: string[][] } // vector hoặc ma trận
	| { kind: 'shape'; rows: string; cols: string; undefined: boolean }
	| { kind: 'choice'; selected: number[] }
	| { kind: 'boolean'; value: boolean | null }
	| { kind: 'solution-set'; particular: string[]; basis: string[][] }
	| { kind: 'expression'; value: string };

export interface Verdict {
	correct: boolean;
	/** Lỗi định dạng (không phải sai đáp án); khi có, chưa tính là một lần nộp. */
	formatError?: string;
	/** Giải thích vì sao sai (dùng cho solution-set). */
	detail?: string;
	/** Với lưới: true tại ô sai. */
	wrongCells?: boolean[][];
}

const FORMAT = (formatError: string): Verdict => ({ correct: false, formatError });

function parseInteger(raw: string): number | null {
	return /^\s*\d+\s*$/.test(raw) ? Number(raw) : null;
}

function parseVector(cells: string[]): Fraction[] | string {
	const out: Fraction[] = [];
	for (let i = 0; i < cells.length; i++) {
		const p = parseNumber(cells[i] ?? '');
		if (!p.ok) return `Phần tử ${i + 1}: ${p.error}`;
		out.push(p.value);
	}
	return out;
}

/** solution-set: nghiệm riêng thỏa Ax = b; mỗi vector hướng thỏa Av = 0; các vector hướng độc lập, đủ số lượng. */
function gradeSolutionSet(answer: AnswerLike, check: CheckLike, user: UserAnswer): Verdict {
	if (user.kind !== 'solution-set') return FORMAT('Sai kiểu đáp án.');
	if (!check.a || !check.b) throw new Error('Thiếu hệ Ax = b trong check.');
	const A = check.a.map((r) => r.map(fromCanonical));
	const b = check.b.map(fromCanonical);
	const xp = parseVector(user.particular);
	if (typeof xp === 'string') return FORMAT(`Nghiệm riêng, ${xp}`);
	const basis: Fraction[][] = [];
	for (let k = 0; k < user.basis.length; k++) {
		const v = parseVector(user.basis[k]);
		if (typeof v === 'string') return FORMAT(`Vector hướng ${k + 1}, ${v}`);
		basis.push(v);
	}
	if (!vecEquals(matVec(A, xp), b)) return { correct: false, detail: 'Nghiệm riêng chưa thỏa hệ Ax = b.' };
	const zero = A.map(() => Fraction.ZERO);
	for (let k = 0; k < basis.length; k++) {
		if (!vecEquals(matVec(A, basis[k]), zero)) {
			return { correct: false, detail: `Vector hướng ${k + 1} chưa thỏa Av = 0 (không thuộc không gian null).` };
		}
	}
	if (rank(basis) !== (answer.cols ?? 0)) {
		return { correct: false, detail: 'Các vector hướng phải độc lập tuyến tính và đủ số lượng (bằng số ẩn tự do).' };
	}
	return { correct: true };
}


/** eigvec: mọi v ≠ 0 thỏa Av = λv đều đúng (bội vô hướng nào cũng được). */
function gradeEigvec(answer: AnswerLike, check: CheckLike, user: UserAnswer): Verdict {
	if (user.kind !== 'grid') return FORMAT('Sai kiểu đáp án.');
	if (!check.a || check.lam == null) throw new Error('Thiếu A hoặc λ trong check.');
	const A = check.a.map((r) => r.map(fromCanonical));
	const lam = fromCanonical(check.lam);
	const v: Fraction[] = [];
	for (let i = 0; i < answer.rows!; i++) {
		const p = parseNumber(user.cells[i]?.[0] ?? '');
		if (!p.ok) return FORMAT(`Phần tử ${i + 1}: ${p.error}`);
		v.push(p.value);
	}
	if (v.every((x) => x.isZero())) return { correct: false, detail: 'Vector riêng phải khác vector 0.' };
	const ok = vecEquals(matVec(A, v), v.map((x) => x.mul(lam)));
	return ok ? { correct: true } : { correct: false, detail: 'Vector này chưa thỏa Av = λv.' };
}

/** span: các cột người học nhập phải độc lập, đúng số lượng và sinh ra cùng không gian với cơ sở chuẩn (so bằng hạng). */
function gradeSpan(answer: AnswerLike, user: UserAnswer): Verdict {
	if (user.kind !== 'grid') return FORMAT('Sai kiểu đáp án.');
	const rows = answer.rows!;
	const cols = answer.cols!;
	const mine: Fraction[][] = [];
	for (let j = 0; j < cols; j++) {
		const col: Fraction[] = [];
		for (let i = 0; i < rows; i++) {
			const p = parseNumber(user.cells[i]?.[j] ?? '');
			if (!p.ok) return FORMAT(`Hàng ${i + 1}, cột ${j + 1}: ${p.error}`);
			col.push(p.value);
		}
		mine.push(col);
	}
	const canon: Fraction[][] = [];
	for (let j = 0; j < cols; j++) canon.push(answer.entries!.map((r) => fromCanonical(r[j])));
	if (rank(mine) !== cols) return { correct: false, detail: 'Các vector (cột) phải độc lập tuyến tính và đủ số lượng.' };
	if (rank([...mine, ...canon]) !== cols) return { correct: false, detail: 'Các vector này chưa sinh ra đúng không gian cần tìm.' };
	return { correct: true };
}

/** numeric: đáp án vô tỉ, so sánh sai số tương đối (mặc định 1e-9; bài có thể khai báo check.tolerance). */
function gradeNumeric(answer: AnswerLike, check: CheckLike, user: UserAnswer): Verdict {
	if (user.kind !== 'number') return FORMAT('Sai kiểu đáp án.');
	const p = parseNumber(user.value);
	if (!p.ok) return FORMAT(p.error);
	const want = Number(answer.value);
	const tol = check.tolerance ?? 1e-9;
	const got = p.value.toNumber();
	const err = want === 0 ? Math.abs(got) : Math.abs(got - want) / Math.abs(want);
	return { correct: err <= tol };
}

export function grade(answer: AnswerLike, check: CheckLike, user: UserAnswer): Verdict {
	if (check.mode === 'solution-set') return gradeSolutionSet(answer, check, user);
	if (check.mode === 'eigvec') return gradeEigvec(answer, check, user);
	if (check.mode === 'span') return gradeSpan(answer, user);
	if (check.mode === 'numeric') return gradeNumeric(answer, check, user);
	switch (answer.kind) {
		case 'number': {
			if (user.kind !== 'number') return FORMAT('Sai kiểu đáp án.');
			const p = parseNumber(user.value);
			if (!p.ok) return FORMAT(p.error);
			return { correct: p.value.equals(fromCanonical(answer.value!)) };
		}
		case 'vector':
		case 'matrix': {
			if (user.kind !== 'grid') return FORMAT('Sai kiểu đáp án.');
			const rows = answer.rows!;
			const cols = answer.cols!;
			const wrong: boolean[][] = [];
			for (let i = 0; i < rows; i++) {
				wrong.push([]);
				for (let j = 0; j < cols; j++) {
					const p = parseNumber(user.cells[i]?.[j] ?? '');
					if (!p.ok) return FORMAT(`Hàng ${i + 1}, cột ${j + 1}: ${p.error}`);
					wrong[i].push(!p.value.equals(fromCanonical(answer.entries![i][j])));
				}
			}
			return { correct: wrong.every((r) => r.every((w) => !w)), wrongCells: wrong };
		}
		case 'shape': {
			if (user.kind !== 'shape') return FORMAT('Sai kiểu đáp án.');
			if (user.undefined) return { correct: !!answer.undefined };
			const r = parseInteger(user.rows);
			const c = parseInteger(user.cols);
			if (r === null || c === null) return FORMAT('Nhập số hàng và số cột là số nguyên dương, hoặc chọn "không xác định".');
			return { correct: !answer.undefined && r === answer.rows && c === answer.cols };
		}
		case 'choice':
		case 'multi': {
			if (user.kind !== 'choice') return FORMAT('Sai kiểu đáp án.');
			if (user.selected.length === 0) return FORMAT('Hãy chọn ít nhất một đáp án.');
			const a = [...(answer.correct ?? [])].sort((x, y) => x - y);
			const u = [...user.selected].sort((x, y) => x - y);
			return { correct: a.length === u.length && a.every((v, i) => v === u[i]) };
		}
		case 'boolean': {
			if (user.kind !== 'boolean') return FORMAT('Sai kiểu đáp án.');
			if (user.value === null) return FORMAT('Hãy chọn Đúng hoặc Sai.');
			return { correct: user.value === (answer.value === 'true') };
		}
		case 'expression': {
			if (user.kind !== 'expression') return FORMAT('Sai kiểu đáp án.');
			const vars = answer.variables ?? [];
			const mine = parseExpr(user.value, vars);
			if (!mine.ok) return FORMAT(mine.error);
			const want = parseExpr(answer.value!, vars);
			if (!want.ok) throw new Error(`Đáp án chuẩn không phân tích được: ${want.error}`);
			return { correct: sameFunction(mine.ast, want.ast, vars) };
		}
		default:
			throw new Error(`Dạng đáp án "${answer.kind}" chưa được cài.`);
	}
}

export { Fraction };
