/**
 * Kiểm thử chéo generator ↔ bộ chấm: mọi bài trong kho phải được chấm ĐÚNG với đáp án chuẩn
 * (và với dạng tương đương), và SAI với một đáp án cố ý làm hỏng.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { Exercise } from '../../types/exercise';
import { grade, type UserAnswer } from './compare';
import { Fraction, fromCanonical } from './fraction';

const root = join(__dirname, '../../../public/exercises');
const files: string[] = [];
(function walk(dir: string) {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) walk(p);
		else if (name.endsWith('.json') && name !== 'manifest.json') files.push(p);
	}
})(root);

const col = (v: string[]) => v;

function correctAnswer(ex: Exercise): UserAnswer {
	const a = ex.answer;
	switch (a.kind) {
		case 'number':
			return { kind: 'number', value: a.value! };
		case 'vector':
		case 'matrix':
			return { kind: 'grid', cells: a.entries! };
		case 'shape':
			return { kind: 'shape', rows: String(a.rows ?? ''), cols: String(a.cols ?? ''), undefined: !!a.undefined };
		case 'choice':
		case 'multi':
			return { kind: 'choice', selected: [...a.correct!] };
		case 'boolean':
			return { kind: 'boolean', value: a.value === 'true' };
		case 'solution-set':
			return { kind: 'solution-set', particular: a.particular!, basis: a.basis! };
		case 'expression':
			return { kind: 'expression', value: a.value! };
		default:
			throw new Error(`dạng ${a.kind} chưa có trong kiểm thử`);
	}
}

/** Đáp án sai có chủ đích, hoặc null nếu không dựng được một đáp án chắc chắn sai. */
function wrongAnswer(ex: Exercise): UserAnswer | null {
	const a = ex.answer;
	// Chế độ chấm đặc biệt: đáp án sai chắc chắn là ma trận/vector toàn 0 (không bao giờ là vector riêng hay cơ sở hợp lệ)
	if (ex.check.mode === 'eigvec') return { kind: 'grid', cells: Array.from({ length: a.rows! }, () => ['0']) };
	if (ex.check.mode === 'span') return { kind: 'grid', cells: Array.from({ length: a.rows! }, () => Array(a.cols!).fill('0')) };
	if (ex.check.mode === 'numeric') return { kind: 'number', value: String(Number(a.value) * 2 + 1) };
	switch (a.kind) {
		case 'number':
			return { kind: 'number', value: fromCanonical(a.value!).add(new Fraction(1, 3)).toString() };
		case 'vector':
		case 'matrix': {
			const cells = a.entries!.map((r) => [...r]);
			cells[0][0] = `${cells[0][0]}7`; // đổi hẳn giá trị ô đầu
			return { kind: 'grid', cells };
		}
		case 'shape':
			return a.undefined
				? { kind: 'shape', rows: '1', cols: '1', undefined: false }
				: { kind: 'shape', rows: String(a.rows! + 1), cols: String(a.cols), undefined: false };
		case 'choice':
		case 'multi': {
			const n = ex.choices!.length;
			const others = [...Array(n).keys()].filter((i) => !a.correct!.includes(i));
			if (others.length === 0) return null;
			return { kind: 'choice', selected: [others[0]] };
		}
		case 'boolean':
			return { kind: 'boolean', value: a.value !== 'true' };
		case 'solution-set':
			return { kind: 'solution-set', particular: a.particular!, basis: a.basis!.map((v) => v.map(() => '0')) };
		case 'expression':
			return { kind: 'expression', value: `(${a.value})+1` };
		default:
			return null;
	}
}

describe('kho bài ↔ bộ chấm', () => {
	for (const f of files) {
		const bank = JSON.parse(readFileSync(f, 'utf8')) as { type: string; exercises: Exercise[] };
		it(`${bank.type}: ${bank.exercises.length} bài chấm đúng/sai nhất quán`, () => {
			for (const ex of bank.exercises) {
				const ok = grade(ex.answer, ex.check, correctAnswer(ex));
				expect(ok.formatError, ex.id).toBeUndefined();
				expect(ok.correct, `${ex.id} đáp án chuẩn bị chấm sai`).toBe(true);
				const bad = wrongAnswer(ex);
				if (bad) {
					const v = grade(ex.answer, ex.check, bad);
					expect(v.formatError, `${ex.id}: đáp án sai phải là sai thật, không phải lỗi định dạng`).toBeUndefined();
					expect(v.correct, `${ex.id} đáp án sai bị chấm đúng`).toBe(false);
				}
			}
		});
	}

	it('eigvec: chấp nhận mọi bội vô hướng khác 0 của vector riêng', () => {
		for (const f of files) {
			const bank = JSON.parse(readFileSync(f, 'utf8')) as { exercises: Exercise[] };
			for (const ex of bank.exercises.filter((e) => e.check.mode === 'eigvec').slice(0, 60)) {
				const scaled = ex.answer.entries!.map((r) => [`${BigInt(r[0].split('/')[0]) * -3n}/${r[0].split('/')[1] ?? '1'}`]);
				expect(grade(ex.answer, ex.check, { kind: 'grid', cells: scaled }).correct, ex.id).toBe(true);
			}
		}
	});

	it('span: chấp nhận cơ sở khác (đổi thứ tự cột, nhân cột với 2)', () => {
		for (const f of files) {
			const bank = JSON.parse(readFileSync(f, 'utf8')) as { exercises: Exercise[] };
			for (const ex of bank.exercises.filter((e) => e.check.mode === 'span').slice(0, 80)) {
				const rows = ex.answer.entries!;
				const cols = ex.answer.cols!;
				const order = [...Array(cols).keys()].reverse();
				const cells = rows.map((r) => order.map((j) => `${BigInt(r[j].split('/')[0]) * 2n}/${r[j].split('/')[1] ?? '1'}`));
				expect(grade(ex.answer, ex.check, { kind: 'grid', cells }).correct, ex.id).toBe(true);
			}
		}
	});

	it('numeric: chấp nhận đáp án làm tròn 3 chữ số thập phân', () => {
		let seen = 0;
		for (const f of files) {
			const bank = JSON.parse(readFileSync(f, 'utf8')) as { exercises: Exercise[] };
			for (const ex of bank.exercises.filter((e) => e.check.mode === 'numeric')) {
				seen++;
				const rounded = Number(ex.answer.value).toFixed(3);
				expect(grade(ex.answer, ex.check, { kind: 'number', value: rounded }).correct, ex.id).toBe(true);
			}
		}
		expect(seen).toBeGreaterThan(0);
	});

	it('chấp nhận dạng tương đương: nhân tử số/mẫu số và thập phân', () => {
		for (const f of files) {
			const bank = JSON.parse(readFileSync(f, 'utf8')) as { exercises: Exercise[] };
			for (const ex of bank.exercises.filter((e) => e.answer.kind === 'number' && e.check.mode === 'exact').slice(0, 50)) {
				const [p, q] = ex.answer.value!.split('/');
				const alt = `${BigInt(p) * 3n}/${BigInt(q ?? '1') * 3n}`;
				expect(grade(ex.answer, ex.check, { kind: 'number', value: alt }).correct, ex.id).toBe(true);
			}
		}
	});

	it('solution-set: chấp nhận nghiệm riêng và hướng khác nhưng đúng', () => {
		for (const f of files) {
			const bank = JSON.parse(readFileSync(f, 'utf8')) as { exercises: Exercise[] };
			for (const ex of bank.exercises.filter((e) => e.answer.kind === 'solution-set').slice(0, 40)) {
				const a = ex.answer;
				const k = 2;
				// nghiệm riêng mới = nghiệm riêng + vector hướng đầu tiên; hướng mới = k × hướng cũ
				const particular = a.particular!.map((s, i) => {
					const [p, q] = s.split('/');
					const [bp, bq] = a.basis![0][i].split('/');
					const num = BigInt(p) * BigInt(bq ?? '1') + BigInt(bp) * BigInt(q ?? '1');
					return `${num}/${BigInt(q ?? '1') * BigInt(bq ?? '1')}`;
				});
				const basis = a.basis!.map((v) => v.map((s) => {
					const [p, q] = s.split('/');
					return `${BigInt(p) * BigInt(k)}/${BigInt(q ?? '1')}`;
				}));
				const v = grade(a, ex.check, { kind: 'solution-set', particular: col(particular), basis });
				expect(v.correct, ex.id).toBe(true);
			}
		}
	});
});
