import { describe, expect, it } from 'vitest';
import { grade } from './compare';
import { Fraction, fromCanonical } from './fraction';
import { rank } from './linalg';
import { parseNumber, splitMatrixText } from './parse';

describe('Fraction', () => {
	it('luôn rút gọn và mẫu dương', () => {
		expect(new Fraction(2, 4).toString()).toBe('1/2');
		expect(new Fraction(1, -2).toString()).toBe('-1/2');
		expect(new Fraction(4, 2).toString()).toBe('2');
	});
	it('cộng, trừ, nhân, chia chính xác', () => {
		const a = new Fraction(1, 3);
		const b = new Fraction(1, 6);
		expect(a.add(b).toString()).toBe('1/2');
		expect(a.sub(b).toString()).toBe('1/6');
		expect(a.mul(b).toString()).toBe('1/18');
		expect(a.div(b).toString()).toBe('2');
	});
	it('so sánh', () => {
		expect(new Fraction(1, 3).cmp(new Fraction(1, 2))).toBe(-1);
		expect(new Fraction(2, 4).equals(new Fraction(1, 2))).toBe(true);
	});
	it('không dính sai số float', () => {
		let s = Fraction.ZERO;
		for (let i = 0; i < 10; i++) s = s.add(new Fraction(1, 10));
		expect(s.equals(Fraction.ONE)).toBe(true);
	});
	it('chia cho 0 báo lỗi', () => {
		expect(() => new Fraction(1, 0)).toThrow();
		expect(() => Fraction.ONE.div(Fraction.ZERO)).toThrow();
	});
	it('fromCanonical', () => {
		expect(fromCanonical('-3/4').toString()).toBe('-3/4');
		expect(fromCanonical('5').toString()).toBe('5');
	});
});

describe('parseNumber', () => {
	const val = (s: string) => {
		const r = parseNumber(s);
		return r.ok ? r.value.toString() : 'ERR';
	};
	it('đọc nguyên, phân số, thập phân, dấu phẩy', () => {
		expect(val('-3')).toBe('-3');
		expect(val('2/4')).toBe('1/2');
		expect(val('0.5')).toBe('1/2');
		expect(val('0,25')).toBe('1/4');
		expect(val(' -3 / 4 ')).toBe('-3/4');
		expect(val('−2')).toBe('-2');
	});
	it('báo lỗi rõ ràng', () => {
		expect(val('')).toBe('ERR');
		expect(val('abc')).toBe('ERR');
		expect(val('1/0')).toBe('ERR');
		expect(val('1/2/3')).toBe('ERR');
	});
	it('tách văn bản ma trận', () => {
		expect(splitMatrixText('1 2 3; 4 5 6')).toEqual([['1', '2', '3'], ['4', '5', '6']]);
	});
});

describe('grade exact', () => {
	const check = { mode: 'exact' as const };
	it('chấp nhận đáp án tương đương 2/4 = 1/2 = 0.5', () => {
		const ans = { kind: 'number' as const, value: '1/2' };
		for (const s of ['2/4', '1/2', '0.5', '0,5']) {
			expect(grade(ans, check, { kind: 'number', value: s }).correct).toBe(true);
		}
		expect(grade(ans, check, { kind: 'number', value: '1/3' }).correct).toBe(false);
	});
	it('tô đúng ô sai trong ma trận', () => {
		const ans = { kind: 'matrix' as const, rows: 2, cols: 2, entries: [['1', '2'], ['3', '4']] };
		const v = grade(ans, check, { kind: 'grid', cells: [['1', '2'], ['3', '5']] });
		expect(v.correct).toBe(false);
		expect(v.wrongCells).toEqual([[false, false], [false, true]]);
	});
	it('lỗi định dạng không bị chấm sai', () => {
		const ans = { kind: 'number' as const, value: '1' };
		const v = grade(ans, check, { kind: 'number', value: 'x' });
		expect(v.formatError).toBeTruthy();
	});
	it('shape: kích thước và không xác định', () => {
		const ok = { kind: 'shape' as const, rows: 2, cols: 3 };
		expect(grade(ok, check, { kind: 'shape', rows: '2', cols: '3', undefined: false }).correct).toBe(true);
		expect(grade(ok, check, { kind: 'shape', rows: '3', cols: '2', undefined: false }).correct).toBe(false);
		expect(grade(ok, check, { kind: 'shape', rows: '', cols: '', undefined: true }).correct).toBe(false);
		const und = { kind: 'shape' as const, undefined: true };
		expect(grade(und, check, { kind: 'shape', rows: '', cols: '', undefined: true }).correct).toBe(true);
	});
	it('multi: phải đúng cả tập', () => {
		const ans = { kind: 'multi' as const, correct: [0, 2] };
		expect(grade(ans, check, { kind: 'choice', selected: [2, 0] }).correct).toBe(true);
		expect(grade(ans, check, { kind: 'choice', selected: [0] }).correct).toBe(false);
	});
	it('boolean', () => {
		const ans = { kind: 'boolean' as const, value: 'false' };
		expect(grade(ans, check, { kind: 'boolean', value: false }).correct).toBe(true);
		expect(grade(ans, check, { kind: 'boolean', value: null }).formatError).toBeTruthy();
	});
});

describe('grade solution-set', () => {
	// x1 + x2 + x3 = 3 và x2 - x3 = 1: nghiệm tổng quát (2,1,0) + t(-2,1,1)
	const check = { mode: 'solution-set' as const, a: [['1', '1', '1'], ['0', '1', '-1']], b: ['3', '1'] };
	const ans = { kind: 'solution-set' as const, rows: 3, cols: 1, particular: ['2', '1', '0'], basis: [['-2', '1', '1']] };
	it('chấp nhận nghiệm riêng và hướng khác nhưng đúng', () => {
		const v = grade(ans, check, { kind: 'solution-set', particular: ['0', '2', '1'], basis: [['4', '-2', '-2']] });
		expect(v.correct).toBe(true);
	});
	it('từ chối nghiệm riêng sai', () => {
		const v = grade(ans, check, { kind: 'solution-set', particular: ['1', '1', '1'], basis: [['-2', '1', '1']] });
		expect(v.correct).toBe(false);
		expect(v.detail).toMatch(/Ax = b/);
	});
	it('từ chối vector hướng không thuộc không gian null', () => {
		const v = grade(ans, check, { kind: 'solution-set', particular: ['2', '1', '0'], basis: [['1', '1', '1']] });
		expect(v.correct).toBe(false);
		expect(v.detail).toMatch(/không gian null/);
	});
	it('từ chối vector hướng bằng 0 (không độc lập)', () => {
		const v = grade(ans, check, { kind: 'solution-set', particular: ['2', '1', '0'], basis: [['0', '0', '0']] });
		expect(v.correct).toBe(false);
	});
	it('báo lỗi định dạng', () => {
		const v = grade(ans, check, { kind: 'solution-set', particular: ['2', 'x', '0'], basis: [['-2', '1', '1']] });
		expect(v.formatError).toBeTruthy();
	});
});

describe('rank', () => {
	it('hạng với phân số', () => {
		const f = (r: string[]) => r.map(fromCanonical);
		expect(rank([f(['1', '2']), f(['2', '4'])])).toBe(1);
		expect(rank([f(['1', '1/2']), f(['0', '1'])])).toBe(2);
		expect(rank([f(['0', '0'])])).toBe(0);
	});
});
