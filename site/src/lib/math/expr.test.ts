import { describe, expect, it } from 'vitest';
import { evalNode, parseExpr, sameFunction, toLatex } from './expr';

const parse = (s: string, v = ['x']) => {
	const r = parseExpr(s, v);
	if (!r.ok) throw new Error(r.error);
	return r.ast;
};
const val = (s: string, x = 2, vars = ['x']) => evalNode(parse(s, vars), { x, y: 3 });

describe('parseExpr / evalNode', () => {
	it('số học cơ bản và thứ tự phép tính', () => {
		expect(val('1+2*3')).toBe(7);
		expect(val('(1+2)*3')).toBe(9);
		expect(val('2^3^2')).toBe(512); // lũy thừa kết hợp phải
		expect(val('-x^2')).toBe(-4);
		expect(val('10/4')).toBe(2.5);
	});
	it('nhân ngầm', () => {
		expect(val('3x')).toBe(6);
		expect(val('3x^2')).toBe(12);
		expect(val('x(x+1)')).toBe(6);
		expect(val('2(x+1)')).toBe(6);
		expect(val('(x+1)(x-1)')).toBe(3);
		expect(val('xy', 2, ['x', 'y'])).toBe(6);
	});
	it('hàm và hằng', () => {
		expect(val('exp(0)')).toBe(1);
		expect(val('ln(e)')).toBeCloseTo(1);
		expect(val('e^x')).toBeCloseTo(Math.E ** 2);
		expect(val('sqrt(x+2)')).toBe(2);
		expect(val('sin(0)+cos(0)')).toBe(1);
		expect(val('x^-1')).toBe(0.5);
	});
	it('thập phân dùng dấu chấm hoặc phẩy', () => {
		expect(val('0.5x')).toBe(1);
		expect(val('0,5x')).toBe(1);
	});
	it('báo lỗi rõ ràng', () => {
		for (const bad of ['', '2+', '(x', 'x)', 'sin x', '3 $ 4', 'k+1', 'x^']) {
			const r = parseExpr(bad, ['x']);
			expect(r.ok, bad).toBe(false);
			if (!r.ok) expect(r.error.length).toBeGreaterThan(0);
		}
	});
});

describe('sameFunction', () => {
	const same = (a: string, b: string, v = ['x']) => sameFunction(parse(a, v), parse(b, v), v);
	it('nhận các dạng tương đương', () => {
		expect(same('3x^2+2x', '2*x + 3*x^2')).toBe(true);
		expect(same('(x+1)^2', 'x^2+2x+1')).toBe(true);
		expect(same('1/x', 'x^-1')).toBe(true);
		expect(same('2*e^(2x)', '2exp(2x)')).toBe(true);
		expect(same('xy+y', 'y(x+1)', ['x', 'y'])).toBe(true);
	});
	it('từ chối biểu thức khác', () => {
		expect(same('3x^2', '3x^2+1')).toBe(false);
		expect(same('x^2', 'x^3')).toBe(false);
		expect(same('xy', 'x+y', ['x', 'y'])).toBe(false);
	});
	it('bỏ qua điểm mẫu ngoài miền của đáp án chuẩn', () => {
		expect(same('ln(x)', 'ln(x)')).toBe(true);
	});
});

describe('toLatex', () => {
	it('tạo LaTeX hợp lý', () => {
		expect(toLatex(parse('3x^2+2x'))).toBe('3\\cdot x^{2} + 2\\cdot x');
		expect(toLatex(parse('1/x'))).toBe('\\frac{1}{x}');
		expect(toLatex(parse('exp(2x)'))).toBe('e^{2\\cdot x}');
		expect(toLatex(parse('(x+1)*2'))).toBe('\\left(x + 1\\right)\\cdot 2');
	});
});
