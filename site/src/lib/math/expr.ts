/**
 * Phân tích, tính giá trị và chấm biểu thức người học nhập (docs/spec.md: dạng `expression`).
 * Cú pháp: + - * / ^, ngoặc, nhân ngầm (2x, x(x+1)), hàm sin cos tan exp ln sqrt abs, hằng e và pi.
 * Chấm bằng cách so giá trị số tại các điểm mẫu cố định (không so sánh chuỗi).
 */
export type Node =
	| { t: 'num'; v: number }
	| { t: 'var'; n: string }
	| { t: 'neg'; a: Node }
	| { t: 'bin'; op: '+' | '-' | '*' | '/' | '^'; a: Node; b: Node }
	| { t: 'fn'; n: string; a: Node };

const FUNCS = new Set(['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'abs']);
const CONSTS: Record<string, number> = { e: Math.E, pi: Math.PI };

type Tok = { k: 'num' | 'id' | 'op' | 'lp' | 'rp'; s: string };

function tokenize(src: string): Tok[] | string {
	const out: Tok[] = [];
	const s = src.replace(/−/g, '-').replace(/×|·/g, '*').replace(/,/g, '.');
	let i = 0;
	while (i < s.length) {
		const c = s[i];
		if (/\s/.test(c)) { i++; continue; }
		if (/[0-9.]/.test(c)) {
			const m = /^\d*\.?\d+(?:[eE][+-]?\d+)?|^\d+\.?/.exec(s.slice(i));
			if (!m) return `Số không hợp lệ gần "${s.slice(i, i + 4)}".`;
			// "2e" không phải ký hiệu khoa học nếu sau e không có số: để e là hằng
			let t = m[0];
			if (/[eE]/.test(t) && !/[eE][+-]?\d+$/.test(t)) t = t.replace(/[eE].*$/, '');
			out.push({ k: 'num', s: t });
			i += t.length;
			continue;
		}
		if (/[a-zA-Z]/.test(c)) {
			const m = /^[a-zA-Z]+/.exec(s.slice(i))!;
			out.push({ k: 'id', s: m[0] });
			i += m[0].length;
			continue;
		}
		if ('+-*/^'.includes(c)) { out.push({ k: 'op', s: c }); i++; continue; }
		if (c === '(') { out.push({ k: 'lp', s: c }); i++; continue; }
		if (c === ')') { out.push({ k: 'rp', s: c }); i++; continue; }
		return `Ký tự lạ "${c}".`;
	}
	return out;
}

export type ParseResult = { ok: true; ast: Node } | { ok: false; error: string };

export function parseExpr(src: string, vars: string[]): ParseResult {
	if (src.trim() === '') return { ok: false, error: 'Ô đang trống.' };
	const tokened = tokenize(src);
	if (typeof tokened === 'string') return { ok: false, error: tokened };
	const toks: Tok[] = tokened;
	let p = 0;
	const peek = () => toks[p];
	const fail = (m: string): never => { throw new Error(m); };

	const startsFactor = (t?: Tok) => !!t && (t.k === 'num' || t.k === 'id' || t.k === 'lp');

	function expr(): Node {
		let a = term();
		while (peek()?.k === 'op' && (peek()!.s === '+' || peek()!.s === '-')) {
			const op = toks[p++].s as '+' | '-';
			a = { t: 'bin', op, a, b: term() };
		}
		return a;
	}
	function term(): Node {
		let a = unary();
		for (;;) {
			const t = peek();
			if (t?.k === 'op' && (t.s === '*' || t.s === '/')) {
				p++;
				a = { t: 'bin', op: t.s as '*' | '/', a, b: unary() };
			} else if (startsFactor(t)) {
				a = { t: 'bin', op: '*', a, b: unary() }; // nhân ngầm
			} else return a;
		}
	}
	function unary(): Node {
		const t = peek();
		if (t?.k === 'op' && (t.s === '-' || t.s === '+')) {
			p++;
			const a = unary();
			return t.s === '-' ? { t: 'neg', a } : a;
		}
		return power();
	}
	function power(): Node {
		const a = atom();
		if (peek()?.k === 'op' && peek()!.s === '^') {
			p++;
			return { t: 'bin', op: '^', a, b: unary() };
		}
		return a;
	}
	function atom(): Node {
		const t = toks[p];
		if (!t) return fail('Biểu thức bị thiếu vế sau.');
		p++;
		if (t.k === 'num') return { t: 'num', v: Number(t.s) };
		if (t.k === 'lp') {
			const a = expr();
			if (toks[p]?.k !== 'rp') fail('Thiếu dấu ngoặc đóng ")".');
			p++;
			return a;
		}
		if (t.k === 'id') {
			const name = t.s.toLowerCase() === t.s ? t.s : t.s; // phân biệt hoa thường: biến có thể là hoa
			if (FUNCS.has(name)) {
				if (toks[p]?.k !== 'lp') fail(`Hàm ${name} cần ngoặc: ${name}(...).`);
				p++;
				const a = expr();
				if (toks[p]?.k !== 'rp') fail('Thiếu dấu ngoặc đóng ")".');
				p++;
				return { t: 'fn', n: name, a };
			}
			if (vars.includes(name)) return { t: 'var', n: name };
			if (name in CONSTS) return { t: 'num', v: CONSTS[name] };
			// chuỗi chữ dính liền như "xy" -> x*y nếu cả hai ký tự đều là biến
			if (name.length > 1 && [...name].every((ch) => vars.includes(ch))) {
				return [...name].map((ch) => ({ t: 'var', n: ch }) as Node).reduce((l, r) => ({ t: 'bin', op: '*', a: l, b: r }));
			}
			return fail(`Không hiểu "${name}". Chỉ dùng biến ${vars.join(', ') || '(không có)'}, hằng e, pi và các hàm sin, cos, tan, exp, ln, sqrt, abs.`);
		}
		return fail(`Dấu "${t.s}" đứng sai chỗ.`);
	}

	try {
		const ast = expr();
		if (p < toks.length) return { ok: false, error: `Thừa "${toks[p].s}" ở cuối biểu thức.` };
		return { ok: true, ast };
	} catch (e) {
		return { ok: false, error: (e as Error).message };
	}
}

export function evalNode(n: Node, env: Record<string, number>): number {
	switch (n.t) {
		case 'num': return n.v;
		case 'var': return env[n.n];
		case 'neg': return -evalNode(n.a, env);
		case 'fn': {
			const x = evalNode(n.a, env);
			switch (n.n) {
				case 'sin': return Math.sin(x);
				case 'cos': return Math.cos(x);
				case 'tan': return Math.tan(x);
				case 'exp': return Math.exp(x);
				case 'ln': return Math.log(x);
				case 'sqrt': return Math.sqrt(x);
				default: return Math.abs(x);
			}
		}
		case 'bin': {
			const a = evalNode(n.a, env);
			const b = evalNode(n.b, env);
			switch (n.op) {
				case '+': return a + b;
				case '-': return a - b;
				case '*': return a * b;
				case '/': return a / b;
				default: return Math.pow(a, b);
			}
		}
	}
}

const PREC: Record<string, number> = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 4 };

/** AST -> LaTeX để xem trước ngay khi gõ. */
export function toLatex(n: Node, parent = 0): string {
	const wrap = (s: string, need: boolean) => (need ? `\\left(${s}\\right)` : s);
	switch (n.t) {
		case 'num': return Number.isInteger(n.v) ? String(n.v) : String(n.v).replace('.', '{,}');
		case 'var': return n.n;
		case 'neg': return wrap(`-${toLatex(n.a, 3)}`, parent > 1);
		case 'fn': {
			const a = toLatex(n.a, 0);
			if (n.n === 'sqrt') return `\\sqrt{${a}}`;
			if (n.n === 'exp') return `e^{${a}}`;
			if (n.n === 'abs') return `\\left|${a}\\right|`;
			return `\\${n.n === 'ln' ? 'ln' : n.n}\\left(${a}\\right)`;
		}
		case 'bin': {
			if (n.op === '/') return `\\frac{${toLatex(n.a, 0)}}{${toLatex(n.b, 0)}}`;
			if (n.op === '^') return `${toLatex(n.a, 5)}^{${toLatex(n.b, 0)}}`;
			const p = PREC[n.op];
			const left = toLatex(n.a, p);
			const right = toLatex(n.b, p + (n.op === '-' ? 1 : 0));
			const sym = n.op === '*' ? '\\cdot ' : ` ${n.op} `;
			return wrap(`${left}${sym}${right}`, p < parent);
		}
	}
}

const POOL = [0.7, 1.3, 2.1, 0.45, 1.9, 2.6, 3.3, 0.85];

/** Các điểm mẫu cố định, đều dương (để ln, sqrt luôn xác định). */
export function samplePoints(vars: string[], count = 8): Record<string, number>[] {
	return Array.from({ length: count }, (_, j) =>
		Object.fromEntries(vars.map((v, i) => [v, POOL[(j * (i + 2) + i * 3) % POOL.length]])),
	);
}

export function sameFunction(a: Node, b: Node, vars: string[]): boolean {
	let valid = 0;
	for (const pt of samplePoints(vars)) {
		const x = evalNode(a, pt);
		const y = evalNode(b, pt);
		if (!Number.isFinite(y)) continue; // điểm mẫu ngoài miền của đáp án chuẩn: bỏ qua
		if (!Number.isFinite(x)) return false;
		if (Math.abs(x - y) > 1e-9 * Math.max(1, Math.abs(y))) return false;
		valid++;
	}
	return valid >= 3;
}
