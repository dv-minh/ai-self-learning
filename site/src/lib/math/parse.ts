import { Fraction } from './fraction';

export type ParseResult = { ok: true; value: Fraction } | { ok: false; error: string };

/**
 * Chuyển chuỗi người học nhập thành Fraction.
 * Chấp nhận: số nguyên "-3", phân số "-3/4", thập phân hữu hạn "0.25" hoặc "0,25".
 * Báo lỗi định dạng rõ ràng thay vì chấm sai.
 */
export function parseNumber(raw: string): ParseResult {
	const s = raw.trim().replace(/\s+/g, '').replace(',', '.').replace('−', '-');
	if (s === '') return { ok: false, error: 'Ô đang trống.' };

	const frac = /^([+-]?\d+(?:\.\d+)?)\/([+-]?\d+(?:\.\d+)?)$/.exec(s);
	if (frac) {
		const num = decimalToFraction(frac[1]);
		const den = decimalToFraction(frac[2]);
		if (den.isZero()) return { ok: false, error: 'Mẫu số không được bằng 0.' };
		return { ok: true, value: num.div(den) };
	}
	if (/^[+-]?\d+(?:\.\d+)?$/.test(s)) return { ok: true, value: decimalToFraction(s) };
	return { ok: false, error: `Không hiểu "${raw.trim()}". Hãy nhập số nguyên (-3), phân số (-3/4) hoặc số thập phân (0.25).` };
}

function decimalToFraction(s: string): Fraction {
	const neg = s.startsWith('-');
	const body = s.replace(/^[+-]/, '');
	const [i, f = ''] = body.split('.');
	const n = BigInt(i + f);
	const d = 10n ** BigInt(f.length);
	return new Fraction(neg ? -n : n, d);
}

/** Đọc "1 2 3; 4 5 6" thành lưới chuỗi (dùng khi dán vào MatrixInput). */
export function splitMatrixText(text: string): string[][] {
	return text
		.trim()
		.split(/\s*[;\n]\s*/)
		.filter((r) => r.length > 0)
		.map((r) => r.split(/\s+/).filter((c) => c.length > 0)); // không tách theo dấu phẩy: "0,25" là số thập phân
}
