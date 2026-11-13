/**
 * Macro KaTeX dùng chung (docs/spec.md mục 7): cấu hình build (astro.config.mjs)
 * và trình duyệt (component Tex) dùng cùng một bản. Generator xuất LaTeX dùng đúng các macro này.
 *
 * - \T là chữ T thẳng nhỏ (mathsf) thay cho ký hiệu \top vốn to và nặng khi làm chỉ số trên.
 * - \frac và \dfrac giữ nguyên hành vi chuẩn (đã thử "thanh chống" vphantom nhưng làm vỡ ô bảng).
 *   Độ thoáng của phân số trong bảng/ma trận đến từ \arraystretch và dfracInEnvs bên dưới.
 */
const frac = (style) =>
	`{\\genfrac{}{}{}{${style}}{#1}{#2}}`;

export const katexMacros = {
	'\\R': '\\mathbb{R}',
	'\\T': '^{\\mathsf{T}}',
	'\\norm': '\\left\\lVert #1 \\right\\rVert',
	'\\rank': '\\operatorname{rank}',
	'\\tr': '\\operatorname{tr}',
	'\\diag': '\\operatorname{diag}',
	'\\arraystretch': '1.3', // giãn dòng vừa phải; dòng có phân số được ROW_STRUT nới thêm
	'\\frac': frac(''),
	'\\dfrac': frac('0'),
};

const ENV = /\\begin\{(array|[pbvBV]?matrix|cases|aligned|align\*?)\}[\s\S]*?\\end\{\1\}/g;

/**
 * Trong bảng/ma trận KaTeX dùng cỡ chữ "text", làm phân số bị co nhỏ và chen sát vạch kẻ.
 * Đổi \frac thành \dfrac bên trong các môi trường này để phân số có cỡ đầy đủ.
 */
// Thanh chống đầu mỗi dòng: dòng có phân số cao ~1,3em phía trên và ~0,7em phía dưới trục; thêm chút đệm để không chạm dòng kẻ hay dòng bên cạnh.
const ROW_STRUT = '\\rule[-1.3em]{0pt}{2.05em}';

export function dfracInEnvs(tex) {
	return tex.replace(ENV, (m) => {
		if (!/\\frac(?![a-zA-Z])/.test(m)) return m;
		let out = m.replace(/\\frac(?![a-zA-Z])/g, '\\dfrac');
		out = out.replace(/^(\\begin\{(?:array\}\{[^}]*\}|[pbvBV]?matrix\}))/, `$1${ROW_STRUT} `);
		out = out.replace(/(\\\\(?:\[[^\]]*\])?\s*(?:\\hline\s*)?)(?!\\end)/g, `$1${ROW_STRUT} `);
		return out;
	});
}

/** Plugin remark: áp dụng dfracInEnvs cho mọi công thức trong MDX trước khi rehype-katex render. */
export function remarkDfrac() {
	const walk = (node) => {
		if ((node.type === 'math' || node.type === 'inlineMath') && typeof node.value === 'string') {
			node.value = dfracInEnvs(node.value);
			// remark-math lưu bản sao nội dung trong data.hChildren (math: pre > code > text); cập nhật mọi nút text
			const fix = (n) => {
				if (n.type === 'text' && typeof n.value === 'string') n.value = dfracInEnvs(n.value);
				if (n.children) n.children.forEach(fix);
			};
			const h = node.data && node.data.hChildren;
			if (Array.isArray(h)) h.forEach(fix);
		}
		if (node.children) node.children.forEach(walk);
	};
	return (tree) => walk(tree);
}
