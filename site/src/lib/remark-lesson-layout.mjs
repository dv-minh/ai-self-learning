/**
 * Plugin remark cho bố cục bài học (chạy sau remarkMath và remarkDfrac):
 *  1. Gộp các công thức display liên tiếp chỉ để định nghĩa vector/ma trận ("u = [...]", "A = [...]")
 *     vào MỘT hàng khi đủ rộng, thay vì mỗi cái một dòng.
 *  2. Bọc các mục chuẩn của bài học (Định nghĩa, Tính chất, Lỗi thường gặp, Liên hệ với ML, Tóm tắt)
 *     thành khung nổi bật, để người học nhìn ra ngay đâu là định nghĩa, kết luận, cảnh báo.
 */

const MATRIX_ENV = '(?:[pbvBV]?matrix)';
// "nhãn = \begin{bmatrix} ... \end{bmatrix}" và không có gì sau \end
const DEF_RE = new RegExp(
	String.raw`^\s*([^=\n]{1,40}?)\s*=\s*(\\begin\{${MATRIX_ENV}\}(?:(?!\\begin)[\s\S])*?\\end\{${MATRIX_ENV}\})\s*$`,
);
const MAX_ROW_WIDTH = 46; // đơn vị ~ số ký tự; vượt quá thì xuống hàng

function plainLength(s) {
	return s.replace(/\\rule\[[^\]]*\]\{[^}]*\}\{[^}]*\}/g, '').replace(/\\[a-zA-Z]+/g, 'x').replace(/[{}\s]/g, '').length;
}

function definitionWidth(match) {
	const [, label, matrix] = match;
	const body = matrix.replace(/^\\begin\{[^}]*\}/, '').replace(/\\end\{[^}]*\}$/, '');
	const rows = body.split(/\\\\(?:\[[^\]]*\])?/);
	let cols = 1;
	let cell = 1;
	for (const row of rows) {
		const cells = row.split('&');
		cols = Math.max(cols, cells.length);
		for (const c of cells) cell = Math.max(cell, plainLength(c));
	}
	return plainLength(label) + 2 + cols * (cell + 1.5);
}

function setMathValue(node, value) {
	node.value = value;
	const fix = (n) => {
		if (n.type === 'text' && typeof n.value === 'string') n.value = value;
		if (n.children) n.children.forEach(fix);
	};
	const h = node.data && node.data.hChildren;
	if (Array.isArray(h)) h.forEach(fix);
}

function mergeDefinitions(parent) {
	const out = [];
	let group = []; // [{node, width, value}]
	const flush = () => {
		if (group.length > 1) {
			const first = group[0].node;
			setMathValue(first, group.map((g) => g.value).join(String.raw` \qquad\quad `));
			out.push(first);
		} else if (group.length === 1) out.push(group[0].node);
		group = [];
	};
	for (const child of parent.children) {
		const m = child.type === 'math' ? DEF_RE.exec(child.value) : null;
		if (!m) {
			flush();
			out.push(child);
			continue;
		}
		const width = definitionWidth(m);
		const total = group.reduce((s, g) => s + g.width, 0) + width;
		if (group.length && total > MAX_ROW_WIDTH) flush();
		group.push({ node: child, width, value: child.value.trim() });
	}
	flush();
	parent.children = out;
}

const SECTIONS = {
	'Định nghĩa': { kind: 'def', split: true },
	'Tính chất': { kind: 'prop', split: false },
	'Lỗi thường gặp': { kind: 'warn', split: true },
	'Liên hệ với ML': { kind: 'ml', split: false },
	'Tóm tắt': { kind: 'sum', split: false },
};

const wrap = (kind, children) => ({
	type: 'callout',
	data: { hName: 'div', hProperties: { className: ['lesson-box', `lesson-box-${kind}`] } },
	children,
});

const isBoldLead = (n) => n.type === 'paragraph' && n.children && n.children[0] && n.children[0].type === 'strong';
const headingText = (n) => (n.children || []).map((c) => c.value || '').join('').trim();

function wrapSections(root) {
	const out = [];
	const kids = root.children;
	let i = 0;
	while (i < kids.length) {
		const node = kids[i];
		const spec = node.type === 'heading' && node.depth === 2 ? SECTIONS[headingText(node)] : undefined;
		if (!spec) {
			out.push(node);
			i++;
			continue;
		}
		out.push(node);
		i++;
		const body = [];
		while (i < kids.length && !(kids[i].type === 'heading' && kids[i].depth <= 2)) body.push(kids[i++]);
		// Giữ nguyên thứ tự; ExercisePlayer/MarkRead là khối MDX nên không bao giờ nằm trong mục này.
		if (!spec.split) {
			if (body.length) out.push(wrap(spec.kind, body));
			continue;
		}
		let seg = [];
		const flush = () => {
			if (seg.length) out.push(wrap(spec.kind, seg));
			seg = [];
		};
		for (const b of body) {
			if (b.type === 'heading') {
				flush();
				out.push(b);
				continue;
			}
			if (isBoldLead(b) && seg.length) flush();
			seg.push(b);
		}
		flush();
	}
	root.children = out;
}

export function remarkLessonLayout() {
	const walk = (node) => {
		if (!node.children) return;
		if (node.children.some((c) => c.type === 'math')) mergeDefinitions(node);
		node.children.forEach(walk);
	};
	return (tree) => {
		walk(tree);
		wrapSections(tree);
	};
}
