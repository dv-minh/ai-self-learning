// Kiểm tra mọi chuỗi LaTeX trong kho bài tập bằng KaTeX (strict): "không lỗi render" (FR2).
import katex from 'katex';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { katexMacros } from '../src/lib/katex-macros.mjs';

const root = fileURLToPath(new URL('../public/exercises', import.meta.url));
const files = [];
(function walk(dir) {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) walk(p);
		else if (name.endsWith('.json') && name !== 'manifest.json') files.push(p);
	}
})(root);

const render = (tex, displayMode) =>
	katex.renderToString(tex, { displayMode, throwOnError: true, strict: 'error', macros: { ...katexMacros } });

let checked = 0;
const errors = [];
const fail = (id, where, tex, e) => errors.push(`${id} [${where}] ${e.message.split('\n')[0]}\n    ${tex.slice(0, 140)}`);

function inline(id, where, text) {
	for (const part of text.split(/(\$[^$]+\$)/g)) {
		if (part.length > 1 && part.startsWith('$') && part.endsWith('$')) {
			checked++;
			try { render(part.slice(1, -1), false); } catch (e) { fail(id, where, part, e); }
		} else if (part.includes('$')) {
			fail(id, where, part, new Error('dấu $ lẻ, công thức chưa đóng'));
		}
	}
}
function display(id, where, tex) {
	checked++;
	try { render(tex, true); } catch (e) { fail(id, where, tex, e); }
}

for (const f of files) {
	const bank = JSON.parse(readFileSync(f, 'utf8'));
	for (const ex of bank.exercises) {
		inline(ex.id, 'prompt.text', ex.prompt.text);
		ex.prompt.display.forEach((d) => display(ex.id, 'prompt.display', d));
		(ex.choices ?? []).forEach((c) => inline(ex.id, 'choices', c));
		ex.hints.forEach((h) => inline(ex.id, 'hints', h));
		ex.steps.forEach((s) => {
			inline(ex.id, 'steps.text', s.text);
			if (s.latex) display(ex.id, 'steps.latex', s.latex);
		});
	}
}

if (errors.length) {
	console.error(`${errors.length} lỗi LaTeX trong kho bài:`);
	console.error(errors.slice(0, 20).join('\n'));
	process.exit(1);
}
console.log(`Kho bài: ${files.length} file, ${checked} công thức, 0 lỗi KaTeX.`);
