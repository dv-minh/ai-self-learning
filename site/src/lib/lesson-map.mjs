// Ánh xạ mã bài ("1.5") -> đường dẫn trang, đọc từ frontmatter `lesson:` của các bài .mdx.
// Dùng ở astro.config.mjs để chèn script đánh dấu bài đã đọc (dấu tích) vào sidebar.
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = fileURLToPath(new URL('../content/docs', import.meta.url));

export function lessonMap() {
	const map = {};
	for (const dir of readdirSync(docsRoot, { withFileTypes: true })) {
		if (!dir.isDirectory()) continue;
		for (const file of readdirSync(join(docsRoot, dir.name))) {
			if (!/\.mdx?$/.test(file)) continue;
			const head = readFileSync(join(docsRoot, dir.name, file), 'utf8').split(/\r?\n---/)[0];
			const m = /^lesson:\s*"?([\d.]+)"?\s*$/m.exec(head);
			if (m) map[m[1]] = `${dir.name}/${file.replace(/\.mdx?$/, '')}/`;
		}
	}
	return map;
}

export function sidebarTickScript(map) {
	return `(function(){try{var map=${JSON.stringify(map)};
function mark(){var p;try{p=JSON.parse(localStorage.getItem('ai-learn-progress')||'{}')}catch(e){p={}}
(p.lessonsRead||[]).forEach(function(l){var s=map[l];if(!s)return;
document.querySelectorAll('.sidebar-content a[href$="'+s+'"]').forEach(function(a){a.classList.add('lesson-read')})})}
document.addEventListener('DOMContentLoaded',mark);window.addEventListener('lesson-read',mark)}catch(e){}})();`;
}
