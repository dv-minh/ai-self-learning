import { useEffect, useState } from 'react';

/** Tô màu code Python bằng Shiki (tải lười, chỉ khi bài có đoạn code). Hai giao diện sáng/tối chuyển bằng CSS. */
let highlighter: Promise<{ codeToHtml: (code: string, opts: object) => string }> | null = null;

function getHighlighter() {
	highlighter ??= (async () => {
		const [{ createHighlighterCore }, { createJavaScriptRegexEngine }, py, light, dark] = await Promise.all([
			import('shiki/core'),
			import('shiki/engine/javascript'),
			import('shiki/langs/python.mjs'),
			import('shiki/themes/github-light.mjs'),
			import('shiki/themes/github-dark.mjs'),
		]);
		return createHighlighterCore({
			themes: [light.default, dark.default],
			langs: [py.default],
			engine: createJavaScriptRegexEngine(),
		}) as unknown as { codeToHtml: (code: string, opts: object) => string };
	})();
	return highlighter;
}

export default function Code({ code }: { code: string }) {
	const [html, setHtml] = useState<string | null>(null);

	useEffect(() => {
		let alive = true;
		getHighlighter()
			.then((h) => h.codeToHtml(code, { lang: 'python', themes: { light: 'github-light', dark: 'github-dark' }, defaultColor: false }))
			.then((out) => alive && setHtml(out))
			.catch(() => alive && setHtml(null)); // lỗi tải: hiện code thường
		return () => {
			alive = false;
		};
	}, [code]);

	if (html) return <div className="code-block" dangerouslySetInnerHTML={{ __html: html }} />;
	return (
		<pre className="code-block plain">
			<code>{code}</code>
		</pre>
	);
}
