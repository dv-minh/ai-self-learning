import katex from 'katex';
import { useMemo } from 'react';
import { katexMacros, dfracInEnvs } from '../../lib/katex-macros.mjs';

function render(tex: string, displayMode: boolean): string {
	// macros được KaTeX ghi thêm vào đối tượng truyền vào, nên luôn truyền bản sao
	return katex.renderToString(dfracInEnvs(tex), { displayMode, throwOnError: false, macros: { ...katexMacros } });
}

/** Công thức display. */
export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
	const html = useMemo(() => render(tex, display), [tex, display]);
	return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Văn bản có chèn công thức $...$, chữ **đậm** (có thể chứa $...$) và `mã` inline. */
export function RichText({ text }: { text: string }) {
	const parts = useMemo(() => text.split(/(\*\*(?:[^*$]|\$[^$]+\$)+\*\*|\$[^$]+\$|`[^`]+`)/g), [text]);
	return (
		<>
			{parts.map((p, i) => {
				if (p.length > 4 && p.startsWith('**') && p.endsWith('**')) return <strong key={i}><RichText text={p.slice(2, -2)} /></strong>;
				if (p.length > 2 && p.startsWith('$') && p.endsWith('$')) return <Tex key={i} tex={p.slice(1, -1)} />;
				if (p.length > 2 && p.startsWith('`') && p.endsWith('`')) return <code key={i}>{p.slice(1, -1)}</code>;
				return <span key={i}>{p}</span>;
			})}
		</>
	);
}
