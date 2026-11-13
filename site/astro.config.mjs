// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { katexMacros, remarkDfrac } from './src/lib/katex-macros.mjs';
import { remarkLessonLayout } from './src/lib/remark-lesson-layout.mjs';
import { lessonMap, sidebarTickScript } from './src/lib/lesson-map.mjs';


const parts = [
	['Phần 1. Đại số tuyến tính', '01-dai-so-tuyen-tinh'],
	['Phần 2. NumPy cho ML', '02-numpy-cho-ml'],
	['Phần 3. Giải tích', '03-giai-tich'],
	['Phần 4. Xác suất – thống kê', '04-xac-suat-thong-ke'],
	['Phần 5. Tối ưu', '05-toi-uu'],
	['Phần 6. ML cổ điển', '06-ml-co-dien'],
	['Phần 7. Deep Learning', '07-deep-learning'],
	['Phần 8. Computer Vision', '08-computer-vision'],
];

// https://astro.build/config
export default defineConfig({
	// CI (GitHub Pages) đặt SITE_URL và BASE_PATH theo tên repo.
	site: process.env.SITE_URL,
	base: process.env.BASE_PATH ?? '/',
	markdown: {
		remarkPlugins: [remarkMath, remarkDfrac, remarkLessonLayout],
		rehypePlugins: [[rehypeKatex, { macros: katexMacros, strict: 'error', throwOnError: true }]],
	},
	integrations: [
		starlight({
			title: 'AI Self-Learning',
			defaultLocale: 'root',
			locales: { root: { label: 'Tiếng Việt', lang: 'vi' } },
			customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
			head: [{ tag: 'script', content: sidebarTickScript(lessonMap()) }],
			sidebar: [
				{ label: 'Giới thiệu', items: [{ label: 'Trang chủ', slug: '' }] },
				...parts.map(([label, directory]) => ({ label, items: [{ autogenerate: { directory } }] })),
				{
					label: 'Công cụ',
					items: [
						{ label: 'Luyện tập', slug: 'luyen-tap' },
						{ label: 'Tiến độ', slug: 'tien-do' },
						{ label: 'Kiểm tra KaTeX', slug: 'kiem-tra-katex' },
					],
				},
			],
		}),
		react(),
	],
});
