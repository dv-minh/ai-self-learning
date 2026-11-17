# site — AI Self-Learning (Astro + Starlight)

Phần giao diện của website tự học AI. Tổng quan dự án, trạng thái và việc cần làm tiếp: xem [`../PROCESS.md`](../PROCESS.md); đặc tả: [`../docs/spec.md`](../docs/spec.md).

## Lệnh

```bash
pnpm dev              # http://localhost:4321
pnpm build            # build tĩnh vào dist/ (KaTeX strict: lỗi công thức làm build thất bại)
pnpm test             # Vitest: Fraction, parse, chấm điểm, biểu thức, tiến độ, kiểm chéo kho bài
pnpm typecheck        # tsc --noEmit
pnpm validate:banks   # render mọi LaTeX trong public/exercises bằng KaTeX
pnpm gen:types        # sinh src/types/exercise.d.ts từ JSON Schema (generator xuất schema)
```

Deploy: site đang chạy trên Cloudflare Workers (https://ai-self-learning.gallery-album.workers.dev). Ở thư mục gốc repo: `pnpm --dir site build` rồi `npx wrangler deploy` (cấu hình ở `../wrangler.jsonc`; lần đầu `npx wrangler login`). Khi build cho Cloudflare không đặt `SITE_URL`/`BASE_PATH`; hai biến này chỉ dùng cho GitHub Pages (CI tự đặt theo tên repo).

## Cấu trúc

- `src/content/docs/<NN-tên-phần>/*.mdx` — bài học (frontmatter có `lesson: "x.y"`); `99-tong-on.mdx` là trang Tổng ôn; `luyen-tap.mdx`, `tien-do.mdx`.
- `src/components/exercise/` — ExercisePlayer, ExerciseCard, PracticeMode, ReviewPlayer, MatrixInput, Code (Shiki), Tex.
- `src/components/` — ProgressView, MarkRead.
- `src/components/viz/` — hình minh họa tương tác (chỉ để xem, không chấm bài): `GridTransform` (lưới 2D), `GdPath`, và 37 hình của Phần 5–8 (`Gd1DViz`, `OptimizerRace`, `BackpropViz`, `AttentionViz`, `ConvViz`, ...). Tiện ích chung ở `common.tsx` (Figure, Slider, Choice, useStepper, Axes, Contours, useSvgDrag, `rng`, màu `--viz-*`). Nhúng bằng `client:only="react"`.
- `src/components/viz/Derivation.astro`, `Step.astro` — khung "Biến đổi từng bước", hiện dần từng bước khi bấm; dùng trong mục cùng tên của bài học.
- `src/lib/math/` — `fraction.ts`, `parse.ts`, `compare.ts` (các chế độ chấm), `expr.ts` (biểu thức), `linalg.ts`.
- `src/lib/` — `progress.ts` (localStorage), `exercises.ts` (tải lười, chọn bài), `katex-macros.mjs`, `lesson-map.mjs`.
- `public/exercises/` — kho bài và `manifest.json`, **do generator sinh, không sửa tay**.
- `src/generated/` — partial MDX ví dụ bài học, **do generator sinh, không sửa tay**.
- `src/types/exercise.d.ts` — **sinh tự động**.
