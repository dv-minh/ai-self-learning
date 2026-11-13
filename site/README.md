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

Biến môi trường khi deploy: `SITE_URL` và `BASE_PATH` (CI GitHub Pages tự đặt theo tên repo).

## Cấu trúc

- `src/content/docs/<NN-tên-phần>/*.mdx` — bài học (frontmatter có `lesson: "x.y"`); `99-tong-on.mdx` là trang Tổng ôn; `luyen-tap.mdx`, `tien-do.mdx`.
- `src/components/exercise/` — ExercisePlayer, ExerciseCard, PracticeMode, ReviewPlayer, MatrixInput, Code (Shiki), Tex.
- `src/components/` — ProgressView, MarkRead; `viz/GridTransform.tsx` (lưới 2D tương tác).
- `src/lib/math/` — `fraction.ts`, `parse.ts`, `compare.ts` (các chế độ chấm), `expr.ts` (biểu thức), `linalg.ts`.
- `src/lib/` — `progress.ts` (localStorage), `exercises.ts` (tải lười, chọn bài), `katex-macros.mjs`, `lesson-map.mjs`.
- `public/exercises/` — kho bài và `manifest.json`, **do generator sinh, không sửa tay**.
- `src/generated/` — partial MDX ví dụ bài học, **do generator sinh, không sửa tay**.
- `src/types/exercise.d.ts` — **sinh tự động**.
