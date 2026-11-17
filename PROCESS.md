# PROCESS.md — Trạng thái dự án và việc cần làm tiếp

Cập nhật lần cuối: 17/11/2026, sau khi thêm hình minh họa và biến đổi từng bước cho Phần 5–8 (mục 3.2c). Đọc file này trước, rồi `CLAUDE.md` và `docs/spec.md` (mục 10 là nhật ký quyết định).

## 1. Tóm tắt một dòng

Website tĩnh tự học AI (Astro + Starlight, generator Python). **GĐ0–GĐ6 đã xong** (Phần 1–8); Phần 5–8 đã có hình tương tác và "Biến đổi từng bước" (mục 3.2c). Còn các việc dọn dẹp và soát chất lượng ở mục 3. Repo đã có commit, remote `origin` trên GitHub; cấu hình deploy Cloudflare Workers (static assets) ở `wrangler.jsonc`.

## 2. Đã làm

| Giai đoạn | Nội dung | Trạng thái |
| --- | --- | --- |
| GĐ0 | Khung Astro + Starlight, KaTeX + macro, sidebar 8 phần, dark mode, tìm kiếm, CI GitHub Pages | Xong |
| GĐ1 | Bài 1.5 nhân ma trận chạy trọn từ generator đến web (ExercisePlayer, MatrixInput, chấm bằng `Fraction`) | Xong |
| GĐ2 | Bài 1.1–1.9, trang `/luyen-tap`, chấm `solution-set` | Xong |
| GĐ3 | Bài 1.10–1.15, chấm `eigvec`/`span`/`numeric`, hình lưới 2D, `/tien-do`, dấu tích sidebar, Tổng ôn Phần 1 | Xong |
| GĐ4 | Phần 2 (NumPy), Phần 3 (Giải tích), Phần 4 (Xác suất–thống kê), ô nhập biểu thức, code Shiki, Tổng ôn mỗi phần | Xong |
| GĐ6 | Phần 7 (Deep Learning, 7 bài, 15 dạng), Phần 8 (Computer Vision, 9 bài, 14 dạng); sửa lỗi bố cục toán lần 2 (căn giữa nội dung, phân số trong bảng/ma trận); khử trùng đề tính theo tập lựa chọn | Xong |
| GĐ5 | Phần 5 (Tối ưu, 7 bài, 21 dạng), Phần 6 (ML cổ điển, 9 bài, 17 dạng), hình `GdPath`, sửa giao diện toán (KaTeX, chuyển vị, phân số, bố cục), `RichText` hiển thị `**đậm**` và `` `mã` `` | Xong |

Quy mô hiện tại: **165 dạng bài tập × 200 bài = 33.000 bài**, 8 phần, 77 bài học + 8 trang Tổng ôn.

| Phần | Mã phần (kho bài) | Bài học | Dạng |
| --- | --- | --- | --- |
| 1. Đại số tuyến tính | `linalg` | 1.1–1.15 | 59 |
| 2. NumPy cho ML | `numpy` | 2.1–2.9 | 15 |
| 3. Giải tích | `calculus` | 3.1–3.7 | 11 |
| 4. Xác suất – thống kê | `probability` | 4.1–4.6 | 13 |
| 5. Tối ưu | `optimization` | 5.1–5.7 | 21 |
| 6. ML cổ điển | `classical-ml` | 6.1–6.9 | 17 |
| 7. Deep Learning | `deep-learning` | 7.1–7.7 | 15 |
| 8. Computer Vision | `computer-vision` | 8.1–8.9 | 14 |

Kiểm tra gần nhất (đều qua, 13/11/2026): `pytest` 339, Vitest 276, `pnpm typecheck`, `pnpm validate:banks` (385.902 công thức, 0 lỗi KaTeX), `pnpm build` (90 trang). Phần 1–4: mỗi dạng đã làm thử trên giao diện thật (3 bài/dạng, nhập sai rồi đúng). **Phần 5–6: chỉ mới làm thử trên giao diện thật** `gd-steps` (vector, nhập sai rồi đúng, phân số và thập phân), `gini-split` (trắc nghiệm có công thức) và `knn-classify`; các dạng còn lại được bảo đảm bằng kiểm chứng độc lập trong generator, `pytest` và Vitest chấm lại toàn bộ kho bằng bộ chấm của site, nhưng **chưa ai làm thử từng dạng trên giao diện** (nên làm, đặc biệt `softmax-grad` ma trận và `convex-hessian`).

## 3. Còn phải làm

### 3.1 GĐ5 — Đã xong (Phần 5 và Phần 6)

Việc đã làm (chi tiết quyết định ở `docs/spec.md` mục 10, "Bổ sung ở GĐ5"): 21 dạng Phần 5 (`generator/gen/topics/opt/`), 17 dạng Phần 6 (`generator/gen/topics/cml/`), ví dụ `examples/part5.py` và `part6.py`, 16 bài học MDX, Tổng ôn hai phần, hình `GdPath` (FR10 đủ ba hình).

Việc còn dở của GĐ5: (1) làm thử từng dạng mới trên giao diện thật như quy định (mới thử vài dạng); (2) Review toán cho 16 bài mới; (3) Phần 6 chưa có hình tương tác riêng (không bắt buộc).

### 3.2 GĐ6 — Đã xong (Phần 7 và Phần 8)

Phần 7: `generator/gen/topics/dl/` (15 dạng), ví dụ `examples/part7.py`, 7 bài MDX. Phần 8: `generator/gen/topics/cv/` (14 dạng), ví dụ `examples/part8.py`, 9 bài MDX. Mã phần trong kho bài: `deep-learning`, `computer-vision`. Softmax/attention dùng điểm dạng k ln 2 để xác suất là số hữu tỉ; khuếch tán dùng ᾱ là bộ ba Pythagoras. Chi tiết quyết định: `docs/spec.md` mục 10, "Bổ sung ở GĐ6".

**Review toán (13/11/2026):** đã chạy cho Phần 5 và Phần 6 (hai phiên agent riêng, chỉ báo cáo; giải lại toàn bộ 4.200 và 3.400 bài, 0 lệch đáp án). **Chưa chạy cho 24 bài mới của Phần 7–8.** Các lỗi nội dung đã sửa ngay: ví dụ support vector 6.9 (không phải nghiệm SVM), ví dụ hinge gõ tay, lời giải MAE, đề kfold-cv mức 4, hint adam-bias, phát biểu Nesterov/PyTorch, điều kiện hội tụ GD, điều kiện g≠0 của Adam, nguồn RMSProp. **Chưa sửa (từ báo cáo review, mức "thiếu"/"văn phong")**: thiếu câu đọc bằng lời dưới nhiều công thức display; "Lỗi thường gặp" chỉ lỗi 1 có ví dụ sai→sửa; dùng `\|...\|` thay `\norm`; thiếu thuật ngữ Anh ở một số chỗ; một số ví dụ nên thêm (R² hiệu chỉnh, F_β, macro-F1, quét ngưỡng Gini, ví dụ K-means thứ hai với phản ví dụ khởi tạo, KNN có trọng số khác khoảng cách); quy ước bước t từ 0 hay 1 và d, W của lịch Transformer chưa định nghĩa; `adamw-step` chưa có công thức trong display của đề; mục "Tham khảo" thiếu URL bài cụ thể cho machinelearningcoban.com; Q xoay của `gd-max-lr` có mẫu số tới 841 (nặng khi tính tay); đề `gd-converge` lệch nhãn (phân kỳ 61%).

### 3.2b Chỉnh giao diện đọc (13/11/2026)

`site/src/lib/remark-lesson-layout.mjs` (đăng ký ở `astro.config.mjs`): (1) gộp các display `nhãn = [ma trận/vector]` liên tiếp vào một hàng khi đủ rộng (ngưỡng `MAX_ROW_WIDTH`); (2) bọc các mục Định nghĩa / Tính chất / Lỗi thường gặp / Liên hệ với ML / Tóm tắt thành khung `.lesson-box-*` (CSS ở `custom.css`, có dark mode). Cỡ chữ thân bài 18px, KaTeX 1,2em; `rraystretch` giảm 1,7 → 1,3; mẫu số phân số được đẩy xuống bằng CSS (`.katex .mfrac .vlist`). Chưa xem trên điện thoại thật cho mọi bài; mới xem 1.1 và 4.4.

### 3.2c Soạn lại Phần 5–8: hình minh họa tương tác và biến đổi từng bước (17/11/2026)

Người dùng phản hồi bài học "chỉ toàn công thức, khó hiểu". Đã thêm vào cả 32 bài của Phần 5–8:

- **Hình tương tác** (React, SVG, `site/src/components/viz/`, dùng chung `common.tsx`): 37 hình mới, đặt trong mục "Trực giác" kèm đoạn dẫn "hãy thử…". Phần 5: `ConvexViz`, `QuadFormViz`, `Gd1DViz` (từng bước GD có tiếp tuyến), `SgdViz` (mây gradient mini-batch), `EmaViz` (momentum / bias correction), `OptimizerRace`, `LrScheduleViz`, `RegGeometryViz` (L1 thoi vs L2 tròn). Phần 6: `LinRegViz` (ô vuông phần dư), `LogisticViz`, `PolyFitViz` (chữ U train/test), `ThresholdViz` (ngưỡng → ma trận nhầm lẫn → ROC), `KnnViz`, `KMeansViz` (từng nửa bước), `TreeSplitViz`, `BoostingViz`, `SvmViz`. Phần 7: `NeuronViz` (XOR, lớp ẩn), `ReluSumViz`, `ActivationViz`, `SoftmaxViz`, `BackpropViz` (đồ thị tính toán forward/backward từng bước), `InitViz`, `NormViz`, `DropoutViz`, `ResidualViz`, `AttentionViz`, `LoraViz`. Phần 8: `ConvViz`, `ReceptiveFieldViz`, `PatchViz`, `AugmentViz`, `IouViz`, `NmsViz`, `SegViz`, `ClipViz`, `DiffusionViz`.
- **Mục mới "## Biến đổi từng bước"** (sau Định nghĩa, trước Ví dụ): các chứng minh/suy luận viết thành `<Derivation>`/`<Step>` (`viz/Derivation.astro`, `viz/Step.astro`), hiện từng bước khi bấm "Bước tiếp"; không có JS thì hiện hết. Chỉ chứa biến đổi ký hiệu, không có số gõ tay.
- Hình dùng `client:only="react"`: số thực tính ở Node và trình duyệt lệch chữ số cuối gây cảnh báo hydration; hình không có ý nghĩa khi tắt JS. Hình chỉ để xem, không dùng để chấm (FR10). `GdPath` cũ vẫn giữ.
- Kiểm tra: `pnpm typecheck`, Vitest 276, `pnpm build` (không lỗi KaTeX), pytest 339 đều qua; đã mở dev server xem và bấm thử các hình chính (GD từng bước, cuộc đua optimizer, SVM, K-means, backprop, attention, tích chập, ViT), xem điện thoại 375px + dark mode bài 5.7.
- **Chưa làm:** chưa bấm thử từng hình một trên giao diện (mới thử khoảng một nửa); chưa có hình cho Phần 1–4; "Review toán" cho các mục "Biến đổi từng bước" mới chưa chạy bằng phiên agent riêng.

### 3.3 Việc dọn dẹp và rủi ro đã biết

- **"Review toán"** (prompt ở `docs/spec.md` mục 9, chạy bằng phiên agent mới; spec yêu cầu trước khi merge): đã chạy cho Phần 5–6 (xem 3.2); đã chạy cho Phần 7–8 (giải lại 3.000 và 2.800 bài, 0 lệch đáp án; đã sửa: công thức `W^\T` lỗi KaTeX, câu giải thích ReLU gõ tay sai, đề BatchNorm với γ theo đặc trưng, đề mlp-forward mức 4 (đầu ra giờ tuyến tính), resid-grad (w=−1, w≥1), LeakyReLU α<1, số gõ tay ở 8.7, hint dilation, ví dụ Hungarian); **chưa chạy cho Phần 1–4 (37 bài)**. Còn lại từ báo cáo 7–8 (mức thiếu/văn phong): `torch-shape` chỉ 185 đề khác nhau, nhiều đề có đáp án 0 (backprop-net, mlp-forward), hint `mlp-params` không theo bias/BN, thiếu điều kiện (He cho lớp đầu, running_var chia n−1, định lý xấp xỉ vạn năng, quy ước đếm tham số Transformer, quy ước U-Net, khoảng GIoU), `clip-retrieval` đề nói "khớp đúng" nhưng embedding ngẫu nhiên, dùng `\norm`.
- **Commit và deploy:** đã có commit đầu (Phần 1–8) và remote `origin`; `wrangler.jsonc` phục vụ `site/dist` bằng Cloudflare Workers (static assets, trang 404 riêng). Workflow GitHub Pages ở `.github/workflows/ci.yml` vẫn còn; chưa xác nhận CI chạy thật trên GitHub. Thay đổi soạn lại Phần 5–8 (mục 3.2c) chưa commit tại thời điểm cập nhật này.
- **Phần "Tham khảo"** của các bài học trỏ tới nguồn theo mục chung, chưa ghi số bài giảng/trang cụ thể; cần soát khi Review toán.
- **Chưa xem tận mắt** giao diện điện thoại (FR11) và dark mode của các thành phần mới (GridTransform, GdPath, ô biểu thức, code Shiki, bảng và ma trận có phân số sau khi sửa macro). Phần 5–8: mới làm thử trên giao diện `gd-steps`, `gini-split`, `knn-classify`, `torch-shape`; các dạng còn lại chỉ được bảo đảm bằng kiểm chứng generator, pytest và Vitest. Khung trình duyệt bị ẩn khi thử nên `client:visible` không kích hoạt được; đã kiểm bằng trang thử tạm hoặc `client:load`. Hình minh họa nay dùng `client:only="react"` nên chạy được cả khi khung ẩn.
- **`np-dtype` ít đa dạng** (khoảng 20 biểu thức khác nhau, đề khác nhau chủ yếu ở thứ tự lựa chọn). Nên mở rộng.
- **Kho bài nặng:** khoảng 42 MB (do ghi cả trường `null`); tải lười theo từng dạng nên chấp nhận được, có thể rút gọn nếu cần.
- **Bài học 1.x trong spec nói "Gram–Schmidt là mục đọc thêm"**: mới chỉ nhắc, chưa có bài đọc thêm riêng.
- **Spec và đặc tả gốc** (`AI Self-Learning Website — Bộ tài liệu đặc tả.md` ở thư mục gốc) giữ nguyên; bản làm việc là `docs/spec.md`.

### 3.4 Việc nên làm tiếp (theo thứ tự gợi ý)

1. Commit phần soạn lại Phần 5–8; xác nhận CI và chọn một đích deploy (Cloudflare Workers hoặc GitHub Pages).
2. Review toán Phần 1–4 (37 bài) và các mục "Biến đổi từng bước" mới của Phần 5–8 bằng phiên agent mới (prompt mục 9 đã thêm bước 5).
2b. Bấm thử từng hình trong `components/viz/` (mới thử khoảng một nửa); thêm hình và "Biến đổi từng bước" cho Phần 1–4 theo cùng quy chuẩn.
3. Làm thử trên giao diện từng dạng Phần 5–8, kể cả nhập sai; xem điện thoại và dark mode.
4. Sửa các mục "thiếu/văn phong" còn lại trong báo cáo review Phần 5–8 (mục 3.2 và 3.3).
5. Mở rộng các dạng ít đề khác nhau (`np-dtype`, `torch-shape`) và giảm đề có đáp án 0.

## 4. Cách chạy

```bash
# Sinh toàn bộ kho bài + manifest + ví dụ bài học (≈ 1–2 phút)
cd generator && uv run python -m gen build --all --count 200
cd generator && uv run python -m gen build --part calculus --type deriv-poly --count 200   # một dạng
cd generator && uv run python -m gen schema          # xuất JSON Schema, rồi: pnpm --dir site gen:types

# Kiểm tra
cd generator && uv run pytest
pnpm --dir site validate:banks      # render mọi LaTeX trong kho bằng KaTeX (strict)
pnpm --dir site typecheck
pnpm --dir site test                # Vitest (gồm kiểm chéo kho ↔ bộ chấm)
pnpm --dir site build

# Chạy thử
pnpm --dir site dev                 # http://localhost:4321 (đã có .claude/launch.json tên "site")
```

Môi trường đã dùng: Node 26, pnpm 12 (cài toàn cục bằng npm), uv, Python 3.12+ (uv dùng 3.13), Astro 7, Starlight 0.42, React 19, Vitest 5.

## 5. Bản đồ mã nguồn

- `generator/gen/schema.py` — nguồn chân lý định dạng bài tập (Pydantic). `site/src/types/exercise.d.ts` là file sinh tự động.
- `generator/gen/cli.py` — `PART_MODULES` (mã phần → module + tiêu đề), sinh kho, manifest, ví dụ.
- `generator/gen/core/` — `latex.py`, `nice.py`, `steps.py`, `gauss.py`, `verify.py`.
- `generator/gen/topics/<phần>/` — mỗi dạng bài = một hàm `gen_*` + `TopicSpec(mã, bài, các mức, hàm)`. Phần 1: `linalg/`; Phần 2: `np_part/`; Phần 3: `calc/`; Phần 4: `prob/`; Phần 5: `opt/`; Phần 6: `cml/` (mỗi nơi có `_o.py`/`_m.py`/`_c.py`/`_p.py` chứa tiện ích dùng chung). Mỗi module `__init__.py` có `SPECS` và `TITLES`.
- `generator/gen/examples/` — ví dụ số cho bài học, ghi vào `site/src/generated/lesson-<bài>/*.mdx` (không sửa tay, file này được sinh lại mỗi lần build).
- `site/public/exercises/<phần>/<bài>/<dạng>.json` và `manifest.json` — kho bài (sinh tự động, đã nằm trong repo để site tải lười).
- `site/src/lib/math/` — `fraction.ts`, `parse.ts`, `compare.ts` (chấm: exact, eigvec, span, numeric, solution-set, expression), `expr.ts` (phân tích biểu thức), `linalg.ts`.
- `site/src/components/exercise/` — `ExerciseCard`, `ExercisePlayer`, `PracticeMode`, `ReviewPlayer`, `MatrixInput`, `Code` (Shiki), `Tex`. Khác: `ProgressView`, `MarkRead`, `viz/GridTransform`. Hình minh họa Phần 5–8 và khung biến đổi từng bước: `viz/` (xem mục 3.2c).
- `site/src/content/docs/<NN-tên-phần>/` — bài học MDX; `99-tong-on.mdx` mỗi phần; `luyen-tap.mdx`, `tien-do.mdx`.
- `site/scripts/validate-banks.mjs` — kiểm LaTeX toàn kho.

## 6. Bài học kinh nghiệm (tránh lặp lại)

1. **Không dùng Bash heredoc (`<<'EOF'`, `python - <<`) để ghi code/LaTeX có dấu `\`**: công cụ làm mất một nửa dấu gạch chéo ngược (`\times` có thể thành Tab). Luôn dùng công cụ Write/Edit. Đã có test (`test_source_has_no_invalid_escapes`, kiểm ký tự điều khiển) để bắt lỗi này.
2. **Không đặt chữ tiếng Việt trong công thức** (cả `\text{}` lẫn ô `array`): KaTeX không có glyph. Dùng nhãn ASCII trong bảng và đặt chữ trong `text` của đề/bước.
3. Mỗi dạng cần ≥ 200 đề **khác nhau** (khử trùng bằng hash của đề); nếu thiếu thì mở rộng tham số hoặc ngữ cảnh, đừng nới điều kiện.
4. `--part` là bắt buộc khi dùng `--type` cho phần khác Phần 1 (mặc định chỉ Phần 1), hoặc dùng `--all`.
5. Kiểm chứng phải **độc lập** với cách dựng đề (SymPy ↔ NumPy ↔ liệt kê ↔ sai phân hữu hạn); test chéo `bank.test.ts` chấm lại mọi bài trong kho.
6. **`rehype-katex` và `katex` phải cùng phiên bản** (đang ghim `katex@0.16.47`): lệch phiên bản làm CSS và HTML không khớp tên class (dấu "khác" hiện thành "/="). Sau khi nâng cấp phải xem lại công thức trên trang thật.
7. Lựa chọn trắc nghiệm có công thức phải bọc `$...$` (`RichText` chỉ render công thức trong cặp `$`); `RichText` hiểu `**đậm**` và `` `mã` ``.
8. Dạng chỉ có vài tham số rời rạc (ví dụ đếm bootstrap) không đủ 200 đề khác nhau: thêm một tham số hoặc nhiều đại lượng hỏi (đã làm ở `forest-average` mức 5).
9. Với công thức cần căn bậc hai, logarit hoặc mũ (Adam, softmax, cosine): chọn số sao cho kết quả hữu tỉ (tìm bằng quét nguyên), hoặc đề cho sẵn giá trị đã tính; đối chiếu bằng NumPy/`math` ở bước kiểm chứng.
10. **Phân số trong bảng/ma trận:** KaTeX dùng cỡ chữ "text" trong ô nên `\frac` bị co nhỏ và chạm dòng kẻ; `site/src/lib/katex-macros.mjs` (`dfracInEnvs`, plugin remark `remarkDfrac`, cùng hàm dùng ở `Tex.tsx`) đổi `\frac` thành `\dfrac` và thêm thanh chống đầu mỗi dòng trong `array`/`matrix`. Thanh chống đặt **trong** phân số (macro `\frac`) làm vỡ ô bảng nên không dùng. Sửa file cấu hình/macro phải **khởi động lại dev server**. Bố cục: Starlight mặc định dồn nội dung sát mục lục phải; `custom.css` đặt `--sl-content-margin-inline: auto` để căn giữa.
11. **Hình tương tác dùng `client:only="react"`**: SSR rồi hydrate làm React báo lệch thuộc tính vì số thực tính ở Node và Chrome khác nhau ở chữ số cuối (`cx`, `fillOpacity`...). Dữ liệu ngẫu nhiên của hình dùng `rng(seed)` trong `viz/common.tsx`, không dùng `Math.random`.
12. **Biến đổi từng bước** viết trong MDX bằng `<Derivation>`/`<Step>` (Astro, có script hiện dần); công thức trong đó vẫn phải không có chữ tiếng Việt — đưa lời giải thích ra ngoài `$$`. Chèn khối lớn vào MDX bằng Write/Edit hoặc script Python đọc file (không heredoc).

## 7. Cách làm tiếp (gợi ý quy trình mỗi giai đoạn)

1. Đọc mục 3 của file này và `docs/spec.md` mục 3, 7, 8, 10.
2. Chốt danh sách bài + mã dạng của phần mới (ghi vào `docs/spec.md` mục 10).
3. Viết generator + test, sinh 200 bài/dạng, chạy `validate:banks`, `pytest`, `pnpm test`.
4. Viết ví dụ (`examples/`) và bài học MDX theo khung mục 7 (gồm hình tương tác ở Trực giác và mục "Biến đổi từng bước"); thêm trang Tổng ôn.
5. `pnpm build`, mở dev server và làm thử ≥ 3 bài/dạng kể cả nhập sai; bấm thử từng hình mới, xem console không lỗi, xem ở 375px và dark mode.
6. Cập nhật file này (mục 2 và 3), `CLAUDE.md` nếu đổi lệnh/quy ước, rồi commit.
