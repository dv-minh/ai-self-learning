# AI Self-Learning Website — Bộ tài liệu đặc tả

Nov 12, 2026 · @Minh

> **Trạng thái triển khai** (cập nhật 13/11/2026): GĐ0–GĐ6 đã xong (8 phần, 165 dạng bài tập, 77 bài học); còn Review toán Phần 1–4, dọn dẹp và soát chất lượng. Chi tiết, việc cần làm tiếp và bản đồ mã nguồn: xem `PROCESS.md`. Các quyết định phát sinh khi làm: mục 10 cuối tài liệu này.

## 0. Cách dùng bộ tài liệu này

Tài liệu này là đặc tả để giao cho AI coding agent (Claude Code, Cursor…): agent đọc nó rồi code, bạn kiểm tra theo Definition of Done của từng giai đoạn. Mọi quyết định đã chốt nằm ở đây, nên agent không cần (và không được) tự chọn lại stack.

1. Tạo repo trống `ai-learn`, export tài liệu này ra Markdown và lưu thành `docs/spec.md`.
2. Copy khối **CLAUDE.md** ở mục 9 vào gốc repo. Agent tự đọc file này ở mọi phiên.
3. Chạy lần lượt các prompt giai đoạn ở mục 9, mỗi lần một giai đoạn. Xong giai đoạn nào, tự kiểm tra checklist ở mục 8 rồi mới commit và sang giai đoạn sau.
4. Khi agent viết bài học, luôn nhắc nó tuân theo mục 7 (Quy chuẩn nội dung). Khi agent thêm dạng bài tập, nhắc mục 5 (Đặc tả bộ sinh bài tập).
5. Sau mỗi bài học, chạy prompt **Review toán** ở mục 9 bằng một phiên agent mới để soát lỗi công thức.

Thứ tự đọc cho người: mục 1 → 3 → 4 để nắm nội dung; mục 2, 5, 6 là phần kỹ thuật dành cho agent.

## 1. PRD — Yêu cầu sản phẩm

Sản phẩm là một website tĩnh tự học AI, đi từ toán nền (ma trận) tới Deep Learning và Computer Vision. Ưu tiên số 1 là đúng toán; ưu tiên số 2 là nhiều bài tập đa dạng, từ dễ tới khó.

**Người dùng:** một người học (chủ dự án), là kỹ sư phần mềm, đã biết Python, chưa từng học ma trận. Mục tiêu là hiểu lý thuyết; không học Python, không làm bài tập viết code. Không có tài khoản, không có người dùng khác.

**Ngôn ngữ:** tiếng Việt, thuật ngữ tiếng Anh trong ngoặc ở lần đầu xuất hiện. **Giao diện:** kiểu trang tài liệu, sidebar mục lục, gọn gàng, có dark mode.

### Yêu cầu chức năng

| ID | Yêu cầu | Ưu tiên |
| --- | --- | --- |
| FR1 | Bài học viết bằng MDX, sidebar theo Phần → Bài, tìm kiếm toàn văn | Bắt buộc |
| FR2 | Công thức render bằng KaTeX, cả inline và display, không lỗi render | Bắt buộc |
| FR3 | Mỗi khái niệm có ít nhất 1 ví dụ số giải chi tiết từng bước | Bắt buộc |
| FR4 | Cuối mỗi bài có bộ bài tập 5 mức (mục 4), lấy ngẫu nhiên từ kho | Bắt buộc |
| FR5 | Dạng câu trả lời: trắc nghiệm một đáp án, nhiều đáp án, đúng/sai, nhập số, nhập vector, nhập ma trận, nhập kích thước, nhập biểu thức; không có bài viết code | Bắt buộc |
| FR6 | Chấm tự động, chấp nhận đáp án tương đương (2/4 = 1/2 = 0.5) | Bắt buộc |
| FR7 | Gợi ý theo bậc; lời giải từng bước hiện sau khi nộp hoặc bấm “Xem lời giải” | Bắt buộc |
| FR8 | Chế độ luyện tập: chọn dạng, kích thước, mức rồi làm liên tục | Bắt buộc |
| FR9 | Lưu tiến độ trong localStorage; xuất/nhập file JSON để sao lưu | Nên có |
| FR10 | Hình minh họa tương tác chỉ để xem (biến đổi lưới 2D, đồ thị hàm, đường gradient descent) | Nên có |
| FR11 | Đọc tốt trên điện thoại (responsive) | Nên có |

### Yêu cầu phi chức năng

- **Tĩnh hoàn toàn:** build ra HTML/JS/JSON, host miễn phí, không server, không database, không auth.
- **Đúng toán:** không có con số nào trong bài học hay bài tập được gõ tay; tất cả do generator tính bằng số hữu tỉ chính xác và được kiểm chứng độc lập (mục 5).
- **Tái lập được:** generator chạy với cùng seed cho ra cùng kho bài tập.
- **Nhẹ:** không tải runtime Python trên trình duyệt; kho bài tập tải theo từng bài, không tải toàn bộ.

### Ngoài phạm vi

Tài khoản và đồng bộ nhiều thiết bị, bình luận, bảng xếp hạng, video, dạy Python, chạy code trên trình duyệt (Pyodide), notebook Colab, bài tập viết code.

## 2. Kiến trúc kỹ thuật

Hệ thống có hai nửa: generator Python sinh và kiểm chứng kho bài tập JSON, còn site Astro gộp bài học MDX với kho đó thành website tĩnh. Ranh giới giữa hai nửa là một JSON Schema duy nhất.

&#91;embedded content: luồng build · generator → kiểm chứng → site tĩnh\]

Bài học và generator là hai nguồn đầu vào; chỉ kho JSON đã qua kiểm chứng mới được Astro gộp vào site.

### Stack

| Thành phần | Công nghệ | Vai trò |
| --- | --- | --- |
| Khung site | Astro + Starlight | Sidebar, tìm kiếm (Pagefind), dark mode, trang tài liệu |
| Bài học | MDX | Markdown nhúng được component bài tập |
| Công thức | KaTeX qua remark-math + rehype-katex | Render LaTeX lúc build, có macro dùng chung |
| Bài tập tương tác | React islands + TypeScript | ExercisePlayer, MatrixInput, PracticeMode |
| Chấm đáp án | Lớp `Fraction` dùng BigInt (tự viết, có test) | So sánh phân số chính xác ở trình duyệt |
| Generator | Python 3.11+, SymPy, NumPy, Pydantic | Sinh đề, đáp án, lời giải từng bước, kiểm chứng; chạy NumPy lúc build để tính đáp án bài Phần 2 |
| Test | pytest + hypothesis (Python), Vitest (TS) | Test tính chất toán và logic chấm |
| Hiển thị code NumPy | Shiki (có sẵn trong Starlight) | Tô màu đoạn code trong bài học và đề bài Phần 2 |
| Quản lý gói | pnpm (site), uv (generator) |  |
| Deploy | Cloudflare Pages hoặc GitHub Pages, CI bằng GitHub Actions | Build + test mỗi lần push |

### Cấu trúc repo

```text
ai-learn/
├─ CLAUDE.md
├─ docs/spec.md                     # tài liệu này
├─ generator/
│  ├─ pyproject.toml
│  ├─ gen/
│  │  ├─ schema.py                  # Pydantic models → xuất JSON Schema
│  │  ├─ core/
│  │  │  ├─ nice.py                 # dựng ma trận "đẹp" (mục 5)
│  │  │  ├─ steps.py                # ghi lời giải từng bước
│  │  │  ├─ latex.py                # SymPy → LaTeX theo quy chuẩn ký hiệu
│  │  │  └─ verify.py               # kiểm chứng độc lập
│  │  ├─ topics/linalg/             # mỗi dạng bài = 1 module
│  │  ├─ examples/                  # sinh ví dụ số cho bài học
│  │  └─ cli.py                     # python -m gen build ...
│  └─ tests/
├─ site/
│  ├─ astro.config.mjs              # KaTeX macros, sidebar
│  ├─ src/content/docs/             # bài học .mdx theo phần
│  │  └─ 01-dai-so-tuyen-tinh/05-nhan-ma-tran.mdx
│  ├─ src/components/exercise/      # ExercisePlayer.tsx, MatrixInput.tsx...
│  ├─ src/lib/math/                 # fraction.ts, parse.ts, compare.ts
│  ├─ src/types/exercise.d.ts       # sinh từ JSON Schema
│  └─ public/exercises/             # JSON do generator xuất ra
└─ .github/workflows/ci.yml
```

### Quy tắc kiến trúc

- `generator/gen/schema.py` là nguồn chân lý của định dạng bài tập. Kiểu TypeScript được sinh tự động từ JSON Schema (json-schema-to-typescript), không viết tay.
- Kho JSON nằm ở `site/public/exercises/<phần>/<bài>/<dạng>.json`, site tải lười (lazy fetch) khi người học mở bài.
- Ví dụ số trong bài học cũng do generator sinh (`gen/examples/`), xuất ra file MDX partial hoặc JSON để bài học import.
- CI chạy theo thứ tự: pytest → sinh kho bài → kiểm tra schema → Vitest → build Astro. Bước nào lỗi thì không deploy.

## 3. Syllabus toàn lộ trình

Lộ trình gồm 8 phần, học lý thuyết là chính; mọi bài tập là điền đáp án hoặc trắc nghiệm, không có phần chạy code. Đại số tuyến tính học trước, NumPy học ngay sau để ánh xạ từng khái niệm toán sang đúng dòng code. Bài tính tay giới hạn ở ma trận tối đa 4×4.

| Phần | Nội dung chính | Dạng bài tập chủ đạo |
| --- | --- | --- |
| 1. Đại số tuyến tính | Vector, tích vô hướng và cosine similarity, ma trận, nhân ma trận, khử Gauss, định thức, nghịch đảo, hạng, trị riêng, ma trận đối xứng và xác định dương, SVD, xấp xỉ hạng thấp, PCA (chi tiết ở mục 4) | Tính tay và điền vector/ma trận 2×2 đến 4×4, đúng/sai tính chất, trắc nghiệm |
| 2. NumPy cho ML (đọc hiểu code) | ndarray ↔ vector/ma trận, shape, chỉ số bắt đầu từ 0, `@` khác `*`, `.T`, `axis`, broadcasting, `np.linalg` (vì sao dùng `solve` thay vì `inv`), sai số dấu phẩy động, vector hóa công thức ML (chi tiết ở mục 4) | Dự đoán shape và giá trị output, chọn dòng code đúng cho một công thức, tìm lỗi trong đoạn code |
| 3. Giải tích | Đạo hàm, quy tắc chuỗi (chain rule), đạo hàm riêng, gradient, Jacobian, Hessian, đạo hàm theo vector/ma trận, đồ thị tính toán (computational graph) và vi phân tự động (autodiff) | Tính đạo hàm, gradient; điền giá trị forward/backward trên đồ thị tính toán |
| 4. Xác suất – thống kê | Xác suất có điều kiện, Bayes (ví dụ Naive Bayes), biến ngẫu nhiên, phân phối Bernoulli, categorical, chuẩn, chuẩn nhiều chiều; kỳ vọng, phương sai, hiệp phương sai; MLE, MAP; entropy, cross-entropy, KL divergence | Tính xác suất bằng phân số chính xác, trắc nghiệm khái niệm |
| 5. Tối ưu | Hàm lồi, gradient descent, SGD, mini-batch, momentum, Adam, AdamW, lịch learning rate (warmup, cosine), regularization L2 và weight decay | Chạy tay 2–3 bước cập nhật, điền giá trị tham số sau mỗi bước |
| 6. ML cổ điển | Linear regression, logistic và softmax regression, overfitting, bias–variance, cross-validation, độ đo (precision, recall, F1, ROC-AUC), KNN, K-means, decision tree, random forest, gradient boosting (XGBoost, LightGBM, CatBoost), PCA, SVM rút gọn (margin, hinge loss) | Tính tay trên tập dữ liệu 4–6 điểm, tính độ đo từ ma trận nhầm lẫn |
| 7. Deep Learning | Perceptron, MLP, ReLU/GELU, softmax + cross-entropy, backpropagation, khởi tạo trọng số, BatchNorm, LayerNorm, dropout, residual connection, embedding, attention và Transformer, fine-tuning và LoRA, đọc hiểu code PyTorch | Forward/backward tay mạng 2 lớp, suy luận shape, tính attention trên ma trận ≤ 4×4, đếm số tham số |
| 8. Computer Vision | Ảnh là tensor; tích chập, padding, stride, pooling; CNN với ResNet làm trọng tâm; Vision Transformer và patch embedding; augmentation, transfer learning; detection anchor-free và end-to-end (IoU, NMS, set prediction kiểu DETR); segmentation (U-Net, SAM); self-supervised (contrastive, DINO); vision–language (CLIP); sinh ảnh bằng diffusion (khái niệm) | Tính tay tích chập trên ảnh 4×4, output shape, IoU, số patch và số tham số |

Mỗi phần có một trang “Tổng ôn” cuối phần, gồm bài tập trộn mọi dạng của phần đó. Phần 6 vẫn có thể đối chiếu với machinelearningcoban.com cho các thuật toán cổ điển.

### Đã lược bỏ hoặc thu gọn, và lý do

| Nội dung | Thay đổi | Lý do |
| --- | --- | --- |
| Phần học Python | Bỏ | Bạn đã biết Python |
| Chạy code trên web (Pyodide), notebook Colab | Bỏ | Học lý thuyết là chính; bài tập chỉ điền và trắc nghiệm |
| Ma trận 5×5, 6×6 | Bỏ; tối đa 4×4 | 4×4 đủ luyện mọi thuật toán mà vẫn tính tay được |
| Gram–Schmidt, phân rã QR tính tay | Thu gọn thành mục đọc thêm | ML gần như không cần tính tay phần này |
| Nhân tử Lagrange, SVM dạng đối ngẫu, kernel trick | Bỏ; SVM chỉ còn margin và hinge loss | Ít dùng trong thực tế; hinge loss nối sang classifier tuyến tính |
| Sigmoid/tanh cho lớp ẩn, AdaGrad, RMSProp | Còn một đoạn lịch sử | Thực tế dùng ReLU/GELU và Adam/AdamW |
| RNN, LSTM | Thu gọn thành một bài khái niệm trước Transformer | Transformer đã thay thế phần lớn; vẫn cần biết để hiểu vì sao có attention |
| LeNet, VGG; R-CNN hai giai đoạn, anchor box | Đoạn lịch sử trong bài | Detector hiện đại là anchor-free và end-to-end, không cần NMS |
| GAN | Khái niệm ngắn | Diffusion là hướng chính cho sinh ảnh |
| `np.matrix` | Không dạy | NumPy khuyến nghị dùng ndarray với toán tử `@` |

**Đã thêm vì đang là chuẩn hiện nay:** gradient boosting cho dữ liệu bảng, attention và Transformer, LayerNorm, AdamW, LoRA, Vision Transformer, self-supervised (DINO), CLIP, SAM, diffusion, cosine similarity, đồ thị tính toán.

Nguồn đối chiếu (đọc ngày 12/11/2026): [lịch học CS231n](https://cs231n.stanford.edu/schedule), [phân tích YOLO26 trên arXiv](https://arxiv.org/html/2601.12882v2), [tổng quan detector 2026 của JetBrains](https://blog.jetbrains.com/pycharm/2026/07/best-object-detection-models-for-machine-learning-in-2026/), [TabArena, NeurIPS 2025](https://arxiv.org/html/2506.16791v4), [báo cáo DINOv3](https://huggingface.co/papers/2508.10104).

## 4. Syllabus chi tiết Phần 1 (Đại số tuyến tính) và Phần 2 (NumPy)

Phần 1 có 15 bài và 59 dạng bài tập, mọi ma trận tối đa 4×4; Phần 2 có 9 bài và 15 dạng, dạy đọc hiểu code NumPy bằng cách ánh xạ ngược về từng bài của Phần 1. Mỗi dạng có kho tối thiểu 200 bài đã kiểm chứng.

### Năm mức bài tập (áp dụng cho mọi phần)

| Mức | Tên | Mô tả | Kích thước điển hình | Giá trị phần tử |
| --- | --- | --- | --- | --- |
| 1 | Nhận biết | Đọc kích thước, phần tử a₂₃, nhận dạng loại ma trận, khái niệm (trắc nghiệm, đúng/sai) | 1×1 đến 4×4 | Số nguyên −9..9 |
| 2 | Tính toán cơ bản | Áp dụng trực tiếp một quy tắc, điền đáp án | 2×2, 3×3, không vuông nhỏ | Số nguyên −5..5 |
| 3 | Tính toán đầy đủ | Cùng dạng mức 2 nhưng lớn hơn, nhiều số âm, có phân số | 4×4, 2×4, 4×3 | Nguyên −9..9, phân số đơn giản |
| 4 | Tổng hợp | Đúng/sai về tính chất, tìm lỗi trong lời giải mẫu, bài có tham số (tìm m để A khả nghịch), kết hợp 2–3 khái niệm | 3×3, 4×4 | Có tham số m |
| 5 | Ứng dụng ML | Suy luận shape trong mô hình, đọc hiểu một công thức ML, chọn hoặc điền đáp án; không viết code | ≤ 4×4 | Tùy bài |

Mỗi lần mở bài, ExercisePlayer phát 15 bài: mức 1–2 mỗi mức 3 bài, mức 3–4 mỗi mức 4 bài, mức 5 một bài (có thể đổi trong cấu hình). Chế độ luyện tập cho làm vô hạn trong kho.

### Phần 1 — Danh sách bài

| Bài | Tên | Khái niệm chính | Mã dạng bài tập | Kích thước |
| --- | --- | --- | --- | --- |
| 1.1 | Vector | Vector cột/hàng, cộng, nhân vô hướng, tổ hợp tuyến tính, chuẩn L1/L2, vector đơn vị | `vec-add`, `vec-scale`, `vec-lincomb`, `vec-norm` | ℝ² đến ℝ⁴ |
| 1.2 | Tích vô hướng, góc, cosine similarity | Dot product, cosin góc, cosine similarity (nền của embedding và attention), trực giao, hình chiếu | `vec-dot`, `vec-cosine`, `vec-orthogonal`, `vec-project` | ℝ² đến ℝ⁴ |
| 1.3 | Ma trận và ký hiệu | Kích thước m×n, phần tử aᵢⱼ, hàng/cột, ma trận vuông, chuyển vị | `mat-shape`, `mat-entry`, `mat-transpose` | 1×1 đến 4×4 |
| 1.4 | Cộng ma trận, nhân vô hướng | Điều kiện cùng kích thước, tính chất giao hoán/kết hợp | `mat-add`, `mat-scale`, `mat-lincomb` | 2×2 đến 4×4 |
| 1.5 | Nhân ma trận | Quy tắc hàng × cột, điều kiện (m×n)(n×p), không giao hoán, ma trận × vector, góc nhìn tổ hợp cột, phân biệt với tích từng phần tử (Hadamard) | `matmul-shape`, `matmul-entry`, `matmul-full`, `matvec`, `matmul-chain-shape`, `matmul-noncommute`, `hadamard-vs-matmul` | m, n, p từ 1 đến 4 |
| 1.6 | Ma trận đặc biệt | Đơn vị, đường chéo, tam giác, đối xứng, trực giao; (AB)ᵀ = BᵀAᵀ; lũy thừa ma trận đường chéo | `mat-identify`, `mat-power-diag`, `transpose-product` | 2×2 đến 4×4 |
| 1.7 | Hệ phương trình và khử Gauss | Ma trận mở rộng, biến đổi sơ cấp, bậc thang, bậc thang rút gọn (RREF), nghiệm duy nhất/vô số/vô nghiệm | `rowop-apply`, `ref`, `rref`, `solve-unique`, `solve-param`, `classify-system` | 2 đến 4 ẩn, 2 đến 4 phương trình |
| 1.8 | Định thức | 2×2, quy tắc Sarrus cho 3×3, khai triển Laplace, tính bằng biến đổi hàng, tính chất, ý nghĩa diện tích/thể tích | `det`, `det-laplace`, `det-rowops`, `det-properties`, `det-param` | 2×2 đến 4×4 (4×4 chỉ ở mức 3–4) |
| 1.9 | Ma trận nghịch đảo | Điều kiện khả nghịch, công thức 2×2, Gauss–Jordan, (AB)⁻¹, giải Ax = b bằng A⁻¹ | `inv-check`, `inv-2x2`, `inv-gauss-jordan`, `inv-solve` | 2×2 đến 4×4 (4×4 luôn có nghịch đảo nguyên) |
| 1.10 | Không gian vector và hạng | Độc lập tuyến tính, span, cơ sở, số chiều, hạng, không gian cột, không gian null | `lin-indep`, `rank`, `basis-col`, `null-space` | Đến 4×4, gồm 3×4 và 4×3 |
| 1.11 | Ánh xạ tuyến tính | Ma trận là phép biến đổi (quay, co giãn, chiếu, trượt), hợp ánh xạ = nhân ma trận | `transform-apply`, `transform-find`, `transform-compose` | ℝ², ℝ³ |
| 1.12 | Trị riêng, vector riêng | Phương trình đặc trưng, tìm trị riêng/vector riêng, chéo hóa A = PDP⁻¹, lũy thừa qua chéo hóa | `eig-values`, `eig-vectors`, `diagonalize`, `power-via-diag` | 2×2, 3×3; 4×4 chỉ dạng tam giác |
| 1.13 | Ma trận đối xứng và xác định dương | Trị riêng thực, vector riêng trực giao, dạng toàn phương, xác định dương (nền của hiệp phương sai và Hessian); Gram–Schmidt là mục đọc thêm | `sym-eig`, `quad-form`, `posdef-check` | 2×2, 3×3 |
| 1.14 | SVD, xấp xỉ hạng thấp, PCA | SVD, xấp xỉ hạng thấp (nền của LoRA), PCA qua ma trận hiệp phương sai | `svd-small`, `low-rank`, `pca-steps` | 2×2, 2×3, 3×2; PCA trên 4 điểm 2D |
| 1.15 | Ma trận trong ML | Ma trận dữ liệu X (n×d), dự đoán Xw + b, normal equation, shape trong mạng neural, shape của QKᵀ trong attention | `ml-shape`, `normal-eq`, `attention-shape` | ≤ 4×4 |

Bài 1.11 và 1.12 có hình minh họa tương tác 2D (lưới bị biến đổi bởi ma trận) để xem, không phải bài tập code.

### Phần 2 — NumPy cho ML

Mỗi bài NumPy mở đầu bằng bảng “Toán ↔ NumPy” trỏ về bài tương ứng của Phần 1. Người học không chạy code: generator chạy NumPy lúc build để tính sẵn đáp án, người học đọc code rồi điền shape, giá trị hoặc chọn đáp án.

| Bài | Tên | Ánh xạ từ Phần 1 | Mã dạng bài tập |
| --- | --- | --- | --- |
| 2.1 | ndarray, shape, dtype | Vector, ma trận, kích thước (1.1, 1.3) | `np-shape`, `np-dtype` |
| 2.2 | Chỉ số và cắt lát | Phần tử aᵢⱼ, hàng, cột; chỉ số bắt đầu từ 0 thay vì 1 (1.3) | `np-index`, `np-slice` |
| 2.3 | `@`, `*` và `.T` | Nhân ma trận, tích Hadamard, chuyển vị (1.5, 1.3) | `np-matmul-vs-mul`, `np-transpose` |
| 2.4 | Rút gọn theo trục | Tổng theo hàng/cột, trung bình; `axis`, `keepdims` | `np-axis` |
| 2.5 | Broadcasting | Cộng một vector vào mọi hàng của ma trận (1.4) | `np-broadcast-shape`, `np-broadcast-value` |
| 2.6 | reshape, stack, concatenate | Ghép vector thành ma trận, đổi góc nhìn dữ liệu | `np-reshape` |
| 2.7 | `np.linalg` | det, inv, solve, matrix\_rank, eig, eigh, svd (1.7–1.14); vì sao `solve` tốt hơn `inv` | `np-linalg-map`, `np-solve-vs-inv` |
| 2.8 | Số thực dấu phẩy động | Vì sao det ra 1e-17 thay vì 0; so sánh bằng `allclose` | `np-float` |
| 2.9 | Vector hóa công thức ML | Xw + b, MSE, softmax ổn định số, khoảng cách giữa các cặp điểm (1.15) | `np-vectorize-choose`, `np-bug-find` |

## 5. Đặc tả bộ sinh bài tập (generator)

Generator dựng mỗi bài tập ngược từ đáp án mong muốn, tính mọi thứ bằng số hữu tỉ chính xác của SymPy, rồi kiểm chứng lại bằng một phương pháp độc lập; bài nào không qua kiểm chứng thì build thất bại.

### Nguyên tắc

- **Không float trong lõi.** Dùng `sympy.Rational`, `sympy.Matrix`. NumPy chỉ dùng ở bước kiểm chứng.
- **Tất định.** Mỗi bài có `seed`; cùng seed cho ra cùng bài.
- **Không tầm thường.** Loại ma trận toàn 0, hàng 0 ngoài ý muốn, ma trận đơn vị, đáp án trùng đề. Khử trùng lặp bằng hash của đề. Ma trận lớn nhất là 4×4.
- **Mỗi dạng = một module** trong `gen/topics/<phần>/`, xuất hàm `generate(rng, level, size) -> Exercise` và khai báo các mức, kích thước hỗ trợ.

### Dựng ma trận “đẹp” để tính tay được

| Mục tiêu | Cách dựng |
| --- | --- |
| Nghịch đảo toàn số nguyên | Nhân ngẫu nhiên các ma trận sơ cấp nguyên (cộng k lần hàng j vào hàng i, đổi hàng) vào I; det = ±1 nên A⁻¹ nguyên |
| Định thức chọn trước | A = Π·L·U: L tam giác dưới nguyên đường chéo 1, U tam giác trên nguyên với đường chéo chọn trước, Π hoán vị |
| Hệ có nghiệm nguyên duy nhất | Chọn x nguyên, A khả nghịch, tính b = Ax |
| Hạng chọn trước r | A = B·C với B (m×r) hạng cột đủ, C (r×n) hạng hàng đủ |
| Hệ vô số nghiệm / vô nghiệm | A hạng r; b = A·x₀ (vô số nghiệm) hoặc b ngoài không gian cột, kiểm rank(\[A｜b\]) = r + 1 (vô nghiệm) |
| Trị riêng nguyên, khác nhau | A = P·D·P⁻¹ với P unimodular nguyên, D đường chéo nguyên |
| Ma trận trực giao hữu tỉ | Biến đổi Cayley từ S phản đối xứng nguyên nhỏ; lọc bài có mẫu số ≤ 25 |
| Đối xứng có trị riêng đẹp; SVD đẹp | A = Q·D·Qᵀ; A = U·Σ·Vᵀ với Q, U, V trực giao hữu tỉ từ Cayley |
| Gram–Schmidt | Bước trực giao hóa cho kết quả hữu tỉ; bước chuẩn hóa cho phép căn thức chính xác |

Các công thức nền của bảng trên:

```latex
A = \Pi L U \quad\Rightarrow\quad \det(A) = \operatorname{sgn}(\Pi)\prod_{i=1}^{n} u_{ii}
```

```latex
A = P D P^{-1},\quad P \in \mathbb{Z}^{n\times n},\ \det(P) = \pm 1 \quad\Rightarrow\quad A \in \mathbb{Z}^{n\times n}
```

```latex
S^{\top} = -S \quad\Rightarrow\quad Q = (I - S)(I + S)^{-1} \text{ thỏa } Q^{\top}Q = I
```

### Schema một bài tập

```json
{
  "id": "linalg.matmul-full.L3.000123",
  "seed": 123,
  "lesson": "1.5",
  "type": "matmul-full",
  "level": 3,
  "tags": ["4x3", "3x3"],
  "prompt": {
    "text": "Tính tích $AB$.",
    "display": ["A = \\begin{bmatrix} ... \\end{bmatrix}", "B = ..."]
  },
  "answer": { "kind": "matrix", "rows": 4, "cols": 3, "entries": [["2", "-1/3", "0"], ["..."]] },
  "check": { "mode": "exact" },
  "choices": null,
  "hints": ["Kích thước của AB là bao nhiêu?", "Phần tử c_{11} = hàng 1 của A nhân cột 1 của B."],
  "steps": [{ "text": "Tính c_{11}", "latex": "c_{11} = 1\\cdot 2 + (-3)\\cdot 0 + \\dots" }],
  "verified": ["sympy", "numpy", "shape"]
}
```

**Các loại đáp án (`answer.kind`):** `number`, `vector`, `matrix`, `shape`, `choice`, `multi`, `boolean`, `expression` (có căn thức hoặc tham số: lưu LaTeX, `srepr` của SymPy và vài giá trị số mẫu), `solution-set` (nghiệm riêng + cơ sở không gian null). Mọi số hữu tỉ lưu dạng chuỗi `"p/q"` đã rút gọn.

**Chế độ chấm (`check.mode`)** cho đáp án không duy nhất:

- `exact`: so từng phần tử bằng phân số.
- `eigvec`: chấp nhận mọi v ≠ 0 thỏa Av = λv (bội vô hướng nào cũng đúng).
- `span`: so cơ sở người học nhập với cơ sở chuẩn bằng hạng (cùng span, đúng số vector).
- `solution-set`: nghiệm riêng thỏa Ax = b và các vector hướng sinh đúng không gian null.
- `numeric`: chỉ dùng cho đáp án vô tỉ, sai số tương đối 1e-9.

### Lời giải từng bước và phương án nhiễu

Thuật toán ghi lại từng bước khi chạy: khử Gauss ghi phép biến đổi (ví dụ R₂ ← R₂ − 3R₁) kèm ma trận sau bước đó; định thức Laplace ghi từng số hạng; nhân ma trận ghi đầy đủ cách tính ít nhất 2 phần tử rồi đưa kết quả. Phương án nhiễu cho trắc nghiệm lấy từ lỗi thường gặp: nhân từng phần tử thay vì nhân ma trận, tính BA thay vì AB, quên đổi dấu khi đổi hàng, quên chuyển vị, sai dấu trong khai triển Laplace.

### Kiểm chứng (bắt buộc với mọi bài)

| Dạng | Kiểm chứng độc lập |
| --- | --- |
| Nhân, cộng, chuyển vị | So với NumPy (sai số 1e-9); kiểm kích thước |
| Định thức | Laplace so với khử Gauss so với `numpy.linalg.det` |
| Nghịch đảo | A·A⁻¹ = I và A⁻¹·A = I chính xác |
| Hệ phương trình | Thế nghiệm vào Ax = b; kiểm hạng của A và \[A｜b\] |
| RREF, hạng | So với `sympy.Matrix.rref`, `rank` và hạng tính bằng SVD của NumPy |
| Trị riêng, chéo hóa | P⁻¹AP = D chính xác; Av = λv cho từng cặp |
| Trực giao, SVD | QᵀQ = I; U·Σ·Vᵀ = A chính xác |

Ngoài ra, pytest + hypothesis chạy các tính chất toán trên dữ liệu ngẫu nhiên: det(AB) = det(A)·det(B), (AB)ᵀ = BᵀAᵀ, (AB)⁻¹ = B⁻¹A⁻¹, det(Aᵀ) = det(A). Chỉ cần một bài không qua kiểm chứng là lệnh build trả lỗi.

### CLI

```bash
uv run python -m gen build --part linalg --count 200   # sinh kho
uv run python -m gen build --type matmul-full --level 3 --count 5 --print   # xem nhanh
uv run python -m gen schema   # xuất JSON Schema cho site
uv run pytest
```

## 6. Đặc tả trình làm bài (ExercisePlayer) và chấm điểm

ExercisePlayer là một React island nhúng cuối mỗi bài học; nó tải kho JSON của bài, phát 15 bài theo 5 mức, chấm ngay trên trình duyệt và lưu kết quả vào localStorage.

### Cách nhúng trong MDX

```mdx
import ExercisePlayer from '@/components/exercise/ExercisePlayer';

<ExercisePlayer client:visible lesson="1.5"
  types={["matmul-shape", "matmul-entry", "matmul-full", "matvec"]}
  perLevel={{ 1: 3, 2: 3, 3: 4, 4: 4, 5: 1 }} />
```

### Luồng làm một bài

1. Hiện đề (KaTeX), nhãn mức và dạng.
2. Người học nhập đáp án bằng ô phù hợp với `answer.kind`.
3. Bấm **Kiểm tra** → báo đúng/sai; với ma trận và vector, tô đỏ đúng những ô sai.
4. Sai thì cho thử lại; **Gợi ý** mở lần lượt từng gợi ý trong `hints`.
5. **Xem lời giải** mở từng bước trong `steps` (bấm để hiện bước tiếp theo).
6. **Bài khác cùng dạng** lấy bài chưa làm cùng dạng, cùng mức.

### Ô nhập liệu

- **MatrixInput:** lưới ô theo kích thước đáp án; Tab/Enter/phím mũi tên để di chuyển; dán được văn bản dạng `1 2 3; 4 5 6`; mỗi ô nhận số nguyên, phân số `-3/4`, số thập phân hữu hạn `0.25`.
- **Dạng `shape`:** hai ô m × n, hoặc chọn “không xác định” khi phép nhân không hợp lệ.
- **Dạng `expression`:** ô nhập kèm xem trước KaTeX; chấm bằng so sánh số tại các giá trị mẫu.
- **Đề bài có code NumPy (Phần 2):** đoạn code tô màu bằng Shiki; người học điền shape, giá trị hoặc chọn đáp án. Không có ô soạn code, không chạy code.

### Chấm điểm ở trình duyệt

- `src/lib/math/fraction.ts`: phân số dùng BigInt, luôn rút gọn, mẫu dương; có test Vitest cho cộng, nhân, so sánh, parse.
- `parse.ts`: chuyển chuỗi người học nhập thành Fraction; báo lỗi định dạng rõ ràng thay vì chấm sai.
- `compare.ts`: cài các chế độ `exact`, `eigvec`, `span`, `solution-set`, `numeric` đúng như mục 5 (cần khử Gauss nhỏ bằng Fraction cho `span` và `solution-set`).

### Chế độ luyện tập

Trang `/luyen-tap` cho chọn phần → dạng → mức → kích thước, rồi phát bài liên tục từ kho, bỏ qua bài đã làm đúng. Có bộ đếm đúng/sai của phiên hiện tại.

### Tiến độ (localStorage)

```json
{
  "version": 1,
  "lessonsRead": ["1.1", "1.2"],
  "attempts": {
    "linalg.matmul-full.L3.000123": { "correct": true, "tries": 2, "usedSolution": false, "at": "2026-11-12" }
  }
}
```

Sidebar hiện dấu tích cho bài đã đọc; trang `/tien-do` thống kê tỉ lệ đúng theo dạng và theo mức, có nút xuất/nhập file JSON.

## 7. Quy chuẩn nội dung

Mọi bài học theo cùng một khung, cùng bộ ký hiệu và cùng bảng thuật ngữ; con số trong bài do generator sinh, công thức đối chiếu với nguồn tham khảo chuẩn.

### Khung một bài học

```markdown
---
title: Nhân ma trận
description: Quy tắc hàng × cột, điều kiện kích thước, ý nghĩa tổ hợp tuyến tính
lesson: "1.5"
prerequisites: ["1.3", "1.4"]
---

## Mục tiêu            (3–5 gạch đầu dòng: học xong làm được gì)
## Trực giác           (ý tưởng bằng lời + hình, chưa công thức)
## Định nghĩa          (công thức chính thức, giải thích từng ký hiệu)
## Ví dụ               (ít nhất 2: một nhỏ 2×2, một lớn hơn/không vuông; giải từng bước)
## Tính chất           (mỗi tính chất kèm 1 ví dụ số hoặc phản ví dụ)
## Lỗi thường gặp      (2–4 lỗi, mỗi lỗi có ví dụ sai → sửa)
## Liên hệ với ML      (bài toán ML nào dùng khái niệm này)
## Tóm tắt             (bảng hoặc 3–5 dòng)
## Bài tập             (<ExercisePlayer ... />)
## Tham khảo           (link tới nguồn đối chiếu)
```

### Ký hiệu toán (KaTeX)

| Đối tượng | Viết | Ví dụ LaTeX |
| --- | --- | --- |
| Số vô hướng | chữ thường nghiêng | `a, \lambda` |
| Vector (mặc định là vector cột) | chữ thường đậm | `\mathbf{x} \in \R^{n}` |
| Ma trận | chữ hoa đậm | `\mathbf{A} \in \R^{m \times n}` |
| Phần tử | chỉ số hàng trước, cột sau | `a_{ij}`, `x_i` |
| Chuyển vị, nghịch đảo |  | `\mathbf{A}\T`, `\mathbf{A}^{-1}` |
| Ma trận đơn vị |  | `\mathbf{I}_n` |
| Chuẩn, tích vô hướng |  | `\norm{\mathbf{x}}_2`, `\mathbf{x}\T\mathbf{y}` |
| Hạng, vết, định thức |  | `\rank(\mathbf{A})`, `\tr(\mathbf{A})`, `\det(\mathbf{A})` |
| Ma trận số | ngoặc vuông `bmatrix`; ma trận mở rộng dùng vạch đứng | `\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}` |

Macro KaTeX khai báo một lần trong `astro.config.mjs`: `\R` → `\mathbb{R}`, `\T` → `^{\mathsf{T}}` (chữ T thẳng nhỏ; macro nằm ở `site/src/lib/katex-macros.mjs`), `\norm{#1}` → `\left\lVert #1 \right\rVert`, `\rank` → `\operatorname{rank}`, `\tr` → `\operatorname{tr}`, `\diag` → `\operatorname{diag}`. Generator dùng đúng các macro này khi xuất LaTeX.

### Thuật ngữ Việt – Anh

Lần đầu xuất hiện trong mỗi bài viết “định thức (determinant)”, các lần sau chỉ viết tiếng Việt. Tên riêng và từ đã quen giữ nguyên tiếng Anh: vector, gradient, loss, batch, epoch.

| Tiếng Việt | Tiếng Anh |
| --- | --- |
| ma trận, ma trận vuông | matrix, square matrix |
| chuyển vị | transpose |
| ma trận đơn vị | identity matrix |
| ma trận đường chéo, tam giác trên/dưới | diagonal, upper/lower triangular matrix |
| ma trận đối xứng, trực giao | symmetric, orthogonal matrix |
| tích vô hướng | dot product (inner product) |
| chuẩn | norm |
| tổ hợp tuyến tính | linear combination |
| độc lập tuyến tính | linear independence |
| không gian con, cơ sở, số chiều | subspace, basis, dimension |
| không gian cột, không gian null | column space, null space |
| hạng | rank |
| khử Gauss | Gaussian elimination |
| dạng bậc thang, bậc thang rút gọn | row echelon form, reduced row echelon form (RREF) |
| định thức | determinant |
| ma trận nghịch đảo, khả nghịch | inverse matrix, invertible |
| ánh xạ tuyến tính | linear transformation |
| trị riêng, vector riêng | eigenvalue, eigenvector |
| chéo hóa | diagonalization |
| phân tích giá trị suy biến | singular value decomposition (SVD) |
| phân tích thành phần chính | principal component analysis (PCA) |
| xác định dương | positive definite |

### Văn phong

- Câu ngắn, xưng “bạn”, giải thích trực giác trước khi đưa công thức.
- Mỗi công thức display có một câu đọc nó bằng lời ngay bên dưới.
- Không viết “dễ thấy”, “hiển nhiên”; nếu một bước cần suy nghĩ thì viết bước đó ra.

### Quy tắc đúng toán cho nội dung

- Không gõ tay kết quả số. Ví dụ trong bài được sinh bởi `gen/examples/` và import vào MDX.
- Mỗi định nghĩa và định lý ghi rõ nguồn đối chiếu trong mục Tham khảo.
- Mỗi bài học phải qua prompt Review toán (mục 9) trước khi merge.

### Nguồn tham khảo chuẩn

- Đại số tuyến tính: MIT 18.06 (Gilbert Strang) trên [ocw.mit.edu](https://ocw.mit.edu); series Essence of Linear Algebra trên [3blue1brown.com](https://www.3blue1brown.com).
- Toán cho ML: sách Mathematics for Machine Learning tại [mml-book.github.io](https://mml-book.github.io).
- ML cổ điển: [machinelearningcoban.com](https://machinelearningcoban.com).
- Deep Learning: Dive into Deep Learning tại [d2l.ai](https://d2l.ai).
- Computer Vision: CS231n tại [cs231n.github.io](https://cs231n.github.io).

## 8. Kế hoạch triển khai và Definition of Done

Làm theo lát cắt dọc: Giai đoạn 1 đưa đúng một dạng bài (nhân ma trận) chạy trọn vẹn từ generator tới web, sau đó mới nhân rộng. Không bắt đầu giai đoạn sau khi checklist giai đoạn trước chưa đạt hết.

### GĐ0 — Khung site

- [ ] Repo theo cấu trúc mục 2; Astro + Starlight chạy được bằng `pnpm dev`
- [ ] KaTeX render đúng một trang mẫu có công thức inline, display, `bmatrix` 4×4 và các macro ở mục 7
- [ ] Sidebar tự sinh theo thư mục 8 phần; dark mode; tìm kiếm hoạt động
- [ ] CI build và deploy lên Cloudflare Pages (hoặc GitHub Pages)

### GĐ1 — Lát cắt đầu tiên: nhân ma trận

- [ ] `schema.py`, xuất JSON Schema, sinh kiểu TypeScript tự động
- [ ] `nice.py`, `steps.py`, `latex.py`, `verify.py` với test pytest + hypothesis
- [ ] 7 dạng của bài 1.5, mỗi dạng 200 bài, 100% qua kiểm chứng
- [ ] `fraction.ts`, `parse.ts`, `compare.ts` (chế độ `exact`) với test Vitest
- [ ] ExercisePlayer + MatrixInput: nhập, kiểm tra, tô ô sai, gợi ý, lời giải từng bước
- [ ] Bài học 1.5 viết đủ khung mục 7, ví dụ do generator sinh, đã qua Review toán

### GĐ2 — Đại số tuyến tính bài 1.1–1.9

- [ ] Nội dung 9 bài theo khung mục 7, đã qua Review toán
- [ ] Toàn bộ dạng bài tập của 1.1–1.9 (mục 4), có `solution-set` cho hệ vô số nghiệm
- [ ] Trang luyện tập `/luyen-tap`

### GĐ3 — Đại số tuyến tính bài 1.10–1.15

- [ ] Chế độ chấm `eigvec`, `span`, `numeric`
- [ ] Hình tương tác biến đổi lưới 2D cho bài 1.11–1.12
- [ ] Trang tiến độ `/tien-do`, xuất/nhập JSON
- [ ] Trang Tổng ôn Phần 1

### GĐ4 — Phần 2 (NumPy), 3, 4

- [ ] Generator Phần 2 chạy NumPy lúc build để tính đáp án; generator giải tích bằng SymPy (đạo hàm, gradient, Jacobian) và xác suất (phân số chính xác)
- [ ] Nội dung và bài tập 3 phần, đã qua Review toán

### GĐ5 — Phần 5, 6

- [ ] Bài tập chạy tay vài bước gradient descent, tính tay thuật toán ML trên dữ liệu 4–6 điểm, tính độ đo từ ma trận nhầm lẫn

### GĐ6 — Phần 7, 8

- [ ] Bài tập tính tay forward/backward, output shape của convolution, IoU
- [ ] Bài tính tay attention trên ma trận ≤ 4×4, số patch của ViT, số tham số của mô hình

### Definition of Done chung cho mọi task

- [ ] `uv run pytest` và `pnpm test` đều qua; `pnpm build` không có cảnh báo KaTeX
- [ ] Không có số nào trong bài học hay bài tập được gõ tay
- [ ] Đã mở trang trên dev server, làm thử ít nhất 3 bài mỗi dạng mới, kể cả nhập sai cố ý
- [ ] Cập nhật `docs/spec.md` nếu có quyết định thay đổi

## 9. Prompt cho AI agent

CLAUDE.md đặt ở gốc repo để agent đọc ở mọi phiên; các prompt bên dưới dán vào agent theo thứ tự giai đoạn, còn ba prompt mẫu cuối dùng lặp lại cho từng bài.

### CLAUDE.md

```markdown
# CLAUDE.md — AI Self-Learning Website

## Bối cảnh
Website tĩnh tự học AI (toán nền → ML → Deep Learning → Computer Vision), tiếng Việt kèm
thuật ngữ tiếng Anh. Đặc tả đầy đủ ở docs/spec.md — đọc mục liên quan trước mỗi task.

## Nguyên tắc bất di bất dịch
1. Đúng toán là ưu tiên số 1. KHÔNG gõ tay bất kỳ con số nào trong bài học hay bài tập.
   Mọi số liệu do generator/ sinh bằng SymPy (số hữu tỉ chính xác) và được kiểm chứng.
2. Không dùng float trong lõi generator. Frontend so đáp án bằng Fraction (BigInt).
3. Mỗi dạng bài tập mới phải có đủ: generator, hints, steps, kiểm chứng độc lập, test.
4. Không thêm backend, database, auth. Không đổi stack nếu chưa hỏi người dùng.
   Không thêm Pyodide, Colab hay bài tập viết code; bài tập chỉ là điền đáp án và trắc nghiệm.
   Ma trận trong bài tập tối đa 4×4.
5. Nội dung theo docs/spec.md mục 7: khung bài học, ký hiệu, bảng thuật ngữ.
6. Định dạng bài tập theo generator/gen/schema.py. Kiểu TS sinh tự động, không sửa tay.
7. Không chắc một công thức → dừng lại và hỏi, không đoán.

## Lệnh
- Site:      pnpm --dir site dev | build | test
- Generator: cd generator && uv run python -m gen build --part linalg --count 200
- Test:      cd generator && uv run pytest
- Schema:    cd generator && uv run python -m gen schema

## Definition of Done mỗi task
- pytest, Vitest, build đều qua; build không có cảnh báo KaTeX.
- Đã chạy dev server và tự làm thử bài tập mới, kể cả nhập sai.
- Báo cáo ngắn: đã làm gì, file nào đổi, còn gì chưa làm.
```

### Prompt GĐ0 — Khung site

```text
Đọc CLAUDE.md và docs/spec.md mục 1, 2, 7, 8 (GĐ0). Khởi tạo repo theo cấu trúc mục 2:
Astro + Starlight trong site/, khung generator/ dùng uv. Cấu hình KaTeX với các macro ở
mục 7. Tạo sidebar 8 phần theo mục 3 (mỗi phần một trang index tạm). Tạo trang
/kiem-tra-katex chứa công thức inline, display, bmatrix 4×4, ma trận mở rộng có vạch
đứng, và mọi macro. Thêm workflow CI build + deploy Cloudflare Pages. Hoàn thành khi mọi
mục GĐ0 ở mục 8 đạt. Trước khi code, liệt kê kế hoạch và hỏi lại nếu có gì chưa rõ.
```

### Prompt GĐ1 — Lát cắt nhân ma trận

```text
Đọc CLAUDE.md và docs/spec.md mục 4, 5, 6, 8 (GĐ1). Làm theo thứ tự, mỗi bước có test
rồi mới sang bước sau:
1. generator/gen/schema.py (Pydantic) + lệnh xuất JSON Schema + sinh kiểu TS cho site.
2. core/nice.py, steps.py, latex.py, verify.py. Test tính chất bằng hypothesis.
3. Bảy dạng của bài 1.5: matmul-shape, matmul-entry, matmul-full, matvec,
   matmul-chain-shape, matmul-noncommute, hadamard-vs-matmul. Đủ 5 mức, kích thước như mục 4.
   Mỗi bài: hints, steps, phương án nhiễu (nếu trắc nghiệm), kiểm chứng NumPy.
4. site/src/lib/math: fraction.ts, parse.ts, compare.ts (mode exact) + Vitest.
5. ExercisePlayer + MatrixInput đúng luồng mục 6.
6. Bài học 1.5 theo khung mục 7; mọi ví dụ số sinh từ gen/examples/.
Sinh 200 bài mỗi dạng, in tỉ lệ qua kiểm chứng (phải là 100%). Cuối cùng in 3 bài mẫu
mức 3 của matmul-full kèm lời giải để tôi tự kiểm tra bằng tay.
```

### Prompt mẫu — Viết một bài học mới

```text
Viết bài học {MÃ BÀI} “{TÊN BÀI}” theo docs/spec.md mục 4 (nội dung, dạng bài) và mục 7
(khung, ký hiệu, thuật ngữ, văn phong). Người đọc chưa từng học chủ đề này. Yêu cầu:
- Trực giác trước, công thức sau; mỗi công thức display có một câu đọc bằng lời.
- Ít nhất 2 ví dụ: một nhỏ, một lớn hơn hoặc không vuông. Mọi số liệu sinh bằng
  script trong gen/examples/{mã bài}.py, KHÔNG gõ tay.
- Mục “Lỗi thường gặp” có ví dụ sai → sửa.
- Mục Tham khảo ghi nguồn đối chiếu cho từng định nghĩa và định lý.
Không chắc công thức nào thì liệt kê ra ở cuối báo cáo thay vì đoán.
```

### Prompt mẫu — Thêm một dạng bài tập

```text
Thêm dạng bài tập {MÃ DẠNG} cho bài {MÃ BÀI} theo docs/spec.md mục 5. Trước khi code,
mô tả: cách dựng đề “đẹp” cho từng mức, cách ghi lời giải từng bước, chế độ chấm
(check.mode), các bất biến dùng để kiểm chứng. Sau khi code: test pytest + hypothesis,
sinh 200 bài, tỉ lệ qua kiểm chứng phải 100%, in 2 bài mẫu mỗi mức kèm lời giải.
```

### Prompt mẫu — Review toán (chạy bằng phiên agent mới)

```text
Bạn là người review toán khó tính. Đọc bài học {ĐƯỜNG DẪN FILE MDX} và kho bài tập
{ĐƯỜNG DẪN JSON}. Với mỗi định nghĩa, công thức, ví dụ và lời giải:
1. Kiểm tra tính đúng; tự tính lại các ví dụ bằng SymPy (viết script, chạy thật).
2. Kiểm tra ký hiệu và thuật ngữ đúng docs/spec.md mục 7.
3. Kiểm tra điều kiện bị thiếu (ví dụ: “nếu A khả nghịch”, “nếu kích thước khớp”).
4. Lấy ngẫu nhiên 20 bài tập, giải lại độc lập, so với đáp án và lời giải trong JSON.
Báo cáo dạng bảng: vị trí, vấn đề, mức độ (sai / thiếu / văn phong), đề xuất sửa.
Không sửa file; chỉ báo cáo.
```

## 10. Nhật ký quyết định trong lúc làm (GĐ0–GĐ2)

Các quyết định dưới đây phát sinh khi triển khai; spec gốc chưa nói rõ nên ghi lại ở đây.

- **Dạng `ref` (dạng bậc thang) không duy nhất.** Để chấm `exact` được, đề nêu rõ quy trình: khử Gauss thuần túy, chỉ dùng phép R_i ← R_i + kR_j, không đổi hàng, không chia hàng pivot. Ma trận đề được dựng dạng A = L·U (L đơn vị tam giác dưới nguyên) nên kết quả đúng bằng U và mọi bội số là số nguyên.
- **`solution-set`:** schema thêm `answer.particular`, `answer.basis` và `check.a`, `check.b` (hệ Ax = b). Người học nhập một nghiệm riêng và các vector hướng; bộ chấm kiểm Ax_p = b, Av_i = 0 và hạng của các v_i bằng số ẩn tự do, nên mọi cách chọn đúng đều được chấp nhận.
- **Cosine similarity chính xác:** các bài `vec-cosine` dùng vector có chuẩn L2 nguyên (ví dụ bộ ba Pythagoras) để cosine là số hữu tỉ, giữ đúng nguyên tắc "không float trong lõi".
- **Hadamard và NumPy:** `*` của NumPy tự broadcast; phép kiểm chứng chặn thủ công để khớp định nghĩa toán (hai ma trận phải cùng kích thước).
- **Kiểm chứng LaTeX:** `pnpm validate:banks` render toàn bộ chuỗi LaTeX trong kho bằng KaTeX (strict). Nó từng bắt được chữ tiếng Việt trong `\text{}` mà KaTeX không hiển thị; quy ước: không đặt chữ tiếng Việt trong công thức, chỉ đặt trong `text` của bước.
- **Kiểm thử chéo kho ↔ bộ chấm:** Vitest chấm lại mọi bài trong kho với đáp án chuẩn (và dạng tương đương) và với một đáp án cố ý sai.
- **Phép xử lý mức thiếu:** nếu một bài học không có dạng nào ở một mức, ExercisePlayer lấy bù từ mức gần nhất để vẫn phát đủ số bài.
- **Phiên bản công cụ:** Python từ 3.12 trở lên; Astro 7 cần thêm `@astrojs/markdown-remark` để dùng remark/rehype (KaTeX).
- **Ví dụ trong bài học** là các partial MDX do `gen/examples/lesson_*.py` sinh vào `site/src/generated/`; bài học chỉ import và viết lời dẫn, không chứa số liệu gõ tay.

### Bổ sung ở GĐ3

- **Chế độ chấm `numeric` có sai số riêng:** spec nêu 1e-9, nhưng người học không thể nhập 10 chữ số. Bài `svd-small` mức 4 (σ lớn nhất vô tỉ) dùng `check.tolerance = 1e-3` và đề ghi rõ "làm tròn 3 chữ số thập phân". Các đáp án còn lại đều hữu tỉ nên dùng `exact`.
- **`span`:** ma trận đáp án có cơ sở ở các **cột**; bộ chấm so cơ sở của người học với cơ sở chuẩn bằng hạng (độc lập, đủ số vector, cùng span). Dùng cho `basis-col` và `null-space`. **`eigvec`:** chấp nhận mọi v ≠ 0 thỏa Av = λv (`check.a`, `check.lam`).
- **Ma trận đối xứng có trị riêng nguyên (bài 1.13):** kho được dựng bằng cách quét mọi ma trận đối xứng phần tử nhỏ bằng NumPy (chỉ để tìm ứng viên), mỗi bài khi dùng đều được kiểm lại chính xác bằng SymPy.
- **SVD (bài 1.14):** ma trận A = UΣVᵀ được dựng từ U, V trực giao hữu tỉ (hoán vị có dấu, quay 3-4-5, Cayley mẫu số nhỏ) và Σ nguyên, chỉ giữ A có phần tử nguyên. PCA dùng hiệp phương sai mẫu (chia n − 1) trên 4 điểm 2D được chọn để trị riêng hữu tỉ.
- **`diagonalize` mức 4** là câu hỏi đúng/sai "chéo hóa được không" (khối Jordan là phản ví dụ); các mức thấp hơn tính D = P⁻¹AP.
- **Schema:** mọi trường (kể cả có giá trị mặc định) xuất hiện trong JSON, nên bắt buộc ở TypeScript; các trường rỗng ghi là `null`. Kho tăng kích thước (khoảng 16 MB tổng) nhưng được tải lười theo từng dạng.
- **Tổng ôn Phần 1** là bộ 15 bài rút ngẫu nhiên từ mọi dạng của phần (tải lười chỉ các kho cần dùng), không phải trang cố định.
- **Tiến độ:** bài học được đánh dấu "đã đọc" khi người học cuộn tới cuối phần Tóm tắt; sidebar hiện dấu tích. Có nút xuất/nhập/xóa tiến độ.
- **Hình tương tác (FR10):** `GridTransform` vẽ lưới 2D bị biến đổi bởi ma trận 2×2 (chọn mẫu hoặc gõ số, kéo thanh trượt từ ma trận đơn vị đến A, hiện det và hướng riêng). Hai hình còn lại của FR10 (đồ thị hàm, đường gradient descent) thuộc Phần 3 và 5.

### Bổ sung ở GĐ4 (Phần 2, 3, 4)

- **Phân bài Phần 3 và 4 do tôi tự đề xuất** (spec chỉ liệt kê chủ đề), theo lựa chọn "tự làm theo bảng mục 3, không cần duyệt":
  - Phần 3: 3.1 Đạo hàm, 3.2 Quy tắc chuỗi, 3.3 Đạo hàm riêng và gradient, 3.4 Jacobian, 3.5 Hessian, 3.6 Đạo hàm theo vector và ma trận, 3.7 Đồ thị tính toán và vi phân tự động (11 dạng).
  - Phần 4: 4.1 Xác suất cơ bản và có điều kiện, 4.2 Bayes và Naive Bayes, 4.3 Kỳ vọng, phương sai, hiệp phương sai, 4.4 Phân phối (Bernoulli, nhị thức, chuẩn, Mahalanobis), 4.5 MLE và MAP, 4.6 Entropy, cross-entropy, KL (13 dạng).
  - Mã phần trong kho bài: `numpy`, `calculus`, `probability` (thư mục `site/public/exercises/<phần>/<bài>/`).
- **Dạng đáp án `expression`:** người học nhập biểu thức (cú pháp `3x^2 + 2*x`, `exp(x)`, `ln(x)`, `sin`, `cos`, `sqrt`; nhân ngầm được hỗ trợ). Bộ chấm (`src/lib/math/expr.ts`) phân tích biểu thức, so giá trị số tại các điểm mẫu cố định (đều dương) với sai số tương đối 1e-9, nên mọi dạng tương đương đều đúng. Có xem trước KaTeX khi gõ và báo lỗi cú pháp. Chuỗi đáp án chuẩn do SymPy xuất được kiểm khứ hồi (parse lại bằng SymPy phải bằng biểu thức gốc).
- **Kiểm chứng đạo hàm** bằng sai phân hữu hạn (độc lập với SymPy) tại các điểm dương; gradient, Jacobian, Hessian tại một điểm cũng được kiểm bằng sai phân.
- **Phần 2 chạy NumPy thật lúc build:** đáp án là kết quả NumPy; các dạng có thể kiểm chéo bằng SymPy hoặc Python thuần thì kiểm thêm. Dạng `np-vectorize-choose` chạy thử **từng lựa chọn** trên dữ liệu ngẫu nhiên (kể cả dữ liệu lớn cho softmax) và chỉ nhận đề khi đúng một lựa chọn cho kết quả đúng; `np-bug-find` chạy cả bản đúng lẫn bản lỗi. Ví dụ trong bài học là code được chạy thật, kết quả in ra được chèn vào MDX.
- **Code NumPy hiển thị bằng Shiki** (tải lười khi bài có đoạn code), hai giao diện sáng và tối.
- **Không đặt chữ tiếng Việt trong công thức LaTeX** (kể cả trong `\text{}` hay ô bảng `array`): KaTeX không có glyph cho các dấu tiếng Việt. Chữ tiếng Việt đặt trong `text` của bước hoặc đề; bảng dùng nhãn ASCII (R_1, C_1, ...) kèm lời giải thích trong đề. `pnpm validate:banks` bắt lỗi này.
- **Phần 4 dùng phân số chính xác.** Entropy, cross-entropy, KL dùng xác suất là lũy thừa của 1/2 để mọi logarit là số nguyên; đáp án đối chiếu với `numpy.log2`. Hiệp phương sai mẫu chia cho n − 1, MLE phương sai chia cho n (đề ghi rõ).
- **Giới hạn đã biết:** dạng `np-dtype` chỉ có khoảng 20 biểu thức khác nhau (các đề khác nhau chủ yếu ở thứ tự lựa chọn); sẽ mở rộng khi có thêm thời gian.

### Bổ sung ở GĐ5 (Phần 5 Tối ưu)

- **Phân bài Phần 5 do tôi tự chia** (mã phần `optimization`, thư mục `05-toi-uu`): 5.1 Hàm lồi, 5.2 Gradient descent, 5.3 SGD và mini-batch, 5.4 Momentum, 5.5 Adam, RMSProp, AdaGrad, 5.6 Lịch learning rate, 5.7 Regularization L1, L2 và weight decay. 21 dạng bài tập.
- **Adam, RMSProp, AdaGrad tính tay chính xác:** đề đặt epsilon = 0 và **chọn gradient** sao cho mọi căn bậc hai là số hữu tỉ (tìm bằng quét nguyên trong |g| ≤ 12, ví dụ AdaGrad dùng các bộ có mọi tổng riêng của bình phương là số chính phương). Bước đầu của Adam luôn có m_hat/sqrt(v_hat) = sign(g), được dùng làm dạng `adamw-step`.
- **Cosine annealing chỉ dùng góc đặc biệt:** t/T thuộc {0, 1/3, 1/2, 2/3, 1} để cosine là số hữu tỉ; mọi đáp án được đối chiếu với `math.cos`. Lịch Transformer dùng d, W, t là số chính phương.
- **Kiểm chứng:** gradient viết tay luôn được so với đạo hàm biểu tượng của SymPy (thế số hữu tỉ); dạng `convex-1d` kiểm bằng bất đẳng thức Jensen chính xác trên lưới Fraction, `convex-hessian` kiểm bằng trị riêng (SymPy hoặc NumPy) và định thức con chính, `sgd-unbiased` liệt kê mọi mini-batch, `l1-prox` kiểm điều kiện tối ưu dưới vi phân.
- **Nesterov** trong bài 5.4 theo định nghĩa gốc (gradient tại x - eta*beta*v), được nêu rõ công thức trong đề; PyTorch dùng cách viết tương đương gần đúng.
- **Phân bài Phần 6 do tôi tự chia** (mã phần `classical-ml`, thư mục `06-ml-co-dien`): 6.1 Hồi quy tuyến tính, 6.2 Hồi quy logistic và softmax, 6.3 Overfitting, bias–variance, cross-validation, 6.4 Độ đo phân loại, 6.5 KNN, 6.6 K-means, 6.7 Cây quyết định và random forest, 6.8 Gradient boosting, 6.9 SVM (17 dạng).
- **Phần 6 tránh số vô tỉ:** hồi quy logistic và softmax cho sẵn xác suất p (không bắt người học tính e^z hay ln), gradient là p - y; cây dùng Gini thay entropy; chuẩn trong SVM dùng w là bộ ba Pythagoras; Gini, F1, AUC, inertia, kết quả K-means đều là phân số chính xác. Cây quyết định và gradient boosting chỉ dùng stump hoặc một phép tách trên một đặc trưng.
- **Kiểm chứng Phần 6:** OLS đối chiếu `numpy.linalg.lstsq` và điều kiện phần dư trực giao; log-loss và softmax đối chiếu sai phân hữu hạn của cross-entropy và đồng nhất thức SymPy dL/dz = p - y; bias–variance đối chiếu bằng liệt kê mọi bộ dữ liệu và nhiễu ±s; AUC đối chiếu tích phân hình thang đường ROC; macro-F1 đối chiếu trung bình điều hòa tính từ danh sách nhãn; OOB đối chiếu liệt kê n^m mẫu bootstrap; K-means, KNN đối chiếu NumPy; SVM đối chiếu sai phân hữu hạn của mục tiêu lề mềm.
- **Ví dụ cho bài học không dùng số thực** (ví dụ sigmoid chỉ dùng giá trị z = 0 và đồng nhất thức); `RichText` của site nay hiển thị `**đậm**` và `` `mã` `` (trước đó các dấu này hiện nguyên văn).
- **Phân bài Phần 7 do tôi tự chia** (mã phần `deep-learning`, thư mục `07-deep-learning`): 7.1 Perceptron và MLP, 7.2 Hàm kích hoạt, softmax, cross-entropy, 7.3 Lan truyền ngược, 7.4 Khởi tạo, BatchNorm, LayerNorm, 7.5 Dropout, residual, embedding, 7.6 Attention và Transformer, 7.7 Fine-tuning, LoRA và đọc code PyTorch (15 dạng).
- **Phân bài Phần 8** (mã phần `computer-vision`, thư mục `08-computer-vision`): 8.1 Ảnh là tensor, tích chập, padding, stride; 8.2 Pooling và trường tiếp nhận; 8.3 CNN và ResNet; 8.4 Vision Transformer; 8.5 Augmentation và transfer learning; 8.6 Phát hiện vật thể (IoU, GIoU, NMS, anchor-free, ghép Hungarian kiểu DETR); 8.7 Phân đoạn (U-Net, SAM); 8.8 Self-supervised, contrastive và CLIP; 8.9 Khuếch tán (14 dạng). Các mô hình lớn (SAM, CLIP, diffusion) chỉ ở mức khái niệm kèm phép tính cốt lõi.
- **Tránh số vô tỉ ở Phần 7–8:** softmax và attention dùng điểm dạng k·ln 2 (nên e^s = 2^k hữu tỉ), cross-entropy hỏi theo bit; khởi tạo dùng căn hữu tỉ bằng cách chọn n là bình phương; BatchNorm/LayerNorm chọn dữ liệu có phương sai là bình phương hữu tỉ (epsilon = 0); khuếch tán chọn ᾱ là bình phương của tỉ số Pythagoras (9/25, 25/169...); CLIP dùng vector đơn vị hữu tỉ.
- **Bài tập đọc code PyTorch (`torch-shape`):** chỉ để đọc, trả lời bằng trắc nghiệm shape; shape đáp án được mô phỏng bằng NumPy (không cần cài PyTorch). Không có bài viết code.
- **Kiểm chứng Phần 7–8:** forward/backward đối chiếu NumPy và sai phân hữu hạn; attention đối chiếu softmax số thực với mặt nạ −∞; tích chập/pooling đối chiếu `sliding_window_view`; trường tiếp nhận đối chiếu lan truyền ngược khoảng chỉ số; IoU đối chiếu đếm ô đơn vị; NMS đối chiếu hai cài đặt (duyệt và loại lặp); ghép Hungarian đối chiếu vét cạn và quy hoạch động theo tập cột; số tham số đối chiếu đếm lại từ hình dạng tensor.
- **Khử trùng đề:** với Phần 5–8, hai đề chỉ khác thứ tự trộn lựa chọn được coi là một (so theo tập lựa chọn đã sắp xếp); Phần 1–4 giữ cách cũ.
- **Review toán Phần 5–6 (13/11/2026):** hai phiên agent riêng giải lại toàn bộ kho (4.200 và 3.400 bài), 0 lệch đáp án; các lỗi nội dung tìm thấy đã sửa hoặc ghi ở PROCESS.md mục 3.2.
- **Hình tương tác thứ hai của FR10:** `GdPath` (đường đồng mức và quỹ đạo gradient descent/momentum, dùng số thực, chỉ để xem), gắn ở bài 5.2 và 5.4.
- **Sửa lỗi giao diện toán (13/11/2026):** rehype-katex dùng KaTeX 0.16 để sinh HTML nhưng CSS lấy từ KaTeX 0.19 (đổi tên class `inner`/`fix` thành `katex-inner`/`katex-fix`), làm dấu "khác" hiện thành "/=". Đã ghim `katex@0.16.47` cho khớp. Ký hiệu chuyển vị dùng macro `\T` = chữ T thẳng nhỏ (mathsf) thay cho `\top`; phân số có "thanh chống" để tử/mẫu không dính vạch; cột mục lục bên phải thu hẹp còn 13,5rem, nội dung giữa rộng 62rem.
