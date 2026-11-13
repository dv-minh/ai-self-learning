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

## Trạng thái
Xem PROCESS.md (đã làm, còn lại, bản đồ mã nguồn, bài học kinh nghiệm). Hiện xong GĐ0–GĐ6 (Phần 1–8); còn dọn dẹp và Review toán (xem mục 3 của PROCESS.md).

## Lệnh
- Site:      pnpm --dir site dev | build | test | typecheck | validate:banks
- Generator: cd generator && uv run python -m gen build --all --count 200
             (một dạng: --part <linalg|numpy|calculus|probability|optimization|classical-ml|deep-learning|computer-vision> --type <mã dạng>)
- Test:      cd generator && uv run pytest
- Schema:    cd generator && uv run python -m gen schema && pnpm --dir site gen:types

## Quy ước rút ra khi làm (chi tiết ở docs/spec.md mục 10 và PROCESS.md mục 6)
- KHÔNG ghi code/LaTeX có dấu `\` bằng Bash heredoc (mất dấu); dùng Write/Edit.
- KHÔNG đặt chữ tiếng Việt trong công thức LaTeX (kể cả text trong công thức và ô array); `pnpm validate:banks` kiểm.
- Mỗi dạng cần ≥ 200 đề khác nhau; kiểm chứng độc lập (SymPy/NumPy/liệt kê/sai phân).
- Cập nhật PROCESS.md khi kết thúc mỗi giai đoạn.

## Definition of Done mỗi task
- pytest, Vitest, build đều qua; build không có cảnh báo KaTeX.
- Đã chạy dev server và tự làm thử bài tập mới, kể cả nhập sai.
- Báo cáo ngắn: đã làm gì, file nào đổi, còn gì chưa làm.
