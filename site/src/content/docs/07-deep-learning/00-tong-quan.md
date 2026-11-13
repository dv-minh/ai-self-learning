---
title: Tổng quan Phần 7
description: Perceptron, MLP, backpropagation, chuẩn hóa, residual, attention, Transformer, LoRA và đọc code PyTorch
sidebar:
  order: 0
---

Phần 7 xây các thành phần của học sâu hiện đại, từ một nơ-ron đến Transformer.

| Bài | Nội dung | Bạn sẽ làm được |
| --- | --- | --- |
| 7.1 | Perceptron và MLP | forward pass, đếm tham số và MAC |
| 7.2 | Hàm kích hoạt, softmax, cross-entropy | ReLU và đạo hàm, softmax, gradient $\mathbf{p} - \mathbf{y}$ |
| 7.3 | Lan truyền ngược | gradient của mạng hai lớp bằng tay |
| 7.4 | Khởi tạo, BatchNorm, LayerNorm | phương sai khởi tạo, chuẩn hóa |
| 7.5 | Dropout, residual, embedding | kỳ vọng dropout, gradient qua residual, tra embedding |
| 7.6 | Attention và Transformer | điểm, đầu ra, mặt nạ, đếm tham số |
| 7.7 | Fine-tuning, LoRA, code PyTorch | tham số LoRA, gộp trọng số, suy luận shape |

Điều kiện tiên quyết: Phần 1 (ma trận), Phần 3 (chain rule, đồ thị tính toán), Phần 5 (tối ưu) và Phần 6 (softmax, cross-entropy). Bài tập chỉ là điền đáp án hoặc trắc nghiệm, tính tay trên ma trận nhỏ (không quá $4\times 4$) bằng số hữu tỉ; khi công thức cần $e^{x}$ đề dùng điểm số dạng $k\ln 2$. Phần "đọc code PyTorch" dùng đoạn code chỉ để đọc, không có bài viết code.

Trang **Tổng ôn Phần 7** trộn mọi dạng bài của phần này.
