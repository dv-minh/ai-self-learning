---
title: Tổng quan Phần 5
description: Hàm lồi, gradient descent, SGD, momentum, Adam, lịch learning rate, regularization
sidebar:
  order: 0
---

Huấn luyện một mô hình ML là bài toán **tối ưu hóa**: tìm tham số làm hàm mất mát nhỏ nhất. Phần 5 đi từ lý thuyết tối thiểu (hàm lồi) đến các thuật toán dùng hằng ngày.

| Bài | Nội dung | Bạn sẽ làm được |
| --- | --- | --- |
| 5.1 | Hàm lồi | phân loại hàm bằng Hessian, tìm cực tiểu hàm bậc hai |
| 5.2 | Gradient descent | chạy tay các bước, chọn learning rate để hội tụ |
| 5.3 | SGD và mini-batch | tính cập nhật trên mini-batch và một epoch |
| 5.4 | Momentum | chạy tay momentum và Nesterov |
| 5.5 | Adam, RMSProp, AdaGrad | tính moment, hiệu chỉnh độ chệch, cập nhật thích ứng |
| 5.6 | Lịch learning rate | step, exponential, warmup, cosine, lịch Transformer |
| 5.7 | Regularization | ridge, weight decay, AdamW, L1 (soft-thresholding) |

Điều kiện tiên quyết: Phần 1 (đặc biệt trị riêng và ma trận xác định dương, bài 1.12 và 1.13) và Phần 3 (gradient, Hessian). Mọi bài tập đều là chạy tay vài bước cập nhật với số hữu tỉ chính xác. Mọi số trong đề được sinh và kiểm chứng bằng chương trình.

Trang **Tổng ôn Phần 5** trộn mọi dạng bài của phần này.
