---
title: Tổng quan Phần 8
description: Tích chập, CNN và ResNet, Vision Transformer, detection, segmentation, self-supervised, CLIP, diffusion
sidebar:
  order: 0
---

Phần 8 đi từ phép tích chập đến các mô hình thị giác hiện đại.

| Bài | Nội dung | Bạn sẽ làm được |
| --- | --- | --- |
| 8.1 | Ảnh là tensor, tích chập, padding, stride | tính tích chập tay, kích thước đầu ra |
| 8.2 | Pooling và trường tiếp nhận | pooling, receptive field |
| 8.3 | CNN và ResNet | đếm tham số, MAC, khối residual |
| 8.4 | Vision Transformer | số patch, tham số patch embedding |
| 8.5 | Augmentation và transfer learning | biến đổi ảnh, tham số huấn luyện được |
| 8.6 | Phát hiện vật thể | IoU, NMS, ghép một-một kiểu DETR |
| 8.7 | Phân đoạn | IoU, Dice, mIoU, shape của U-Net |
| 8.8 | Self-supervised và CLIP | độ tương tự, truy hồi, contrastive |
| 8.9 | Khuếch tán | khuếch tán thuận, $\bar{\alpha}_t$, khôi phục |

Điều kiện tiên quyết: Phần 1 (ma trận), Phần 7 (MLP, backprop, residual, attention). Bài tập chỉ là điền đáp án hoặc trắc nghiệm, tính tay trên ảnh $4\times 4$ và ma trận nhỏ bằng số hữu tỉ chính xác; không có bài viết code. Các mô hình lớn (CLIP, SAM, diffusion) được trình bày ở mức khái niệm kèm các phép tính cốt lõi.

Trang **Tổng ôn Phần 8** trộn mọi dạng bài của phần này.
