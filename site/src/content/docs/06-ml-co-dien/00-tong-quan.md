---
title: Tổng quan Phần 6
description: Hồi quy, phân loại, đánh giá mô hình, KNN, K-means, cây, boosting, SVM
sidebar:
  order: 0
---

Phần 6 gom các thuật toán học máy cổ điển. Chúng vẫn là xương sống của nhiều hệ thống thực tế (đặc biệt với dữ liệu dạng bảng) và là nền tảng để hiểu học sâu ở Phần 7.

| Bài | Nội dung | Bạn sẽ làm được |
| --- | --- | --- |
| 6.1 | Hồi quy tuyến tính | giải phương trình chuẩn, tính MSE, $R^2$ |
| 6.2 | Hồi quy logistic và softmax | tính gradient $p - y$, biên quyết định |
| 6.3 | Overfitting, bias–variance, cross-validation | phân rã sai số, chia fold, chọn siêu tham số |
| 6.4 | Độ đo phân loại | precision, recall, F1, AUC |
| 6.5 | KNN | phân loại và hồi quy bằng láng giềng gần nhất |
| 6.6 | K-means | một vòng gán và cập nhật tâm |
| 6.7 | Cây quyết định và random forest | Gini, chọn ngưỡng tách, OOB |
| 6.8 | Gradient boosting | một vòng khớp phần dư |
| 6.9 | SVM | lề, support vector, hinge loss |

Điều kiện tiên quyết: Phần 1 (ma trận, hạng, trị riêng), Phần 3 (gradient), Phần 4 (xác suất) và Phần 5 (gradient descent, regularization). Mọi bài tập đều tính tay trên vài điểm dữ liệu bằng số hữu tỉ chính xác; khi công thức thật cần $e^{x}$ hay logarit thì đề cho sẵn xác suất đã tính.

Trang **Tổng ôn Phần 6** trộn mọi dạng bài của phần này.
