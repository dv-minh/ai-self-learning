"""Bài 8.4 vit-count. 8.5 aug-transform, transfer-params. 8.6 iou-nms, set-match. 8.7 seg-metrics. 8.8 clip-retrieval. 8.9 diffusion-forward."""
from __future__ import annotations

import random
from itertools import combinations, permutations

import numpy as np

from ..dl._d import (F, NUM_HINT, VEC_HINT, Prompt, Step, TopicSpec, check, choice_exercise, cv, fp, ft, mat_ans, mat_tex, mm, num_ans, pick, sqrt_q, vec_ans, vt)

L84, L85, L86, L87, L88, L89 = "8.4", "8.5", "8.6", "8.7", "8.8", "8.9"


def _arr(m):
    return np.array([[float(v) for v in r] for r in m])


# --------------------------------------------------------------------- vit-count
def gen_vit_count(rng: random.Random, level: int, seed: int):
    P = rng.choice([4, 8, 14, 16, 32])
    g = rng.randint(2, 16)
    H = P * g
    D = 64 * rng.randint(1, 16)
    heads = rng.choice([2, 4, 8, 12, 16])
    N = g * g
    if level == 1:
        val, text, st = N, f"Vision Transformer chia ảnh ${H}\\times {H}$ thành các patch không chồng lấn ${P}\\times {P}$. Có bao nhiêu patch?", f"({H}/{P})^2 = {g}^2 = {N}"
        recount = sum(1 for i in range(0, H, P) for j in range(0, H, P))
    elif level == 2:
        val, text, st = N + 1, f"ViT chia ảnh ${H}\\times {H}$ thành patch ${P}\\times {P}$ và thêm một token [CLS]. Độ dài chuỗi token đưa vào Transformer?", f"({H}/{P})^2 + 1 = {N + 1}"
        recount = sum(1 for i in range(0, H, P) for j in range(0, H, P)) + 1
    elif level == 3:
        val = (P * P * 3 + 1) * D
        text = f"Patch embedding của ViT: mỗi patch RGB ${P}\\times {P}$ được duỗi thành vector rồi qua một lớp tuyến tính (có chặn) lên $D = {D}$ chiều. Số tham số của lớp này? (Tương đương conv kernel ${P}$ stride ${P}$.)"
        st = f"({P}\\cdot {P}\\cdot 3 + 1)\\cdot {D} = {val}"
        recount = sum(P * P * 3 + 1 for _ in range(D))
    elif level == 4:
        val = heads * (N + 1) ** 2
        text = f"ViT với ảnh ${H}\\times {H}$, patch ${P}$, có [CLS], ${heads}$ đầu attention. Một lớp tạo ra tổng cộng bao nhiêu phần tử ma trận attention (mọi đầu, một ảnh)?"
        st = f"{heads}\\cdot ({N} + 1)^2 = {val}"
        recount = heads * sum(1 for _ in range(N + 1) for _ in range(N + 1))
    else:
        val = (P * P * 3 + 1) * D + D + (N + 1) * D
        text = (f"ViT với ảnh ${H}\\times {H}$, patch ${P}$, $D = {D}$. Tính tổng số tham số của patch embedding (có chặn), token [CLS] học được ($D$ tham số) và embedding vị trí học được cho mọi token (gồm [CLS]).")
        st = f"({P}^2\\cdot 3 + 1){D} + {D} + ({N} + 1){D} = {val}"
        recount = sum(P * P * 3 + 1 for _ in range(D)) + D + sum(D for _ in range(N + 1))
    check(recount == val, "đếm lại")
    return cv(type_="vit-count", lesson=L84, level=level, seed=seed, prompt=Prompt(text=text + " " + NUM_HINT), answer=num_ans(F(val)),
              hints=["Số patch $= (H/P)^2$ khi $P$ chia hết $H$.", "Token [CLS] cộng thêm $1$ vào độ dài chuỗi; attention có ma trận $(N+1)\\times (N+1)$ cho mỗi đầu.", "Patch embedding: $P^2C$ trọng số mỗi chiều đầu ra (+ chặn)."],
              steps=[Step(text="Tính:", latex=st)], verified=["recount"], tags=["vit", f"level{level}"])


# ---------------------------------------------------------------- aug-transform
def gen_aug_transform(rng: random.Random, level: int, seed: int):
    n = 4
    img = [[F(rng.randint(0, 9)) for _ in range(n)] for _ in range(n)]
    a = _arr(img)
    ops = {1: ("flip-h", "lật ngang (trái–phải)", lambda m: np.fliplr(m)), 2: ("flip-v", "lật dọc (trên–dưới)", lambda m: np.flipud(m)),
           3: ("rot90", "quay $90^\\circ$ **cùng chiều kim đồng hồ**", lambda m: np.rot90(m, -1))}
    if level <= 3:
        name, desc, fn = ops[level]
        out = fn(a)
    elif level == 4:
        i, j = rng.randint(0, 2), rng.randint(0, 2)
        name, desc = "crop-flip", f"cắt vùng $2\\times 2$ có góc trên-trái ở hàng ${i + 1}$, cột ${j + 1}$ rồi lật ngang"
        out = np.fliplr(a[i:i + 2, j:j + 2])
    else:
        name, desc = "compose", "lật dọc rồi quay $90^\\circ$ cùng chiều kim đồng hồ"
        out = np.rot90(np.flipud(a), -1)
    res = [[F(int(v)) for v in r] for r in out]
    # kiểm chứng độc lập bằng chỉ số: phần tử đầu ra tính từ vị trí gốc
    if level == 1:
        check(all(res[i][j] == img[i][n - 1 - j] for i in range(n) for j in range(n)), "lật ngang theo chỉ số")
    elif level == 2:
        check(all(res[i][j] == img[n - 1 - i][j] for i in range(n) for j in range(n)), "lật dọc theo chỉ số")
    elif level == 3:
        check(all(res[i][j] == img[n - 1 - j][i] for i in range(n) for j in range(n)), "quay 90° theo chỉ số")
    return cv(type_="aug-transform", lesson=L85, level=level, seed=seed,
              prompt=Prompt(text=f"Áp dụng phép tăng cường dữ liệu (augmentation) sau lên ảnh xám $\\mathbf{{X}}$ cỡ $4\\times 4$: {desc}. Tính ảnh kết quả. Nhập từng phần tử.", display=["\\mathbf{X} = " + mat_tex(img)]),
              answer=mat_ans(res), hints=["Lật ngang: đảo thứ tự các cột. Lật dọc: đảo thứ tự các hàng.", "Quay $90^\\circ$ cùng chiều kim đồng hồ: cột đầu (đọc từ dưới lên) thành hàng đầu.", "Với phép ghép, áp dụng lần lượt từ trái sang phải."],
              steps=[Step(text="Kết quả:", latex="\\mathbf{Y} = " + mat_tex(res))], verified=["numpy-and-index-check"], tags=["aug", name])


# ------------------------------------------------------------- transfer-params
def gen_transfer_params(rng: random.Random, level: int, seed: int):
    feat = rng.choice([512, 768, 1024, 2048])
    ncls = rng.randint(2, 200)
    total_m = rng.randint(5, 600)
    block = rng.randint(1, 40)
    k = rng.randint(1, 4)
    head = (feat + 1) * ncls
    if level == 1:
        val = head
        text = f"Linear probe: đóng băng toàn bộ backbone, chỉ huấn luyện một lớp tuyến tính mới từ ${feat}$ đặc trưng đến ${ncls}$ lớp (có chặn). Số tham số huấn luyện được?"
        st = f"({feat} + 1)\\cdot {ncls} = {val}"
    elif level == 2:
        val = head + block * 10**6 * k
        text = f"Fine-tune phần cuối: đóng băng backbone ngoại trừ ${k}$ khối cuối (mỗi khối ${block}$ triệu tham số) và thay đầu phân loại bằng lớp tuyến tính ${feat}\\to {ncls}$ (có chặn). Số tham số huấn luyện được?"
        st = f"{k}\\cdot {block}\\,000\\,000 + {head} = {val}"
    elif level == 3:
        base = total_m * 10**6
        val = F(head, base + head)
        text = f"Backbone có ${total_m}$ triệu tham số (đóng băng); thêm lớp tuyến tính ${feat}\\to {ncls}$ (có chặn) và chỉ huấn luyện lớp mới. Tỉ lệ (tham số huấn luyện) / (tổng tham số mô hình) là bao nhiêu (phân số rút gọn)?"
        st = f"\\dfrac{{{head}}}{{{base} + {head}}} = {ft(val)}"
    elif level == 4:
        keep = rng.choice([F(1, 2), F(1, 4), F(1, 10)])
        base = total_m * 10**6
        val = base * keep + head
        text = f"Mô hình có ${total_m}$ triệu tham số ở backbone; mở khóa (huấn luyện) ${ft(keep)}$ số tham số cuối và thêm đầu phân loại ${feat}\\to {ncls}$ (có chặn). Tổng số tham số huấn luyện được?"
        st = f"{ft(keep)}\\cdot {base} + {head} = {ft(val)}"
    else:
        r = rng.choice([4, 8, 16])
        nmat = rng.choice([24, 48, 96])
        val = nmat * r * 2 * feat + head
        text = f"Backbone đóng băng; áp dụng LoRA hạng $r = {r}$ lên ${nmat}$ ma trận $d\\times d$ ($d = {feat}$) và thêm đầu phân loại ${feat}\\to {ncls}$ (có chặn). Tổng số tham số huấn luyện được?"
        st = f"{nmat}\\cdot {r}\\cdot 2\\cdot {feat} + {head} = {val}"
    return cv(type_="transfer-params", lesson=L85, level=level, seed=seed, prompt=Prompt(text=text + " " + NUM_HINT), answer=num_ans(F(val)),
              hints=["Tham số bị đóng băng không được cập nhật và không tính vào tham số huấn luyện.", "Lớp tuyến tính $a\\to b$ có chặn: $(a + 1)b$ tham số.", "LoRA trên ma trận $d\\times d$ hạng $r$: $2rd$ tham số."],
              steps=[Step(text="Cộng các phần huấn luyện được:", latex=st)], verified=["recompute"], tags=["transfer", f"level{level}"])


# ------------------------------------------------------------------------- iou-nms
def _iou(a, b):
    iw = max(F(0), min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(F(0), min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua, inter, ua


def _box(rng, lo=0, hi=12):
    x1, y1 = rng.randint(lo, hi - 3), rng.randint(lo, hi - 3)
    return [F(x1), F(y1), F(x1 + rng.randint(2, 6)), F(y1 + rng.randint(2, 6))]


def _iou_pixels(a, b):
    """IoU độc lập: đếm ô đơn vị bằng tập hợp."""
    sa = {(x, y) for x in range(int(a[0]), int(a[2])) for y in range(int(a[1]), int(a[3]))}
    sb = {(x, y) for x in range(int(b[0]), int(b[2])) for y in range(int(b[1]), int(b[3]))}
    return F(len(sa & sb), len(sa | sb))


def gen_iou_nms(rng: random.Random, level: int, seed: int):
    if level <= 3:
        for _ in range(500):
            a, b = _box(rng), _box(rng)
            iou, inter, ua = _iou(a, b)
            if level == 1 and iou > 0 and iou < 1:
                break
            if level == 2 and rng.random() < 0.5 and (inter == 0 or iou > 0):
                break
            if level == 3 and iou > 0:
                break
        check(iou == _iou_pixels(a, b), "IoU đếm ô ≠ công thức")
        bx = lambda v: f"({', '.join(ft(c) for c in v)})"
        if level == 3:
            cx1, cy1, cx2, cy2 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
            carea = (cx2 - cx1) * (cy2 - cy1)
            giou = iou - (carea - ua) / carea
            check(giou <= iou and giou > -1, "GIoU hợp lệ")
            text = (f"Hai hộp $(x_1, y_1, x_2, y_2)$: $A = {bx(a)}$, $B = {bx(b)}$ (trục $y$ hướng xuống, cạnh phải/dưới là biên ngoài). Tính **GIoU** $= \\mathrm{{IoU}} - \\dfrac{{|C| - |A\\cup B|}}{{|C|}}$ với $C$ là hộp bao nhỏ nhất chứa cả $A$ và $B$. " + NUM_HINT)
            steps = [Step(text=f"$|A\\cap B| = {ft(inter)}$, $|A\\cup B| = {ft(ua)}$, $\\mathrm{{IoU}} = {ft(iou)}$; hộp bao $C$ có diện tích ${ft(carea)}$:", latex=f"\\mathrm{{GIoU}} = {ft(iou)} - \\dfrac{{{ft(carea)} - {ft(ua)}}}{{{ft(carea)}}} = {ft(giou)}")]
            val = giou
        else:
            text = f"Hai hộp $(x_1, y_1, x_2, y_2)$: $A = {bx(a)}$, $B = {bx(b)}$ (trục $y$ hướng xuống). Tính IoU (intersection over union). " + NUM_HINT
            steps = [Step(text=f"Giao có diện tích ${ft(inter)}$, hợp có diện tích ${ft(ua)}$:", latex=f"\\mathrm{{IoU}} = \\dfrac{{{ft(inter)}}}{{{ft(ua)}}} = {ft(iou)}")]
            val = iou
        return cv(type_="iou-nms", lesson=L86, level=level, seed=seed, prompt=Prompt(text=text), answer=num_ans(val),
                  hints=["Giao: chiều rộng $= \\max(0, \\min(x_2) - \\max(x_1))$, chiều cao tương tự; diện tích giao $=$ rộng $\\times$ cao.", "Hợp $= |A| + |B| - |A\\cap B|$.", "IoU $= 0$ khi hai hộp không giao nhau."],
                  steps=steps, verified=["pixel-set-count"], tags=["iou", f"level{level}"])
    nb = 4 if level == 4 else 5
    thr = F(1, 2)
    for _ in range(2000):
        boxes = [_box(rng, 0, 10) for _ in range(nb)]
        scores = rng.sample([F(k, 10) for k in range(1, 10)], nb)
        order = sorted(range(nb), key=lambda i: -scores[i])
        ious = {(i, j): _iou(boxes[i], boxes[j])[0] for i, j in combinations(range(nb), 2)}
        if any(v == thr for v in ious.values()):
            continue
        keep = []
        for i in order:
            if all((min(i, k), max(i, k)) not in ious or ious[(min(i, k), max(i, k))] <= thr for k in keep):
                keep.append(i)
        if 1 < len(keep) < nb:
            break
    # kiểm chứng độc lập: loại lặp lại (xóa phần tử bị trùm) bằng danh sách còn lại
    rest, kept2 = list(order), []
    while rest:
        top = rest.pop(0)
        kept2.append(top)
        rest = [j for j in rest if _iou_pixels(boxes[top], boxes[j]) <= thr]
    check(kept2 == keep, "NMS lặp ≠ NMS duyệt")
    rows = " \\\\ ".join(f"{i + 1} & " + " & ".join(ft(c) for c in b) + f" & {ft(sc)}" for i, (b, sc) in enumerate(zip(boxes, scores)))
    tbl = "\\begin{array}{c|cccc|c} i & x_1 & y_1 & x_2 & y_2 & \\mathrm{score} \\\\ \\hline " + rows + " \\end{array}"
    return cv(type_="iou-nms", lesson=L86, level=level, seed=seed,
              prompt=Prompt(text=f"Non-maximum suppression (NMS) với ngưỡng IoU $= {ft(thr)}$: lần lượt chọn hộp điểm cao nhất còn lại và loại các hộp còn lại có IoU **lớn hơn** ngưỡng với hộp vừa chọn. Với ${nb}$ hộp dưới đây, còn lại bao nhiêu hộp?", display=[tbl]),
              answer=num_ans(F(len(keep))), hints=["Sắp xếp theo điểm giảm dần; chọn hộp đầu, tính IoU với các hộp còn lại và loại những hộp vượt ngưỡng.", "Lặp lại với hộp điểm cao nhất còn lại.", "IoU bằng ngưỡng thì **không** bị loại (điều kiện là lớn hơn ngưỡng)."],
              steps=[Step(text="Thứ tự điểm giảm dần: " + ", ".join(str(i + 1) for i in order) + ". Các hộp được giữ: " + ", ".join(str(i + 1) for i in keep) + ".")], verified=["greedy-nms-two-ways"], tags=["nms"])


# --------------------------------------------------------------------- set-match
def gen_set_match(rng: random.Random, level: int, seed: int):
    n = {1: 2, 2: 3, 3: 2, 4: 3, 5: 4}[level]
    if level in (1, 2, 5):
        for _ in range(500):
            c = [[F(rng.randint(1, 9)) for _ in range(n)] for _ in range(n)]
            costs = sorted(sum(c[i][p[i]] for i in range(n)) for p in permutations(range(n)))
            if costs[0] != costs[1]:
                break
        best = costs[0]
        # kiểm chứng độc lập: quy hoạch động trên tập cột đã dùng
        dp = {0: F(0)}
        for i in range(n):
            nd = {}
            for mask, v in dp.items():
                for j in range(n):
                    if not mask >> j & 1:
                        nm = mask | 1 << j
                        nd[nm] = min(nd.get(nm, F(10**9)), v + c[i][j])
            dp = nd
        check(dp[(1 << n) - 1] == best, "DP ≠ vét cạn")
        bp = min(permutations(range(n)), key=lambda p: sum(c[i][p[i]] for i in range(n)))
        return cv(type_="set-match", lesson=L86, level=level, seed=seed,
                  prompt=Prompt(text=f"DETR ghép mỗi dự đoán với đúng một vật thật (một-một) sao cho tổng chi phí nhỏ nhất (Hungarian matching). Ma trận chi phí $\\mathbf{{C}}$ (hàng: dự đoán, cột: vật thật). Tính tổng chi phí nhỏ nhất. " + NUM_HINT,
                                display=["\\mathbf{C} = " + mat_tex(c)]),
                  answer=num_ans(best), hints=["Thử mọi hoán vị (có $n!$ cách); với ma trận nhỏ liệt kê hết.", "Mỗi dự đoán và mỗi vật thật dùng đúng một lần.", "Chọn phần tử nhỏ nhất của từng hàng chưa chắc hợp lệ vì hai hàng có thể chọn cùng một cột."],
                  steps=[Step(text="Ghép tối ưu (dự đoán $\\to$ vật thật): " + ", ".join(f"{i + 1}\\to{bp[i] + 1}" for i in range(n)).replace("\\to", "→") + ".", latex=f"\\min = {ft(best)}")], verified=["brute-force-and-dp"], tags=["hungarian", f"{n}x{n}"])
    # level 3-4: chi phí = -p(lớp) + lambda * L1(hộp)
    for _ in range(500):
        probs = [[F(rng.randint(1, 9), 10) for _ in range(n)] for _ in range(n)]  # p dự đoán i cho lớp của vật thật j
        dist = [[F(rng.randint(0, 6)) for _ in range(n)] for _ in range(n)]
        lam = F(rng.choice([1, 2]), 10)
        c = [[-probs[i][j] + lam * dist[i][j] for j in range(n)] for i in range(n)]
        costs = sorted(sum(c[i][p[i]] for i in range(n)) for p in permutations(range(n)))
        if len(costs) < 2 or costs[0] != costs[1]:
            break
    best = costs[0]
    text = (f"Chi phí ghép của DETR $C_{{ij}} = -p_{{ij}} + \\lambda\\,d_{{ij}}$, với $p_{{ij}}$ là xác suất dự đoán $i$ cho lớp của vật thật $j$ và $d_{{ij}}$ là khoảng cách $L_1$ giữa hộp, $\\lambda = {ft(lam)}$. "
            "Tìm tổng chi phí nhỏ nhất của phép ghép một-một. " + NUM_HINT)
    return cv(type_="set-match", lesson=L86, level=level, seed=seed,
              prompt=Prompt(text=text, display=["\\mathbf{P} = " + mat_tex(probs) + ",\\quad \\mathbf{D} = " + mat_tex(dist)]), answer=num_ans(best),
              hints=["Dựng $\\mathbf{C} = -\\mathbf{P} + \\lambda\\mathbf{D}$ trước.", "Liệt kê các hoán vị và chọn tổng nhỏ nhất.", "Chi phí có thể âm (vì có số hạng $-p$)."],
              steps=[Step(text="Ma trận chi phí $\\mathbf{C}$ và tổng nhỏ nhất:", latex="\\mathbf{C} = " + mat_tex(c) + f",\\quad \\min = {ft(best)}")], verified=["brute-force"], tags=["hungarian", "cost"])


# --------------------------------------------------------------------- seg-metrics
def gen_seg_metrics(rng: random.Random, level: int, seed: int):
    n = 4
    ncls = 3 if level == 4 else 2
    if level == 5:
        den = rng.randint(2, 60)
        iou = F(rng.randint(1, den - 1), den)
        val = 2 * iou / (1 + iou)
        # kiểm chứng: dựng hai tập có IoU cho trước rồi tính Dice theo định nghĩa
        inter_, uni_ = iou.numerator, iou.denominator
        sa_ = inter_ + (uni_ - inter_) // 2
        sb_ = uni_ + inter_ - sa_
        check(2 * F(inter_, 1) / (sa_ + sb_) == val, "Dice theo định nghĩa ≠ 2 IoU/(1+IoU)")
        return cv(type_="seg-metrics", lesson=L87, level=level, seed=seed,
                  prompt=Prompt(text=f"Với một mặt nạ dự đoán, IoU với mặt nạ thật là ${ft(iou)}$. Tính hệ số Dice (F1 trên pixel), biết $\\mathrm{{Dice}} = \\dfrac{{2\\,\\mathrm{{IoU}}}}{{1 + \\mathrm{{IoU}}}}$. " + NUM_HINT),
                  answer=num_ans(val), hints=["$\\mathrm{Dice} = 2|A\\cap B|/(|A| + |B|)$ và $|A| + |B| = |A\\cup B| + |A\\cap B|$.", "Chia tử và mẫu cho $|A\\cup B|$ để đưa về IoU."],
                  steps=[Step(text="Thế vào:", latex=f"\\dfrac{{2\\cdot {ft(iou)}}}{{1 + {ft(iou)}}} = {ft(val)}")], verified=["identity-check"], tags=["dice"])
    for _ in range(500):
        gt = [[rng.randrange(ncls) for _ in range(n)] for _ in range(n)]
        pr = [[g if rng.random() < 0.65 else rng.randrange(ncls) for g in r] for r in gt]
        g, p = np.array(gt), np.array(pr)
        if level <= 3 and (g == 1).sum() > 0 and (p == 1).sum() > 0 and (g != p).sum() > 0:
            break
        if level == 4 and all((g == c).sum() + (p == c).sum() > 0 for c in range(ncls)) and (g != p).sum() > 0:
            break
    fm = lambda m: "\\begin{bmatrix} " + " \\\\ ".join(" & ".join(str(v) for v in r) for r in m) + " \\end{bmatrix}"
    disp = [f"\\text{{GT}} = {fm(gt)},\\quad \\text{{Pred}} = {fm(pr)}"]
    if level == 1:
        val = F(int((g == p).sum()), n * n)
        text, steps = "Hai bản đồ nhãn pixel $4\\times 4$ (nhãn lớp $0$ hoặc $1$): thật (GT) và dự đoán (Pred). Tính độ chính xác pixel (pixel accuracy). " + NUM_HINT, [Step(text=f"Số pixel đúng ${int((g == p).sum())}$ trên ${n * n}$:", latex=f"{ft(val)}")]
    elif level in (2, 3):
        inter = int(((g == 1) & (p == 1)).sum())
        union = int(((g == 1) | (p == 1)).sum())
        sa, sb = int((g == 1).sum()), int((p == 1).sum())
        if level == 2:
            val = F(inter, union)
            text, steps = "Hai bản đồ nhãn $4\\times 4$ (lớp $0$ và $1$). Tính IoU của lớp $1$: $|GT\\cap Pred| / |GT\\cup Pred|$ trên các pixel mang nhãn $1$. " + NUM_HINT, [Step(text=f"Giao ${inter}$, hợp ${union}$:", latex=f"\\mathrm{{IoU}} = {ft(val)}")]
        else:
            val = F(2 * inter, sa + sb)
            text, steps = "Hai bản đồ nhãn $4\\times 4$ (lớp $0$ và $1$). Tính hệ số Dice của lớp $1$: $2|GT\\cap Pred| / (|GT| + |Pred|)$. " + NUM_HINT, [Step(text=f"$|GT\\cap Pred| = {inter}$, $|GT| = {sa}$, $|Pred| = {sb}$:", latex=f"\\mathrm{{Dice}} = {ft(val)}")]
    else:
        ious = []
        for c in range(ncls):
            i_, u_ = int(((g == c) & (p == c)).sum()), int(((g == c) | (p == c)).sum())
            ious.append(F(i_, u_))
        val = sum(ious) / ncls
        text, steps = "Hai bản đồ nhãn $4\\times 4$ với $3$ lớp ($0, 1, 2$). Tính mIoU: trung bình cộng IoU của từng lớp. " + NUM_HINT, [Step(text="IoU từng lớp rồi trung bình:", latex="(" + ", ".join(ft(v) for v in ious) + f")\\Rightarrow\\ \\mathrm{{mIoU}} = {ft(val)}")]
    return cv(type_="seg-metrics", lesson=L87, level=level, seed=seed, prompt=Prompt(text=text, display=disp), answer=num_ans(val),
              hints=["Phân đoạn gán một nhãn cho **mỗi pixel**.", "IoU $= |A\\cap B|/|A\\cup B|$, Dice $= 2|A\\cap B|/(|A| + |B|)$; mIoU là trung bình IoU các lớp.", "Đếm pixel bằng cách so sánh hai ma trận từng ô."],
              steps=steps, verified=["numpy-masks"], tags=["seg", f"level{level}"])


# ------------------------------------------------------------------ clip-retrieval
UNITS = [(F(1), F(0)), (F(0), F(1)), (F(3, 5), F(4, 5)), (F(4, 5), F(3, 5)), (F(-3, 5), F(4, 5)), (F(5, 13), F(12, 13)), (F(12, 13), F(5, 13)), (F(-4, 5), F(3, 5)), (F(0), F(-1)), (F(-1), F(0))]


def gen_clip_retrieval(rng: random.Random, level: int, seed: int):
    n = 3
    img = rng.sample(UNITS, n)
    txt = rng.sample(UNITS, n)
    sim = [[a[0] * b[0] + a[1] * b[1] for b in txt] for a in img]  # cosine vì đã chuẩn hóa
    for v in img + txt:
        check(v[0] ** 2 + v[1] ** 2 == 1, "vector đơn vị")
    check(np.allclose(_arr(sim), _arr(img) @ _arr(txt).T), "NumPy")
    base = f"Mô hình kiểu CLIP: embedding ảnh đã chuẩn hóa $\\mathbf{{i}}_1, \\mathbf{{i}}_2, \\mathbf{{i}}_3$ và embedding chú thích $\\mathbf{{t}}_1, \\mathbf{{t}}_2, \\mathbf{{t}}_3$ (ảnh $k$ khớp đúng chú thích $k$). Ảnh: ${mat_tex([list(v) for v in img])}$ (mỗi hàng một ảnh), chú thích: ${mat_tex([list(v) for v in txt])}$. "
    argmax_row = [max(range(n), key=lambda j: sim[i][j]) for i in range(n)]
    argmax_col = [max(range(n), key=lambda i: sim[i][j]) for j in range(n)]
    tie = any(sorted(sim[i], reverse=True)[0] == sorted(sim[i], reverse=True)[1] for i in range(n)) or any(sorted([sim[i][j] for i in range(n)], reverse=True)[0] == sorted([sim[i][j] for i in range(n)], reverse=True)[1] for j in range(n))
    if tie:
        return gen_clip_retrieval(random.Random(rng.random()), level, seed)
    if level == 1:
        i, j = rng.randrange(n), rng.randrange(n)
        val, text = sim[i][j], base + f"Độ tương tự cosine giữa ảnh ${i + 1}$ và chú thích ${j + 1}$ là bao nhiêu? " + NUM_HINT
        ans, steps = num_ans(val), [Step(text="Vector đã chuẩn hóa nên cosine bằng tích vô hướng:", latex=f"\\mathbf{{i}}_{i + 1}\\T\\mathbf{{t}}_{j + 1} = {ft(val)}")]
    elif level == 2:
        text = base + "Tính ma trận độ tương tự $\\mathbf{S} = \\mathbf{I}\\mathbf{T}\\T$ ($3\\times 3$, hàng: ảnh, cột: chú thích). Nhập từng phần tử."
        ans, steps = mat_ans(sim), [Step(text="$S_{ij} = \\mathbf{i}_i\\T\\mathbf{t}_j$:", latex="\\mathbf{S} = " + mat_tex(sim))]
    elif level == 3:
        i = rng.randrange(n)
        label = f"Chú thích {argmax_row[i] + 1}"
        choices, ans = choice_exercise(rng, label, [f"Chú thích {j + 1}" for j in range(n) if j != argmax_row[i]])
        text = base + f"Truy hồi ảnh $\\to$ chú thích: chú thích nào có độ tương tự cao nhất với ảnh ${i + 1}$?"
        return cv(type_="clip-retrieval", lesson=L88, level=level, seed=seed, prompt=Prompt(text=text), answer=ans, choices=choices,
                  hints=["Tính hàng $i$ của $\\mathbf{S} = \\mathbf{I}\\mathbf{T}\\T$ rồi lấy chỉ số lớn nhất.", "Vector đã chuẩn hóa nên cosine = tích vô hướng."],
                  steps=[Step(text="Hàng độ tương tự: $" + vt(sim[i]) + f"$; lớn nhất ở chú thích {argmax_row[i] + 1}.")], verified=["numpy-argmax"], tags=["retrieval"])
    elif level == 4:
        correct = sum(1 for i in range(n) if argmax_row[i] == i)
        val = F(correct, n)
        text = base + "Độ chính xác top-1 của truy hồi ảnh $\\to$ chú thích (mỗi ảnh chọn chú thích tương tự nhất; đúng nếu trùng chú thích cùng chỉ số) là bao nhiêu? " + NUM_HINT
        ans, steps = num_ans(val), [Step(text="Chú thích được chọn cho từng ảnh: " + ", ".join(str(j + 1) for j in argmax_row) + f"; đúng ${correct}/{n}$.", latex=f"\\text{{acc}} = {ft(val)}")]
    else:
        tau = F(rng.choice([1, 2, 5]), 10)
        logits = [[v / tau for v in r] for r in sim]
        c1 = sum(1 for i in range(n) if argmax_row[i] == i)
        c2 = sum(1 for j in range(n) if argmax_col[j] == j)
        val = F(c1 + c2, 2 * n)
        text = (base + f"CLIP chia điểm cho nhiệt độ $\\tau = {ft(tau)}$ (logit $= S_{{ij}}/\\tau$; không đổi thứ tự trong mỗi hàng/cột). Tính độ chính xác top-1 **trung bình hai chiều**: (số ảnh truy hồi đúng chú thích + số chú thích truy hồi đúng ảnh) / $2n$. " + NUM_HINT)
        ans, steps = num_ans(val), [Step(text=f"Logit $\\mathbf{{S}}/\\tau = {mat_tex(logits)}$. Ảnh→chú thích đúng ${c1}$, chú thích→ảnh đúng ${c2}$:", latex=f"\\dfrac{{{c1} + {c2}}}{{{2 * n}}} = {ft(val)}")]
    return cv(type_="clip-retrieval", lesson=L88, level=level, seed=seed, prompt=Prompt(text=text), answer=ans,
              hints=["Ảnh và chú thích được chiếu vào cùng một không gian; cặp khớp có cosine cao.", "Ma trận độ tương tự: $\\mathbf{S} = \\mathbf{I}\\mathbf{T}\\T$; truy hồi là argmax theo hàng (ảnh→chữ) hoặc theo cột (chữ→ảnh).", "Chia cho nhiệt độ $\\tau$ không đổi argmax."],
              steps=steps, verified=["numpy"], tags=["clip", f"level{level}"])


# ------------------------------------------------------------- diffusion-forward
ABARS = [(F(9, 25), F(3, 5), F(4, 5)), (F(16, 25), F(4, 5), F(3, 5)), (F(25, 169), F(5, 13), F(12, 13)), (F(144, 169), F(12, 13), F(5, 13)), (F(64, 289), F(8, 17), F(15, 17)), (F(225, 289), F(15, 17), F(8, 17))]


def gen_diffusion_forward(rng: random.Random, level: int, seed: int):
    abar, sa, sn = rng.choice(ABARS)
    check(sa * sa == abar and sn * sn == 1 - abar, "căn hữu tỉ")
    d = 1 if level == 1 else 3
    x0 = [F(rng.randint(-5, 5)) for _ in range(d)]
    eps = [F(rng.randint(-3, 3)) for _ in range(d)]
    xt = [sa * a + sn * b for a, b in zip(x0, eps)]
    if level in (1, 2):
        ans = num_ans(xt[0]) if d == 1 else vec_ans(xt)
        fm = (lambda a: ft(a[0])) if d == 1 else vt
        text = (f"Quá trình khuếch tán thuận (DDPM): $\\mathbf{{x}}_t = \\sqrt{{\\bar{{\\alpha}}_t}}\\,\\mathbf{{x}}_0 + \\sqrt{{1 - \\bar{{\\alpha}}_t}}\\,\\boldsymbol{{\\varepsilon}}$ với $\\bar{{\\alpha}}_t = {ft(abar)}$, "
                f"$\\mathbf{{x}}_0 = {fm(x0)}$ và nhiễu $\\boldsymbol{{\\varepsilon}} = {fm(eps)}$. Tính $\\mathbf{{x}}_t$. " + (NUM_HINT if d == 1 else VEC_HINT))
        steps = [Step(text=f"$\\sqrt{{\\bar{{\\alpha}}_t}} = {ft(sa)}$ và $\\sqrt{{1 - \\bar{{\\alpha}}_t}} = {ft(sn)}$:", latex="\\mathbf{x}_t = " + fm(xt))]
    elif level == 3:
        ts = rng.randint(2, 4)
        betas = [F(rng.randint(1, 9), 100) for _ in range(ts)]
        val = F(1)
        for b in betas:
            val *= (1 - b)
        check(abs(float(val) - float(np.prod([1 - float(b) for b in betas]))) < 1e-12, "tích NumPy")
        text = f"Lịch nhiễu DDPM: $\\alpha_s = 1 - \\beta_s$ và $\\bar{{\\alpha}}_t = \\prod_{{s=1}}^{{t}}\\alpha_s$. Với $\\beta = ({', '.join(ft(b) for b in betas)})$, tính $\\bar{{\\alpha}}_{ts}$. " + NUM_HINT
        ans = num_ans(val)
        steps = [Step(text="Nhân các $\\alpha_s = 1 - \\beta_s$:", latex=" \\cdot ".join(f"{ft(1 - b)}" for b in betas) + f" = {ft(val)}")]
    elif level == 4:
        text = (f"Cho $\\bar{{\\alpha}}_t = {ft(abar)}$, $\\mathbf{{x}}_0 = {vt(x0)}$ và $\\mathbf{{x}}_t = {vt(xt)}$ (với $\\mathbf{{x}}_t = \\sqrt{{\\bar{{\\alpha}}_t}}\\mathbf{{x}}_0 + \\sqrt{{1 - \\bar{{\\alpha}}_t}}\\boldsymbol{{\\varepsilon}}$). Tìm nhiễu $\\boldsymbol{{\\varepsilon}}$ đã thêm. " + VEC_HINT)
        ans = vec_ans(eps)
        steps = [Step(text="Giải $\\boldsymbol{\\varepsilon} = (\\mathbf{x}_t - \\sqrt{\\bar{\\alpha}_t}\\mathbf{x}_0)/\\sqrt{1 - \\bar{\\alpha}_t}$:", latex="\\boldsymbol{\\varepsilon} = " + vt(eps))]
    else:
        x0hat = [(a - sn * b) / sa for a, b in zip(xt, eps)]
        check(x0hat == x0, "dự đoán x0 khớp")
        text = (f"Mạng khử nhiễu dự đoán nhiễu $\\hat{{\\boldsymbol{{\\varepsilon}}}} = {vt(eps)}$ cho $\\mathbf{{x}}_t = {vt(xt)}$ với $\\bar{{\\alpha}}_t = {ft(abar)}$. Ước lượng $\\hat{{\\mathbf{{x}}}}_0 = (\\mathbf{{x}}_t - \\sqrt{{1 - \\bar{{\\alpha}}_t}}\\,\\hat{{\\boldsymbol{{\\varepsilon}}}})/\\sqrt{{\\bar{{\\alpha}}_t}}$. " + VEC_HINT)
        ans = vec_ans(x0hat)
        steps = [Step(text=f"$\\sqrt{{\\bar{{\\alpha}}_t}} = {ft(sa)}$, $\\sqrt{{1 - \\bar{{\\alpha}}_t}} = {ft(sn)}$:", latex="\\hat{\\mathbf{x}}_0 = " + vt(x0hat))]
    return cv(type_="diffusion-forward", lesson=L89, level=level, seed=seed, prompt=Prompt(text=text), answer=ans,
              hints=["$\\bar{\\alpha}_t$ giảm dần theo $t$: $t$ càng lớn thì tín hiệu gốc càng bị che bởi nhiễu.", "Khuếch tán thuận có công thức đóng: $\\mathbf{x}_t$ tính trực tiếp từ $\\mathbf{x}_0$ và một nhiễu Gauss $\\boldsymbol{\\varepsilon}$.", "Đề chọn $\\bar{\\alpha}_t$ để cả $\\sqrt{\\bar{\\alpha}_t}$ và $\\sqrt{1-\\bar{\\alpha}_t}$ là số hữu tỉ (bộ ba Pythagoras)."],
              steps=steps, verified=["roundtrip-and-numpy"], tags=["diffusion", f"level{level}"])


SPECS = [
    TopicSpec("vit-count", L84, (1, 2, 3, 4, 5), gen_vit_count),
    TopicSpec("aug-transform", L85, (1, 2, 3, 4, 5), gen_aug_transform),
    TopicSpec("transfer-params", L85, (1, 2, 3, 4, 5), gen_transfer_params),
    TopicSpec("iou-nms", L86, (1, 2, 3, 4, 5), gen_iou_nms),
    TopicSpec("set-match", L86, (1, 2, 3, 4, 5), gen_set_match),
    TopicSpec("seg-metrics", L87, (1, 2, 3, 4, 5), gen_seg_metrics),
    TopicSpec("clip-retrieval", L88, (1, 2, 3, 4, 5), gen_clip_retrieval),
    TopicSpec("diffusion-forward", L89, (1, 2, 3, 4, 5), gen_diffusion_forward),
]
