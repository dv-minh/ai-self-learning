r"""Ví dụ cho Phần 8 (Computer Vision). Mọi số do chương trình tính bằng Fraction và kiểm chứng; MDX chỉ chứa kết quả."""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations, permutations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..core.verify import check
from ..topics.cv.c8_1_3_cnn import _conv, _pad
from ..topics.cv.c8_4_9_modern import _iou, _iou_pixels
from ..topics.opt._o import fp, ft, mat_tex, vt
from .common import display, write_examples

EX: dict[str, dict] = {}


def lesson(code: str):
    def deco(fn):
        EX.setdefault(code, {})[fn.__name__.replace("_", "-")] = fn
        return fn
    return deco


def table(header: list[str], rows: list[list[str]]) -> str:
    cols = "c" * len(header)
    body = " \\\\ ".join(" & ".join(r) for r in rows)
    return f"\\begin{{array}}{{{cols}}} {' & '.join(header)} \\\\ \\hline {body} \\end{{array}}"


def _arr(m):
    return np.array([[float(v) for v in r] for r in m])


def _m(rows):
    return [[F(v) for v in r] for r in rows]


# ------------------------------------------------------------------ 8.1
IMG = _m([[1, 2, 0, 1], [3, 1, 2, 2], [0, 4, 1, 0], [2, 1, 3, 1]])
KER = _m([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


@lesson("8.1")
def conv_worked():
    out = _conv(IMG, KER)
    w = sliding_window_view(_arr(IMG), (3, 3))
    check(np.allclose((w * _arr(KER)).sum(axis=(2, 3)), _arr(out)), "NumPy")
    win = [IMG[a][b] for a in range(3) for b in range(3)]
    return display("\\mathbf{X} = " + mat_tex(IMG) + ",\\quad \\mathbf{K} = " + mat_tex(KER), f"Y_{{11}} = \\sum_{{a,b}} K_{{ab}}X_{{a,b}} = 1\\cdot 1 + 0\\cdot 2 + (-1)\\cdot 0 + 2\\cdot 3 + 0\\cdot 1 + (-2)\\cdot 2 + 1\\cdot 0 + 0\\cdot 4 + (-1)\\cdot 1 = {ft(out[0][0])}", "\\mathbf{Y} = " + mat_tex(out)) + \
        "\n\nKernel này (Sobel dọc) cho giá trị lớn ở nơi cường độ thay đổi theo chiều ngang, tức phát hiện cạnh đứng."


@lesson("8.1")
def conv_padding():
    img = IMG[:3]
    img = [r[:3] for r in img]
    ker = _m([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    out = _conv(_pad(img, 1), ker)
    a = np.pad(_arr(img), 1)
    check(np.allclose((sliding_window_view(a, (3, 3)) * _arr(ker)).sum(axis=(2, 3)), _arr(out)), "NumPy")
    return display("\\mathbf{X} = " + mat_tex(img) + ",\\quad \\mathbf{K} = " + mat_tex(ker) + ",\\quad p = 1", "\\text{padded } \\mathbf{X} = " + mat_tex(_pad(img, 1)), "\\mathbf{Y} = " + mat_tex(out)) + "\n\nPadding $1$ với kernel $3\\times 3$ giữ nguyên kích thước $3\\times 3$ (\"same\")."


@lesson("8.1")
def shape_table():
    rows = []
    for n, k, p, s in [(32, 3, 1, 1), (32, 3, 0, 1), (32, 5, 2, 2), (224, 7, 3, 2), (28, 3, 1, 2)]:
        out = (n + 2 * p - k) // s + 1
        pos = sum(1 for st in range(0, n + 2 * p, s) if st + k - 1 < n + 2 * p)
        check(pos == out, "đếm cửa sổ")
        rows.append([str(n), str(k), str(p), str(s), str(out)])
    return display("\\text{out} = \\left\\lfloor \\dfrac{n + 2p - k}{s} \\right\\rfloor + 1", table(["n", "k", "p", "s", "\\mathrm{out}"], rows))


@lesson("8.1")
def prop_translation():
    x = _m([[0, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    xs = _m([[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    k = _m([[1, 2], [3, 4]])
    y, ys = _conv(x, k), _conv(xs, k)
    check(all(ys[i][j + 1] == y[i][j] for i in range(3) for j in range(2)), "đồng biến với phép tịnh tiến")
    return display("\\mathbf{K} = " + mat_tex(k), "\\mathbf{Y}(\\mathbf{X}) = " + mat_tex(y), "\\mathbf{Y}(\\text{shifted }\\mathbf{X}) = " + mat_tex(ys)) + "\n\nDịch ảnh một cột thì đầu ra cũng dịch một cột (tính đồng biến tịnh tiến): cùng một mẫu hình được phát hiện ở bất kỳ vị trí nào."


@lesson("8.1")
def mistake_flip():
    x = _m([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = _m([[1, 0, 0], [0, 0, 0], [0, 0, 2]])
    corr = _conv(x, k)[0][0]
    kf = [r[::-1] for r in k[::-1]]
    true_conv = _conv(x, kf)[0][0]
    return f"**Sai:** khi tính tay, lật kernel (như tích chập toán học) trong khi framework dùng tương quan chéo.\n\nVới $\\mathbf{{X}} = {mat_tex(x)}$, $\\mathbf{{K}} = {mat_tex(k)}$: tương quan chéo (không lật, đúng như `nn.Conv2d`) cho ${ft(corr)}$, còn lật kernel cho ${ft(true_conv)}$.\n\n**Đúng:** trong học sâu, \"tích chập\" là tương quan chéo: **không** lật kernel (kernel là tham số học được nên việc lật không quan trọng)."


# ------------------------------------------------------------------ 8.2
@lesson("8.2")
def maxpool_example():
    out = [[max(IMG[2 * i + a][2 * j + b] for a in range(2) for b in range(2)) for j in range(2)] for i in range(2)]
    check(np.allclose(sliding_window_view(_arr(IMG), (2, 2))[::2, ::2].max(axis=(2, 3)), _arr(out)), "NumPy")
    return display("\\mathbf{X} = " + mat_tex(IMG), "\\text{max pool } 2\\times 2,\\ s = 2:\\quad " + mat_tex(out))


@lesson("8.2")
def avgpool_example():
    out = [[sum(IMG[2 * i + a][2 * j + b] for a in range(2) for b in range(2)) / 4 for j in range(2)] for i in range(2)]
    check(np.allclose(sliding_window_view(_arr(IMG), (2, 2))[::2, ::2].mean(axis=(2, 3)), _arr(out)), "NumPy")
    return display("\\text{average pool } 2\\times 2,\\ s = 2:\\quad " + mat_tex(out))


@lesson("8.2")
def receptive_field_example():
    layers = [(3, 1), (3, 1), (3, 2), (3, 1)]
    r, j = 1, 1
    rows = []
    for i, (k, s) in enumerate(layers, 1):
        r += (k - 1) * j
        j *= s
        rows.append([str(i), str(k), str(s), str(r), str(j)])
    lo, hi = 0, 0
    for k, s in reversed(layers):
        lo, hi = lo * s, hi * s + (k - 1)
    check(hi - lo + 1 == r, "ngược")
    return display("r \\leftarrow r + (k - 1)\\,j,\\quad j \\leftarrow j\\,s", table(["\\text{layer}", "k", "s", "r", "j"], rows)) + "\n\nStride $2$ làm bước nhảy $j$ tăng gấp đôi nên các lớp sau mở rộng trường tiếp nhận nhanh hơn."


@lesson("8.2")
def prop_two_3x3():
    r = 1 + 2 + 2
    return display(f"\\text{{two }}3\\times 3\\ (s = 1):\\ r = 1 + 2 + 2 = {r},\\qquad \\text{{one }}5\\times 5:\\ r = 1 + 4 = 5") + f"\n\nHai lớp $3\\times 3$ có trường tiếp nhận bằng một lớp $5\\times 5$ nhưng ít tham số hơn (${2 * 9}$ so với ${5 * 5}$ trọng số mỗi cặp kênh) và có thêm một phi tuyến."


@lesson("8.2")
def mistake_pool_params():
    return "**Sai:** nghĩ rằng pooling có tham số học được hoặc làm tăng số kênh.\n\n**Đúng:** max/average pooling không có tham số, giữ nguyên số kênh và chỉ giảm kích thước không gian (cạnh chia theo stride)."


# ------------------------------------------------------------------ 8.3
@lesson("8.3")
def params_compare():
    cin, cout, k = 64, 128, 3
    std = cin * cout * k * k
    sep = cin * k * k + cin * cout
    return display(f"C_{{\\mathrm{{in}}}} = {cin},\\ C_{{\\mathrm{{out}}}} = {cout},\\ k = {k}", f"\\text{{standard}} = {cin}\\cdot {cout}\\cdot {k}^2 = {std}", f"\\text{{depthwise separable}} = {cin}\\cdot {k}^2 + {cin}\\cdot {cout} = {sep}\\quad(\\approx {ft(F(sep, std))}\\ \\text{{of standard}})")


@lesson("8.3")
def resnet_block_params():
    c = 64
    basic = 2 * 9 * c * c
    bottleneck = 4 * c * c + 9 * c * c + 4 * c * c
    return display(f"C = {c}", f"\\text{{basic}} = 2\\cdot 9C^2 = {basic}\\ (\\text{{input and output }} {c}\\text{{ channels}})", f"\\text{{bottleneck}} = 4C^2 + 9C^2 + 4C^2 = {bottleneck}\\ (\\text{{input and output }} {4 * c}\\text{{ channels}})") + "\n\nBottleneck nén xuống $C$ kênh trước khi làm conv $3\\times 3$ đắt đỏ rồi mở lại $4C$ kênh, nên làm việc với biểu diễn rộng hơn mà vẫn tiết kiệm."


@lesson("8.3")
def cnn_shapes():
    n = 224
    rows = []
    c = 3
    for name, k, s, p, cout in [("conv1", 7, 2, 3, 64), ("pool", 3, 2, 1, 64), ("conv2", 3, 1, 1, 64), ("conv3", 3, 2, 1, 128)]:
        n = (n + 2 * p - k) // s + 1
        c = cout
        rows.append([f"\\mathrm{{{name}}}", f"{k}/{s}/{p}", f"{c}\\times {n}\\times {n}"])
    return display("224\\times 224\\times 3\\ \\text{input}", table(["\\text{layer}", "k/s/p", "C\\times H\\times W"], rows))


@lesson("8.3")
def prop_skip_gradient():
    return display("y = x + F(x) \\Rightarrow \\dfrac{\\partial L}{\\partial x} = \\dfrac{\\partial L}{\\partial y}\\left(1 + \\dfrac{\\partial F}{\\partial x}\\right)") + "\n\nNhánh tắt luôn đưa gradient từ $y$ về $x$ không bị suy giảm; đó là lý do ResNet huấn luyện được hàng trăm lớp (bài 7.5)."


@lesson("8.3")
def mistake_param_depends_on_size():
    c, k = 64, 3
    p = c * c * k * k
    return f"**Sai:** nghĩ rằng số tham số của lớp tích chập phụ thuộc kích thước ảnh. Với ảnh $32\\times 32$ hay $224\\times 224$, lớp conv $3\\times 3$ ${c}\\to {c}$ vẫn có đúng ${p}$ trọng số.\n\n**Đúng:** tham số chỉ phụ thuộc $C_{{\\mathrm{{in}}}}$, $C_{{\\mathrm{{out}}}}$, $k$; kích thước ảnh chỉ làm thay đổi **số phép tính** (MAC) và bộ nhớ kích hoạt."


# ------------------------------------------------------------------ 8.4
@lesson("8.4")
def vit_numbers():
    H, P, D = 224, 16, 768
    N = (H // P) ** 2
    embed = (P * P * 3 + 1) * D
    return display(f"H = W = {H},\\ P = {P},\\ D = {D}", f"N = (H/P)^2 = {H // P}^2 = {N},\\quad \\text{{tokens}} = N + 1 = {N + 1}", f"\\text{{patch embedding}} = ({P}^2\\cdot 3 + 1)\\cdot {D} = {embed}")


@lesson("8.4")
def vit_attention_cost():
    rows = []
    for P in (32, 16, 8):
        N = (224 // P) ** 2
        rows.append([str(P), str(N), str((N + 1) ** 2)])
    return display(table(["P", "N", "(N+1)^2"], rows)) + "\n\nPatch nhỏ cho nhiều token hơn (chi tiết hơn) nhưng ma trận attention tăng theo bình phương số token."


@lesson("8.4")
def patch_as_conv():
    P, D = 16, 4
    x = np.arange(3 * P * P, dtype=float).reshape(3, P, P)
    w = np.ones((D, 3, P, P))
    patch_vec = x.reshape(-1)
    lin = np.array([w[d].reshape(-1) @ patch_vec for d in range(D)])
    conv = np.array([(w[d] * x).sum() for d in range(D)])
    check(np.allclose(lin, conv), "linear trên patch duỗi = conv kernel P stride P")
    return display(f"\\text{{patch}}\\ {P}\\times {P}\\times 3\\ \\xrightarrow{{\\text{{flatten}}}}\\ {P * P * 3}\\ \\xrightarrow{{\\text{{linear}}}}\\ D") + "\n\nLớp tuyến tính trên patch đã duỗi cho kết quả đúng bằng một conv kernel $P\\times P$, stride $P$ (đã kiểm tra bằng số)."


@lesson("8.4")
def mistake_non_divisible():
    return "**Sai:** dùng patch $16$ cho ảnh $230\\times 230$: $230/16$ không nguyên, một phần ảnh bị bỏ hoặc phải đệm.\n\n**Đúng:** chọn kích thước ảnh chia hết cho $P$ (resize hoặc crop về $224$), hoặc đệm/cắt có chủ ý; số patch là $(H/P)(W/P)$ khi chia hết."


# ------------------------------------------------------------------ 8.5
@lesson("8.5")
def aug_example():
    x = _m([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    a = _arr(x)
    fh, fv, rot = np.fliplr(a), np.flipud(a), np.rot90(a, -1)
    return display("\\mathbf{X} = " + mat_tex(x), "\\text{flip-h} = " + mat_tex(_m(fh.astype(int))), "\\text{flip-v} = " + mat_tex(_m(fv.astype(int))), "\\text{rot90 (cw)} = " + mat_tex(_m(rot.astype(int))))


@lesson("8.5")
def transfer_example():
    feat, ncls = 2048, 10
    head = (feat + 1) * ncls
    backbone = 25_000_000
    return display(f"\\text{{backbone}} = {backbone}\\ (\\text{{frozen}}),\\quad \\text{{head}} = ({feat} + 1)\\cdot {ncls} = {head}", f"\\text{{trainable ratio}} = \\dfrac{{{head}}}{{{backbone + head}}} = {ft(F(head, backbone + head))}") + "\n\nLinear probe chỉ huấn luyện khoảng $0{,}08\\%$ tham số, nhanh và ít overfit khi dữ liệu mới ít."


@lesson("8.5")
def label_preserving():
    return display("\\text{image } 6\\ \\xrightarrow{\\text{flip-h}}\\ \\text{still } 6?\\qquad \\text{digit } 6\\ \\xrightarrow{\\text{rot }180^\\circ}\\ 9") + "\n\nPhép biến đổi phải **giữ nhãn**: lật ngang một con mèo vẫn là con mèo, nhưng quay $180^\\circ$ chữ số $6$ thành $9$ làm đổi nhãn."


@lesson("8.5")
def mistake_augment_test():
    return "**Sai:** áp dụng augmentation ngẫu nhiên lên tập validation/test (hoặc dùng cùng phép biến đổi cho cả hai).\n\n**Đúng:** chỉ augment tập **huấn luyện**; tập validation/test chỉ qua tiền xử lý xác định (resize, chuẩn hóa), để số đo công bằng và lặp lại được."


# ------------------------------------------------------------------ 8.6
@lesson("8.6")
def iou_example():
    a, b = [F(0), F(0), F(4), F(4)], [F(2), F(1), F(6), F(5)]
    iou, inter, ua = _iou(a, b)
    check(iou == _iou_pixels(a, b), "IoU đếm ô")
    return display("A = (0, 0, 4, 4),\\ B = (2, 1, 6, 5)", f"|A\\cap B| = (4 - 2)(4 - 1) = {ft(inter)},\\quad |A\\cup B| = 16 + 16 - {ft(inter)} = {ft(ua)}", f"\\mathrm{{IoU}} = {ft(iou)}")


@lesson("8.6")
def nms_example():
    boxes = [[F(0), F(0), F(4), F(4)], [F(1), F(0), F(5), F(4)], [F(8), F(8), F(12), F(12)], [F(0), F(1), F(4), F(5)]]
    scores = [F(9, 10), F(8, 10), F(7, 10), F(6, 10)]
    thr = F(1, 2)
    order = sorted(range(4), key=lambda i: -scores[i])
    keep = []
    rows = []
    for i in order:
        ok = all(_iou(boxes[i], boxes[k])[0] <= thr for k in keep)
        rows.append([str(i + 1), ft(scores[i]), ", ".join(f"{ft(_iou(boxes[i], boxes[k])[0])}" for k in keep) or "-", "keep" if ok else "drop"])
        if ok:
            keep.append(i)
    return display("\\text{IoU threshold} = \\tfrac{1}{2}", table(["i", "\\mathrm{score}", "\\mathrm{IoU\\ with\\ kept}", "\\mathrm{decision}"], rows)) + f"\n\nGiữ các hộp {', '.join(str(i + 1) for i in keep)}."


@lesson("8.6")
def hungarian_example():
    c = _m([[1, 2, 9], [1, 5, 9], [9, 9, 1]])
    best = min(permutations(range(3)), key=lambda p: sum(c[i][p[i]] for i in range(3)))
    tot = sum(c[i][best[i]] for i in range(3))
    greedy = []
    used = set()
    for i in range(3):
        j = min((j for j in range(3) if j not in used), key=lambda j: c[i][j])
        used.add(j)
        greedy.append(j)
    gtot = sum(c[i][greedy[i]] for i in range(3))
    check(gtot > tot, "ví dụ phải cho thấy ghép tham lam tệ hơn tối ưu")
    return display("\\mathbf{C} = " + mat_tex(c), "\\text{optimal matching: } " + ", ".join(f"{i + 1}\\to{best[i] + 1}" for i in range(3)) + f",\\ \\text{{cost}} = {ft(tot)}", f"\\text{{greedy by row}} = {ft(gtot)}") + "\n\nDETR dùng ghép một-một tối ưu (Hungarian) nên không cần NMS: mỗi vật thật chỉ được gán cho đúng một dự đoán."


@lesson("8.6")
def anchor_free_ltrb():
    x, y = F(5), F(4)
    box = [F(2), F(1), F(9), F(7)]
    l, t, r, b = x - box[0], y - box[1], box[2] - x, box[3] - y
    return display(f"\\text{{point }} ({ft(x)}, {ft(y)}),\\ \\text{{box }} (2, 1, 9, 7)", f"(l, t, r, b) = (x - x_1,\\ y - y_1,\\ x_2 - x,\\ y_2 - y) = ({ft(l)}, {ft(t)}, {ft(r)}, {ft(b)})") + "\n\nDetector anchor-free (như FCOS) dự đoán trực tiếp khoảng cách $(l, t, r, b)$ từ điểm trên bản đồ đặc trưng tới bốn cạnh hộp, không cần hộp neo (anchor) dựng sẵn."


@lesson("8.6")
def mistake_iou_empty():
    return "**Sai:** tính diện tích giao bằng $\\min(x_2) - \\max(x_1)$ mà không chặn dưới bằng $0$. Hai hộp không giao nhau khi đó cho diện tích giao **âm**.\n\n**Đúng:** chiều rộng giao $= \\max(0, \\min(x_2) - \\max(x_1))$, chiều cao tương tự; hai hộp rời nhau có IoU $= 0$."


# ------------------------------------------------------------------ 8.7
@lesson("8.7")
def seg_metrics_example():
    gt = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pr = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    inter = int(((gt == 1) & (pr == 1)).sum())
    union = int(((gt == 1) | (pr == 1)).sum())
    sa, sb = int((gt == 1).sum()), int((pr == 1).sum())
    acc = F(int((gt == pr).sum()), 16)
    fm = lambda m: "\\begin{bmatrix} " + " \\\\ ".join(" & ".join(str(v) for v in r) for r in m) + " \\end{bmatrix}"
    return display(f"\\text{{GT}} = {fm(gt)},\\quad \\text{{Pred}} = {fm(pr)}", f"\\text{{pixel accuracy}} = {ft(acc)},\\quad \\mathrm{{IoU}} = \\dfrac{{{inter}}}{{{union}}} = {ft(F(inter, union))},\\quad \\mathrm{{Dice}} = \\dfrac{{2\\cdot {inter}}}{{{sa} + {sb}}} = {ft(F(2 * inter, sa + sb))}") + f"\n\nPixel accuracy cao (${ft(acc)}$) dù IoU chỉ ${ft(F(inter, union))}$, vì phần lớn pixel là nền."


@lesson("8.7")
def unet_shapes():
    rows = []
    h, c = 64, 32
    enc = []
    for i in range(3):
        enc.append((h, c))
        rows.append([f"\\mathrm{{enc}}{i + 1}", f"{c}\\times {h}\\times {h}"])
        h //= 2
        c *= 2
    rows.append(["\\mathrm{bottleneck}", f"{c}\\times {h}\\times {h}"])
    for i in reversed(range(3)):
        h *= 2
        c //= 2
        cat = c + enc[i][1]
        check(enc[i][0] == h, "kích thước khớp để nối")
        rows.append([f"\\mathrm{{dec}}{3 - i}\\ (\\mathrm{{upsample + concat}})", f"{cat}\\times {h}\\times {h}"])
    return display(table(["\\text{stage}", "C\\times H\\times W"], rows)) + "\n\nU-Net nối (concat) đặc trưng của bộ mã hóa cùng độ phân giải vào bộ giải mã nên số kênh sau nối là tổng hai nhánh."


@lesson("8.7")
def mistake_pixel_accuracy():
    n, fg = 100, 5
    acc = F(n - fg, n)
    return f"**Sai:** chỉ dùng pixel accuracy. Dự đoán toàn bộ là nền trên ảnh có ${fg}$ pixel vật thể trong ${n}$ pixel cho accuracy ${ft(acc)}$ nhưng IoU của vật thể bằng $0$.\n\n**Đúng:** dùng IoU/Dice theo từng lớp (mIoU), đặc biệt khi lớp mất cân bằng."


# ------------------------------------------------------------------ 8.8
@lesson("8.8")
def clip_similarity():
    img = [(F(1), F(0)), (F(3, 5), F(4, 5)), (F(0), F(1))]
    txt = [(F(4, 5), F(3, 5)), (F(0), F(1)), (F(1), F(0))]
    sim = [[a[0] * b[0] + a[1] * b[1] for b in txt] for a in img]
    check(np.allclose(_arr(sim), _arr([list(v) for v in img]) @ _arr([list(v) for v in txt]).T), "NumPy")
    return display("\\mathbf{S} = \\mathbf{I}\\mathbf{T}\\T = " + mat_tex(sim)) + "\n\nHàng $i$: độ tương tự của ảnh $i$ với từng chú thích. Mục tiêu huấn luyện là làm các phần tử trên đường chéo (cặp đúng) lớn hơn phần còn lại cùng hàng và cùng cột."


@lesson("8.8")
def contrastive_loss_shape():
    return display("\\ell_{i \\to t} = -\\ln\\dfrac{e^{S_{ii}/\\tau}}{\\sum_j e^{S_{ij}/\\tau}}") + "\n\nĐây là cross-entropy trên hàng $i$ của $\\mathbf{S}/\\tau$ với nhãn đúng là cột $i$ (bài 7.2); lấy trung bình hai chiều (ảnh$\\to$chữ, chữ$\\to$ảnh) cho mất mát CLIP (InfoNCE đối xứng)."


@lesson("8.8")
def prop_temperature():
    s = [F(3, 5), F(1, 5), F(0)]
    rows = []
    for tau in (F(1), F(1, 2), F(1, 10)):
        z = [v / tau for v in s]
        rows.append([ft(tau)] + [ft(v) for v in z])
    return display("S = (\\tfrac{3}{5}, \\tfrac{1}{5}, 0)", table(["\\tau", "z_1", "z_2", "z_3"], rows)) + "\n\n$\\tau$ nhỏ khuếch đại chênh lệch giữa các logit nên softmax nhọn hơn, nhưng thứ tự (argmax) không đổi."


@lesson("8.8")
def mistake_unnormalized():
    a, b = [F(3), F(4)], [F(1), F(0)]
    dot = a[0] * b[0] + a[1] * b[1]
    return f"**Sai:** dùng tích vô hướng thay cho cosine khi vector chưa chuẩn hóa: $\\mathbf{{a}} = {vt(a)}$, $\\mathbf{{b}} = {vt(b)}$ cho ${ft(dot)}$ chỉ vì $\\|\\mathbf{{a}}\\| = 5$ lớn.\n\n**Đúng:** cosine $= \\dfrac{{\\mathbf{{a}}\\T\\mathbf{{b}}}}{{\\|\\mathbf{{a}}\\|\\|\\mathbf{{b}}\\|}} = {ft(dot / 5)}$. CLIP chuẩn hóa embedding trước khi nhân."


# ------------------------------------------------------------------ 8.9
@lesson("8.9")
def diffusion_example():
    abar, sa, sn = F(9, 25), F(3, 5), F(4, 5)
    x0, eps = [F(5), F(-5), F(0)], [F(5), F(0), F(-5)]
    xt = [sa * a + sn * b for a, b in zip(x0, eps)]
    back = [(a - sn * b) / sa for a, b in zip(xt, eps)]
    check(back == x0, "khôi phục x0")
    return display(f"\\bar{{\\alpha}}_t = {ft(abar)},\\ \\sqrt{{\\bar{{\\alpha}}_t}} = {ft(sa)},\\ \\sqrt{{1 - \\bar{{\\alpha}}_t}} = {ft(sn)}", f"\\mathbf{{x}}_t = {ft(sa)}\\,{vt(x0)} + {ft(sn)}\\,{vt(eps)} = {vt(xt)}", f"\\hat{{\\mathbf{{x}}}}_0 = \\dfrac{{\\mathbf{{x}}_t - {ft(sn)}\\boldsymbol{{\\varepsilon}}}}{{{ft(sa)}}} = {vt(back)}")


@lesson("8.9")
def alpha_bar_table():
    betas = [F(1, 100), F(2, 100), F(4, 100), F(8, 100)]
    ab = F(1)
    rows = []
    for t, b in enumerate(betas, 1):
        ab *= 1 - b
        rows.append([str(t), ft(b), ft(1 - b), ft(ab)])
    check(float(ab) == float(np.prod([1 - float(b) for b in betas])) or abs(float(ab) - np.prod([1 - float(b) for b in betas])) < 1e-12, "NumPy")
    return display(table(["t", "\\beta_t", "\\alpha_t", "\\bar{\\alpha}_t"], rows)) + "\n\n$\\bar{\\alpha}_t$ giảm dần: càng về cuối quá trình thuận, $\\mathbf{x}_t$ càng gần nhiễu thuần túy."


@lesson("8.9")
def prop_variance_preserving():
    abar = F(9, 25)
    var = abar + (1 - abar)
    check(var == 1, "phương sai bảo toàn")
    return display(f"\\mathrm{{Var}}(\\mathbf{{x}}_t) = \\bar{{\\alpha}}_t\\mathrm{{Var}}(\\mathbf{{x}}_0) + (1 - \\bar{{\\alpha}}_t)\\mathrm{{Var}}(\\boldsymbol{{\\varepsilon}}) = {ft(abar)} + {ft(1 - abar)} = {ft(var)}") + "\n\nVới $\\mathrm{Var}(\\mathbf{x}_0) = \\mathrm{Var}(\\boldsymbol{\\varepsilon}) = 1$ độc lập, phương sai của $\\mathbf{x}_t$ giữ nguyên bằng $1$ (variance-preserving)."


@lesson("8.9")
def mistake_what_is_predicted():
    return "**Sai:** nghĩ rằng mạng khuếch tán dự đoán trực tiếp ảnh sạch trong một bước từ nhiễu thuần.\n\n**Đúng:** mạng thường dự đoán **nhiễu** $\\boldsymbol{\\varepsilon}$ (hoặc một đại lượng tương đương) ở một mức $t$; ảnh được sinh bằng cách lặp nhiều bước khử nhiễu nhỏ từ $t = T$ về $t = 0$."


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, examples)
    return paths
