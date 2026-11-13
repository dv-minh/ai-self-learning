"""Bài 1.14 SVD, xấp xỉ hạng thấp, PCA: svd-small, low-rank, pca-steps."""
from __future__ import annotations

import random
from fractions import Fraction as PyFraction
from math import isqrt

import numpy as np
from sympy import Matrix, Rational, eye, sqrt, zeros

from ...core import latex as L
from ...core.nice import int_norm_vector, nontrivial_matrix
from ...core.verify import check, np_close, to_np, verify_matmul
from ...schema import Answer, Check, Exercise, Prompt, Step
from ._base import TopicSpec, build, mat_ans, num_ans, shape_answer, size_tag, vec_ans
from .nice_orth import rand_orth

LESSON = "1.14"
B, N = L.bold, L.num_paren


def _sigma(m: int, n: int, s: list) -> Matrix:
    sg = zeros(m, n)
    for i, v in enumerate(s):
        sg[i, i] = v
    return sg


def _build_svd(rng: random.Random, m: int, n: int, smax: int = 9, integer: bool = True) -> tuple[Matrix, Matrix, Matrix, Matrix, list]:
    """Trả về (A, U, Σ, V, σ giảm dần) với A = UΣVᵀ; nếu integer thì A có phần tử nguyên."""
    k = min(m, n)
    for _ in range(2000):
        u, v = rand_orth(rng, m), rand_orth(rng, n)
        s = sorted((rng.randint(1, smax) for _ in range(k)), reverse=True)
        if len(set(s)) < k:
            continue
        a = u * _sigma(m, n, s) * v.T
        if integer and not all(x.is_integer for x in a):
            continue
        if any(all(x == 0 for x in a[i, :]) for i in range(m)):
            continue
        return a, u, _sigma(m, n, s), v, s
    raise RuntimeError("không dựng được SVD đẹp")


def _verify_svd(a: Matrix, s: list) -> None:
    sv = np.linalg.svd(to_np(a), compute_uv=False)
    np_close(Matrix(s), sv[: len(s)])


# ------------------------------------------------------------------ svd-small
def gen_svd(rng: random.Random, level: int, seed: int) -> Exercise:
    m, n = rng.choice([(2, 2), (2, 2), (2, 3), (3, 2)])
    if level == 4:  # σ lớn nhất của ma trận nguyên bất kỳ (số vô tỉ): chấm numeric
        for _ in range(300):
            a = nontrivial_matrix(rng, 2, 2, -4, 4)
            g = a.T * a
            tr, det = g.trace(), g.det()
            disc = tr * tr - 4 * det
            if disc > 0 and not sqrt(disc).is_Rational:
                break
        lam = (tr + sqrt(disc)) / 2
        smax = sqrt(lam)
        val = smax.evalf(30)
        sv = np.linalg.svd(to_np(a), compute_uv=False)
        check(abs(float(val) - sv[0]) < 1e-9, "σ_max numpy")
        return build(
            type_="svd-small", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text="Tính giá trị suy biến (singular value) lớn nhất $\\sigma_1$ của $\\mathbf{A}$, tức căn bậc hai của trị riêng lớn nhất của $\\mathbf{A}\\T\\mathbf{A}$. "
                               "Nhập số thập phân làm tròn $3$ chữ số; sai số tương đối chấp nhận được là $0{,}1\\%$.",
                          display=[L.mat(a, B("A"))]),
            answer=Answer(kind="number", value=str(val)[:22]), check=Check(mode="numeric", tolerance=1e-3),
            hints=["Tính $\\mathbf{A}\\T\\mathbf{A}$ (ma trận đối xứng $2\\times 2$).", "Giải $\\lambda^2 - \\tr(\\mathbf{G})\\lambda + \\det(\\mathbf{G}) = 0$ rồi lấy căn trị riêng lớn."],
            steps=[Step(text="Ma trận Gram:", latex=L.mat(g, "\\mathbf{A}\\T\\mathbf{A}")),
                   Step(text="Vết và định thức:", latex=f"\\tr = {L.num(tr)},\\quad \\det = {L.num(det)}"),
                   Step(text="Trị riêng lớn nhất:", latex=f"\\lambda_{{\\max}} = \\frac{{{L.num(tr)} + \\sqrt{{{disc}}}}}{{2}} \\approx {float(lam):.4f}"),
                   Step(text="Căn bậc hai:", latex=f"\\sigma_1 = \\sqrt{{\\lambda_{{\\max}}}} \\approx {float(val):.3f}")],
            verified=["sympy", "numpy-svd"], tags=["2x2", "numeric"],
        )
    a, u, sg, v, s = _build_svd(rng, m, n, 9 if level == 2 else 12)
    _verify_svd(a, s)
    k = min(m, n)
    if level == 2:  # ghép lại A = UΣVᵀ
        us = u * sg
        verify_matmul(u, sg, us)
        verify_matmul(us, v.T, a)
        return build(
            type_="svd-small", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text="Cho phân tích SVD $\\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}\\T$ với các thừa số dưới đây. Tính $\\mathbf{A}$.",
                          display=[L.mat(u, B("U")), L.mat(sg, "\\boldsymbol{\\Sigma}"), L.mat(v, B("V"))]),
            answer=mat_ans(a),
            hints=["Nhân $\\mathbf{U}\\boldsymbol{\\Sigma}$ trước, rồi nhân với $\\mathbf{V}\\T$.", "Nhớ chuyển vị $\\mathbf{V}$."],
            steps=[Step(text="$\\mathbf{U}\\boldsymbol{\\Sigma}$:", latex=L.mat(us)), Step(text="$\\mathbf{V}\\T$:", latex=L.mat(v.T)),
                   Step(text="Nhân hai ma trận:", latex=L.mat(a, "\\mathbf{A}"))],
            verified=["sympy", "numpy", "numpy-svd"], tags=[size_tag(a)],
        )
    # level 3: tìm các giá trị suy biến
    g = a.T * a if m >= n else a * a.T
    eig = sorted(g.eigenvals().keys(), reverse=True)
    check([sqrt(e) for e in eig[:k]] == [Rational(x) for x in s], "σ = √λ(AᵀA)")
    return build(
        type_="svd-small", lesson=LESSON, level=3, seed=seed,
        prompt=Prompt(text=f"Tìm các giá trị suy biến của $\\mathbf{{A}}$ (đều là số nguyên), xếp **giảm dần**: $\\sigma_1 \\ge \\sigma_2$. "
                           f"Có {k} giá trị khác $0$.", display=[L.mat(a, B("A"))]),
        answer=vec_ans(Matrix(s)),
        hints=["$\\sigma_i = \\sqrt{\\lambda_i}$ với $\\lambda_i$ là trị riêng của $\\mathbf{A}\\T\\mathbf{A}$ (hoặc $\\mathbf{A}\\mathbf{A}\\T$, chọn ma trận cỡ nhỏ hơn).",
               "Hai ma trận $\\mathbf{A}\\T\\mathbf{A}$ và $\\mathbf{A}\\mathbf{A}\\T$ có chung các trị riêng khác $0$."],
        steps=[Step(text="Ma trận Gram cỡ nhỏ:", latex=L.mat(g)),
               Step(text="Các trị riêng (giảm dần):", latex=", ".join(L.num(e) for e in eig[:k])),
               Step(text="Lấy căn bậc hai:", latex=", ".join(f"\\sigma_{{{i + 1}}} = {x}" for i, x in enumerate(s)))],
        verified=["sympy", "numpy-svd"], tags=[size_tag(a)],
    )


# ------------------------------------------------------------------ low-rank
def _unit(rng: random.Random, n: int) -> Matrix:
    v = int_norm_vector(rng, n, hi=12 if n < 3 else 8)
    return v / sqrt(sum(x * x for x in v))


def gen_low_rank(rng: random.Random, level: int, seed: int) -> Exercise:
    if level == 5:  # LoRA
        d, k, r = rng.randint(5, 12), rng.randint(5, 12), rng.randint(1, 3)
        full = d * k
        lora = d * r + r * k
        check((np.zeros((d, r)) @ np.zeros((r, k))).shape == (d, k), "BA shape")
        return build(
            type_="low-rank", lesson=LESSON, level=5, seed=seed,
            prompt=Prompt(text=f"LoRA thay việc cập nhật trực tiếp ma trận trọng số $\\mathbf{{W}} \\in \\R^{{{d}\\times {k}}}$ bằng một cập nhật hạng thấp "
                               f"$\\Delta\\mathbf{{W}} = \\mathbf{{B}}\\mathbf{{A}}$ với $\\mathbf{{B}} \\in \\R^{{{d}\\times {r}}}$ và $\\mathbf{{A}} \\in \\R^{{{r}\\times {k}}}$. "
                               f"Số tham số cần huấn luyện **giảm đi bao nhiêu** so với huấn luyện cả $\\mathbf{{W}}$?"),
            answer=num_ans(full - lora),
            hints=["Số tham số của $\\mathbf{W}$ là số hàng nhân số cột.", "LoRA huấn luyện $\\mathbf{B}$ và $\\mathbf{A}$: cộng số phần tử hai ma trận."],
            steps=[Step(text="Cả $\\mathbf{W}$:", latex=f"{d}\\cdot {k} = {full}"),
                   Step(text="LoRA:", latex=f"{d}\\cdot {r} + {r}\\cdot {k} = {lora}"),
                   Step(text="Giảm đi:", latex=f"{full} - {lora} = {full - lora}")],
            verified=["numpy", "shape"], tags=["ml", "lora"],
        )
    m, n = (2, 2) if level in (2, 3) else rng.choice([(2, 2), (3, 3), (2, 3)])
    a, u, sg, v, s = _build_svd(rng, m, n, 9, integer=(level != 2))
    if level == 2:  # tích ngoài σ u vᵀ
        u1, v1 = _unit(rng, m), _unit(rng, n)
        sig = Rational(rng.randint(2, 9))
        res = sig * u1 * v1.T
        np_close(res, float(sig) * to_np(u1) @ to_np(v1).T)
        check(res.rank() == 1, "hạng 1")
        return build(
            type_="low-rank", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text="Một ma trận hạng $1$ có dạng $\\sigma\\,\\mathbf{u}\\mathbf{v}\\T$ (tích ngoài, outer product). Tính ma trận đó.",
                          display=[f"\\sigma = {L.num(sig)}", L.mat(u1, B("u")), L.mat(v1, B("v"))]),
            answer=mat_ans(res),
            hints=["Phần tử $(i, j)$ bằng $\\sigma u_i v_j$.", "Đây là tích của vector cột $\\mathbf{u}$ với vector hàng $\\mathbf{v}\\T$."],
            steps=[Step(text="Mỗi phần tử là $\\sigma\\,u_iv_j$:", latex=L.mat(res, "\\sigma\\mathbf{u}\\mathbf{v}\\T"))],
            verified=["sympy", "numpy", "rank"], tags=[size_tag(res), "outer"],
        )
    k = min(m, n)
    if level == 3:  # xấp xỉ hạng 1 từ SVD
        a1 = s[0] * u[:, 0] * v[:, 0].T
        err = a - a1
        check(err.rank() <= k - 1 and a1.rank() == 1, "rank")
        sv = np.linalg.svd(to_np(a), compute_uv=False)
        np_close(Matrix([[sv[0]]]), np.array([[float(s[0])]]))
        return build(
            type_="low-rank", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text="Cho SVD $\\mathbf{A} = \\sum_i \\sigma_i\\mathbf{u}_i\\mathbf{v}_i\\T$. Xấp xỉ hạng $1$ tốt nhất của $\\mathbf{A}$ (Eckart–Young) là $\\mathbf{A}_1 = \\sigma_1\\mathbf{u}_1\\mathbf{v}_1\\T$. "
                               "Tính $\\mathbf{A}_1$ từ các thừa số dưới đây (cột $i$ của $\\mathbf{U}$, $\\mathbf{V}$ là $\\mathbf{u}_i$, $\\mathbf{v}_i$).",
                          display=[L.mat(u, B("U")), L.mat(sg, "\\boldsymbol{\\Sigma}"), L.mat(v, B("V"))]),
            answer=mat_ans(a1),
            hints=["Chỉ lấy giá trị suy biến lớn nhất $\\sigma_1$ và cột đầu của $\\mathbf{U}$, $\\mathbf{V}$.", "Tính tích ngoài $\\mathbf{u}_1\\mathbf{v}_1\\T$ rồi nhân với $\\sigma_1$."],
            steps=[Step(text="Cột đầu của $\\mathbf{U}$ và $\\mathbf{V}$:", latex=f"{L.mat(u[:, 0], chr(92) + 'mathbf{u}_1')},\\quad {L.mat(v[:, 0], chr(92) + 'mathbf{v}_1')}"),
                   Step(text="$\\mathbf{A}_1 = \\sigma_1\\mathbf{u}_1\\mathbf{v}_1\\T$ với $\\sigma_1 = " + str(s[0]) + "$:", latex=L.mat(a1, "\\mathbf{A}_1"))],
            verified=["sympy", "numpy-svd", "rank"], tags=[size_tag(a), "rank1"],
        )
    # level 4: sai số Frobenius của xấp xỉ hạng r
    r = rng.randint(1, k - 1)
    err2 = sum(Rational(x) ** 2 for x in s[r:])
    ar = sum((s[i] * u[:, i] * v[:, i].T for i in range(r)), zeros(m, n))
    fro2 = sum(x * x for x in (a - ar))
    check(fro2 == err2, "‖A − A_r‖_F² = Σ σ_i² bỏ đi")
    check(np.isclose(np.linalg.norm(to_np(a) - to_np(ar)) ** 2, float(err2)), "numpy Frobenius")
    return build(
        type_="low-rank", lesson=LESSON, level=4, seed=seed,
        prompt=Prompt(text=f"Ma trận $\\mathbf{{A}}$ có các giá trị suy biến (giảm dần) cho dưới đây. $\\mathbf{{A}}_{{{r}}}$ là xấp xỉ hạng ${r}$ tốt nhất (giữ ${r}$ giá trị suy biến lớn nhất). "
                           f"Tính sai số $\\norm{{\\mathbf{{A}} - \\mathbf{{A}}_{{{r}}}}}_F^2$ (bình phương chuẩn Frobenius).",
                      display=["\\sigma = " + ", ".join(str(x) for x in s)]),
        answer=num_ans(err2),
        hints=["Định lý Eckart–Young: sai số bình phương bằng tổng bình phương các giá trị suy biến **bị bỏ đi**."],
        steps=[Step(text=f"Bỏ đi các giá trị từ $\\sigma_{{{r + 1}}}$ trở đi:", latex=" + ".join(f"{x}^2" for x in s[r:]) + f" = {L.num(err2)}")],
        verified=["sympy", "numpy", "numpy-svd"], tags=[f"r={r}", size_tag(a)],
    )


# ------------------------------------------------------------------ pca-steps
def _pca_data(rng: random.Random) -> tuple[Matrix, Matrix, Matrix, Matrix, list]:
    """4 điểm 2D nguyên có ma trận hiệp phương sai mẫu với trị riêng hữu tỉ khác nhau."""
    for _ in range(20000):
        x = Matrix(4, 2, lambda i, j: rng.randint(-4, 4))
        if x.rank() < 2:
            continue
        mu = Matrix([[sum(x[:, j]) / 4 for j in range(2)]])
        xc = x - Matrix.ones(4, 1) * mu
        cov = xc.T * xc / 3
        if cov[0, 1] == 0:
            continue
        a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
        disc = (a - c) ** 2 + 4 * b ** 2
        if disc != 0 and sqrt(disc).is_Rational:
            l1, l2 = (a + c + sqrt(disc)) / 2, (a + c - sqrt(disc)) / 2
            if l2 > 0 and l1 != l2:
                return x, mu.T, xc, cov, [l1, l2]
    raise RuntimeError("pca data")


def gen_pca(rng: random.Random, level: int, seed: int) -> Exercise:
    x, mu, xc, cov, lam = _pca_data(rng)
    check(np.allclose(to_np(cov), np.cov(to_np(x).T)), "cov numpy (ddof=1)")
    np_close(Matrix(sorted(lam, reverse=True)), np.sort(np.linalg.eigvalsh(to_np(cov)))[::-1])
    disp = [L.mat(x, B("X"))]
    base = ("Ma trận dữ liệu $\\mathbf{X}$ gồm $4$ điểm trong $\\R^{2}$ (mỗi hàng là một điểm). "
            "Quy ước hiệp phương sai mẫu: chia cho $n - 1$.")
    mean_step = Step(text="Vector trung bình (trung bình từng cột):", latex=L.mat(mu, "\\bar{\\mathbf{x}}"))
    cen_step = Step(text="Trừ trung bình khỏi từng hàng:", latex=L.mat(xc, "\\mathbf{X}_c"))
    cov_step = Step(text="Hiệp phương sai mẫu $\\mathbf{C} = \\frac{1}{n-1}\\mathbf{X}_c\\T\\mathbf{X}_c$:", latex=L.mat(cov, "\\mathbf{C}"))
    if level == 2:
        return build(
            type_="pca-steps", lesson=LESSON, level=2, seed=seed,
            prompt=Prompt(text=base + " Bước 1: tính vector trung bình $\\bar{\\mathbf{x}}$.", display=disp),
            answer=vec_ans(mu), hints=["Trung bình từng cột (mỗi đặc trưng)."], steps=[mean_step],
            verified=["sympy", "numpy"], tags=["4x2", "mean"],
        )
    if level == 3:
        return build(
            type_="pca-steps", lesson=LESSON, level=3, seed=seed,
            prompt=Prompt(text=base + " Tính ma trận hiệp phương sai mẫu $\\mathbf{C}$ (sau khi trừ trung bình).", display=disp),
            answer=mat_ans(cov),
            hints=["Trừ vector trung bình khỏi mỗi hàng để được $\\mathbf{X}_c$.", "$\\mathbf{C} = \\frac{1}{n-1}\\mathbf{X}_c\\T\\mathbf{X}_c$; $n = 4$."],
            steps=[mean_step, cen_step, cov_step], verified=["sympy", "numpy"], tags=["4x2", "covariance"],
        )
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    lam_sorted = sorted(lam, reverse=True)
    eig_steps = [Step(text="Phương trình đặc trưng của $\\mathbf{C}$:", latex=f"\\lambda^2 - {N(a + c)}\\lambda + {N(a * c - b * b)} = 0"),
                 Step(text="Hai nghiệm (giảm dần):", latex=f"\\lambda_1 = {L.num(lam_sorted[0])},\\quad \\lambda_2 = {L.num(lam_sorted[1])}")]
    if level == 4:
        return build(
            type_="pca-steps", lesson=LESSON, level=4, seed=seed,
            prompt=Prompt(text=base + " Tính hai trị riêng của ma trận hiệp phương sai $\\mathbf{C}$ (phương sai dọc theo hai thành phần chính), xếp giảm dần.", display=disp),
            answer=vec_ans(Matrix(lam_sorted)),
            hints=["Tính $\\mathbf{C}$ trước.", "Trị riêng của ma trận $2\\times 2$: nghiệm của $\\lambda^2 - \\tr(\\mathbf{C})\\lambda + \\det(\\mathbf{C}) = 0$."],
            steps=[mean_step, cen_step, cov_step] + eig_steps, verified=["sympy", "numpy"], tags=["4x2", "eigenvalues"],
        )
    ratio = lam_sorted[0] / (lam_sorted[0] + lam_sorted[1])
    check(lam_sorted[0] + lam_sorted[1] == a + c, "tổng phương sai = vết")
    return build(
        type_="pca-steps", lesson=LESSON, level=5, seed=seed,
        prompt=Prompt(text=base + " Thành phần chính thứ nhất (PC1) giải thích bao nhiêu phần trăm tổng phương sai? Nhập đáp án dạng phân số hoặc số thập phân của tỉ lệ $\\lambda_1/(\\lambda_1+\\lambda_2)$.",
                      display=disp),
        answer=num_ans(ratio),
        hints=["Tổng phương sai $\\lambda_1 + \\lambda_2$ bằng vết của $\\mathbf{C}$.", "Tỉ lệ phương sai giải thích bởi PC1 là $\\lambda_1/(\\lambda_1+\\lambda_2)$."],
        steps=[mean_step, cen_step, cov_step] + eig_steps + [Step(text="Tỉ lệ phương sai giải thích bởi PC1:", latex=f"\\frac{{{L.num(lam_sorted[0])}}}{{{L.num(a + c)}}} = {L.num(ratio)}")],
        verified=["sympy", "numpy"], tags=["ml", "4x2"],
    )


SPECS = [
    TopicSpec("svd-small", LESSON, (2, 3, 4), gen_svd),
    TopicSpec("low-rank", LESSON, (2, 3, 4, 5), gen_low_rank),
    TopicSpec("pca-steps", LESSON, (2, 3, 4, 5), gen_pca),
]
