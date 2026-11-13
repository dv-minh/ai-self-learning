"""Bài 2.7 (np.linalg), 2.8 (số thực dấu phẩy động), 2.9 (vector hóa công thức ML)."""
from __future__ import annotations

from sympy import Matrix

from ._np import (Answer, Step, TopicSpec, array_answer, check, choice_answer, code_prompt, exec_code, make_choices, nb, np, number_answer, random, rat,
                  run, tup)

L27, L28, L29 = "2.7", "2.8", "2.9"

# (mô tả toán, hàm NumPy, mô tả ngắn để chọn ngược)
LINALG = [
    ("định thức của ma trận vuông", "np.linalg.det", "định thức"),
    ("nghịch đảo của ma trận vuông khả nghịch", "np.linalg.inv", "ma trận nghịch đảo"),
    ("nghiệm của hệ $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$", "np.linalg.solve", "nghiệm của hệ phương trình tuyến tính"),
    ("hạng của ma trận", "np.linalg.matrix_rank", "hạng của ma trận"),
    ("trị riêng và vector riêng của ma trận vuông bất kỳ", "np.linalg.eig", "trị riêng và vector riêng"),
    ("trị riêng và vector riêng của ma trận đối xứng (ổn định hơn, trị riêng xếp tăng dần)", "np.linalg.eigh", "trị riêng, vector riêng của ma trận đối xứng"),
    ("phân tích giá trị suy biến (SVD)", "np.linalg.svd", "phân tích SVD"),
    ("chuẩn L2 của một vector", "np.linalg.norm", "chuẩn của vector hoặc ma trận"),
    ("số điều kiện của ma trận", "np.linalg.cond", "số điều kiện"),
]


def _a(rng, n, lo=-3, hi=4):
    return np.array([[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)])


def gen_linalg_map(rng: random.Random, level: int, seed: int):
    if level in (1, 2):
        math_desc, fn, short = rng.choice(LINALG)
        check(hasattr(np.linalg, fn.split(".")[-1]), f"không có {fn}")
        if level == 1:
            choices, idx = make_choices(rng, [f"`{fn}`"], [f"`{f}`" for _, f, _ in LINALG if f != fn], 4)
            return nb(type_="np-linalg-map", lesson=L27, level=1, seed=seed,
                      prompt=code_prompt(f"Hàm NumPy nào tính **{math_desc}**?", "import numpy as np"), answer=choice_answer(idx), choices=choices,
                      hints=["Các hàm nằm trong `np.linalg`."], steps=[Step(text=f"`{fn}` tính {short}.")], verified=["numpy"], tags=["map"])
        choices, idx = make_choices(rng, [short], [s for _, _, s in LINALG if s != short], 4)
        return nb(type_="np-linalg-map", lesson=L27, level=2, seed=seed,
                  prompt=code_prompt(f"Dòng code `{fn}(...)` tính đại lượng nào?", f"import numpy as np\n\nresult = {fn}(A)"), answer=choice_answer(idx), choices=choices,
                  hints=["Tên hàm gợi đại lượng toán tương ứng (bảng \"Toán ↔ NumPy\" trong bài)."], steps=[Step(text=f"`{fn}` tính {short}.")], verified=["numpy"], tags=["reverse"])
    if level == 3:  # giá trị hạng
        n = rng.choice([3, 4])
        for _ in range(200):
            a = _a(rng, n)
            if rng.random() < 0.5:
                a[n - 1] = a[0] + a[1]
            r = int(np.linalg.matrix_rank(a))
            check(r == Matrix(a.tolist()).rank(), "rank numpy ≠ sympy")
            if r < n or rng.random() < 0.3:
                break
        code = f"import numpy as np\n\nA = np.array({a.tolist()})\nprint(np.linalg.matrix_rank(A))"
        return nb(type_="np-linalg-map", lesson=L27, level=3, seed=seed, prompt=code_prompt("Điền số được in ra.", code), answer=number_answer(r),
                  hints=["`matrix_rank` đếm số cột (hàng) độc lập tuyến tính (bài 1.10)."], steps=[Step(text=f"Hạng của ma trận là {r}.")], verified=["numpy", "sympy"], tags=["rank"])
    # level 4: shape đầu ra của svd
    m, n = rng.randint(2, 5), rng.randint(2, 5)
    A = np.arange(1, m * n + 1, dtype=float).reshape(m, n) + np.eye(m, n)
    U, s, Vt = np.linalg.svd(A)
    which = rng.choice(["U", "s", "Vt"])
    res = {"U": U, "s": s, "Vt": Vt}[which].shape
    cands = [res, (m, n), (n, m), (min(m, n),), (m, m), (n, n), (min(m, n), min(m, n))]
    choices, idx = make_choices(rng, [tup(res)], [tup(c) for c in cands if c != res], 4)
    code = f"import numpy as np\n\nA = np.ones(({m}, {n}))\nU, s, Vt = np.linalg.svd(A)\nprint({which}.shape)"
    return nb(type_="np-linalg-map", lesson=L27, level=4, seed=seed,
              prompt=code_prompt("`shape` nào được in ra? (mặc định `full_matrices=True`)", code), answer=choice_answer(idx), choices=choices,
              hints=["$\\mathbf{U}$ cỡ $m\\times m$, $\\mathbf{V}\\T$ cỡ $n\\times n$; `s` là vector các giá trị suy biến, dài $\\min(m, n)$."],
              steps=[Step(text=f"Với `A` cỡ `({m}, {n})`: `U` `{tup(U.shape)}`, `s` `{tup(s.shape)}`, `Vt` `{tup(Vt.shape)}`.")], verified=["numpy", "shape"], tags=["svd"])


def _singular(rng, n):
    a = _a(rng, n)
    a[n - 1] = a[0] * 2
    return a


def gen_solve_vs_inv(rng: random.Random, level: int, seed: int):
    if level == 2:
        n = rng.choice([2, 3])
        for _ in range(200):
            a = _a(rng, n)
            if abs(np.linalg.det(a)) > 0.5:
                break
        x = np.array([rng.randint(-4, 4) for _ in range(n)])
        b = a @ x
        res = np.linalg.solve(a, b)
        check(np.allclose(res, x), "solve")
        check(Matrix(a.tolist()) * Matrix(x.tolist()) == Matrix(b.tolist()), "A x = b chính xác")
        code = f"import numpy as np\n\nA = np.array({a.tolist()})\nb = np.array({b.tolist()})\nprint(np.linalg.solve(A, b))"
        return nb(type_="np-solve-vs-inv", lesson=L27, level=2, seed=seed,
                  prompt=code_prompt("Điền vector nghiệm in ra (làm tròn đến số nguyên).", code), answer=array_answer(x),
                  hints=["`np.linalg.solve(A, b)` giải $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$."], steps=[Step(text=f"Nghiệm là `{x.tolist()}`; kiểm tra $\\mathbf{{A}}\\mathbf{{x}} = \\mathbf{{b}}$.")],
                  verified=["numpy", "sympy"], tags=["value"])
    if level == 3:
        s = np.array([[2.0, 1.0], [1.0, 3.0]])
        try:
            np.linalg.inv(np.zeros((2, 3)))
            raise AssertionError
        except np.linalg.LinAlgError:
            pass
        try:
            np.linalg.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))
            raise AssertionError
        except np.linalg.LinAlgError:
            pass
        true_stmts = [
            "Để giải $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$, nên dùng `np.linalg.solve(A, b)` thay vì `np.linalg.inv(A) @ b` (nhanh hơn và ổn định số hơn).",
            "`np.linalg.solve(A, b)` báo `LinAlgError` khi `A` là ma trận suy biến (ví dụ hai hàng tỉ lệ).",
            "`np.linalg.inv(A)` báo `LinAlgError` khi `A` không phải ma trận vuông.",
            "Về lý thuyết $\\mathbf{x} = \\mathbf{A}^{-1}\\mathbf{b}$, nhưng trong code ta tránh tính $\\mathbf{A}^{-1}$ tường minh nếu chỉ cần nghiệm.",
        ]
        false_stmts = [
            "`np.linalg.inv(A) @ b` luôn chính xác tuyệt đối vì NumPy dùng số thực chính xác vô hạn.",
            "`np.linalg.solve(A, b)` chỉ dùng được khi `A` đối xứng.",
            "Với `A` suy biến, `np.linalg.solve(A, b)` luôn trả về nghiệm có độ dài nhỏ nhất.",
            "`np.linalg.inv(A)` nhanh hơn `np.linalg.solve(A, b)` khi chỉ cần một nghiệm.",
            "`A * b` cho nghiệm của hệ $\\mathbf{A}\\mathbf{x} = \\mathbf{b}$.",
        ]
        t = rng.choice(true_stmts)
        choices, idx = make_choices(rng, [t], false_stmts, 4)
        return nb(type_="np-solve-vs-inv", lesson=L27, level=3, seed=seed,
                  prompt=code_prompt("Phát biểu nào **đúng**?", "import numpy as np"), answer=choice_answer(idx), choices=choices,
                  hints=["`solve` dùng phân rã LU, không tạo ma trận nghịch đảo tường minh."], steps=[Step(text="Đáp án đúng là phát biểu về `solve`/`inv` khớp với hành vi của NumPy (đã kiểm chứng bằng cách chạy thử các trường hợp lỗi).")],
                  verified=["numpy"], tags=["concept"])
    # level 4: có báo lỗi không?
    n = rng.choice([2, 3])
    sing = rng.random() < 0.5
    a = _singular(rng, n) if sing else _a(rng, n)
    while not sing and abs(np.linalg.det(a)) < 0.5:
        a = _a(rng, n)
    b = np.array([rng.randint(-3, 3) for _ in range(n)])
    try:
        np.linalg.solve(a.astype(float), b.astype(float))
        errors = False
    except np.linalg.LinAlgError:
        errors = True
    check(errors == (Matrix(a.tolist()).det() == 0), "singular ⇔ LinAlgError")
    code = f"import numpy as np\n\nA = np.array({a.tolist()})\nb = np.array({b.tolist()})\nx = np.linalg.solve(A, b)"
    return nb(type_="np-solve-vs-inv", lesson=L27, level=4, seed=seed,
              prompt=code_prompt("Đoạn code trên có báo `LinAlgError: Singular matrix` không?", code), answer=Answer(kind="boolean", value="true" if errors else "false"),
              hints=["`solve` cần ma trận hệ số khả nghịch: định thức khác $0$ (bài 1.9)."], steps=[Step(text=f"$\\det(\\mathbf{{A}}) = {Matrix(a.tolist()).det()}$: " + ("suy biến nên báo lỗi." if errors else "khả nghịch nên chạy được."))],
              verified=["numpy", "sympy"], tags=["singular" if errors else "ok"])


# ----------------------------------------------------------------- 2.8 float
FLOAT_TRUE = ["0.5 + 0.25 == 0.75", "np.isclose(0.1 + 0.2, 0.3)", "abs((0.1 + 0.2) - 0.3) < 1e-9", "np.nan != np.nan", "1e16 + 1 == 1e16", "np.allclose([1e-10], [0])",
              "0.1 * 3 != 0.3", "float(2 ** 53) + 1 == float(2 ** 53)", "np.float32(0.1) != 0.1", "(0.1 + 0.2) - 0.3 > 0"]
FLOAT_FALSE = ["0.1 + 0.2 == 0.3", "np.nan == np.nan", "0.1 * 3 == 0.3", "np.float32(0.1) == 0.1", "np.isclose(1e-10, 0, atol=0)", "1e15 + 1 == 1e15",
               "2 ** 0.5 * 2 ** 0.5 == 2", "(0.1 + 0.2) - 0.3 == 0", "np.sqrt(2) ** 2 == 2", "0.7 + 0.1 == 0.8"]


def _ev(e: str) -> bool:
    return bool(run(e))


def _generated_float_exprs() -> list[str]:
    out = []
    for a in range(1, 10):
        for b in range(1, 10):
            out.append(f"{a / 10} + {b / 10} == {(a + b) / 10}")
    for a in range(1, 10):
        for k in (3, 6, 7):
            out.append(f"{a / 10} * {k} == {a * k / 10}")
    return out


GENERATED = _generated_float_exprs()
FLOAT_TRUE += [e for e in GENERATED if run(e)]
FLOAT_FALSE += [e for e in GENERATED if not run(e)]


def gen_float(rng: random.Random, level: int, seed: int):
    if level == 2:
        pool = FLOAT_TRUE + FLOAT_FALSE
        e = rng.choice(pool)
        val = _ev(e)
        return nb(type_="np-float", lesson=L28, level=2, seed=seed,
                  prompt=code_prompt("Dòng code in ra `True` hay `False`?", f"import numpy as np\n\nprint({e})"), answer=Answer(kind="boolean", value="true" if val else "false"),
                  hints=["Số thực nhị phân không biểu diễn chính xác nhiều số thập phân (ví dụ $0{,}1$); phép toán có sai số làm tròn.", "`nan` không bằng chính nó."],
                  steps=[Step(text=f"`{e}` cho `{val}` (đã chạy thật).")], verified=["numpy"], tags=["bool"])
    if level == 3:
        t = rng.sample([x for x in FLOAT_TRUE + FLOAT_FALSE if _ev(x)], rng.randint(1, 2))
        f = rng.sample([x for x in FLOAT_TRUE + FLOAT_FALSE if not _ev(x)], 4 - len(t))
        choices, idx = make_choices(rng, [f"`{x}`" for x in t], [f"`{x}`" for x in f], 4)
        code = "import numpy as np"
        return nb(type_="np-float", lesson=L28, level=3, seed=seed,
                  prompt=code_prompt("Chọn **mọi** biểu thức cho `True`.", code), answer=choice_answer(idx), choices=choices,
                  hints=["Hãy nghĩ biểu thức nào tránh được sai số làm tròn, và biểu thức nào bị mất độ chính xác."],
                  steps=[Step(text=f"`{x}` → `{_ev(x)}`.") for x in t + f], verified=["numpy"], tags=["multi"])
    # level 4: cách so sánh đúng
    a, b = 0.1 + 0.2, 0.3
    cands = {"a == b": a == b, "np.isclose(a, b)": bool(np.isclose(a, b)), "abs(a - b) < 1e-9": abs(a - b) < 1e-9, "a - b == 0": (a - b) == 0,
             "np.allclose(a, b)": bool(np.allclose(a, b)), "a is b": a is b}
    good = [k for k, v in cands.items() if v]
    bad = [k for k, v in cands.items() if not v]
    check(len(good) >= 2 and len(bad) >= 2, "cần cả đúng và sai")
    g = rng.choice(good)
    choices, idx = make_choices(rng, [f"`{g}`"], [f"`{x}`" for x in bad], 4)
    code = "import numpy as np\n\na = 0.1 + 0.2\nb = 0.3"
    return nb(type_="np-float", lesson=L28, level=4, seed=seed,
              prompt=code_prompt("Về mặt toán $a = b$, nhưng với số thực máy tính, cách nào **nhận đúng** rằng `a` và `b` \"bằng nhau\" (cho `True`)?", code),
              answer=choice_answer(idx), choices=choices,
              hints=["So sánh số thực nên dùng dung sai: `np.isclose` / `np.allclose`."], steps=[Step(text=f"`{g}` cho `True`; các phép so sánh bằng chính xác cho `False` vì $0{{,}}1 + 0{{,}}2 \\neq 0{{,}}3$ trong dấu phẩy động.")],
              verified=["numpy"], tags=["compare"])


# --------------------------------------------------------- 2.9 vectorize
def _data(rng, n=5, d=3, k=4, big=False):
    r = np.random.RandomState(rng.randint(0, 10**6))
    X = r.randn(n, d)
    z = r.randn(n, k) * (30 if big else 1) + (np.arange(n)[:, None] * (500 if big else 0))
    return {"X": X, "w": r.randn(d), "b": float(r.randn()), "y": r.randn(n), "y_hat": r.randn(n), "z": z, "pred": r.randint(0, 3, n), "yy": r.randint(0, 3, n)}


def _safe(code: str, env: dict):
    with np.errstate(all="ignore"):
        try:
            return run(code, **env)
        except Exception:
            return None


SCENARIOS = {
    "linear": dict(text="Dự đoán tuyến tính cho cả batch: $\\hat{\\mathbf{y}} = \\mathbf{X}\\mathbf{w} + b$.", note="`X` cỡ (N, d), `w` có d phần tử, `b` là số.",
                   ref=lambda e: e["X"] @ e["w"] + e["b"], ok="X @ w + b", bad=["X * w + b", "w @ X + b", "X.T @ w + b"], level=3, big=False),
    "mse": dict(text="Sai số bình phương trung bình: $\\frac{1}{N}\\sum_i (\\hat{y}_i - y_i)^2$.", note="`y_hat` và `y` là vector N phần tử.",
                ref=lambda e: np.mean((e["y_hat"] - e["y"]) ** 2), ok="np.mean((y_hat - y) ** 2)", bad=["np.mean(y_hat - y) ** 2", "np.sum((y_hat - y) ** 2)", "np.mean(np.abs(y_hat - y))"], level=3, big=False),
    "acc": dict(text="Độ chính xác: tỉ lệ mẫu mà nhãn dự đoán trùng nhãn thật.", note="`pred` và `yy` là hai vector nhãn nguyên cùng độ dài.",
                ref=lambda e: np.mean(e["pred"] == e["yy"]), ok="np.mean(pred == yy)", bad=["np.sum(pred == yy)", "np.mean(pred) == np.mean(yy)", "pred == yy"], level=3, big=False),
    "standardize": dict(text="Chuẩn hóa từng cột (đặc trưng) về trung bình $0$, độ lệch chuẩn $1$.", note="`X` cỡ (N, d).",
                        ref=lambda e: (e["X"] - e["X"].mean(axis=0)) / e["X"].std(axis=0), ok="(X - X.mean(axis=0)) / X.std(axis=0)",
                        bad=["(X - X.mean(axis=1)) / X.std(axis=1)", "(X - X.mean()) / X.std()", "X - X.mean(axis=0) / X.std(axis=0)"], level=4, big=False),
    "softmax": dict(text="Softmax theo từng hàng, **ổn định số**: $p_{ij} = e^{z_{ij}} / \\sum_k e^{z_{ik}}$ (không để $e^{z}$ tràn số).", note="`z` cỡ (N, k) có thể chứa giá trị rất lớn.",
                    ref=lambda e: (lambda s: s / s.sum(axis=1, keepdims=True))(np.exp(e["z"] - e["z"].max(axis=1, keepdims=True))),
                    ok="np.exp(z - z.max(axis=1, keepdims=True)) / np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)",
                    bad=["np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)", "np.exp(z - z.max()) / np.exp(z - z.max()).sum()", "np.exp(z - z.max(axis=1)) / np.exp(z - z.max(axis=1)).sum(axis=1)"], level=4, big=True),
    "pairwise": dict(text="Ma trận khoảng cách bình phương giữa mọi cặp mẫu: $D_{ij} = \\lVert \\mathbf{x}_i - \\mathbf{x}_j \\rVert_2^2$.", note="`X` cỡ (N, d).",
                     ref=lambda e: np.array([[np.sum((a - b) ** 2) for b in e["X"]] for a in e["X"]]), ok="((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)",
                     bad=["((X - X.T) ** 2).sum(axis=1)", "(X @ X.T)", "np.sum((X - X) ** 2)"], level=5, big=False),
    "cosine": dict(text="Ma trận cosine similarity giữa mọi cặp mẫu (các hàng của `X`).", note="`X` cỡ (N, d).",
                   ref=lambda e: (lambda U: U @ U.T)(e["X"] / np.linalg.norm(e["X"], axis=1, keepdims=True)),
                   ok="(X / np.linalg.norm(X, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T",
                   bad=["X @ X.T", "(X / np.linalg.norm(X)) @ (X / np.linalg.norm(X)).T", "X / np.linalg.norm(X, axis=0)"], level=5, big=False),
}


def gen_vectorize(rng: random.Random, level: int, seed: int):
    names = [k for k, v in SCENARIOS.items() if v["level"] == level or (level == 5 and k == "softmax")]
    key = rng.choice(names)
    sc = SCENARIOS[key]
    flags = []
    for _attempt in range(50):  # thử lại đến khi dữ liệu phân biệt được mọi lựa chọn (tránh trùng hợp như tổng = trung bình)
        env = _data(rng, n=rng.choice([4, 5, 6]), d=rng.choice([2, 3]), k=rng.choice([3, 4]), big=sc["big"])
        env2 = _data(rng, n=env["X"].shape[0], d=env["X"].shape[1], k=env["z"].shape[1], big=sc["big"])
        flags = []
        for code in [sc["ok"]] + sc["bad"]:
            good = True
            for e in (env, env2):
                ref = sc["ref"](e)
                got = _safe(code, e)
                if got is None or np.shape(got) != np.shape(ref) or not np.all(np.isfinite(np.asarray(got, dtype=float))) or not np.allclose(got, ref):
                    good = False
            flags.append(good)
        if flags[0] and not any(flags[1:]):
            break
    check(flags[0] and not any(flags[1:]), f"{key}: đúng một lựa chọn đúng ({flags})")
    cands = [sc["ok"]] + sc["bad"]
    choices, idx = make_choices(rng, [f"`{sc['ok']}`"], [f"`{c}`" for c in sc["bad"]], 4)
    n, d = env["X"].shape
    code = f"import numpy as np\n\n# {sc['note']}\n# N = {n}, d = {d}, k = {env['z'].shape[1]}"
    return nb(type_="np-vectorize-choose", lesson=L29, level=level, seed=seed,
              prompt=code_prompt(f"Biểu thức NumPy nào tính đúng yêu cầu sau (kết quả đúng và cùng shape với định nghĩa)? {sc['text']}", code),
              answer=choice_answer(idx), choices=choices,
              hints=["Theo dõi shape ở từng bước và chọn `axis` đúng.", "Các lựa chọn sai đã được chạy thử: chúng báo lỗi hoặc cho kết quả khác."],
              steps=[Step(text=f"Lựa chọn đúng: `{sc['ok']}`. Các lựa chọn khác sai vì lỗi shape/`axis`, thiếu `keepdims`, hoặc không ổn định số.")],
              verified=["numpy", "executed-candidates"], tags=["ml", key])


# --------------------------------------------------------------- bug find
def _buggy_variants(n: int, d: int, h: int, k0: int = 0):
    """Mỗi template: (tiêu đề, danh sách dòng đúng, chỉ số dòng lỗi, dòng lỗi, kiểm tra kết quả)."""
    return {
        "linear": dict(level=3,
                       good=[f"X = np.arange({n * d}).reshape({n}, {d}) + {k0}", f"W = np.ones(({d}, {h}))", "b = np.zeros(" + str(h) + ")", "Z = X @ W + b   # mục tiêu: Z = XW + b"],
                       bug=(3, "Z = X * W + b   # mục tiêu: Z = XW + b"), target="Z"),
        "gram": dict(level=3,
                     good=[f"X = np.arange({n * d}).reshape({n}, {d}) + {k0}", "G = X.T @ X   # mục tiêu: ma trận Gram cỡ (d, d)", "print(G.shape)"],
                     bug=(1, "G = X @ X   # mục tiêu: ma trận Gram cỡ (d, d)"), target="G"),
        "prob": dict(level=3,
                     good=[f"counts = np.array({[i + 1 for i in range(d)]})", "total = counts.sum()", "p = counts / total   # mục tiêu: xác suất, tổng bằng 1"],
                     bug=(2, "p = counts // total   # mục tiêu: xác suất, tổng bằng 1"), target="p"),
        "softmax_axis": dict(level=4,
                             good=[f"z = np.arange({n * d}).reshape({n}, {d}) / 10 + {k0}", "e = np.exp(z)", "P = e / e.sum(axis=1, keepdims=True)   # mục tiêu: mỗi hàng tổng bằng 1"],
                             bug=(2, "P = e / e.sum(axis=0, keepdims=True)   # mục tiêu: mỗi hàng tổng bằng 1"), target="P"),
        "center": dict(level=4,
                       good=[f"X = np.arange({n * d}).reshape({n}, {d}) + {k0}", "mu = X.mean(axis=0)", "Xc = X - mu   # mục tiêu: trừ trung bình từng cột"],
                       bug=(1, "mu = X.mean(axis=1)"), target="Xc"),
        "reshape": dict(level=5,
                        good=[f"X = np.arange({n * d}).reshape({n}, {d}) + {k0}", f"Y = X.reshape({n * d}, 1)   # mục tiêu: trải thành cột", "print(Y.shape)"],
                        bug=(1, f"Y = X.reshape({n}, {d + 1})   # mục tiêu: trải thành cột"), target="Y"),
        "mean_loss": dict(level=5,
                          good=[f"yh = np.arange({n}) + 1.0", f"y = np.arange({n}) * 1.0", "loss = ((yh - y) ** 2).mean()   # mục tiêu: một số (MSE)"],
                          bug=(2, "loss = ((yh - y) ** 2)   # mục tiêu: một số (MSE)"), target="loss"),
    }


def gen_bug_find(rng: random.Random, level: int, seed: int):
    while True:
        n, d, h = rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6)
        if n != d:
            break
    pool = {k: v for k, v in _buggy_variants(n, d, h, rng.randint(0, 9)).items() if v["level"] == level}
    key = rng.choice(list(pool))
    t = pool[key]
    good_lines = [l for l in t["good"]]
    bi, bline = t["bug"]
    buggy_lines = good_lines[:bi] + [bline] + good_lines[bi + 1:]
    # chạy hai phiên bản: bản đúng phải chạy được; bản sai phải lỗi hoặc cho kết quả khác
    def result(lines):
        try:
            with np.errstate(all="ignore"):
                env = exec_code("\n".join(l.split("#")[0].rstrip() for l in lines if not l.startswith("print")))
            return env[t["target"]]
        except Exception as e:
            return e
    good_res, bug_res = result(good_lines), result(buggy_lines)
    check(not isinstance(good_res, Exception), "bản đúng phải chạy được")
    differs = isinstance(bug_res, Exception) or np.shape(bug_res) != np.shape(good_res) or not np.allclose(bug_res, good_res)
    check(differs, f"{key}: bản lỗi không khác bản đúng")
    lines = ["import numpy as np", ""] + buggy_lines
    code = "\n".join(lines)
    nlines = len(lines)
    ids = list(range(1, nlines + 1))
    bug_lineno = 3 + bi  # có dòng import và dòng trống
    choices = [f"Dòng {i}" for i in ids if lines[i - 1].strip()]
    idx = [choices.index(f"Dòng {bug_lineno}")]
    err = f"báo `{type(bug_res).__name__}`" if isinstance(bug_res, Exception) else "chạy được nhưng cho kết quả sai"
    return nb(type_="np-bug-find", lesson=L29, level=level, seed=seed,
              prompt=code_prompt("Đoạn code có **đúng một dòng** làm sai mục tiêu ghi trong chú thích (dòng được đánh số từ $1$, kể cả dòng `import`). Dòng nào?", code),
              answer=choice_answer(idx), choices=choices,
              hints=["Đọc chú thích mục tiêu rồi kiểm shape và `axis` của từng phép toán.", "Có thể dòng đó báo lỗi, hoặc chạy được nhưng cho kết quả sai (khó phát hiện hơn)."],
              steps=[Step(text=f"Dòng lỗi là dòng {bug_lineno}: `{bline.split('#')[0].strip()}`; khi chạy nó {err}. Bản đúng: `{good_lines[bi].split('#')[0].strip()}`.")],
              verified=["numpy", "executed-both-versions"], tags=["ml", key])


SPECS = [
    TopicSpec("np-linalg-map", L27, (1, 2, 3, 4), gen_linalg_map),
    TopicSpec("np-solve-vs-inv", L27, (2, 3, 4), gen_solve_vs_inv),
    TopicSpec("np-float", L28, (2, 3, 4), gen_float),
    TopicSpec("np-vectorize-choose", L29, (3, 4, 5), gen_vectorize),
    TopicSpec("np-bug-find", L29, (3, 4, 5), gen_bug_find),
]
