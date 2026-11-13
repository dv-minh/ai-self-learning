r"""Ví dụ cho Phần 2 (NumPy): mỗi ví dụ là một đoạn code được CHẠY THẬT lúc build; MDX chứa cả code lẫn kết quả in ra.

Không có kết quả nào gõ tay: đổi code ở đây thì kết quả tự cập nhật.
"""
from __future__ import annotations

import contextlib
import io
import textwrap

import numpy as np

from .common import write_examples


def show(code: str, note: str = "") -> str:
    """Chạy code, trả về MDX gồm khối code và khối kết quả. Ngoại lệ được in dưới dạng `Loại: thông điệp`."""
    code = textwrap.dedent(code).strip("\n")
    buf = io.StringIO()
    env: dict = {}
    with contextlib.redirect_stdout(buf):
        try:
            exec(code, {"__builtins__": __builtins__}, env)
        except Exception as e:  # noqa: BLE001 - ví dụ cố tình gây lỗi
            print(f"{type(e).__name__}: {e}")
    out = buf.getvalue().rstrip("\n")
    parts = [f"```python\n{code}\n```"]
    if out:
        parts.append(f"Kết quả in ra:\n\n```text\n{out}\n```")
    if note:
        parts.append(note)
    return "\n\n".join(parts)


EX: dict[str, dict[str, str]] = {}


def lesson(code: str):
    def deco(fn):
        EX.setdefault(code, {})[fn.__name__.replace("_", "-")] = fn
        return fn
    return deco


# ---------------------------------------------------------------- 2.1
@lesson("2.1")
def array_shape():
    return show("""
        import numpy as np

        v = np.array([3, 1, 4, 1, 5])          # vector: 1 chiều
        A = np.array([[1, 2, 3], [4, 5, 6]])   # ma trận 2 hàng, 3 cột

        print(v.shape, v.ndim, v.size)
        print(A.shape, A.ndim, A.size)
    """, "`shape` là tuple độ dài từng chiều; vector một chiều có shape dạng `(n,)`, **không phải** `(n, 1)`.")


@lesson("2.1")
def dtype_promotion():
    return show("""
        import numpy as np

        print(np.array([1, 2, 3]).dtype)        # toàn số nguyên
        print(np.array([1, 2, 3.5]).dtype)      # có một số thực -> cả mảng là thực
        print((np.array([1, 2, 3]) / 2).dtype)  # chia '/' luôn cho số thực
        print(np.array([True, False]).dtype)
    """)


@lesson("2.1")
def mistake_vector_shape():
    return show("""
        import numpy as np

        v = np.array([1, 2, 3])
        col = v[:, None]       # thêm trục để thành vector cột

        print(v.shape, v.T.shape)   # .T không làm gì với mảng 1 chiều
        print(col.shape, col.T.shape)
    """, "Vector một chiều không có \"hàng\" hay \"cột\"; muốn vector cột hoặc hàng phải thêm trục (`v[:, None]`, `v[None, :]`).")


# ---------------------------------------------------------------- 2.2
@lesson("2.2")
def indexing():
    return show("""
        import numpy as np

        A = np.array([[10, 11, 12, 13],
                      [20, 21, 22, 23],
                      [30, 31, 32, 33]])

        print(A[1, 2])      # hàng 1, cột 2 (đếm từ 0)
        print(A[-1, -1])    # phần tử cuối
        print(A[1][2])      # cho cùng kết quả
        print(A[A > 31])    # lọc bằng mặt nạ bool
    """)


@lesson("2.2")
def slicing():
    return show("""
        import numpy as np

        A = np.arange(12).reshape(3, 4)
        print(A)
        print(A[1:, :2])    # từ hàng 1 trở đi, 2 cột đầu
        print(A[:, 1])      # cả cột 1 (thành vector 1 chiều)
        print(A[::2, ::-1]) # cách 1 hàng, đảo cột
    """)


@lesson("2.2")
def index_from_zero():
    return show("""
        import numpy as np

        A = np.arange(1, 10).reshape(3, 3)   # trong toán: a_11 ... a_33

        print(A[0, 0])   # a_11
        print(A[1, 2])   # a_23: hàng 2, cột 3 trong toán
    """, "Toán đếm từ $1$ ($a_{23}$ là hàng $2$, cột $3$), NumPy đếm từ $0$ (`A[1, 2]`).")


# ---------------------------------------------------------------- 2.3
@lesson("2.3")
def matmul_vs_mul():
    return show("""
        import numpy as np

        A = np.array([[1, 2], [3, 4]])
        B = np.array([[0, 1], [1, 0]])

        print(A @ B)   # tích ma trận (hàng nhân cột)
        print(A * B)   # nhân từng phần tử (Hadamard)
    """)


@lesson("2.3")
def transpose_gram():
    return show("""
        import numpy as np

        X = np.array([[1, 2], [3, 4], [5, 6]])   # 3 mẫu, 2 đặc trưng

        G = X.T @ X
        print(G, G.shape)
        print(np.array_equal(G, G.T))   # ma trận Gram đối xứng
    """)


@lesson("2.3")
def mistake_matmul_shape():
    return show("""
        import numpy as np

        X = np.zeros((4, 3))    # 4 mẫu, 3 đặc trưng
        w = np.zeros(3)

        print((X @ w).shape)    # đúng: (4,)
        print((w @ X).shape)    # sai thứ tự: 3 không khớp 4
    """)


# ---------------------------------------------------------------- 2.4
@lesson("2.4")
def axis_sum():
    return show("""
        import numpy as np

        A = np.array([[1, 2, 3],
                      [4, 5, 6]])

        print(A.sum(axis=0))   # rút gọn theo trục 0: một giá trị cho mỗi cột
        print(A.sum(axis=1))   # rút gọn theo trục 1: một giá trị cho mỗi hàng
        print(A.sum())         # không chỉ trục: tổng tất cả
    """)


@lesson("2.4")
def axis_keepdims():
    return show("""
        import numpy as np

        A = np.array([[1, 2, 3],
                      [4, 5, 6]])

        s = A.sum(axis=1, keepdims=True)
        print(s.shape)         # (2, 1): giữ trục để broadcast được
        print(A / s)           # chia mỗi hàng cho tổng của nó
    """)


@lesson("2.4")
def mistake_axis():
    return show("""
        import numpy as np

        A = np.array([[1, 2, 3], [4, 5, 6]])

        print(A.mean(axis=0))   # trung bình từng cột (từng đặc trưng)
        print(A.mean(axis=1))   # trung bình từng hàng (từng mẫu)
    """, "`axis=0` rút gọn **dọc theo hàng** (kết quả có một số cho mỗi cột); đừng nhầm với \"theo hàng\" nghĩa là tính cho từng hàng.")


# ---------------------------------------------------------------- 2.5
@lesson("2.5")
def broadcast_row():
    return show("""
        import numpy as np

        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        v = np.array([10, 20, 30])

        print(A + v)   # v được cộng vào MỌI hàng của A
    """)


@lesson("2.5")
def broadcast_outer():
    return show("""
        import numpy as np

        col = np.array([[1], [2], [3]])   # shape (3, 1)
        row = np.array([10, 20])          # shape (2,)

        print((col + row).shape)
        print(col + row)
    """)


@lesson("2.5")
def mistake_broadcast():
    return show("""
        import numpy as np

        A = np.zeros((3, 4))
        v = np.zeros(3)

        print((A + v).shape)   # (3, 4) và (3,) -> so 4 với 3: lỗi
    """, "Căn phải các shape: `(3, 4)` và `(3,)` ghép `4` với `3`, không khớp. Muốn cộng theo cột cần `v[:, None]` có shape `(3, 1)`.")


# ---------------------------------------------------------------- 2.6
@lesson("2.6")
def reshape_example():
    return show("""
        import numpy as np

        a = np.arange(6)
        print(a.reshape(2, 3))     # đọc theo thứ tự hàng trước
        print(a.reshape(3, -1))    # -1: tự suy ra (ở đây là 2)
        print(a.reshape(2, 3).T.reshape(-1))   # chuyển vị rồi trải phẳng
    """)


@lesson("2.6")
def stack_concat():
    return show("""
        import numpy as np

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        print(np.concatenate([a, b], axis=0).shape)   # nối thêm hàng
        print(np.concatenate([a, b], axis=1).shape)   # nối thêm cột
        print(np.stack([a, b], axis=0).shape)         # tạo trục MỚI
    """)


@lesson("2.6")
def mistake_reshape():
    return show("""
        import numpy as np

        a = np.arange(6)
        print(a.reshape(4, 2))   # 6 phần tử không chia được thành 4x2
    """)


# ---------------------------------------------------------------- 2.7
@lesson("2.7")
def linalg_tour():
    return show("""
        import numpy as np

        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([3.0, 5.0])

        print(np.round(np.linalg.det(A), 6))
        print(np.round(np.linalg.inv(A), 6))
        print(np.round(np.linalg.solve(A, b), 6))
        print(np.linalg.matrix_rank(A))
        w, V = np.linalg.eigh(A)            # A đối xứng -> dùng eigh
        print(np.round(w, 6))               # trị riêng, tăng dần
        print(np.round(np.linalg.svd(A, compute_uv=False), 6))   # giá trị suy biến
    """)


@lesson("2.7")
def solve_vs_inv():
    return show("""
        import numpy as np

        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([3.0, 5.0])

        x1 = np.linalg.solve(A, b)          # nên dùng
        x2 = np.linalg.inv(A) @ b           # tính nghịch đảo tường minh
        print(np.allclose(x1, x2))
    """, "Hai cách cho cùng nghiệm ở bài toán nhỏ, nhưng `solve` nhanh hơn và ổn định số hơn khi ma trận lớn hoặc gần suy biến.")


@lesson("2.7")
def mistake_singular():
    return show("""
        import numpy as np

        A = np.array([[1.0, 2.0], [2.0, 4.0]])   # hàng 2 = 2 x hàng 1: suy biến
        b = np.array([1.0, 2.0])
        print(np.linalg.solve(A, b))
    """)


# ---------------------------------------------------------------- 2.8
@lesson("2.8")
def float_demo():
    return show("""
        import numpy as np

        print(0.1 + 0.2 == 0.3)
        print(0.1 + 0.2)
        print(np.isclose(0.1 + 0.2, 0.3))
        print(1e16 + 1 == 1e16)     # số nhỏ bị "nuốt" khi cộng với số rất lớn
    """)


@lesson("2.8")
def det_tiny():
    return show("""
        import numpy as np

        A = np.array([[1.0, 2.0], [2.0, 4.0]])   # suy biến thật sự: định thức bằng 0
        print(np.linalg.det(A) == 0)
        print(np.linalg.matrix_rank(A))
    """, "Với ma trận lớn hơn, `np.linalg.det` có thể cho giá trị rất nhỏ như $10^{-16}$ thay vì $0$. Vì vậy hãy dùng `np.linalg.matrix_rank` hoặc `np.isclose(det, 0)` thay vì so bằng `0`.")


@lesson("2.8")
def mistake_equal():
    return show("""
        import numpy as np

        a = 0.1 + 0.2
        b = 0.3
        print(a == b)               # sai: so bằng chính xác
        print(np.isclose(a, b))     # đúng: so với dung sai
    """)


# ---------------------------------------------------------------- 2.9
@lesson("2.9")
def softmax_stable():
    return show("""
        import numpy as np

        z = np.array([[1000.0, 1001.0, 1002.0]])

        bad = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)   # e^1000 tràn số
        e = np.exp(z - z.max(axis=1, keepdims=True))             # trừ max trước
        good = e / e.sum(axis=1, keepdims=True)

        print(bad)
        print(np.round(good, 6))
    """, "Trừ đi giá trị lớn nhất của từng hàng không đổi kết quả softmax (tử và mẫu cùng nhân một hệ số) nhưng tránh tràn số.")


@lesson("2.9")
def mse_vectorized():
    return show("""
        import numpy as np

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        w = np.array([0.5, -0.5])
        b = 1.0
        y = np.array([0.0, 1.0, 2.0])

        y_hat = X @ w + b                    # dự đoán cho cả batch
        mse = np.mean((y_hat - y) ** 2)
        print(y_hat)
        print(mse)
    """)


@lesson("2.9")
def pairwise():
    return show("""
        import numpy as np

        X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])

        D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
        print(D)    # D[i, j] = bình phương khoảng cách giữa mẫu i và mẫu j
    """, "`X[:, None, :]` có shape `(N, 1, d)` và `X[None, :, :]` có shape `(1, N, d)`; broadcasting tạo ra mọi cặp.")


def build():
    paths = []
    for code, examples in EX.items():
        paths += write_examples(code, {name: fn for name, fn in examples.items()})
    return paths
