r"""Ví dụ số bài 1.10 Không gian vector và hạng."""
from __future__ import annotations

from sympy import Matrix

from ..core import latex as L
from ..core.gauss import eliminate
from ..core.verify import check, verify_rank, verify_rref
from ..topics.linalg.l1_7_gauss import _nice_from_rref
from .common import M, V, display, rng, steps_md, write_examples

LESSON = "1.10"
B, N = L.bold, L.num_paren


def _vecs(*vs):
    return display(*[L.mat(v, f"\\mathbf{{v}}_{{{i + 1}}}") for i, v in enumerate(vs)])


def dependent() -> str:
    v1, v2 = V(11, 3, -4, 4), V(12, 3, -4, 4)
    v3 = 2 * v1 - v2
    a = Matrix.hstack(v1, v2, v3)
    el = eliminate(a, mode="ref")
    verify_rank(a, a.rank())
    check(2 * v1 - v2 - v3 == Matrix.zeros(3, 1), "quan hệ")
    return (
        _vecs(v1, v2, v3)
        + "\n\nXếp thành các cột rồi khử Gauss:\n\n" + steps_md(el.steps)
        + f"\n\nHạng là {a.rank()}, nhỏ hơn số vector (3) nên chúng **phụ thuộc tuyến tính**. Một quan hệ cụ thể:\n\n"
        + display("2\\mathbf{v}_1 - \\mathbf{v}_2 - \\mathbf{v}_3 = " + L.mat(2 * v1 - v2 - v3))
    )


def independent() -> str:
    for seed in range(100):
        v1, v2, v3 = V(21 + seed, 3, -4, 4), V(121 + seed, 3, -4, 4), V(221 + seed, 3, -4, 4)
        a = Matrix.hstack(v1, v2, v3)
        if a.rank() == 3:
            break
    el = eliminate(a, mode="ref")
    return _vecs(v1, v2, v3) + "\n\n" + steps_md(el.steps) + f"\n\nHạng bằng {a.rank()}, đúng bằng số vector, nên chúng **độc lập tuyến tính** (và tạo thành một cơ sở của $\\R^{{3}}$)."


def rank_example() -> str:
    a, rr, piv = _nice_from_rref(rng(31), 3, 4, 2, bound=10, free_max=3)
    el = eliminate(a, mode="rref")
    verify_rref(a, el.matrix, el.pivots)
    return display(L.mat(a, B("A"))) + "\n\n" + steps_md(el.steps) + f"\n\nCó {len(el.pivots)} cột pivot (cột " + ", ".join(str(c + 1) for c in el.pivots) + f"), nên $\\rank(\\mathbf{{A}}) = {len(el.pivots)}$."


def basis_col() -> str:
    a, rr, piv = _nice_from_rref(rng(41), 3, 4, 2, bound=10, free_max=3)
    el = eliminate(a, mode="rref")
    basis = Matrix.hstack(*[a[:, j] for j in el.pivots])
    check(Matrix.hstack(a, basis).rank() == basis.rank() == len(el.pivots), "cơ sở không gian cột")
    return (
        display(L.mat(a, B("A")), f"\\mathrm{{rref}}(\\mathbf{{A}}) = {L.mat(el.matrix)}")
        + f"\n\nCác cột pivot là cột " + ", ".join(str(c + 1) for c in el.pivots) + ". Lấy các cột tương ứng của $\\mathbf{A}$ **gốc**:\n\n"
        + display(L.mat(basis, "\\mathbf{B}")) + f"\n\nSố chiều của không gian cột bằng {len(el.pivots)}."
    )


def null_space() -> str:
    a, rr, piv = _nice_from_rref(rng(51), 3, 4, 2, bound=10, free_max=3)
    el = eliminate(a, mode="rref")
    free = [j for j in range(4) if j not in el.pivots]
    basis = []
    for f in free:
        v = Matrix.zeros(4, 1)
        v[f] = 1
        for i, pc in enumerate(el.pivots):
            v[pc] = -el.matrix[i, f]
        basis.append(v)
    nb = Matrix.hstack(*basis)
    check(a * nb == Matrix.zeros(3, len(basis)), "A·N = 0")
    return (
        display(L.mat(a, B("A")), f"\\mathrm{{rref}}(\\mathbf{{A}}) = {L.mat(el.matrix)}")
        + f"\n\nẨn tự do: " + ", ".join(f"$x_{{{j + 1}}}$" for j in free) + ". Mỗi ẩn tự do cho một vector cơ sở của không gian null:\n\n"
        + display(L.mat(nb, "\\mathbf{N}"), f"\\mathbf{{A}}\\mathbf{{N}} = {L.mat(a * nb)}")
        + f"\n\nSố chiều không gian null bằng {len(basis)} và $\\rank(\\mathbf{{A}}) + {len(basis)} = {a.rank() + len(basis)} = n$."
    )


def prop_rank_nullity() -> str:
    a, rr, piv = _nice_from_rref(rng(61), 4, 4, 2, bound=12, free_max=3)
    r = a.rank()
    nul = len(a.nullspace())
    check(r + nul == a.cols, "rank-nullity")
    return display(L.mat(a, B("A"))) + f"\n\n$\\rank(\\mathbf{{A}}) = {r}$ và số chiều không gian null là ${nul}$. Tổng là ${r + nul}$, đúng bằng số cột ${a.cols}$."


def prop_rank_transpose() -> str:
    a = M(71, 3, 4, -4, 4)
    check(a.rank() == a.T.rank(), "rank A = rank A^T")
    return display(L.mat(a, B("A"))) + f"\n\n$\\rank(\\mathbf{{A}}) = {a.rank()}$ và $\\rank(\\mathbf{{A}}\\T) = {a.T.rank()}$: hai hạng bằng nhau (số cột độc lập bằng số hàng độc lập)."


def prop_rank_bound() -> str:
    a = M(81, 4, 3, -4, 4)
    check(a.rank() <= min(a.shape), "rank ≤ min")
    return display(L.mat(a, B("A"))) + f"\n\n$\\mathbf{{A}}$ cỡ ${a.rows}\\times {a.cols}$ nên $\\rank(\\mathbf{{A}}) \\le \\min({a.rows}, {a.cols}) = {min(a.shape)}$; thực tế $\\rank(\\mathbf{{A}}) = {a.rank()}$."


def mistake_basis() -> str:
    a, rr, piv = _nice_from_rref(rng(91), 3, 3, 2, bound=10, free_max=3)
    el = eliminate(a, mode="rref")
    wrong = Matrix.hstack(*[el.matrix[:, j] for j in el.pivots])
    right = Matrix.hstack(*[a[:, j] for j in el.pivots])
    in_col = Matrix.hstack(a, wrong).rank() == a.rank()
    return (
        display(L.mat(a, B("A")))
        + "\n\n**Sai:** lấy các cột pivot của **dạng bậc thang rút gọn** làm cơ sở không gian cột:\n\n"
        + display(L.mat(wrong))
        + ("\n\nPhép biến đổi hàng làm đổi không gian cột, nên các cột này nói chung **không** thuộc không gian cột của $\\mathbf{A}$ ban đầu." if not in_col else "\n\n")
        + "\n\n**Đúng:** dùng RREF chỉ để *tìm vị trí* các cột pivot, rồi lấy cột tương ứng của $\\mathbf{A}$ gốc:\n\n"
        + display(L.mat(right))
    )


def mistake_rank() -> str:
    a, rr, piv = _nice_from_rref(rng(95), 3, 3, 2, bound=10, free_max=3)
    nonzero_rows = sum(1 for i in range(3) if any(x != 0 for x in a[i, :]))
    return (
        display(L.mat(a, B("A")))
        + f"\n\n**Sai:** đếm số hàng khác $0$ của $\\mathbf{{A}}$ ngay từ đầu và kết luận hạng là ${nonzero_rows}$.\n\n"
        + f"**Đúng:** phải khử Gauss trước; các hàng có thể phụ thuộc nhau. Sau khi khử, số pivot mới là hạng: $\\rank(\\mathbf{{A}}) = {a.rank()}$."
    )


EXAMPLES = {"dependent": dependent, "independent": independent, "rank": rank_example, "basis-col": basis_col, "null-space": null_space,
            "prop-rank-nullity": prop_rank_nullity, "prop-rank-transpose": prop_rank_transpose, "prop-rank-bound": prop_rank_bound,
            "mistake-basis": mistake_basis, "mistake-rank": mistake_rank}


def build():
    return write_examples(LESSON, EXAMPLES)
