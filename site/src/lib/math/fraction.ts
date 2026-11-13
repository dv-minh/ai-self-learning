/** Phân số chính xác bằng BigInt: luôn rút gọn, mẫu dương. */
function gcd(a: bigint, b: bigint): bigint {
	a = a < 0n ? -a : a;
	b = b < 0n ? -b : b;
	while (b) [a, b] = [b, a % b];
	return a;
}

export class Fraction {
	readonly n: bigint;
	readonly d: bigint;

	constructor(n: bigint | number, d: bigint | number = 1n) {
		let nn = BigInt(n);
		let dd = BigInt(d);
		if (dd === 0n) throw new RangeError('Mẫu số bằng 0');
		if (dd < 0n) {
			nn = -nn;
			dd = -dd;
		}
		const g = gcd(nn, dd) || 1n;
		this.n = nn / g;
		this.d = dd / g;
	}

	static readonly ZERO = new Fraction(0n);
	static readonly ONE = new Fraction(1n);

	add(o: Fraction): Fraction {
		return new Fraction(this.n * o.d + o.n * this.d, this.d * o.d);
	}
	sub(o: Fraction): Fraction {
		return new Fraction(this.n * o.d - o.n * this.d, this.d * o.d);
	}
	mul(o: Fraction): Fraction {
		return new Fraction(this.n * o.n, this.d * o.d);
	}
	div(o: Fraction): Fraction {
		if (o.n === 0n) throw new RangeError('Chia cho 0');
		return new Fraction(this.n * o.d, this.d * o.n);
	}
	neg(): Fraction {
		return new Fraction(-this.n, this.d);
	}
	isZero(): boolean {
		return this.n === 0n;
	}
	equals(o: Fraction): boolean {
		return this.n === o.n && this.d === o.d;
	}
	cmp(o: Fraction): number {
		const l = this.n * o.d;
		const r = o.n * this.d;
		return l < r ? -1 : l > r ? 1 : 0;
	}
	toNumber(): number {
		return Number(this.n) / Number(this.d);
	}
	toString(): string {
		return this.d === 1n ? `${this.n}` : `${this.n}/${this.d}`;
	}
}

/** Đọc chuỗi "p/q" hoặc "p" do generator xuất ra. */
export function fromCanonical(s: string): Fraction {
	const [p, q] = s.split('/');
	return new Fraction(BigInt(p), q === undefined ? 1n : BigInt(q));
}
