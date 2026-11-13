import { useState } from 'react';
import type { Exercise } from '../../types/exercise';
import { grade, type UserAnswer, type Verdict } from '../../lib/math/compare';
import { recordAttempt } from '../../lib/progress';
import Code from './Code';
import MatrixInput from './MatrixInput';
import { RichText, Tex } from './Tex';
import { parseExpr, toLatex } from '../../lib/math/expr';

const LEVEL_NAMES = ['', 'Nhận biết', 'Tính toán cơ bản', 'Tính toán đầy đủ', 'Tổng hợp', 'Ứng dụng ML'];

function emptyAnswer(ex: Exercise): UserAnswer {
	const a = ex.answer;
	switch (a.kind) {
		case 'number':
			return { kind: 'number', value: '' };
		case 'vector':
		case 'matrix':
			return { kind: 'grid', cells: Array.from({ length: a.rows! }, () => Array(a.cols!).fill('')) };
		case 'shape':
			return { kind: 'shape', rows: '', cols: '', undefined: false };
		case 'choice':
		case 'multi':
			return { kind: 'choice', selected: [] };
		case 'boolean':
			return { kind: 'boolean', value: null };
		case 'expression':
			return { kind: 'expression', value: '' };
		case 'solution-set':
			return {
				kind: 'solution-set',
				particular: Array(a.rows!).fill(''),
				basis: Array.from({ length: a.cols! }, () => Array(a.rows!).fill('')),
			};
		default:
			throw new Error(`Chưa hỗ trợ dạng đáp án ${a.kind}`);
	}
}

interface Props {
	exercise: Exercise;
	onDone?: (r: { correct: boolean; usedSolution: boolean; tries: number }) => void;
}

/** Một bài tập: nhập đáp án, kiểm tra, gợi ý theo bậc, lời giải từng bước. */
export default function ExerciseCard({ exercise: ex, onDone }: Props) {
	const [user, setUser] = useState<UserAnswer>(() => emptyAnswer(ex));
	const [verdict, setVerdict] = useState<Verdict | null>(null);
	const [tries, setTries] = useState(0);
	const [hints, setHints] = useState(0);
	const [steps, setSteps] = useState(0);
	const [usedSolution, setUsedSolution] = useState(false);
	const [solved, setSolved] = useState(false);

	const locked = solved;
	const shownSteps = solved ? ex.steps.length : steps;

	const submit = () => {
		const v = grade(ex.answer, ex.check, user);
		setVerdict(v);
		if (v.formatError) return;
		const t = tries + 1;
		setTries(t);
		recordAttempt(ex.id, { correct: v.correct, tries: t, usedSolution });
		if (v.correct) {
			setSolved(true);
			onDone?.({ correct: true, usedSolution, tries: t });
		}
	};

	const revealStep = () => {
		if (!solved && !usedSolution) {
			setUsedSolution(true);
			recordAttempt(ex.id, { correct: false, tries, usedSolution: true });
			onDone?.({ correct: false, usedSolution: true, tries });
		}
		setSteps((s) => Math.min(s + 1, ex.steps.length));
	};

	const showAll = () => {
		if (!solved && !usedSolution) {
			setUsedSolution(true);
			recordAttempt(ex.id, { correct: false, tries, usedSolution: true });
			onDone?.({ correct: false, usedSolution: true, tries });
		}
		setSteps(ex.steps.length);
	};

	const input = (() => {
		const a = ex.answer;
		const set = (u: UserAnswer) => {
			setUser(u);
			setVerdict(null);
		};
		switch (a.kind) {
			case 'number':
				return (
					<input
						className="num-input"
						aria-label="Đáp án"
						disabled={locked}
						value={(user as { value: string }).value}
						placeholder="vd: -3, 1/2, 0.25"
						onChange={(e) => set({ kind: 'number', value: e.target.value })}
						onKeyDown={(e) => e.key === 'Enter' && submit()}
					/>
				);
			case 'vector':
			case 'matrix': {
				const u = user as { cells: string[][] };
				return (
					<MatrixInput
						rows={a.rows!}
						cols={a.cols!}
						cells={u.cells}
						disabled={locked}
						wrong={verdict?.wrongCells}
						onChange={(cells) => set({ kind: 'grid', cells })}
					/>
				);
			}
			case 'shape': {
				const u = user as Extract<UserAnswer, { kind: 'shape' }>;
				return (
					<div className="shape-input">
						<input aria-label="Số hàng" inputMode="numeric" disabled={locked || u.undefined} value={u.rows}
							onChange={(e) => set({ ...u, rows: e.target.value })} />
						<span>×</span>
						<input aria-label="Số cột" inputMode="numeric" disabled={locked || u.undefined} value={u.cols}
							onChange={(e) => set({ ...u, cols: e.target.value })} />
						<label>
							<input type="checkbox" disabled={locked} checked={u.undefined}
								onChange={(e) => set({ ...u, undefined: e.target.checked })} />{' '}
							Không xác định
						</label>
					</div>
				);
			}
			case 'choice':
			case 'multi': {
				const u = user as { selected: number[] };
				const multi = a.kind === 'multi';
				return (
					<ul className="choices">
						{ex.choices!.map((c, i) => (
							<li key={i}>
								<label>
									<input
										type={multi ? 'checkbox' : 'radio'}
										name={ex.id}
										disabled={locked}
										checked={u.selected.includes(i)}
										onChange={(e) =>
											set({
												kind: 'choice',
												selected: multi
													? e.target.checked ? [...u.selected, i] : u.selected.filter((x) => x !== i)
													: [i],
											})
										}
									/>{' '}
									<RichText text={c} />
								</label>
							</li>
						))}
					</ul>
				);
			}
			case 'expression': {
				const u = user as Extract<UserAnswer, { kind: 'expression' }>;
				const parsed = u.value.trim() ? parseExpr(u.value, a.variables ?? []) : null;
				return (
					<div className="expr-input">
						<input
							className="num-input expr"
							aria-label="Biểu thức"
							disabled={locked}
							value={u.value}
							placeholder="vd: 3x^2 + 2x, e^(2x), 1/x"
							onChange={(e) => set({ kind: 'expression', value: e.target.value })}
							onKeyDown={(e) => e.key === 'Enter' && submit()}
						/>
						<div className="expr-preview" aria-live="polite">
							{parsed?.ok ? <Tex tex={toLatex(parsed.ast)} /> : parsed ? <span className="msg warn">{parsed.error}</span> : <span className="hint">Xem trước công thức sẽ hiện ở đây.</span>}
						</div>
					</div>
				);
			}
			case 'solution-set': {
				const u = user as Extract<UserAnswer, { kind: 'solution-set' }>;
				const col = (v: string[]) => v.map((c) => [c]);
				return (
					<div className="solution-set-input">
						<div className="ss-row">
							<span className="ss-label"><Tex tex="\mathbf{x}_p =" /></span>
							<MatrixInput
								rows={a.rows!}
								cols={1}
								cells={col(u.particular)}
								disabled={locked}
								label="Nghiệm riêng"
								onChange={(cells) => set({ ...u, particular: cells.map((r) => r[0]) })}
							/>
						</div>
						{u.basis.map((vec, k) => (
							<div className="ss-row" key={k}>
								<span className="ss-label"><Tex tex={`\\mathbf{v}_{${k + 1}} =`} /></span>
								<MatrixInput
									rows={a.rows!}
									cols={1}
									cells={col(vec)}
									disabled={locked}
									label={`Vector hướng ${k + 1}`}
									onChange={(cells) => set({ ...u, basis: u.basis.map((b, i) => (i === k ? cells.map((r) => r[0]) : b)) })}
								/>
							</div>
						))}
					</div>
				);
			}
			case 'boolean': {
				const u = user as { value: boolean | null };
				return (
					<div className="bool-input">
						{[true, false].map((b) => (
							<button key={String(b)} type="button" disabled={locked}
								className={u.value === b ? 'chosen' : ''}
								onClick={() => set({ kind: 'boolean', value: b })}>
								{b ? 'Đúng' : 'Sai'}
							</button>
						))}
					</div>
				);
			}
		}
	})();

	return (
		<article className="exercise-card" data-exercise-id={ex.id}>
			<header>
				<span className="badge">Mức {ex.level}: {LEVEL_NAMES[ex.level]}</span>
				<code className="type">{ex.type}</code>
			</header>
			<p className="prompt"><RichText text={ex.prompt.text} /></p>
			{ex.prompt.display.map((d, i) => (
				<div key={i} className="display"><Tex tex={d} display /></div>
			))}
			{ex.prompt.code && <Code code={ex.prompt.code} />}

			<div className="answer-area">{input}</div>

			{verdict?.formatError && <p className="msg warn" role="alert">{verdict.formatError}</p>}
			{verdict && !verdict.formatError && (
				<p className={verdict.correct ? 'msg ok' : 'msg bad'} role="status">
					{verdict.correct ? 'Chính xác!' : `Chưa đúng. ${verdict.detail ?? ''} Hãy thử lại${verdict.wrongCells ? ' (các ô đỏ là ô sai).' : '.'}`}
				</p>
			)}

			<div className="actions">
				{!solved && <button type="button" className="primary" onClick={submit}>Kiểm tra</button>}
				{!solved && hints < ex.hints.length && (
					<button type="button" onClick={() => setHints(hints + 1)}>Gợi ý ({hints}/{ex.hints.length})</button>
				)}
				{!solved && shownSteps < ex.steps.length && (
					<button type="button" onClick={revealStep}>{shownSteps === 0 ? 'Xem lời giải' : 'Bước tiếp theo'}</button>
				)}
				{!solved && shownSteps > 0 && shownSteps < ex.steps.length && (
					<button type="button" onClick={showAll}>Hiện hết</button>
				)}
			</div>

			{hints > 0 && (
				<ol className="hints">
					{ex.hints.slice(0, hints).map((h, i) => <li key={i}><RichText text={h} /></li>)}
				</ol>
			)}

			{shownSteps > 0 && (
				<section className="solution" aria-label="Lời giải">
					<h4>Lời giải</h4>
					<ol>
						{ex.steps.slice(0, shownSteps).map((s, i) => (
							<li key={i}>
								<RichText text={s.text} />
								{s.latex && <div className="display"><Tex tex={s.latex} display /></div>}
							</li>
						))}
					</ol>
				</section>
			)}
		</article>
	);
}
