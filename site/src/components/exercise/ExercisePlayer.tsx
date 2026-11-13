import { useEffect, useState } from 'react';
import type { Exercise } from '../../types/exercise';
import { loadBank, pickSet, shuffle } from '../../lib/exercises';
import ExerciseCard from './ExerciseCard';

interface Props {
	lesson: string;
	types: string[];
	part?: string;
	perLevel?: Record<number, number>;
}

const DEFAULT_PER_LEVEL = { 1: 3, 2: 3, 3: 4, 4: 4, 5: 1 };

type Result = { correct: boolean; usedSolution: boolean };

/** Phát 15 bài theo 5 mức cho một bài học (docs/spec.md mục 6). */
export default function ExercisePlayer({ lesson, types, part = 'linalg', perLevel = DEFAULT_PER_LEVEL }: Props) {
	const [pool, setPool] = useState<Exercise[] | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [set, setSet] = useState<Exercise[]>([]);
	const [index, setIndex] = useState(0);
	const [results, setResults] = useState<Record<string, Result>>({});
	const [nonce, setNonce] = useState(0); // đổi để làm mới thẻ bài

	useEffect(() => {
		let alive = true;
		Promise.all(types.map((t) => loadBank(part, lesson, t)))
			.then((banks) => {
				if (!alive) return;
				const all = banks.flat();
				setPool(all);
				setSet(pickSet(all, perLevel));
			})
			.catch((e) => alive && setError(String(e.message ?? e)));
		return () => {
			alive = false;
		};
	}, [lesson, part, types.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

	if (error) return <p className="msg bad" role="alert">{error}</p>;
	if (!pool) return <p>Đang tải bài tập…</p>;
	if (set.length === 0) return <p>Chưa có bài tập cho bài này.</p>;

	const ex = set[index];
	const done = Object.keys(results).length;
	const right = Object.values(results).filter((r) => r.correct && !r.usedSolution).length;

	const another = () => {
		// bài khác cùng dạng, cùng mức, chưa nằm trong bộ hiện tại
		const inSet = new Set(set.map((e) => e.id));
		const cands = shuffle(pool.filter((e) => e.type === ex.type && e.level === ex.level && !inSet.has(e.id)));
		if (cands.length === 0) return;
		const next = [...set];
		next[index] = cands[0];
		setSet(next);
		setNonce((n) => n + 1);
	};

	const regenerate = () => {
		setSet(pickSet(pool, perLevel));
		setIndex(0);
		setResults({});
		setNonce((n) => n + 1);
	};

	return (
		<div className="exercise-player">
			<div className="player-bar">
				<strong>Bài {index + 1}/{set.length}</strong>
				<span>Đúng không cần xem lời giải: {right}/{set.length}</span>
				<progress value={done} max={set.length} aria-label="Tiến độ bộ bài" />
			</div>
			<ExerciseCard
				key={`${ex.id}-${nonce}`}
				exercise={ex}
				onDone={(r) => setResults((prev) => ({ ...prev, [ex.id]: { correct: r.correct, usedSolution: r.usedSolution } }))}
			/>
			<div className="player-nav">
				<button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>← Trước</button>
				<button type="button" onClick={another}>Bài khác cùng dạng</button>
				{index + 1 < set.length ? (
					<button type="button" className="primary" onClick={() => setIndex(index + 1)}>Bài tiếp →</button>
				) : (
					<button type="button" className="primary" onClick={regenerate}>Bộ bài mới</button>
				)}
			</div>
		</div>
	);
}
