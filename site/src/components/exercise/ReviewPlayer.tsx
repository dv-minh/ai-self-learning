import { useEffect, useRef, useState } from 'react';
import type { Exercise } from '../../types/exercise';
import { loadBank, shuffle } from '../../lib/exercises';
import { loadProgress } from '../../lib/progress';
import ExerciseCard from './ExerciseCard';

interface Manifest {
	parts: Record<string, { lessons: Record<string, { types: Record<string, { levels: number[] }> }> }>;
}

const DEFAULT_PER_LEVEL: Record<number, number> = { 1: 3, 2: 3, 3: 4, 4: 4, 5: 1 };

/** Trang "Tổng ôn" cuối phần: trộn mọi dạng bài của cả phần, tải lười chỉ các kho cần dùng. */
export default function ReviewPlayer({ part, perLevel = DEFAULT_PER_LEVEL }: { part: string; perLevel?: Record<number, number> }) {
	const [manifest, setManifest] = useState<Manifest | null>(null);
	const [set, setSet] = useState<Exercise[]>([]);
	const [index, setIndex] = useState(0);
	const [results, setResults] = useState<Record<string, { correct: boolean; usedSolution: boolean; tries: number }>>({});
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const cache = useRef(new Map<string, Promise<Exercise[]>>());

	useEffect(() => {
		const base = import.meta.env.BASE_URL.replace(/\/?$/, '/');
		fetch(`${base}exercises/manifest.json`)
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Không tải được danh mục (${r.status}).`))))
			.then(setManifest)
			.catch((e) => setError(String(e.message ?? e)));
	}, []);

	const draw = async () => {
		if (!manifest) return;
		setLoading(true);
		setError(null);
		try {
			const entries = Object.entries(manifest.parts[part].lessons).flatMap(([lesson, l]) =>
				Object.entries(l.types).map(([type, info]) => ({ lesson, type, levels: info.levels })),
			);
			const done = loadProgress().attempts;
			const chosen: Exercise[] = [];
			for (const [lvStr, want] of Object.entries(perLevel)) {
				const lv = Number(lvStr);
				const types = shuffle(entries.filter((e) => e.levels.includes(lv)));
				for (let i = 0; i < want && types.length; i++) {
					const e = types[i % types.length];
					const key = `${e.lesson}/${e.type}`;
					if (!cache.current.has(key)) cache.current.set(key, loadBank(part, e.lesson, e.type));
					const bank = await cache.current.get(key)!;
					const pool = shuffle(bank.filter((x) => x.level === lv && !chosen.some((c) => c.id === x.id)));
					const pick = pool.find((x) => !done[x.id]?.correct) ?? pool[0];
					if (pick) chosen.push(pick);
				}
			}
			setSet(chosen);
			setIndex(0);
			setResults({});
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setLoading(false);
		}
	};

	if (error) return <p className="msg bad" role="alert">{error}</p>;
	if (!manifest) return <p>Đang tải danh mục…</p>;

	const finals = Object.values(results);
	const right = finals.filter((r) => r.correct && !r.usedSolution && r.tries === 1).length;
	const ex = set[index];

	return (
		<div className="exercise-player">
			{set.length === 0 ? (
				<p>
					<button type="button" className="primary" onClick={draw} disabled={loading}
						style={{ padding: '0.5rem 1.2rem', borderRadius: '0.375rem', border: '1px solid var(--sl-color-accent)', background: 'var(--sl-color-accent)', color: 'var(--sl-color-black)', cursor: 'pointer', font: 'inherit' }}>
						{loading ? 'Đang chuẩn bị bộ bài…' : 'Bắt đầu tổng ôn'}
					</button>
				</p>
			) : (
				<>
					<div className="player-bar">
						<strong>Bài {index + 1}/{set.length}</strong>
						<span>Đúng ngay lần đầu: {right}/{set.length}</span>
						<progress value={finals.length} max={set.length} aria-label="Tiến độ bộ bài" />
					</div>
					<ExerciseCard key={ex.id} exercise={ex} onDone={(r) => setResults((prev) => ({ ...prev, [ex.id]: r }))} />
					<div className="player-nav">
						<button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>← Trước</button>
						{index + 1 < set.length ? (
							<button type="button" className="primary" onClick={() => setIndex(index + 1)}>Bài tiếp →</button>
						) : (
							<button type="button" className="primary" onClick={draw} disabled={loading}>{loading ? 'Đang tải…' : 'Bộ bài mới'}</button>
						)}
					</div>
				</>
			)}
		</div>
	);
}
