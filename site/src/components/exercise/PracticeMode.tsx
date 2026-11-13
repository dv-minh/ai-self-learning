import { useEffect, useMemo, useState } from 'react';
import type { Exercise } from '../../types/exercise';
import { loadBank, shuffle } from '../../lib/exercises';
import { loadProgress } from '../../lib/progress';
import ExerciseCard from './ExerciseCard';

interface TypeInfo {
	levels: number[];
	sizes: string[];
	count: number;
}
interface Manifest {
	parts: Record<string, { title: string; lessons: Record<string, { title: string; types: Record<string, TypeInfo> }> }>;
}

const LEVELS = ['', 'Nhận biết', 'Tính toán cơ bản', 'Tính toán đầy đủ', 'Tổng hợp', 'Ứng dụng ML'];
const ALL = '*';

/** Trang /luyen-tap: chọn phần → bài → dạng → mức → kích thước, rồi làm liên tục (docs/spec.md mục 6). */
export default function PracticeMode() {
	const [manifest, setManifest] = useState<Manifest | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [part, setPart] = useState('');
	const [lesson, setLesson] = useState(ALL);
	const [type, setType] = useState(ALL);
	const [level, setLevel] = useState(ALL);
	const [size, setSize] = useState(ALL);

	const [queue, setQueue] = useState<Exercise[] | null>(null);
	const [index, setIndex] = useState(0);
	const [results, setResults] = useState<Record<string, { correct: boolean; usedSolution: boolean; tries: number }>>({});
	const [loading, setLoading] = useState(false);
	const [skipped, setSkipped] = useState(0);

	useEffect(() => {
		const base = import.meta.env.BASE_URL.replace(/\/?$/, '/');
		fetch(`${base}exercises/manifest.json`)
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Không tải được danh mục bài (${r.status}).`))))
			.then((m: Manifest) => {
				setManifest(m);
				setPart(Object.keys(m.parts)[0] ?? '');
			})
			.catch((e) => setError(String(e.message ?? e)));
	}, []);

	const lessons = manifest?.parts[part]?.lessons ?? {};
	const typeEntries = useMemo(() => {
		const out: { lesson: string; type: string; info: TypeInfo }[] = [];
		for (const [ls, l] of Object.entries(lessons)) {
			if (lesson !== ALL && ls !== lesson) continue;
			for (const [t, info] of Object.entries(l.types)) out.push({ lesson: ls, type: t, info });
		}
		return out;
	}, [lessons, lesson]);
	const chosen = typeEntries.filter((e) => type === ALL || e.type === type);
	const levelOptions = [...new Set(chosen.flatMap((e) => e.info.levels))].sort();
	const sizeOptions = [...new Set(chosen.flatMap((e) => e.info.sizes))].sort(
		(a, b) => Number(a.split('x')[0]) - Number(b.split('x')[0]) || Number(a.split('x')[1]) - Number(b.split('x')[1]),
	);

	const start = async () => {
		setLoading(true);
		setError(null);
		try {
			const banks = await Promise.all(chosen.map((e) => loadBank(part, e.lesson, e.type)));
			const done = loadProgress().attempts;
			let pool = banks.flat();
			if (level !== ALL) pool = pool.filter((e) => e.level === Number(level));
			if (size !== ALL) pool = pool.filter((e) => e.tags.includes(size));
			const fresh = pool.filter((e) => !done[e.id]?.correct); // bỏ qua bài đã làm đúng
			setSkipped(pool.length - fresh.length);
			setQueue(shuffle(fresh));
			setIndex(0);
			setResults({});
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setLoading(false);
		}
	};

	if (error && !manifest) return <p className="msg bad" role="alert">{error}</p>;
	if (!manifest) return <p>Đang tải danh mục…</p>;

	const finals = Object.values(results);
	const right = finals.filter((r) => r.correct && !r.usedSolution && r.tries === 1).length;
	const wrong = finals.length - right;
	const ex = queue?.[index];

	return (
		<div className="practice">
			<fieldset className="practice-filters">
				<legend>Chọn bài luyện</legend>
				<label>
					Phần
					<select value={part} onChange={(e) => { setPart(e.target.value); setLesson(ALL); setType(ALL); setLevel(ALL); setSize(ALL); }}>
						{Object.entries(manifest.parts).map(([k, p]) => <option key={k} value={k}>{p.title}</option>)}
					</select>
				</label>
				<label>
					Bài
					<select value={lesson} onChange={(e) => { setLesson(e.target.value); setType(ALL); setLevel(ALL); setSize(ALL); }}>
						<option value={ALL}>Tất cả các bài</option>
						{Object.entries(lessons).map(([k, l]) => <option key={k} value={k}>{k} {l.title}</option>)}
					</select>
				</label>
				<label>
					Dạng
					<select value={type} onChange={(e) => { setType(e.target.value); setLevel(ALL); setSize(ALL); }}>
						<option value={ALL}>Tất cả các dạng</option>
						{typeEntries.map((e) => <option key={e.type} value={e.type}>{e.type}</option>)}
					</select>
				</label>
				<label>
					Mức
					<select value={level} onChange={(e) => setLevel(e.target.value)}>
						<option value={ALL}>Tất cả các mức</option>
						{levelOptions.map((l) => <option key={l} value={l}>Mức {l}: {LEVELS[l]}</option>)}
					</select>
				</label>
				<label>
					Kích thước
					<select value={size} onChange={(e) => setSize(e.target.value)}>
						<option value={ALL}>Mọi kích thước</option>
						{sizeOptions.map((s) => <option key={s} value={s}>{s.replace('x', ' × ')}</option>)}
					</select>
				</label>
				<button type="button" className="primary" onClick={start} disabled={loading || chosen.length === 0}>
					{loading ? 'Đang tải…' : queue ? 'Bắt đầu lại' : 'Bắt đầu'}
				</button>
			</fieldset>

			{error && <p className="msg bad" role="alert">{error}</p>}

			{queue && (
				<>
					<div className="player-bar" aria-live="polite">
						<strong>Phiên này:</strong>
						<span className="msg ok" style={{ margin: 0 }}>Đúng ngay: {right}</span>
						<span className="msg bad" style={{ margin: 0 }}>Chưa đúng / xem lời giải: {wrong}</span>
						<span>Còn lại: {Math.max(queue.length - index - 1, 0)}</span>
						{skipped > 0 && <span>(đã bỏ qua {skipped} bài làm đúng trước đó)</span>}
					</div>
					{ex ? (
						<>
							<ExerciseCard
								key={ex.id}
								exercise={ex}
								onDone={(r) => setResults((prev) => ({ ...prev, [ex.id]: r }))}
							/>
							<div className="player-nav">
								<span />
								<button type="button" className="primary" onClick={() => setIndex(index + 1)}>
									{index + 1 < queue.length ? 'Bài tiếp →' : 'Hết bài'}
								</button>
							</div>
						</>
					) : (
						<p className="msg ok">
							{queue.length === 0
								? 'Không còn bài nào khớp bộ lọc (bạn đã làm đúng hết, hoặc bộ lọc quá hẹp).'
								: 'Bạn đã làm hết các bài khớp bộ lọc. Chọn lại bộ lọc hoặc bấm "Bắt đầu lại".'}
						</p>
					)}
				</>
			)}
		</div>
	);
}
