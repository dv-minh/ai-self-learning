import { useEffect, useMemo, useRef, useState } from 'react';
import { exportProgress, importProgress, loadProgress, saveProgress, type Progress } from '../lib/progress';

interface Manifest {
	parts: Record<string, { title: string; lessons: Record<string, { title: string; types: Record<string, { count: number }> }> }>;
}

const LEVELS = ['', 'Nhận biết', 'Tính toán cơ bản', 'Tính toán đầy đủ', 'Tổng hợp', 'Ứng dụng ML'];

/** id dạng "linalg.matmul-full.L3.000123" -> { part, type, level } */
function parseId(id: string) {
	const [part, type, lv] = id.split('.');
	return { part, type, level: Number(lv?.replace('L', '')) || 0 };
}

interface Tally { done: number; correct: number; clean: number }
const empty = (): Tally => ({ done: 0, correct: 0, clean: 0 });

/** Trang /tien-do: tỉ lệ đúng theo dạng và theo mức, bài đã đọc, xuất/nhập JSON (docs/spec.md mục 6). */
export default function ProgressView() {
	const [progress, setProgress] = useState<Progress | null>(null);
	const [manifest, setManifest] = useState<Manifest | null>(null);
	const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
	const fileRef = useRef<HTMLInputElement>(null);

	useEffect(() => {
		setProgress(loadProgress());
		const base = import.meta.env.BASE_URL.replace(/\/?$/, '/');
		fetch(`${base}exercises/manifest.json`).then((r) => r.json()).then(setManifest).catch(() => setManifest(null));
	}, []);

	const stats = useMemo(() => {
		const byType: Record<string, Tally> = {};
		const byLevel: Record<number, Tally> = {};
		if (!progress) return { byType, byLevel };
		for (const [id, a] of Object.entries(progress.attempts)) {
			const { type, level } = parseId(id);
			for (const t of [(byType[type] ??= empty()), (byLevel[level] ??= empty())]) {
				t.done++;
				if (a.correct) t.correct++;
				if (a.correct && !a.usedSolution && a.tries === 1) t.clean++;
			}
		}
		return { byType, byLevel };
	}, [progress]);

	if (!progress) return <p>Đang tải tiến độ…</p>;

	const total = Object.values(progress.attempts);
	const correct = total.filter((a) => a.correct).length;
	const clean = total.filter((a) => a.correct && !a.usedSolution && a.tries === 1).length;

	const download = () => {
		const blob = new Blob([exportProgress()], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `ai-learn-tien-do-${new Date().toISOString().slice(0, 10)}.json`;
		a.click();
		URL.revokeObjectURL(url);
	};

	const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const f = e.target.files?.[0];
		if (!f) return;
		const r = importProgress(await f.text());
		setMsg(r.ok ? { ok: true, text: 'Đã nhập tiến độ.' } : { ok: false, text: r.error });
		if (r.ok) setProgress(loadProgress());
		e.target.value = '';
	};

	const reset = () => {
		if (!window.confirm('Xóa toàn bộ tiến độ trên trình duyệt này? Hành động này không thể hoàn tác (hãy xuất file sao lưu trước).')) return;
		saveProgress({ version: 1, lessonsRead: [], attempts: {} });
		setProgress(loadProgress());
		setMsg({ ok: true, text: 'Đã xóa tiến độ.' });
	};

	const pct = (t?: Tally) => (t && t.done ? `${Math.round((100 * t.correct) / t.done)}%` : '–');

	return (
		<div className="progress-view">
			<p>
				Bài đã làm: <strong>{total.length}</strong> · đúng: <strong>{correct}</strong> · đúng ngay lần đầu (không xem lời giải): <strong>{clean}</strong> ·
				bài học đã đọc: <strong>{progress.lessonsRead.length}</strong>
			</p>

			<div className="progress-actions">
				<button type="button" onClick={download}>Xuất file JSON</button>
				<label className="btn">
					Nhập file JSON
					<input ref={fileRef} type="file" accept="application/json,.json" onChange={onFile} style={{ display: 'none' }} />
				</label>
				<button type="button" onClick={reset}>Xóa tiến độ</button>
				{msg && <span className={msg.ok ? 'msg ok' : 'msg bad'} role="status">{msg.text}</span>}
			</div>
			<p style={{ fontSize: 'var(--sl-text-sm)' }}>Tiến độ chỉ được lưu trong trình duyệt này. Xuất file để sao lưu hoặc chuyển sang thiết bị khác.</p>

			<h2>Theo mức</h2>
			<table className="progress-table">
				<thead><tr><th>Mức</th><th>Đã làm</th><th>Đúng</th><th>Tỉ lệ đúng</th><th>Đúng ngay lần đầu</th></tr></thead>
				<tbody>
					{[1, 2, 3, 4, 5].map((lv) => {
						const t = stats.byLevel[lv];
						return (
							<tr key={lv}>
								<td>Mức {lv}: {LEVELS[lv]}</td><td>{t?.done ?? 0}</td><td>{t?.correct ?? 0}</td><td>{pct(t)}</td><td>{t?.clean ?? 0}</td>
							</tr>
						);
					})}
				</tbody>
			</table>

			<h2>Theo dạng bài</h2>
			{!manifest && <p>Đang tải danh mục…</p>}
			{manifest && Object.entries(manifest.parts).map(([part, p]) => (
				<section key={part}>
					<h3>{p.title}</h3>
					<table className="progress-table">
						<thead><tr><th>Bài</th><th>Dạng</th><th>Đã làm</th><th>Tỉ lệ đúng</th><th className="progress-bar-cell">Tiến độ trong kho</th></tr></thead>
						<tbody>
							{Object.entries(p.lessons).flatMap(([ls, l]) =>
								Object.entries(l.types).map(([type, info]) => {
									const t = stats.byType[type];
									return (
										<tr key={type}>
											<td>{ls} {l.title}</td><td><code>{type}</code></td><td>{t?.done ?? 0}</td><td>{pct(t)}</td>
											<td className="progress-bar-cell"><progress value={t?.correct ?? 0} max={info.count} aria-label={`Đã làm đúng ${t?.correct ?? 0} trên ${info.count} bài`} /></td>
										</tr>
									);
								}),
							)}
						</tbody>
					</table>
				</section>
			))}
		</div>
	);
}
