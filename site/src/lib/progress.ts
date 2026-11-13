/** Tiến độ lưu trong localStorage (docs/spec.md mục 6). Mọi truy cập bọc try/catch. */
export interface Attempt {
	correct: boolean;
	tries: number;
	usedSolution: boolean;
	at: string;
}
export interface Progress {
	version: 1;
	lessonsRead: string[];
	attempts: Record<string, Attempt>;
}

const KEY = 'ai-learn-progress';
const empty = (): Progress => ({ version: 1, lessonsRead: [], attempts: {} });

export function loadProgress(): Progress {
	try {
		const raw = localStorage.getItem(KEY);
		if (!raw) return empty();
		const p = JSON.parse(raw);
		if (p?.version === 1 && Array.isArray(p.lessonsRead) && typeof p.attempts === 'object') return p;
	} catch {
		/* localStorage không dùng được: chạy không lưu */
	}
	return empty();
}

export function saveProgress(p: Progress): void {
	try {
		localStorage.setItem(KEY, JSON.stringify(p));
	} catch {
		/* bỏ qua */
	}
}

const today = () => new Date().toISOString().slice(0, 10);

export function recordAttempt(id: string, a: Omit<Attempt, 'at'>): void {
	const p = loadProgress();
	const prev = p.attempts[id];
	// Giữ kết quả đúng đã có; không để lần làm lại sai ghi đè.
	if (prev?.correct && !a.correct) return;
	p.attempts[id] = { ...a, at: today() };
	saveProgress(p);
}

export function markLessonRead(lesson: string): void {
	const p = loadProgress();
	if (!p.lessonsRead.includes(lesson)) {
		p.lessonsRead.push(lesson);
		saveProgress(p);
	}
}

export function exportProgress(): string {
	return JSON.stringify(loadProgress(), null, 2);
}

/** Nhập file sao lưu; trả về lỗi nếu định dạng sai (không ghi đè dữ liệu hiện có). */
export function importProgress(text: string): { ok: true } | { ok: false; error: string } {
	try {
		const p = JSON.parse(text);
		if (p?.version !== 1 || !Array.isArray(p.lessonsRead) || typeof p.attempts !== 'object' || p.attempts === null) {
			return { ok: false, error: 'File không đúng định dạng tiến độ (version 1).' };
		}
		saveProgress(p as Progress);
		return { ok: true };
	} catch {
		return { ok: false, error: 'Không đọc được file JSON.' };
	}
}
