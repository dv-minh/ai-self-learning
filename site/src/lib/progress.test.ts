import { beforeEach, describe, expect, it } from 'vitest';
import { exportProgress, importProgress, loadProgress, markLessonRead, recordAttempt } from './progress';

// localStorage giả trong bộ nhớ (môi trường Node không có)
class MemoryStorage {
	private d = new Map<string, string>();
	getItem(k: string) { return this.d.get(k) ?? null; }
	setItem(k: string, v: string) { this.d.set(k, v); }
	removeItem(k: string) { this.d.delete(k); }
	clear() { this.d.clear(); }
}

beforeEach(() => {
	(globalThis as unknown as { localStorage: MemoryStorage }).localStorage = new MemoryStorage();
});

describe('progress', () => {
	it('bắt đầu rỗng', () => {
		const p = loadProgress();
		expect(p).toEqual({ version: 1, lessonsRead: [], attempts: {} });
	});

	it('ghi nhận bài đã đọc, không trùng', () => {
		markLessonRead('1.5');
		markLessonRead('1.5');
		markLessonRead('1.6');
		expect(loadProgress().lessonsRead).toEqual(['1.5', '1.6']);
	});

	it('ghi nhận lần làm và không để lần sai ghi đè kết quả đúng', () => {
		recordAttempt('linalg.det.L1.000001', { correct: true, tries: 1, usedSolution: false });
		recordAttempt('linalg.det.L1.000001', { correct: false, tries: 3, usedSolution: true });
		const a = loadProgress().attempts['linalg.det.L1.000001'];
		expect(a.correct).toBe(true);
		expect(a.tries).toBe(1);
		expect(a.at).toMatch(/^\d{4}-\d{2}-\d{2}$/);
	});

	it('xuất rồi nhập lại cho kết quả giống hệt', () => {
		markLessonRead('1.1');
		recordAttempt('linalg.rank.L2.000009', { correct: false, tries: 2, usedSolution: true });
		const text = exportProgress();
		localStorage.clear();
		expect(importProgress(text)).toEqual({ ok: true });
		expect(loadProgress()).toEqual(JSON.parse(text));
	});

	it('nhập file sai định dạng báo lỗi và giữ nguyên dữ liệu hiện có', () => {
		markLessonRead('1.2');
		const before = exportProgress();
		expect(importProgress('không phải json').ok).toBe(false);
		expect(importProgress('{"version":2}').ok).toBe(false);
		expect(importProgress('{"version":1,"lessonsRead":"x","attempts":{}}').ok).toBe(false);
		expect(exportProgress()).toBe(before);
	});

	it('hỏng dữ liệu cũ trong localStorage thì quay về rỗng', () => {
		localStorage.setItem('ai-learn-progress', '{hỏng');
		expect(loadProgress().lessonsRead).toEqual([]);
	});
});
