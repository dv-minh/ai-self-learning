import type { Exercise, ExerciseBank } from '../types/exercise';
import { loadProgress } from './progress';

/** Tải lười kho JSON của một dạng (docs/spec.md mục 2). */
export async function loadBank(part: string, lesson: string, type: string): Promise<Exercise[]> {
	const base = import.meta.env.BASE_URL.replace(/\/?$/, '/');
	const res = await fetch(`${base}exercises/${part}/${lesson}/${type}.json`);
	if (!res.ok) throw new Error(`Không tải được kho bài "${type}" (${res.status}).`);
	const bank = (await res.json()) as ExerciseBank;
	return bank.exercises;
}

export function shuffle<T>(arr: T[], rnd: () => number = Math.random): T[] {
	const a = [...arr];
	for (let i = a.length - 1; i > 0; i--) {
		const j = Math.floor(rnd() * (i + 1));
		[a[i], a[j]] = [a[j], a[i]];
	}
	return a;
}

/**
 * Chọn bài theo mức: ưu tiên bài chưa làm đúng, cố gắng rải đều các dạng.
 * perLevel: số bài mỗi mức. Mức nào kho không đủ thì lấy được bao nhiêu hay bấy nhiêu.
 */
export function pickSet(
	pool: Exercise[],
	perLevel: Record<number, number>,
	rnd: () => number = Math.random,
): Exercise[] {
	const attempts = loadProgress().attempts;
	const out: Exercise[] = [];
	const shortages: number[] = [];
	for (const [lvStr, want] of Object.entries(perLevel)) {
		const lv = Number(lvStr);
		const atLevel = shuffle(pool.filter((e) => e.level === lv), rnd);
		const fresh = atLevel.filter((e) => !attempts[e.id]?.correct);
		const ordered = [...fresh, ...atLevel.filter((e) => attempts[e.id]?.correct)];
		const chosen: Exercise[] = [];
		const usedTypes = new Set<string>();
		// vòng 1: mỗi dạng một bài; vòng 2: lấp phần còn thiếu
		for (const e of ordered) {
			if (chosen.length >= want) break;
			if (!usedTypes.has(e.type)) {
				chosen.push(e);
				usedTypes.add(e.type);
			}
		}
		for (const e of ordered) {
			if (chosen.length >= want) break;
			if (!chosen.includes(e)) chosen.push(e);
		}
		out.push(...chosen);
		for (let i = chosen.length; i < want; i++) shortages.push(lv);
	}
	// Mức nào kho không đủ bài (ví dụ bài chưa có dạng mức 1) thì lấy bù từ mức gần nhất còn bài.
	const taken = new Set(out.map((e) => e.id));
	for (const lv of shortages) {
		const rest = shuffle(pool.filter((e) => !taken.has(e.id)), rnd);
		if (rest.length === 0) break;
		const fresh = rest.filter((e) => !attempts[e.id]?.correct);
		const candidates = fresh.length ? fresh : rest;
		const best = candidates.reduce((a, b) => (Math.abs(b.level - lv) < Math.abs(a.level - lv) ? b : a));
		out.push(best);
		taken.add(best.id);
	}
	return out.sort((a, b) => a.level - b.level);
}
