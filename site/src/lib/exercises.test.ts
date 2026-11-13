import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { Exercise } from '../types/exercise';
import { pickSet } from './exercises';

const root = join(__dirname, '../../public/exercises');
const manifest = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf8')) as {
	parts: Record<string, { lessons: Record<string, { types: Record<string, unknown> }> }>;
};
const perLevel = { 1: 3, 2: 3, 3: 4, 4: 4, 5: 1 };

function lessonPool(part: string, lesson: string): Exercise[] {
	return Object.keys(manifest.parts[part].lessons[lesson].types).flatMap(
		(t) => JSON.parse(readFileSync(join(root, part, lesson, `${t}.json`), 'utf8')).exercises as Exercise[],
	);
}

describe('pickSet', () => {
	for (const [part, p] of Object.entries(manifest.parts)) {
		for (const lesson of Object.keys(p.lessons)) {
			it(`bài ${lesson}: đủ 15 bài khác nhau, theo thứ tự mức`, () => {
				const set = pickSet(lessonPool(part, lesson), perLevel);
				expect(set.length).toBe(15);
				expect(new Set(set.map((e) => e.id)).size).toBe(15);
				const levels = set.map((e) => e.level);
				expect(levels).toEqual([...levels].sort((a, b) => a - b));
			});
		}
	}

	it('phân bổ đúng khi kho có đủ mọi mức (bài 1.5)', () => {
		const set = pickSet(lessonPool('linalg', '1.5'), perLevel);
		const count = (lv: number) => set.filter((e) => e.level === lv).length;
		expect([1, 2, 3, 4, 5].map(count)).toEqual([3, 3, 4, 4, 1]);
	});
});
