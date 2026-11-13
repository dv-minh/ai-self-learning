import { useEffect } from 'react';
import { markLessonRead } from '../lib/progress';

/** Không hiển thị gì; khi cuộn tới (client:visible) thì ghi nhận bài học đã đọc vào localStorage. */
export default function MarkRead({ lesson }: { lesson: string }) {
	useEffect(() => {
		markLessonRead(lesson);
		window.dispatchEvent(new Event('lesson-read'));
	}, [lesson]);
	return null;
}
