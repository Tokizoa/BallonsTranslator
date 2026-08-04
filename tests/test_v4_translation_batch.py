import concurrent.futures
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from modules.translators import trans_llm_api_v4 as v4


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def advance(self, seconds):
        with self.lock:
            self.now += seconds


class _Block:
    def __init__(self, text, translation='old'):
        self.text = text
        self.translation = translation

    def get_text(self):
        return self.text


def _make_translator():
    translator = v4.LLM_API_Translator_V4('日本語', '한국어')
    translator.params = copy.deepcopy(v4.LLM_API_Translator_V4.params)
    translator.params['invalid repeat count']['value'] = 1
    translator.params['retry attempts']['value'] = 1
    translator.params['retry timeout']['value'] = 0
    translator.params['fallback model']['value'] = ''
    translator.params['content_encryption']['value'] = '없음'
    translator.params['input_format']['value'] = 'JSON'
    return translator


class V4TranslationBatchTests(unittest.TestCase):
    def test_batch_settings_defaults_and_legacy_mode(self):
        params = v4.LLM_API_Translator_V4.params
        self.assertEqual(params['translation batch pages']['value'], 10)
        self.assertEqual(params['concurrent images']['value'], 3)
        self.assertEqual(
            params['concurrent images']['display_name'],
            '동시 번역 배치 수',
        )

        translator = _make_translator()
        self.assertEqual(translator.translation_batch_pages, 10)
        translator.updateParam('translation batch pages', 1)
        self.assertEqual(translator.translation_batch_pages, 1)

    def test_25_pages_are_collected_as_10_10_5(self):
        queue = [f'p{i}' for i in range(25)]
        scheduled = 0
        batches = []
        while queue:
            batch = v4._v4_pop_translation_batch(queue, 10, scheduled, 25)
            self.assertTrue(batch)
            batches.append(batch)
            scheduled += len(batch)
        self.assertEqual([len(batch) for batch in batches], [10, 10, 5])

        queue = [f'p{i}' for i in range(4)]
        scheduled = 0
        sizes = []
        while queue:
            batch = v4._v4_pop_translation_batch(queue, 1, scheduled, 4)
            sizes.append(len(batch))
            scheduled += len(batch)
        self.assertEqual(sizes, [1, 1, 1, 1])

    def test_final_partial_waits_until_all_remaining_pages_arrive(self):
        queue = ['p0', 'p1', 'p2']
        self.assertEqual(v4._v4_pop_translation_batch(queue, 10, 0, 5), [])
        queue.extend(['p3', 'p4'])
        self.assertEqual(
            v4._v4_pop_translation_batch(queue, 10, 0, 5),
            ['p0', 'p1', 'p2', 'p3', 'p4'],
        )

    def test_out_of_order_response_is_restored_by_global_id(self):
        translator = _make_translator()
        response = v4.TranslationResponse.model_validate([
            {'id': 3, 'text': '셋'},
            {'id': 1, 'text': '하나'},
            {'id': 2, 'text': '둘'},
        ])
        prompt = '# Input\nOriginalText:\n' + (
            '[{"id":1,"text":"one"},{"id":2,"text":"two"},'
            '{"id":3,"text":"three"}]'
        )
        with mock.patch.object(
            translator, '_request_translation_sync', return_value=response
        ):
            result = translator._process_batch_sync(
                prompt, 3, allow_individual_recovery=False
            )
        self.assertEqual(result, ['하나', '둘', '셋'])

    def test_duplicate_or_missing_ids_reject_the_entire_batch(self):
        translator = _make_translator()
        prompt = (
            '# Input\nOriginalText:\n'
            '[{"id":1,"text":"one"},{"id":2,"text":"two"}]'
        )
        invalid_responses = (
            [{'id': 1, 'text': '하나'}, {'id': 1, 'text': '둘'}],
            [{'id': 1, 'text': '하나'}],
        )
        for raw_response in invalid_responses:
            with self.subTest(raw_response=raw_response):
                response = v4.TranslationResponse.model_validate(raw_response)
                with mock.patch.object(
                    translator, '_request_translation_sync', return_value=response
                ):
                    with self.assertRaises(v4.BatchTranslationError):
                        translator._process_batch_sync(
                            prompt, 2, allow_individual_recovery=False
                        )

    def test_stop_interrupt_does_not_start_fallback_or_recovery_requests(self):
        translator = _make_translator()
        translator.params['fallback model']['value'] = 'fallback-model'
        prompt = '# Input\nOriginalText:\n[{"id":1,"text":"one"}]'
        with mock.patch.object(
            translator,
            '_request_translation_sync',
            side_effect=InterruptedError('stopped'),
        ) as request:
            with self.assertRaises(InterruptedError):
                translator._process_batch_sync(
                    prompt, 1, allow_individual_recovery=False
                )
        request.assert_called_once()

    def test_atomic_commit_maps_same_local_block_numbers_across_pages(self):
        translator = _make_translator()
        page_one = [_Block('one'), _Block('two')]
        page_two = [_Block('three'), _Block('four')]
        blocks = page_one + page_two

        with mock.patch.object(
            translator,
            '_process_batch_sync',
            return_value=['하나', '둘', '셋', '넷'],
        ):
            translator.translate_textblk_batch_atomic(blocks)

        self.assertEqual(
            [block.translation for block in blocks],
            ['하나', '둘', '셋', '넷'],
        )

    def test_atomic_commit_preserves_old_values_on_failure_and_skips_empty(self):
        translator = _make_translator()
        blocks = [_Block('one', 'keep-1'), _Block('two', 'keep-2')]
        with mock.patch.object(
            translator,
            '_process_batch_sync',
            side_effect=v4.BatchTranslationError('failed'),
        ):
            with self.assertRaises(v4.BatchTranslationError):
                translator.translate_textblk_batch_atomic(blocks)
        self.assertEqual([block.translation for block in blocks], ['keep-1', 'keep-2'])

        with mock.patch.object(translator, '_process_batch_sync') as request:
            self.assertEqual(
                translator.translate_textblk_batch_atomic([_Block('  '), _Block('')]),
                [],
            )
        request.assert_not_called()

    def test_failed_ten_page_batch_splits_to_two_successful_halves_once(self):
        translator = _make_translator()
        pages = {f'p{i}': [_Block(f'text-{i}')] for i in range(10)}
        calls = []

        def translate_atomic(blocks, should_stop=None):
            calls.append(len(blocks))
            if len(blocks) == 10:
                raise v4.BatchTranslationError('too large')
            for block in blocks:
                block.translation = f'ok:{block.text}'

        translator.translate_textblk_batch_atomic = translate_atomic
        thread = SimpleNamespace(
            translator=translator,
            imgtrans_proj=SimpleNamespace(pages=pages),
            stop_requested=False,
        )
        with mock.patch.object(v4, '_V4_STOP_REQUESTED', False):
            result = v4._v4_translate_page_batch(thread, list(pages))

        self.assertEqual(calls, [10, 5, 5])
        self.assertTrue(all(result.values()))
        self.assertEqual(translator.request_metrics()['batch_splits'], 1)

    def test_rpm_gate_uses_a_thread_safe_rolling_window_per_key(self):
        clock = _FakeClock()
        gate = v4._PerKeyRPMGate(clock=clock, wait_hook=clock.advance)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            reservations = list(
                executor.map(lambda _index: gate.reserve(['key-a'], 20), range(20))
            )
        self.assertEqual(len(gate._request_times['key-a']), 20)
        self.assertTrue(all(key == 'key-a' and waited == 0 for key, waited in reservations))

        key, waited = gate.reserve(['key-a'], 20)
        self.assertEqual(key, 'key-a')
        self.assertAlmostEqual(waited, 60.0)
        self.assertAlmostEqual(clock(), 60.0)

    def test_rpm_gate_limits_multiple_keys_independently(self):
        clock = _FakeClock()
        gate = v4._PerKeyRPMGate(clock=clock, wait_hook=clock.advance)
        selected = [gate.reserve(['key-a', 'key-b'], 20)[0] for _ in range(40)]
        self.assertEqual(selected.count('key-a'), 20)
        self.assertEqual(selected.count('key-b'), 20)
        self.assertEqual(clock(), 0.0)

        _key, waited = gate.reserve(['key-a', 'key-b'], 20)
        self.assertAlmostEqual(waited, 60.0)
        self.assertAlmostEqual(clock(), 60.0)


if __name__ == '__main__':
    unittest.main()
