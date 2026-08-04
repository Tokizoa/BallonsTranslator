import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from modules.translators import trans_llm_api_v4 as v4_mod
from utils.config import RunStatus
from utils.proj_imgtrans import ProjImgTrans
from utils.textblock import TextBlock


class V4ProgressJournalTests(unittest.TestCase):
    def _project(self, directory):
        Image.new('RGB', (8, 8), 'white').save(os.path.join(directory, 'page.png'))
        project = ProjImgTrans(directory)
        project.pages['page.png'] = [
            TextBlock(text=['source'], translation='old translation')
        ]
        project.save()
        return project

    def test_interrupted_page_progress_is_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            project.enable_progress_journal(True)
            project.pages['page.png'][0].translation = 'recovered translation'
            project.update_page_progress('page.png', RunStatus.FIN_OCR)

            with open(project.progress_journal_path(), 'ab') as journal:
                journal.write(b'{"truncated":')

            recovered = ProjImgTrans(directory)
            self.assertEqual(
                recovered.pages['page.png'][0].translation,
                'recovered translation',
            )
            self.assertTrue(
                recovered._image_info['page.png']['finish_code'] & RunStatus.FIN_OCR
            )

    def test_compaction_preserves_records_newer_than_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            project.enable_progress_journal(True)
            project.pages['page.png'][0].translation = 'first'
            project.append_progress_journal('page.png')
            cutoff = project.progress_journal_cutoff()

            project.pages['page.png'][0].translation = 'second'
            project.append_progress_journal('page.png')
            project.compact_progress_journal(cutoff)

            with open(project.progress_journal_path(), encoding='utf-8') as journal:
                records = [json.loads(line) for line in journal if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['blocks'][0]['translation'], 'second')

    def test_full_save_compacts_included_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            project.enable_progress_journal(True)
            project.pages['page.png'][0].translation = 'checkpointed'
            project.append_progress_journal('page.png')

            project.save()

            self.assertFalse(os.path.exists(project.progress_journal_path()))
            with open(project.proj_path, encoding='utf-8') as saved:
                data = json.load(saved)
            self.assertEqual(
                data['pages']['page.png'][0]['translation'],
                'checkpointed',
            )

    def test_existing_interval_debounces_full_json_but_force_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            project._v4_save_interval = 60
            project.pages['page.png'][0].translation = 'first checkpoint'

            globals_patch = (
                mock.patch.object(v4_mod, '_PIPELINE_ACTIVE', True),
                mock.patch.object(v4_mod, '_HEADLESS_SAVE_IN_PROGRESS', False),
                mock.patch.object(v4_mod, '_DEBUG_PROFILING', False),
            )
            with globals_patch[0], globals_patch[1], globals_patch[2]:
                v4_mod._v4_save_project_checkpoint(project, (), {}, False)
                project.pages['page.png'][0].translation = 'too soon'
                v4_mod._v4_save_project_checkpoint(project, (), {}, False)

                with open(project.proj_path, encoding='utf-8') as saved:
                    self.assertEqual(
                        json.load(saved)['pages']['page.png'][0]['translation'],
                        'first checkpoint',
                    )

                v4_mod._v4_save_project_checkpoint(project, (), {}, True)
                with open(project.proj_path, encoding='utf-8') as saved:
                    self.assertEqual(
                        json.load(saved)['pages']['page.png'][0]['translation'],
                        'too soon',
                    )


if __name__ == '__main__':
    unittest.main()
