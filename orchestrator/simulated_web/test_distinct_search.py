"""Five-source presentation without changing BM25 scoring or browser tools."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair


def page(i, title='Shared title', text='needle source'):
    return {'url': f'https://docs.test/{i}', 'title': title, 'text': text}


class DistinctSearchTests(unittest.TestCase):
    def test_cap_aliases_and_same_title_different_body(self):
        pages = [page(i,text='needle duplicate') for i in range(12)]
        pages += [page(i,text=f'needle distinct paragraph {i}') for i in range(12,20)]
        with tempfile.TemporaryDirectory() as tmp:
            browser=Browser(pages,Path(tmp)/'db',search_policy='distinct_sources_5')
            try:
                found=browser.search('needle')['results']
                self.assertEqual(len(found),5)
                bodies=[browser.pages[r['url']]['text'] for r in found]
                self.assertEqual(len(set(bodies)),5)
                self.assertEqual(len({r['title'] for r in found}),1)
            finally:browser.close()

    def test_original_group_uses_best_chunk_after_hidden_filter(self):
        pages=[page('a0',text='unrelated first chunk'),page('a1',text='needle later chunk'),
               page('b1',text='needle later chunk'),page('c',text='needle other paragraph')]
        groups={p['url']: 'original-A' for p in pages[:3]}
        groups[pages[3]['url']]='original-B'
        browser=Browser(pages,':memory:',search_policy='distinct_sources_5',search_source_groups=groups,
                        discovery_hidden_urls={'agent-2':[pages[1]['url']]})
        try:
            found=browser.search('needle','agent-1')['results']
            self.assertEqual([r['url'] for r in found],[pages[1]['url'],pages[3]['url']])
            found=browser.search('needle','agent-2')['results']
            self.assertEqual({r['url'] for r in found},{pages[2]['url'],pages[3]['url']})
            browser.discovery_hidden_urls['agent-2'].add(pages[2]['url'])
            found=browser.search('needle','agent-2')['results']
            self.assertEqual([r['url'] for r in found],[pages[3]['url']])
        finally:browser.close()

    def test_current_edit_text_controls_search(self):
        pages=[page('a',text='oldword'),page('b',text='oldword')]
        selector=(pages[0]['title'],hashlib.sha256(b'oldword').hexdigest())
        browser=Browser(pages,':memory:',editable_sources=[selector],search_policy='distinct_sources_5',
                        search_source_groups={p['url']:'original' for p in pages})
        try:
            self.assertEqual(len(browser.search('oldword')['results']),1)
            browser.db.execute('UPDATE source_pages SET body=?',('newword',))
            self.assertEqual(browser.search('oldword')['results'],[])
            self.assertEqual(len(browser.search('newword')['results']),1)
        finally:browser.close()

    def test_pair_policy_saved_and_legacy_missing_field_resume(self):
        data,selectors=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run_token_pair(root/'fresh',data,'topic',Client(),pair_policy(compaction_enabled=False),selectors)
            cp_path=root/'fresh/checkpoints/rounds-005'
            cp=load_pair_checkpoint(cp_path)
            try:self.assertEqual(cp.browser.search_policy,'distinct_sources_5')
            finally:cp.close()
            # Model old-format fixture: remove newly added policy fields and rehash.
            for name in ('settings.json','browser.json'):
                p=cp_path/name; value=json.loads(p.read_text());value.pop('search_policy');p.write_text(json.dumps(value))
            m=cp_path/'checkpoint.json';manifest=json.loads(m.read_text())
            for name in ('settings.json','browser.json'):
                manifest['files_sha256'][name]=hashlib.sha256((cp_path/name).read_bytes()).hexdigest()
            m.write_text(json.dumps(manifest))
            cp=load_pair_checkpoint(cp_path)
            try:self.assertEqual(cp.browser.search_policy,'legacy_pages_10')
            finally:cp.close()
            run_token_pair(root/'resumed',None,None,Client(),resume_from=cp_path)
            settings=json.loads((root/'resumed/settings.json').read_text())
            self.assertEqual(settings['search_policy'],'legacy_pages_10')
            with self.assertRaises(ValueError):
                run_token_pair(root/'bad',None,None,Client(),resume_from=cp_path,search_policy='distinct_sources_5')
            self.assertFalse((root/'bad').exists())

    def test_legacy_ten_urls_and_invalid_policy_no_database(self):
        pages=[page(i) for i in range(12)]
        browser=Browser(pages,':memory:')
        try:self.assertEqual(len(browser.search('needle')['results']),10)
        finally:browser.close()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'db'
            with self.assertRaises(ValueError):Browser(pages,p,search_policy='invalid')
            self.assertFalse(p.exists())
