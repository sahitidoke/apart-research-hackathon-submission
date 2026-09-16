"""Offline integrity and browser integration checks for the real-document corpus."""
import hashlib
import json
from pathlib import Path

from orchestrator.simulated_web.browser import Browser

ROOT = Path(__file__).resolve().parents[1] / 'corpora' / 'yellowstone-records'


def test_source_provenance_and_complete_page_coverage():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    pages = json.loads((ROOT / 'pages.json').read_text())
    assert len(manifest['sources']) == 9
    for source in manifest['sources']:
        raw = (ROOT / source['text_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['text_sha256']
        assert source['source_url'].startswith('https://www.nps.gov/yell/learn/news/')
        assert source['retrieved_at'] and source['release_date_label']
        text = raw.decode()
        chunks = [p for p in pages if p.get('source_id') == source['id']]
        cursor = 0
        for page in chunks:
            assert page['source_char_start'] == cursor
            end = page['source_char_end']
            assert page['text'].endswith(text[cursor:end])
            assert source['source_url'] in page['text']
            cursor = end
        assert cursor == len(text)


def test_all_pages_reachable_and_browser_loads(tmp_path):
    pages = json.loads((ROOT / 'pages.json').read_text())
    by_url = {page['url']: page for page in pages}
    assert len(by_url) == len(pages)
    browser = Browser(pages, tmp_path / 'wiki.sqlite3')
    try:
        pending = ['https://docs.test/yellowstone/']
        seen = set()
        while pending:
            url = pending.pop()
            if url in seen or url.startswith('https://wiki.test/'):
                continue
            seen.add(url)
            page = browser.call('reader', 'open', {'url': url})
            assert 'error' not in page
            assert page['text'] == by_url[url]['text']
            pending.extend(link['url'] for link in page['links'])
        assert seen == set(by_url)
        assert browser.call('reader', 'search', {'query': 'Lewis River Bridge'})['results']
        assert 'error' in browser.call('reader', 'open', {'url': 'https://docs.test/yellowstone/tasks/funding-and-continuation.json'})
        assert browser.db.execute('SELECT count(*) FROM pages').fetchone()[0] == 0
    finally:
        browser.close()


def test_task_supporting_excerpts_exist_in_separate_sources():
    tasks = list((ROOT / 'tasks').glob('*.json'))
    assert len(tasks) == 6
    pages = json.loads((ROOT / 'pages.json').read_text())
    urls = {page['url'] for page in pages}
    for path in tasks:
        task = json.loads(path.read_text())
        assert task['question'] and task['reference_answer']
        assert len(task['evidence']) >= 2
        for evidence in task['evidence']:
            text = (ROOT / 'sources' / (evidence['source_id'] + '.txt')).read_text()
            assert evidence['simulated_url'] in urls
            for excerpt in evidence['supporting_excerpts']:
                assert excerpt in text, (task['id'], excerpt)
    assert all('reference_answer' not in p and 'evidence' not in p for p in pages)
