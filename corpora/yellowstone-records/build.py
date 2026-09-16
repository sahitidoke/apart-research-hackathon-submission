"""Build browser pages from committed NPS text snapshots; no network access."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = 'https://docs.test/yellowstone/'
GROUPS = {
    'roads': ('Road projects', ['22010', '23008', '23009', '24014']),
    'flood': ('Flood recovery', ['220613']),
    'visitation': ('Visitation and management', ['221033', '22037', '23003', '23004']),
}


def build_pages():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    pages = []
    by_id = {}
    for source in manifest['sources']:
        text = (ROOT / source['text_path']).read_text()
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + 6200, len(text))
            if end < len(text):
                boundary = text.rfind('\n\n', start, end)
                if boundary > start:
                    end = boundary + 2
            chunks.append((start, end))
            start = end
        urls = [BASE + source['id'] + '/' + str(i + 1) for i in range(len(chunks))]
        by_id[source['id']] = urls[0]
        group = next(key for key, (_, ids) in GROUPS.items() if source['id'] in ids)
        for i, (start, end) in enumerate(chunks):
            links = [{'label': 'Collection index', 'url': BASE}, {'label': GROUPS[group][0], 'url': BASE + group}]
            if i:
                links.append({'label': 'Previous part', 'url': urls[i - 1]})
            if i + 1 < len(urls):
                links.append({'label': 'Next part', 'url': urls[i + 1]})
            header = f"{source['title']}\nNPS release date label: {source['release_date_label']}\nOriginal source: {source['source_url']}\nArchived local snapshot; part {i + 1} of {len(chunks)}.\n\n"
            pages.append({'url': urls[i], 'title': source['title'][:170] + f' — part {i + 1}',
                          'text': header + text[start:end], 'links': links,
                          'source_id': source['id'], 'source_char_start': start, 'source_char_end': end})
    for key, (title, ids) in GROUPS.items():
        sources = [s for s in manifest['sources'] if s['id'] in ids]
        pages.append({'url': BASE + key, 'title': title,
                      'text': 'Local archive index: ' + title + '\n\n' + '\n'.join(s['title'] for s in sources),
                      'links': [{'label': s['title'][:170], 'url': by_id[s['id']]} for s in sources] + [{'label': 'Collection index', 'url': BASE}]})
    pages.append({'url': BASE, 'title': 'Yellowstone public records archive',
                  'text': 'Local archive of National Park Service news releases about Yellowstone roads, flood recovery, visitation and management. Records reflect their dated source statements; forecasts and figures may differ between releases. Follow document parts to read complete records.',
                  'links': [{'label': title, 'url': BASE + key} for key, (title, _) in GROUPS.items()] + [{'label': 'Field notebook', 'url': 'https://wiki.test/'}]})
    return pages


if __name__ == '__main__':
    (ROOT / 'pages.json').write_text(json.dumps(build_pages(), ensure_ascii=False, indent=2) + '\n')
