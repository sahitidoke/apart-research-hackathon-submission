"""Opt-in answer JSON shape validation; never judges evidence or correctness."""
import json

EVIDENCE_SCHEMA = {
    'type':'object', 'additionalProperties':False, 'required':['status','answer','citations'],
    'properties':{
        'status':{'type':'string','enum':['answered','insufficient_evidence']},
        'answer':{'type':'string'},
        'citations':{'type':'array','items':{'type':'object','additionalProperties':False,
                    'required':['url','quote'],'properties':{'url':{'type':'string','minLength':1},'quote':{'type':'string','minLength':1}}}}}}
EVIDENCE_INSTRUCTION = ('Return only one JSON object with exactly status, answer, and citations. '
    'Use status "answered" with a nonempty answer and at least one citation, each containing only url and quote strings. '
    'Use exact source quotes. If evidence is insufficient, use status "insufficient_evidence", answer "", and citations []. '
    'Do not add Markdown fences or additional fields.')


def format_settings(value):
    if value not in ('text','json_evidence'):
        raise ValueError('Invalid final answer format')
    return None if value == 'text' else {'schema':EVIDENCE_SCHEMA,'instruction':EVIDENCE_INSTRUCTION,
        'validation':'JSON shape and status consistency only; no correctness or support judgment',
        'generation':'schema on existing final-only requests; prompt and host validation on early natural finals',
        'automatic_retries':False}


def unique_object(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_evidence_answer(content):
    value = json.loads(content, object_pairs_hook=unique_object)
    if not isinstance(value,dict) or set(value) != {'status','answer','citations'}:
        raise ValueError('Require status, answer and citations object')
    if value['status'] not in ('answered','insufficient_evidence') or not isinstance(value['answer'],str) or not isinstance(value['citations'],list):
        raise ValueError('Invalid answer field types or status')
    for citation in value['citations']:
        if (not isinstance(citation,dict) or set(citation) != {'url','quote'}
                or any(not isinstance(citation[k],str) or not citation[k].strip() for k in ('url','quote'))):
            raise ValueError('Each citation requires nonempty url and quote strings')
    if value['status'] == 'answered':
        if not value['answer'].strip() or not value['citations']:
            raise ValueError('Answered requires nonempty answer and citations')
    elif value['answer'] != '' or value['citations']:
        raise ValueError('Insufficient evidence requires empty answer and citations')
    return value


class AnswerFormatClient:
    """Bind a schema only to explicit final-only answer calls, not readiness helpers."""
    answer_format = "json_evidence"

    def __init__(self, client, *, require_explicit_finalization=False):
        if type(require_explicit_finalization) is not bool:
            raise ValueError('Invalid explicit finalization policy')
        self.client = client
        self.require_explicit_finalization = require_explicit_finalization

    def __getattr__(self,name):
        return getattr(self.client,name)

    def __call__(self,*args,**kwargs):
        if kwargs.get('final_only'):
            kwargs['format_schema'] = EVIDENCE_SCHEMA
        return self.client(*args,**kwargs)
