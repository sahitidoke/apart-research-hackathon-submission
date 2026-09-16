# Fixtures

`smoke.jsonl` is three synthetic questions with fictional entities (2-, 3- and
4-hop), shaped like MuSiQue-Ans records including `question_decomposition`.
It exists so the whole package runs offline with no dataset download and no
GPU.

It is not a dataset. Three invented questions cannot show anything about
deference, provenance or difficulty scaling, and a number produced from it is
not a result. Real runs need MuSiQue-Ans via `--dataset`.
