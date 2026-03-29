"""
Generate normalized chief complaint data.

4-step normalization pipeline:
  1. Case-insensitive + whitespace trim
  2. Medical abbreviation expansion (40+ mappings)
  3. Punctuation removal + whitespace collapse
  4. Fuzzy spelling correction (90% similarity threshold)

Input:  raw_data/triage_{train,valid,test}.csv
Output: temp_data/triage_{train,valid,test}_relabeled.csv
"""

import pandas as pd
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw_data"
TEMP_DIR = Path(__file__).resolve().parent / "temp_data"

# Medical abbreviation → full form
ABBREV_MAP = {
    r'\babd\b': 'abdominal', r'\bsob\b': 'shortness of breath',
    r'\bdyspnea\b': 'shortness of breath', r'\bdoe\b': 'shortness of breath on exertion',
    r'\bcp\b': 'chest pain', r'\bn/v/d\b': 'nausea vomiting diarrhea',
    r'\bn/v\b': 'nausea vomiting', r'\bams\b': 'altered mental status',
    r'\bloc\b': 'loss of consciousness', r'\bmvc\b': 'motor vehicle collision',
    r'\bmva\b': 'motor vehicle accident', r'\bs/p\b': 'status post',
    r'\bfx\b': 'fracture', r'\bbrbpr\b': 'bright red blood per rectum',
    r'\bhtn\b': 'hypertension', r'\bdm\b': 'diabetes',
    r'\bha\b': 'headache', r'\bh/a\b': 'headache',
    r'\buti\b': 'urinary tract infection', r'\betoh\b': 'alcohol',
    r'\bsi\b': 'suicidal ideation', r'\bgi\b': 'gastrointestinal',
    r'\bgib\b': 'gastrointestinal bleed', r'\bugib\b': 'upper gastrointestinal bleed',
    r'\blgib\b': 'lower gastrointestinal bleed', r'\bcva\b': 'cerebrovascular accident',
    r'\btia\b': 'transient ischemic attack', r'\bich\b': 'intracranial hemorrhage',
    r'\bpe\b': 'pulmonary embolism', r'\bdvt\b': 'deep vein thrombosis',
    r'\bchf\b': 'congestive heart failure', r'\bafib\b': 'atrial fibrillation',
    r'\ba-?fib\b': 'atrial fibrillation', r'\bcopd\b': 'chronic obstructive pulmonary disease',
    r'\buri\b': 'upper respiratory infection', r'\bped\b': 'pedestrian',
    r'\blac\b': 'laceration', r'\bdsb\b': 'deliberate self harm',
    r'\boi\b': 'overdose ingestion', r'\bod\b': 'overdose',
    r'\brlq\b': 'right lower quadrant', r'\bruq\b': 'right upper quadrant',
    r'\bllq\b': 'left lower quadrant', r'\bluq\b': 'left upper quadrant',
    r'\bl\b': 'left', r'\br\b': 'right', r'\blt\b': 'left', r'\brt\b': 'right',
    r'\bpalps\b': 'palpitations', r'\bsz\b': 'seizure',
    r'\bw/\b': 'with', r'\bb/l\b': 'bilateral',
    r'\ble\b': 'lower extremity', r'\bue\b': 'upper extremity',
}

FUZZY_THRESHOLD = 0.90
FUZZY_MAX_WORDS = 3
FUZZY_BLOCK_LIMIT = 100
FILES = ['triage_train.csv', 'triage_valid.csv', 'triage_test.csv']


def expand_and_clean(text):
    """Steps 1-3: lowercase, expand abbreviations, remove punctuation, collapse spaces."""
    if pd.isna(text):
        return text
    t = str(text).lower().strip()
    if not t:
        return t
    for pattern, replacement in ABBREV_MAP.items():
        t = re.sub(pattern, replacement, t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def sort_key(text):
    """Word-order-invariant key for grouping."""
    return ' '.join(sorted(text.split())) if text else ''


def build_fuzzy_groups(freq):
    """Step 4: fuzzy match similar short complaints (90% threshold)."""
    # Group by sorted-word key, pick most frequent natural form as canonical
    sort_groups = defaultdict(list)
    for cleaned, count in freq.items():
        sort_groups[sort_key(cleaned)].append((cleaned, count))

    canonical_map = {}
    for sk, members in sort_groups.items():
        canonical_map[sk] = max(members, key=lambda x: x[1])[0]

    # Union-Find for fuzzy merging
    parent = {sk: sk for sk in canonical_map}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Block by shared word, match only same-length short keys
    short_keys = [sk for sk in canonical_map if 0 < len(sk.split()) <= FUZZY_MAX_WORDS]
    blocks = defaultdict(list)
    for sk in short_keys:
        for word in sk.split():
            blocks[word].append(sk)

    for word, members in blocks.items():
        if len(members) > FUZZY_BLOCK_LIMIT:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if find(a) == find(b):
                    continue
                if len(a.split()) != len(b.split()):
                    continue
                if SequenceMatcher(None, a, b).ratio() >= FUZZY_THRESHOLD:
                    union(a, b)

    # For each fuzzy group, pick the canonical with highest total frequency
    fuzzy_groups = defaultdict(list)
    for sk in canonical_map:
        fuzzy_groups[find(sk)].append(sk)

    final_label = {}
    for root, skeys in fuzzy_groups.items():
        best_sk = max(skeys, key=lambda sk: sum(c for _, c in sort_groups[sk]))
        label = canonical_map[best_sk]
        for sk in skeys:
            final_label[sk] = label

    return final_label, sort_groups


def main():
    TEMP_DIR.mkdir(exist_ok=True)

    # Load all files and count frequencies of cleaned forms
    all_dfs = {f: pd.read_csv(RAW_DIR / f) for f in FILES}
    from collections import Counter
    freq = Counter()
    for df in all_dfs.values():
        for c in df['chiefcomplaint'].dropna():
            freq[expand_and_clean(c)] += 1

    # Build fuzzy mapping
    final_label, sort_groups = build_fuzzy_groups(freq)
    print(f"Unique after normalization: {len(sort_groups)}")
    print(f"Unique after fuzzy grouping: {len(set(final_label.values()))}")

    # Apply to each file
    def map_complaint(text):
        if pd.isna(text):
            return text
        cleaned = expand_and_clean(text)
        sk = sort_key(cleaned)
        return final_label.get(sk, cleaned)

    for fname in FILES:
        df = all_dfs[fname].copy()
        df['chiefcomplaint'] = df['chiefcomplaint'].apply(map_complaint)
        outname = fname.replace('.csv', '_relabeled.csv')
        df.to_csv(TEMP_DIR / outname, index=False)
        unique = df['chiefcomplaint'].nunique()
        print(f"{fname} -> {outname} | unique: {unique}")

    print(f"\nSaved to {TEMP_DIR}/")


if __name__ == '__main__':
    main()
