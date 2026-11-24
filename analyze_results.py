"""Recalculate NLP HW2 MCQA results from saved raw inference responses (CPU only).

Example:
    python analyze_results.py --results results/main_1280_results.csv

This file does not execute Qwen inference or fetch/redistribute any KMMLU question text.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map({'true': True, 'false': False})
    if mapped.isna().any():
        raise ValueError(f'Unexpected boolean entries in {series.name}')
    return mapped.astype(bool)


def paired_test(differences: np.ndarray, rng: np.random.Generator, bootstrap_n=2000, signflip_n=10000):
    d = np.asarray(differences, dtype=float)
    n = len(d)
    observed = d.mean()
    indices = rng.integers(0, n, size=(bootstrap_n, n))
    boots = d[indices].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    signs = rng.choice([-1, 1], size=(signflip_n, n))
    null_means = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null_means) >= abs(observed)) + 1) / (signflip_n + 1)
    return dict(difference=observed, ci_low=lo, ci_high=hi, p_value=p)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def analyze(results_file: Path, output_dir: Path):
    df = pd.read_csv(results_file, keep_default_na=False)
    required = {'experiment_id','question_id','subject','rotation','label_format',
                'is_valid','is_correct','predicted_original_id','correct_original_id','correct_position'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if len(df) != 1280 or df.experiment_id.nunique() != 1280 or df.question_id.nunique() != 160:
        raise ValueError('Expected 1,280 unique experiments across 160 unique questions')
    for col in ['is_valid','is_correct']:
        df[col] = parse_bool(df[col])
    df['predicted_original_id'] = pd.to_numeric(df['predicted_original_id'], errors='coerce')
    df['correct_original_id'] = pd.to_numeric(df['correct_original_id'], errors='raise')
    df['correct_position'] = pd.to_numeric(df['correct_position'], errors='raise')
    df['rotation'] = pd.to_numeric(df['rotation'], errors='raise')
    if not df['predicted_original_id'].eq(df['correct_original_id']).equals(df['is_correct']):
        raise ValueError('Correctness values do not match decoded original option IDs')
    if df['label_format'].value_counts().to_dict() != {'Latin':640, 'Hangul':640}:
        raise ValueError('Expected 640 records per label format')

    rows = []
    for (qid, fmt), group in df.groupby(['question_id','label_format']):
        if len(group) != 4 or sorted(group.rotation.tolist()) != [0,1,2,3]:
            raise ValueError(f'Incomplete or duplicated rotations: {qid} {fmt}')
        if sorted(group.correct_position.tolist()) != [1,2,3,4]:
            raise ValueError(f'Unbalanced answer positions: {qid} {fmt}')
        consistent = bool(group.is_valid.all() and group.predicted_original_id.nunique() == 1)
        rows.append(dict(question_id=qid, subject=group.subject.iloc[0],
                         label_format=fmt, accuracy=group.is_correct.mean(),
                         consistency=int(consistent),invalid_rate=1-group.is_valid.mean()))
    item_df = pd.DataFrame(rows)
    if len(item_df) != 320:
        raise ValueError('Expected 320 question × format records')
    latin = item_df[item_df.label_format=='Latin'].set_index('question_id').sort_index()
    hangul = item_df[item_df.label_format=='Hangul'].set_index('question_id').sort_index()
    if not latin.index.equals(hangul.index):
        raise ValueError('Unmatched paired questions')
    rng = np.random.default_rng(42)
    h1 = paired_test((hangul.accuracy-latin.accuracy).to_numpy(),rng)
    h2 = paired_test((hangul.consistency-latin.consistency).to_numpy(),rng)
    corrected = holm_adjust([h1['p_value'],h2['p_value']])
    h1['p_holm'] = corrected[0]
    h2['p_holm'] = corrected[1]
    tests = pd.DataFrame([{'hypothesis':'H1 Accuracy',**h1},
                          {'hypothesis':'H2 Consistency',**h2}])
    subj = (item_df.groupby(['subject','label_format'])
            .agg(accuracy=('accuracy','mean'),consistency=('consistency','mean'),
                 invalid_rate=('invalid_rate','mean'),questions=('question_id','nunique'))
            .reset_index())
    subj[['accuracy','consistency','invalid_rate']] *= 100
    position = (df.groupby(['label_format','correct_position'])
                .agg(total=('is_correct','size'),accuracy=('is_correct','mean')).reset_index())
    position['accuracy'] *= 100
    overall = (df.groupby('label_format')
               .agg(total=('is_correct','size'),valid=('is_valid','sum'),correct=('is_correct','sum'),
                    accuracy=('is_correct','mean')))
    overall['invalid'] = overall.total-overall.valid
    output_dir.mkdir(parents=True,exist_ok=True)
    item_df.to_csv(output_dir/'item_level_metrics.csv',index=False)
    tests.to_csv(output_dir/'hypothesis_tests.csv',index=False)
    subj.to_csv(output_dir/'subject_analysis.csv',index=False)
    position.to_csv(output_dir/'position_analysis.csv',index=False)
    print('MAIN RESULTS\n', overall.to_string())
    print('\nSTATISTICAL TESTS\n',tests.to_string(index=False))
    print('\nPOSITION ACCURACY\n',position.to_string(index=False))
    print('\nSaved tables to',output_dir)
    return overall,tests

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/main_1280_results.csv'))
    parser.add_argument('--output',type=Path,default=Path('derived_tables'))
    args=parser.parse_args()
    if not args.results.is_file():
        parser.error(f'File not found: {args.results}. Download the saved inference CSV first.')
    analyze(args.results,args.output)
