#!/usr/bin/env python3
import vcf
from collections import defaultdict
from functools import reduce

import argparse
import numpy as np

import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics
import sklearn.ensemble
import sklearn.svm

from datetime import datetime
import sys

def make_variants(vcf_filename):
  variants = defaultdict(list)
  def add_bool(attribs, variant, key):
    attribs[key.lower()] = key in variant.INFO and variant.INFO[key] == True

  # http://www.ncbi.nlm.nih.gov/variation/docs/human_variation_vcf/#clinvar
  # Encoding must be specified to prevent UnicodeDecodeError. See
  # https://github.com/jamescasbon/PyVCF/issues/201.
  vcfr = vcf.Reader(filename=vcf_filename, encoding='utf-8')
  cnt = 0
  for variant in vcfr:
    cnt += 1
    var = {}
    clnsig = tuple(variant.INFO['CLNSIG'])
    if 'CAF' in variant.INFO:
      cafs = [float(v) for v in variant.INFO['CAF'][1:] if v is not None]
      var['MAF'] = sum(cafs)
    add_bool(var, variant, 'NSF')
    add_bool(var, variant, 'NSM')
    add_bool(var, variant, 'NSN')
    add_bool(var, variant, 'REF')
    add_bool(var, variant, 'SYN')
    add_bool(var, variant, 'U3')
    add_bool(var, variant, 'U5')
    add_bool(var, variant, 'ASS')
    add_bool(var, variant, 'DSS')
    add_bool(var, variant, 'INT')
    add_bool(var, variant, 'R3')
    add_bool(var, variant, 'R5')

    variants[clnsig].append(var)

  return variants

def assign_classes(variants):
  munged = {}
  mapping = {
    '5': 'pathogenic',
    '4': 'likely_pathogenic',
    '3': 'likely_benign',
    '2': 'benign',
    '0': 'unknown',
  }

  # Merge multiple identical reports together.
  for idxs, vars in variants.items():
    idxs = set(idxs)
    if len(idxs) > 1:
      continue
    idx = idxs.pop()
    if idx not in mapping:
      continue

    label = mapping[idx]
    if label not in munged:
      munged[label] = []
    munged[label] += vars

  return munged

def print_feature_counts(variants):
  all_features = set()
  counts = {}

  for label, vars in variants.items():
    counts[label] = defaultdict(int)
    for var in vars:
      for key in var.keys():
        counts[label][key] += 1
        all_features.add(key)

  sorted_features = tuple(sorted(all_features))
  print('\t'.join(('label',) + sorted_features + ('total',)), file=sys.stderr)
  for label in sorted(counts.keys()):
    total = len(variants[label])
    feature_counts = tuple(['%.3f' % (counts[label][f] / float(total)) for f in sorted_features])
    print('\t'.join((label,) + feature_counts + (str(total),)), file=sys.stderr)

def impute_missing(variants):
  mafs = []

  for label, vars in variants.items():
    for var in vars:
      if 'MAF' in var:
        mafs.append(var['MAF'])

  missing_maf = np.mean(mafs)
  # Best values
  #missing_maf = 1.0

  for label, vars in variants.items():
    for var in vars:
      if 'MAF' not in var:
        var['MAF'] = missing_maf

def vectorize_variants(variants):
  vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
  # List all variants (which are dictionaries) in single list regardless of
  # what class (pathogenic, benign, ...) they fall in.
  all_vars = reduce(lambda a, b: a + b, variants.values())
  vectorizer.fit(all_vars)

  vectorized_vars = {}
  for clnsig, vars in variants.items():
    vectorized_vars[clnsig] = vectorizer.transform(vars)

  return vectorized_vars

def predict(model, data):
  return model.predict_proba(data)[:,1]

def create_models():
  models = (
    ('Logistic regression', sklearn.linear_model.LogisticRegression(class_weight='balanced', penalty='l2')),
    ('Random forest', sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=16)),
    ('SVM', sklearn.svm.SVC(probability=True)),
  )
  return models

def concat_training_data(variants):
  # Labels: 1=pathogenic, 0=benign
  vars = np.vstack((variants['pathogenic'], variants['benign']))
  labels_pathogenic = np.ones(variants['pathogenic'].shape[0])
  labels_benign = np.zeros(variants['benign'].shape[0])
  labels = np.concatenate((labels_pathogenic, labels_benign))
  return (vars, labels)

def train_model(variants, model):
  vars, labels = concat_training_data(variants)
  skf = sklearn.cross_validation.StratifiedKFold(labels, n_folds=3, shuffle=True)

  shape = labels.shape
  validation_probs = np.zeros(shape)
  training_probs = np.zeros(shape)
  validation_times_used = np.zeros(shape)
  training_times_used = np.zeros(shape)

  for train_index, validation_index in skf:
    training_data, validation_data = vars[train_index], vars[validation_index]
    training_labels, validation_labels = labels[train_index], labels[validation_index]

    validation_fold_probs = np.zeros(validation_labels.shape)
    training_fold_probs   = np.zeros(training_labels.shape)

    model.fit(training_data, training_labels)

    validation_fold_probs = predict(model, validation_data)
    validation_probs[validation_index] += validation_fold_probs
    validation_times_used[validation_index] += 1

    training_fold_probs = predict(model, training_data)
    training_probs[train_index] += training_fold_probs
    # This should always be n_folds - 1, but I track it to be sure this
    # assumption is correct.
    training_times_used[train_index] += 1

    #validation_preds = np.round(validation_probs)
    #print('F1 (fold):', sklearn.metrics.f1_score(validation_labels, validation_preds))

  training_probs /= training_times_used
  validation_probs /= validation_times_used

  return (labels, training_probs, validation_probs)

def evaluate_model(variants):
  for model_type, model in create_models():
    print('Training %s ...' % model_type)
    labels, training_probs, validation_probs = train_model(variants, model)

    probs = validation_probs
    metrics = {}
    metrics['roc_auc'] = sklearn.metrics.roc_auc_score(labels, probs)
    metrics['pr_auc'] = sklearn.metrics.average_precision_score(labels, probs)
    metrics['f1'] = sklearn.metrics.f1_score(labels, np.round(probs))
    print(metrics)

def logmsg(msg, fd=sys.stdout):
  now = datetime.now()
  if logmsg.last_log is None:
    time_delta = 0
  else:
    time_delta = now - logmsg.last_log
  print('[%s] %s' % (time_delta, msg), file=fd)
  logmsg.last_log = now
logmsg.last_log = None

def main():
  np.set_printoptions(threshold=np.nan)
  np.random.seed(1)

  parser = argparse.ArgumentParser(
    description='Various classifiers to predict variant pathogenicity',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('vcf_file')
  args = parser.parse_args()

  logmsg('Loading variants ...')
  variants = make_variants(args.vcf_file)

  logmsg('Vectorizing variants ...')
  variants = assign_classes(variants)
  print_feature_counts(variants)
  impute_missing(variants)
  variants = vectorize_variants(variants)

  evaluate_model(variants)

if __name__ == '__main__':
  main()
