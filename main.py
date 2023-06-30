from sybil import Serie, Sybil
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import argparse
import logging
import sys
sys.path.append('/home/local/VANDERBILT/litz/github/MASILab/toolkit/')
from statutils import bootstrap
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--nifti', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--limit', default=None, type=int)
args = parser.parse_args()

cohort = pd.read_csv('/home/local/VANDERBILT/litz/data/nlst/nlst_dicom_fpaths.csv', dtype={'pid':str})
ardila = pd.read_csv('/home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/cohorts/nlst/nlst_ardila_test.csv', dtype={'pid':str})
ardila['year'] = pd.to_datetime(ardila['scandate']).apply(lambda x: x.year)
ardila_dicom = ardila.merge(cohort, on=['pid', 'year'])
if args.limit:
    ardila_dicom = ardila_dicom.iloc[:args.limit]

# ardila_dicom = ardila_dicom.iloc[:10]

phase = 'test' if args.test else 'eval'
ftype = 'nifti' if args.nifti else 'dicom'
logger = logging.basicConfig(filename=f'log/main_{ftype}.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', level=logging.DEBUG)

if phase == 'test':
    if ftype == 'dicom':
        paths = [[i.path for i in os.scandir(n)] for n in ardila_dicom['series_path'].tolist()]
    elif ftype == 'nifti':
        paths = [n for n in ardila_dicom['fpath'].tolist()]

    # Load a trained model
    model = Sybil("sybil_base")

    # Get risk scores
    ardila_scores = np.zeros((len(ardila_dicom), 6))
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        try:
            serie = Serie([path], file_type='nifti') if ftype=='nifti' else Serie(path)
            scores = model.predict([serie])
            ardila_scores[i] = scores.scores[0]
        except Exception as e:
            logging.exception(e)
            continue

    # ardila_dicom['sybil_risk'] = scores.scores
    if args.limit:
        np.save(f'./data/nlst/{ftype}_ardila_scores_n{str(args.limit)}.npy', ardila_scores)
    else:
        np.save(f'./data/nlst/{ftype}_ardila_scores.npy', ardila_scores)

else:
    dicom_scores = np.load('./data/nlst/dicom_ardila_scores.npy')
    nifti_scores = np.load('./data/nlst/nifti_ardila_scores.npy')
    dicom_year1 = dicom_scores[:,0]
    nifti_year1 = nifti_scores[:,0]
    ardila_dicom['sybil_dicom'] = dicom_year1
    ardila_dicom['sybil_nifti'] = nifti_year1
    dicom_metrics = bootstrap(ardila_dicom, lambda x: roc_auc_score(x['cancer_year1'], x['sybil_dicom']), n=1000)
    nifti_metrics = bootstrap(ardila_dicom, lambda x: roc_auc_score(x['cancer_year1'], x['sybil_nifti']), n=1000)

    print('helo')