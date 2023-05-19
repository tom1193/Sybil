from sybil import Serie, Sybil
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--nifti', default=False, action='store_true')
parser.add_argument('--test', default=True, action='store_true')
args = parser.parse_args()

cohort = pd.read_csv('/home/local/VANDERBILT/litz/data/nlst/nlst_dicom_fpaths.csv', dtype={'pid':str})
ardila = pd.read_csv('/home/local/VANDERBILT/litz/github/MASILab/DeepLungScreening/cohorts/nlst/nlst_ardila_test.csv', dtype={'pid':str})

ardila_dicom = ardila.merge(cohort, left_on=['pid', 'scandate'], right_on=['pid', 'year']).iloc[3005:3008]
# ardila_dicom = ardila_dicom.iloc[:10]

phase = 'test' if args.test else 'eval'
ftype = 'nifti' if args.nifti else 'dicom'
logger = logging.basicConfig(filename='logs/main.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', level=logging.DEBUG)


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
    np.save(f'./data/nlst/{ftype}_ardila_scores.npy', ardila_scores)

else:
    dicom_scores = np.load('./data/nlst/dicom_ardila_scores.npy')
    # nifti_scores = np.load('./data/nlst/nifti_ardila_scores.npy')
    max_dicom_scores = np.max(dicom_scores, axis=1)
    # max_nifti_scores = np.max(nifti_scores, axis=1)
    ardila_dicom['sybil_dicom'] = max_dicom_scores
    print('helo')