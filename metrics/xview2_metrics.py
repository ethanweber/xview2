# xview2 metrics
# 
# the total score is calculated a weighted average of the localization f1 score (lf1) and the damage f1 score (df1)
# score = .3 * lf1 + .7 * df1
# 
# the df1 is calculated by taking the harmonic mean of the 4 damage f1 scores (no damage, minor damage, major damage, and destroyed)
# df1 = 4 / sum((f1+epsilon)**-1 for f1 in [no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1]), where epsilon = 1e-6
# 
# Abbreviations used in this file:
# l: localization
# d: damage
# p: prediction
# t: target (ground truth)
# x: usually a numpy array

import os, json

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image
from typing import Union, List

class PathHandler:
    def __init__(self, pred_dir:Path, targ_dir:Path, img_id:str, test_hold:str):
        """
        Args:
            pred_dir  (Path): directory of localization and damage predictions
            targ_dir  (Path): directory of localization and damage targets
            img_id    (str) : 5 digit string of image id
            test_hold (str) : either 'test' or 'hold'. Most likely 'test' unless you have access to holdout set
        """
        assert isinstance(pred_dir, Path), f"pred_dir should be of type Path, got {type(pred_dir)}"
        assert pred_dir.is_dir(), f"Directory '{pred_dir}' does not exist or is not a directory"
        
        assert isinstance(targ_dir, Path), f"targ_dir '{targ_dir}' should be of type Path, got {type(pred_dir)}"
        assert targ_dir.is_dir(), f"Directory '{targ_dir}' does not exist or is not a directory"
        
        assert test_hold in ['test', 'hold'], f"test_hold '{test_hold}' was not one of 'test' or 'hold'"
        
        self.lp = pred_dir/f"{test_hold}_localization_{img_id}_prediction.png" # localization prediction
        self.dp = pred_dir/f"{test_hold}_damage_{img_id}_prediction.png" # damage prediction
        self.lt = targ_dir/f"{test_hold}_localization_{img_id}_target.png" # localization target
        self.dt = targ_dir/f"{test_hold}_damage_{img_id}_target.png" # damage target
        self.paths = (self.lp, self.dp, self.lt, self.dt)
        
    def load_and_validate_image(self, path):
        assert path.is_file(), f"file '{path}' does not exist or is not a file"
        img = np.array(Image.open(path))
        assert img.dtype == np.uint8, f"{path.name} is of wrong format {img.dtype} - should be np.uint8"
        assert set(np.unique(img)) <= {0,1,2,3,4}, f"values must ints 0-4, found {np.unique(img)}, path: {path}"
        assert img.shape == (1024,1024), f"{path} must be a 1024x1024 image"
        return img
    
    def load_images(self):
        return [self.load_and_validate_image(path) for path in self.paths]

class RowPairCalculator:
    """
    Contains all the information and functions necessary to calculate the true positives (TPs),
    false negatives (FNs), and false positives (FPs), for a pair of localization/damage predictions 
    """
    @staticmethod
    def extract_buildings(x:np.ndarray):
        """ Returns a mask of the buildings in x """
        buildings = x.copy()
        buildings[x>0] = 1
        return buildings
    
    @staticmethod
    def compute_tp_fn_fp(pred:np.ndarray, targ:np.ndarray, c:int) -> List[int]:
        """
        Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)
        Args:
            pred (np.ndarray): prediction
            targ (np.ndarray): target
            c (int): positive class
        """
        TP = np.logical_and(pred == c, targ == c).sum()
        FN = np.logical_and(pred != c, targ == c).sum()
        FP = np.logical_and(pred == c, targ != c).sum()
        return [TP, FN, FP]
        
    @classmethod
    def get_row_pair(cls, ph:PathHandler):
        """
        Builds a row of TPs, FNs, and FPs for both the localization dataframe and the damage dataframe.
        This pair of rows are built in the same function as damages are only assessed where buildings are predicted. 
        Args:
            ph (PathHandler): used to load the required prediction and target images
        """
        lp,dp,lt,dt = ph.load_images()
        lp_b,lt_b,dt_b = map(cls.extract_buildings, (lp,lt,dt)) # convert all damage scores 1-4 to 1

        dp = dp*lp_b # only give credit to damages where buildings are predicted
        dp, dt = dp[dt_b==1], dt[dt_b==1] # only score damage where there exist buildings in target damage

        lrow = cls.compute_tp_fn_fp(lp_b, lt_b, 1)
        drow = []
        for i in range(1,5): drow += cls.compute_tp_fn_fp(dp, dt, i)
        return lrow, drow

class F1Recorder:
    """
    Records the precision and recall when calculating the f1 score.
    Read about the f1 score here: https://en.wikipedia.org/wiki/F1_score
    """
    
    def __init__(self, TP, FP, FN, name=''):
        """
        Args:
            TP (int): true positives
            FP (int): false positives
            FN (int): false negatives
            name (str): optional name when printing
        """
        self.TP,self.FN,self.FP,self.name = TP,FN,FP,name
        self.P = self.precision()
        self.R = self.recall()
        self.f1 = self.f1()
        
    def __repr__(self):
        return f'{self.name} | f1: {self.f1:.4f}, precision: {self.P:.4f}, recall: {self.R:.4f}'
        
    def precision(self):
        """ calculates the precision using the true positives (self.TP) and false positives (self.FP)"""
        assert self.TP >= 0 and self.FP >= 0
        if self.TP == 0: return 0
        else: return self.TP/(self.TP+self.FP)

    def recall(self):
        """ calculates recall using the true positives (self.TP) and false negatives (self.FN) """
        assert self.TP >= 0 and self.FN >= 0
        if self.TP == 0: return 0
        return self.TP/(self.TP+self.FN)

    def f1(self):
        """ calculates the f1 score using precision (self.P) and recall (self.R) """
        assert 0 <= self.P <= 1 and 0 <= self.R <= 1
        if self.P == 0 or self.R == 0: return 0
        return (2*self.P*self.R)/(self.P+self.R)

class XviewMetrics:
    """
    Calculates the xview2 metrics given a directory of predictions and a directory of targets
    
    Directory of predictions and directory of targets must be two separate directories. These
    could be structured as followed:
        .
        ├── predictions
        │   ├── test_damage_00000_prediction.png
        │   ├── test_damage_00001_prediction.png
        │   ├── test_localization_00000_prediction.png
        │   ├── test_localization_00001_prediction.png
        │   └── ...
        └── targets
            ├── test_damage_00000_target.png
            ├── test_damage_00001_target.png
            ├── test_localization_00000_target.png
            ├── test_localization_00001_target.png
            └── ...
    """
    
    def __init__(self, pred_dir, targ_dir):
        self.pred_dir, self.targ_dir = Path(pred_dir), Path(targ_dir)
        assert self.pred_dir.is_dir(), f"Could not find prediction directory: '{pred_dir}'"
        assert self.targ_dir.is_dir(), f"Could not find target directory: '{targ_dir}'"
        
        self.dmg2str = {1:f'No damage     (1) ',
                        2:f'Minor damage  (2) ',
                        3:f'Major damage  (3) ',
                        4:f'Destroyed     (4) '}
        
        self.get_path_handlers()
        self.get_dfs()
        self.get_lf1r()
        self.get_df1rs()
        
    def __repr__(self):
        s = 'Localization:\n'
        s += f'    {self.lf1r}\n'
        
        s += '\nDamage:\n'
        for F1Rec in self.df1rs:
            s += f'    {F1Rec}\n'
        s += f'    Harmonic mean dmgs | f1: {self.df1:.4f}\n'
        
        s += '\nScore:\n'
        s += f'    Score | f1: {self.score:.4f}\n'
        return s.rstrip()
        
    def get_path_handlers(self):
        self.path_handlers = []
        for path in self.targ_dir.glob('*.png'):
            test_hold, loc_dmg, img_id, target = path.name.rstrip('.png').split('_')
            assert loc_dmg in ['localization', 'damage'], f"target filenames must have 'localization' or 'damage' in filename, got {path}"
            assert target == 'target', f"{target} should equal 'target' when getting path handlers"
            if loc_dmg == 'localization': # localization or damage is fine here
                self.path_handlers.append(PathHandler(self.pred_dir, self.targ_dir, img_id, test_hold))
        
    def get_dfs(self):
        """
        builds the localization dataframe (self.ldf) and damage dataframe (self.ddf) from
        path handlers (self.path_handlers)
        """
        with Pool() as p:
            all_rows = p.map(RowPairCalculator.get_row_pair, self.path_handlers)
        
        lcolumns = ['lTP','lFN','lFP']
        self.ldf = pd.DataFrame([lrow for lrow,drow in all_rows], columns=lcolumns)
        
        dcolumns = ['dTP1','dFN1','dFP1','dTP2','dFN2','dFP2',
                    'dTP3','dFN3','dFP3','dTP4','dFN4','dFP4']
        self.ddf = pd.DataFrame([drow for lrow,drow in all_rows], columns=dcolumns)
    
    def get_lf1r(self):
        """ localization f1 recorder """
        TP = self.ldf['lTP'].sum()
        FP = self.ldf['lFP'].sum()
        FN = self.ldf['lFN'].sum()
        self.lf1r = F1Recorder(TP, FP, FN, 'Buildings')
        
    @property
    def lf1(self):
        """ localization f1 """
        return self.lf1r.f1
        
    def get_df1rs(self):
        """ damage f1 recorders """
        self.df1rs = []
        for i in range(1,5):
            TP = self.ddf[f'dTP{i}'].sum()
            FP = self.ddf[f'dFP{i}'].sum()
            FN = self.ddf[f'dFN{i}'].sum()
            self.df1rs.append(F1Recorder(TP, FP, FN, self.dmg2str[i]))
    
    @property
    def df1s(self):
        """ damage f1s """
        return [F1.f1 for F1 in self.df1rs]
    
    @property
    def df1(self):
        """ damage f1. Computed using harmonic mean of damage f1s """
        harmonic_mean = lambda xs: len(xs) / sum((x+1e-6)**-1 for x in xs)
        return harmonic_mean(self.df1s)
        
    @property
    def score(self):
        """ xview2 score computed as a weighted average of the localization f1 and damage f1 """
        return 0.3 * self.lf1 + 0.7 * self.df1
        
    @classmethod
    def compute_score(cls, pred_dir, targ_dir, out_fp):
        """
        Args:
            pred_dir (str): directory of localization and damage predictions
            targ_dir (str): directory of localization and damage targets
            out_fp   (str): output json - folder must already exist
        """
        print(f"Calculating metrics using {cpu_count()} cpus...")
        
        self = cls(pred_dir, targ_dir)
        
        d = {'score':self.score,
             'damage_f1':self.df1,
             'localization_f1':self.lf1}
        d['damage_f1_no_damage'] = self.df1s[0]
        d['damage_f1_minor_damage'] = self.df1s[1]
        d['damage_f1_major_damage'] = self.df1s[2]
        d['damage_f1_destroyed'] = self.df1s[3]
        
        with open(out_fp, 'w') as f: json.dump(d, f)
        print(f"Wrote metrics to {out_fp}")

if __name__ == '__main__':
    import fire
    fire.Fire(XviewMetrics.compute_score)