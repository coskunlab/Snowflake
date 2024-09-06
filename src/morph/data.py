import cv2
import os
import numpy as np
import PIL
import pandas as pd
import skimage.io
import multiprocessing
import math

from tqdm.auto import tqdm
from collections import defaultdict
from scipy import interpolate
from joblib import Parallel, delayed
from copy import deepcopy
import time
import pickle

PIL.Image.MAX_IMAGE_PIXELS = 933120000

class MORPH(object):
    def __init__(self, path, foll_data_path, gc_data_path):
        self.get_info(path)
        self.foll_data_path = foll_data_path
        self.GC_data_path = gc_data_path
        
    def load_boundaries(self):
        if not self.foll_data_path.exists():
            print('No Foll and GC boundaries found')
            self.extract_boundaries()

        foll_data = np.load(self.foll_data_path)
        gc_data = np.load(self.GC_data_path)

        # Get id
        foll_keys = foll_data.keys()
        print("We found %i foll segmentations"%len(foll_keys))
        self.df_bd = pd.DataFrame({'id': list(foll_keys), 'foll': foll_data.values(), 'gc': gc_data.values()})
        
        # get follcile with and without GCs
        df_gc = self.df_bd[self.df_bd.gc.apply(len) > 80]
        df_no_gc = self.df_bd[self.df_bd.gc.apply(len) <= 80]

        # One hot encoding of presence or not of GC
        self.gc_one_hot = (self.df_bd.gc.apply(len) > 80).to_numpy().astype(np.uint8) 

        self.with_gc_id = df_gc.index.tolist()
        self.no_gc_id = df_no_gc.index.tolist()

    def extract_boundaries(self):
        def _mask2boundary(mask):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = np.empty(2)
            for i in range(len(contours[0])):
                contour = np.vstack((contours[0][i][0], contour))
            contour = contour[0:-1]
            contour.T[0] = contour.T[0] + 1
            contour.T[1] = contour.T[1] + 1
            boundary = np.empty((2, len(contour.T[1])))
            boundary[0] = contour.T[1]
            boundary[1] = contour.T[0]
            boundary = boundary.T.astype(int)
            return boundary

        folls = {}
        gcs = {}

        for row in tqdm(self.df_info.itertuples(), total=len(self.df_info), desc='Dataset'):
            # Get mask info
            GC_mask_path = row.Path_GC
            foll_mask_path = row.Path_foll
            name = row.Dataset + row.Subset
            
            # Get foll and GC mask
            GC_mask = skimage.io.imread(GC_mask_path)
            foll_mask = skimage.io.imread(foll_mask_path)
            
            # Iterate through each follicles
            labels = np.unique(foll_mask)[1:]
            print(f'{name} has {len(labels)} follicles')
            
            for _, lab in tqdm(enumerate(labels), total=len(labels), desc='Follicles', leave=False):
                if name == '05' and lab==73:
                    continue
                
                # Extract follicle boundary
                mask = (foll_mask == lab).astype(np.uint8)
                boundary = mask2boundary(mask)
                folls[name+'_id'+str(lab)] = boundary
                
                # Extract GC boundary
                mask_gc = (GC_mask == lab).astype(np.uint8)
                if mask_gc.sum() == 0:
                    print(f'{name} follicle {lab} have no GC')
                    # If no GC mask use centroid of follicle as single contour entry
                    boundary_gc = boundary.mean(axis=0)[np.newaxis,:] 
                else:
                    boundary_gc = mask2boundary(mask_gc)
                gcs[name+'_id'+str(lab)] = boundary_gc

        np.savez(self.foll_data_path, **folls) 
        np.savez(self.gc_data_path, **gcs) 

    def get_info(self, path):
        info = defaultdict(list)

        print('Reading file info for all follicle and GC masks')
        # Loop through image folder
        for (dirpath, dirnames, filenames) in os.walk(path):
            for name in sorted(filenames):
                if "png" not in name or 'foll' not in name:
                    continue
                    
                n_split = name.split('_')
                if len(n_split)>2:
                    dataset = n_split[0]
                    subset = '_' + n_split[1]
                else:
                    dataset = n_split[0]
                    subset = ''
                
                path_foll = os.path.join(dirpath, name)
                path_GC = os.path.join(dirpath, dataset + subset + '_GC.png')
                
                info['Path_foll'].append(path_foll)
                info['Path_GC'].append(path_GC)
                info['Dataset'].append(dataset)
                info['Subset'].append(subset)

        df_info = pd.DataFrame(info)
        self.df_info = df_info

    def resample_boundaries(self, num_points=120, foll_col='foll', gc_col='gc'):
        print('Resampling boundaries')
        self.num_points = num_points
        n_contours = len(self.df_bd) # Number of contours 
        bd_sampled_foll = np.zeros([n_contours, 2 * self.num_points]) # Boundary coordinates
        bd_sampled_gc = np.zeros([n_contours, 2 * self.num_points])
        df_new = deepcopy(self.df_bd)
        df_new['R'] = np.nan
        df_new['t0'] = np.nan
        df_new['t1'] = np.nan
        ori = defaultdict(list)

        # For each boundary resample to N points and register
        for i in range(n_contours):  
            # Get max axis angle
            foll_ori = self.df_bd.iloc[i]['foll'].transpose()
            gc_ori = self.df_bd.iloc[i]['gc'].transpose()
            xc,yc = np.mean(gc_ori, axis=1)
            diff = foll_ori-np.array([xc,yc])[:, np.newaxis]
            x=diff[1]
            y=diff[0]
            angles = np.arctan2(y,x)
            lengths = np.apply_along_axis(lambda ele: np.sqrt(ele[0]**2 + ele[1]**2), 0, diff)
            a_max = np.argmax(lengths)
            df_new.loc[i, 't0'] = angles[a_max]

            bd_foll_resampled = self.bd_resample(self.df_bd.loc[i, foll_col], self.num_points)
            bd_gc_resampled = self.bd_resample(self.df_bd.loc[i, gc_col], self.num_points)
            a, b, R, t1 = self.reg_bd_SVD(bd_foll_resampled, bd_gc_resampled)
            
            ori[foll_col].append(bd_foll_resampled)
            ori[gc_col].append(bd_gc_resampled)
            df_new.loc[i, 'R'] = R
            df_new.loc[i, 't1'] = t1
            bd_sampled_foll[i] = np.append([a[1]], [a[0]], axis=1)    
            bd_sampled_gc[i] = np.append([b[1]], [b[0]], axis=1) 
            df_new.at[i, foll_col] = a
            df_new.at[i, gc_col] = b

        # Get average along each (x,y) direction from resampled data
        mean_foll = np.mean(bd_sampled_foll, axis=0)
        bd_foll_m = np.append([mean_foll[num_points:]], [mean_foll[0:num_points]], axis=0)
        
        start = time.time()
        # Foll
        bdreg = Parallel()(delayed(self.reg_bd3)(df_new.loc[k], bd_foll_m) for k in range(n_contours))
        bd_reg_foll = [np.append(bdreg[i][0][1],bdreg[i][0][0]) for i in range(len(bdreg))]
        bd_reg_foll = np.array(bd_reg_foll)
        
        bd_reg_gc = [np.append(bdreg[i][1][1],bdreg[i][1][0]) for i in range(len(bdreg))]
        bd_reg_gc = np.array(bd_reg_gc)
        end = time.time()
        print('For parallel of bdreg, elapsed time is ' + str(end - start) + 'seconds...')

        self.df_new = df_new
        self.ori = ori
        self.bd_reg_foll = bd_reg_foll
        self.bd_reg_gc = bd_reg_gc
    
    def bd_resample(self, bd, n, s=0):
        if len(bd) == 1:
            bd_tiled = np.tile(bd, (n,1))
            return bd_tiled.transpose()
        
        # Get list of x and y coordinates
        x = bd.transpose()[1]
        y = bd.transpose()[0]
        
        # Check if last element is the same as first element and add it if not
        if x[-1] != x[0] and y[-1] != y[0]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        
        # Get distance between coord
        d = np.sqrt(np.power(x[1:]-x[0:-1], 2) + np.power(y[1:]-y[0:-1], 2))
        d = np.append([1], d)
        cum_d = np.cumsum(d)
        
        # Sampled with spline representation
        sampled = np.linspace(1, cum_d[-1], n + 1)
        sampled = sampled[0:-1]
        splinerone = interpolate.splrep(cum_d, x, s=s)
        sx = interpolate.splev(sampled, splinerone)
        splinertwo = interpolate.splrep(cum_d, y, s=s)
        sy = interpolate.splev(sampled, splinertwo)
        
        sampled_bd = np.append([sy], [sx], axis=0)
        return sampled_bd    

    def reg_bd_SVD(self, bd_foll, bd_gc):
        yc, xc = np.mean(bd_gc, axis=1)

        # Centered boundary 
        yi_foll, xi_foll = np.append([bd_foll[0]] - yc, [bd_foll[1]] - xc, axis=0)
        yi_gc, xi_gc = np.append([bd_gc[0]] - yc, [bd_gc[1]] - xc, axis=0)
        
        # Get R parameter and normalize boudaries by R
        R = np.sqrt((sum(np.power(xi_foll, 2)) + sum(np.power(yi_foll, 2))) / len(xi_foll))
        xi_foll, yi_foll = xi_foll / R, yi_foll / R
        xi_gc, yi_gc = xi_gc / R, yi_gc / R

        # concatenate x and y coordinates into two rows
        xiyi_foll = np.append([xi_foll], [yi_foll], axis=0).transpose()
        xiyi_gc = np.append([xi_gc], [yi_gc], axis=0).transpose()

        # SVD decomposition to get Rotation matrix from foll boundaries
        u, S, rm = np.linalg.svd(xiyi_foll, full_matrices=True)
        if np.isnan(rm).any(): # Quick fix if nan values 
            rm[rm!=1]=1

        foll_new = np.dot(xiyi_foll, rm.transpose()).transpose()
        gc_new = np.dot(xiyi_gc, rm.transpose()).transpose()

        # Center new boundaries
        xc, yc = np.mean(gc_new, axis=1)
        x1,y1 = np.append([foll_new[0]]-xc, [foll_new[1]]-yc, axis=0)
        x2,y2 = np.append([gc_new[0]]-xc, [gc_new[1]]-yc, axis=0)

        # Reorder by angle
        theta = np.arctan2(y1,x1)
        i_min = np.argmin(abs(theta))
        theta_new = theta[i_min]
        new_indices = np.append(range(i_min, len(x1)), range(0, i_min)).astype(int)

        x1, y1 = x1[new_indices], y1[new_indices]
        x2, y2 = x2[new_indices], y2[new_indices]
        theta = theta[new_indices]

        if theta[4] - theta[0] < 0:
            x1 = np.append(x1[-1:0:-1], x1[0])
            y1 = np.append(y1[-1:0:-1], y1[0])
            x2 = np.append(x2[-1:0:-1], x2[0])
            y2 = np.append(y2[-1:0:-1], y2[0])
            theta = np.append(theta[-1:0:-1], theta[0])
            theta_new = theta[0] 

        bd_foll_registered = np.append([y1], [x1], axis=0)
        bd_gc_registered = np.append([y2], [x2], axis=0)
        return bd_foll_registered, bd_gc_registered, R, theta_new

    def reg_bd3(self, df_k, bdr0, foll_col='foll', gc_col='gc'):
        bd0 = df_k[foll_col]
        bd0_gc = df_k[gc_col]

        # Normalize coordinate by average
        xc = np.sum(np.dot(bd0[1], abs(bd0[0]))) / np.sum(abs(bd0[0]))
        yc = np.sum(np.dot(bd0[0], abs(bd0[1]))) / np.sum(abs(bd0[1]))
        
        bd = np.append([bd0[1] - xc], [bd0[0] - yc], axis=0)
        bd_gc = np.append([bd0_gc[1] - xc], [bd0_gc[0] - yc], axis=0)

        xc = np.sum(np.dot(bdr0[1], abs(bdr0[0]))) / np.sum(abs(bdr0[0]))
        yc = np.sum(np.dot(bdr0[0], abs(bdr0[1]))) / np.sum(abs(bdr0[1]))
        bdr = np.append([bdr0[1] - xc], [bdr0[0] - yc], axis=0)

        N = len(bd[0])
        
        costold = np.mean(sum(sum(np.power((bdr - bd), 2))))
        bdout = deepcopy(bd)
        bdout_gc = deepcopy(bd_gc)

        # print('regbd3')
        for k in range(1, N + 1):
            idk = np.append(range(k, N + 1), range(1, k))
            bdt = np.empty([len(idk), 2]) * np.nan
            bdt2 = np.empty([len(idk), 2]) * np.nan
            for i in range(len(bd.transpose())):
                ind = int(idk[i] - 1)
                bdt[i] = bd.transpose()[ind]
                bdt2[i] = bd_gc.transpose()[ind]
            temp = np.dot(bdr, bdt)
            u, _, v = np.linalg.svd(temp)
            v = v.T
            q = np.dot(v, u.transpose())
            bdtemp = np.dot(bdt, q)
            costnew = np.mean(sum(sum(np.power((bdr.transpose() - bdtemp), 2))))
            if costnew < costold:
                bdout = bdtemp 
                bdout_gc = np.dot(bdt2, q)
                costold = deepcopy(costnew)

        regbd = deepcopy(bdout.T)
        regbd[:] = np.nan
        regbd[0] = deepcopy(bdout.T[1])
        regbd[1] = deepcopy(bdout.T[0])
        
        regbd_gc = deepcopy(bdout_gc.T)
        regbd_gc[:] = np.nan
        regbd_gc[0] = deepcopy(bdout_gc.T[1])
        regbd_gc[1] = deepcopy(bdout_gc.T[0])
        return regbd, regbd_gc

    def save_pickle(self, path):
        try:
            os.remove(path)
            print("File exist. Deleted")
        except FileNotFoundError:
            pass

        # Open a file and use dump()
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

