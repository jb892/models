# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import logging
import argparse
import csv
import json
from plyfile import PlyData, PlyElement


__all__ = ["read_mesh_vertices", "read_aggregation", "read_segmentation", "read_label_mapping", "export"]

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("ScanNetV2 preprocess script")
parser.add_argument('--config_file', default='config.json', required=False, help='Configuration JSON file.')
args = parser.parse_args()

class ScanNetPreprocessor(object):
    def __init__(self, config_file):
        assert os.path.isfile(config_file)

        self.config_file = config_file

        # Parse config.json file
        logger.info('Parsing {} file...'.format(config_file))

        with open(config_file) as f:
            data = json.load(f)
            logger.info('Dataset_name: {}'.format(data['dataset_name']))

            self.SCANNET_DIR = print_h(data['scans_dir'])
            self.SAMPLE_FILE_PATH_LIST = print_h(data['sample_file_path'])
            self.LABEL_MAP_FILE_PATH = print_h(data['label_map_file_path'])
            self.OBJECT_CLASS_IDS = print_h(data['object_class_ids'])
            self.DO_NOT_CARE_CLASS_IDS = print_h(data['do_not_care_class_ids'])
            self.MAX_NUM_POINT = print_h(data['max_num_point'])
            self.OUTPUT_DIR = print_h(data['output_dir'])
            self.EXPORT_BBOX = print_h(data['export_bbox'])
            self.LABEL_MAP = read_label_mapping(self.LABEL_MAP_FILE_PATH, label_from='raw_category', label_to='nyu40id')

            assert os.path.isdir(self.SCANNET_DIR)
            for item in self.SAMPLE_FILE_PATH_LIST:
                assert os.path.isfile(item)
            assert os.path.isfile(self.LABEL_MAP_FILE_PATH)

        logger.info('Done config file parsing!')

    def run(self):

        for sample_file in self.SAMPLE_FILE_PATH_LIST:

            # Read scene name list
            scene_name_list = [line.strip('\n') for line in open(sample_file).readlines()]

def read_mesh_vertices(filename, load_color=False):
    """ read XYZ or RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        channel = 6 if load_color else 3
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, channel], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        if load_color:
            vertices[:, 3] = plydata['vertex'].data['red']
            vertices[:, 4] = plydata['vertex'].data['green']
            vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices

def read_aggregation(filename):
    """
    read aggregation file which contains the label and object_id for each segment group
    """
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    """
    read segmentation file which contains segment group IDs for each vertex
    """
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def represents_int(s):
    ''' if string s represents an int. '''
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping

def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = read_label_mapping(label_map_file,
        label_from='raw_category', label_to='nyu40id')
    mesh_vertices = read_mesh_vertices(mesh_file, load_color=True)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances,7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox

    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids,\
        instance_bboxes, object_id_to_label_id

def print_h(data):
    # print helper funtion
    logger.info(data)
    return data

def test_export():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(opt.scan_path, scan_name + '.txt')  # includes axisAlignment info for the train set scans.
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file, opt.output_file)

if __name__ == '__main__':

    # Test read_segmentation function
    seg_filename = 'scans/scene0000_00/scene0000_00_vh_clean_2.0.010000.segs.json'
    seg_to_verts, num_verts = read_segmentation(seg_filename)

    # Test read aggregation function
    agg_filename = 'scans/scene0000_00/scene0000_00.aggregation.json'
    object_id_to_segs, label_to_segs = read_aggregation(agg_filename)

    # Test read_mesh_vertices function
    mesh_filename = 'scans/scene0000_00/scene0000_00_vh_clean_2.ply'
    vertices = read_mesh_vertices(mesh_filename, load_color=True)

    test_export()