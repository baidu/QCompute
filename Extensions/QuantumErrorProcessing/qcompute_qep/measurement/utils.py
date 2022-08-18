#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
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

"""
Utility functions used in the `qcompute_qep.measurement` package.

**Examples preparation**

    >>> import numpy as np
    >>> from qcompute_qep.measurement.utils import extract_substr, init_cal_data, dict2vector, vector2dict, state_labels

"""
import copy
import json
import math
from typing import List, Dict, Union, Iterable, Tuple

import numpy as np
from collections import OrderedDict

from qcompute_qep.exceptions.QEPError import ArgumentError

try:
    from matplotlib import pyplot as plt
    import pylab
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas
    import seaborn
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def extract_substr(target_str: str, indices: Iterable) -> Tuple[str, str]:
    """
    Extract a sub string from a given target string. Namely, divide the target string into two parts:
    the sub-string extracted according to @indices and the remaining substring.
    Notice that the target string is given in MSB order. That is, ::

         target_str:   "1       1        0        1"
            indices:    3       2        1        0

    **Example**

        >>> target_str = '1011001'
        >>> indices = [0, 1, 2]
        >>> sub_str = extract_substr(target_str, indices)
        >>> print('sub_str: ', sub_str)
        sub_str:  ('001', '1011')

    :param target_str: str, the target string which will be extracted
    :param indices: Iterable, indices of the extracted bits.
    :return: Tuple[str, str], a tuple of the extracted string and the remaining string.
            Notice that the extracted substring is sorted based on the indices in increasing order.
    """
    if (indices is None) or (len(indices) == 0):
        raise ArgumentError("in extract_substr(): the extracting indices of the target string is not given!")

    k = len(indices)
    n = len(target_str)
    if k > n:
        raise ArgumentError("in extract_substr(): the extracted substring must be no longer than the target string!")

    if (max(indices) >= n) or (min(indices) < 0):
        raise ArgumentError("in extract_substr(): the extracting indices exceed the range!")

    str_e: str = ''
    str_r = ''
    target_str = target_str[::-1]  # reverse the target string order
    for i in range(n):
        if i in indices:
            str_e += target_str[i]
        else:
            str_r += target_str[i]

    return str_e[::-1], str_r[::-1]


def init_cal_data(n: int, layer: int, init_value: int) -> Union[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Initialize dictionary of calibration data.

    :param n: int, the size of qubits list whose calibration data is loaded, composed of integers.
    :param layer: int, the layer of the dictionary.
    :param init_value: int, the value of the inner dictionary.
    :return: Union[Dict[str, int], Dict[str, Dict[str, int]]],
            initialized dictionary of the calibration data corresponding to @qubits.

    **Example**

        >>> two_layer = init_cal_data(n=2, layer=2, init_value=0)
        >>> one_layer = init_cal_data(n=2, layer=1, init_value=0)
        >>>
        >>> print(two_layer)
        {'00': {'00': 0, '01': 0, '10': 0, '11': 0},
        '01': {'00': 0, '01': 0, '10': 0, '11': 0},
        '10': {'00': 0, '01': 0, '10': 0, '11': 0},
        '11': {'00': 0, '01': 0, '10': 0, '11': 0}}
        >>> print(one_layer)
        {'00': 0, '01': 0, '10': 0, '11': 0}
    """
    a1 = []
    b1 = {}
    c1 = {}
    if (layer != 1) and (layer != 2):
        raise ArgumentError("layer should be 1 or 2")
    if (init_value != 0) and (init_value != 1):
        raise ArgumentError("init_value should be 0 or 1")

    for i in range(2 ** n):
        a1.append(bin(i).split('b')[1].zfill(n))
        b1[a1[i]] = init_value

    if layer == 1:
        return b1
    else:
        for i in range(2 ** n):
            c1[a1[i]] = copy.deepcopy(b1)
        return c1


def vector2dict(origin_v: np.ndarray) -> dict:
    """
    Transform a vector into a dictionary.

    :param origin_v: np.ndarray, array that needs to be converted into a dictionary, v for vector.
    :return: dictionary that corresponds to the vector.

    **Example**

        >>> example_vec = np.array([1, 2, 3, 4])
        >>> print(vector2dict(example_vec))
        {'00': 1, '01': 2, '10': 3, '11': 4}
    """
    # Origin_v has been completed when use dic2vector().
    nqubits = int(math.log2(origin_v.size))
    new_dict = init_cal_data(n=nqubits, layer=1, init_value=0)

    for k, key in enumerate(new_dict.keys()):
        new_dict[key] = origin_v[k]

    return new_dict


def dict2vector(origin_d: dict) -> np.ndarray:
    """
    Transform a dictionary into a vector.

    :param origin_d: dict, dictionary that needs to be converted into a vector, d for dictionary.
    :return: vector that corresponds to the dictionary.

    **Example**

        >>> example_dict = {'00': 0, '01': 0, '10': 0, '11': 0}
        >>> print(dict2vector(example_dict))
        [0. 0. 0. 0.]
    """
    nqubits = len(next(iter(origin_d)))
    raw_data = np.zeros(2 ** nqubits)

    #  We use a new_raw_data to guarantee a complete basis no matter what kind of input is.
    new_raw_data = init_cal_data(n=nqubits, layer=1, init_value=0)

    for key, value in origin_d.items():
        new_raw_data[key] = value

    for k, value in enumerate(new_raw_data.values()):
        raw_data[k] = value
    return raw_data


def preprocess_cal_data(cal_data: Union[str, Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    """
    Preprocess the file storing the calibration data. The following procedures will be carried out:

    1. Sort increasingly the calibration data according to the input states (converted to int value).
    2. Sort increasingly the calibration data according to the output states for each input state.

    :param cal_data: Optional[str, Dict[str, Dict[str, int]]], the name of the file storing the calibration data
    :return: Dict[str, Dict[str, int]], the processed calibration data
    """
    file_name = None
    # If the input is a file, load the calibration data from the file
    if isinstance(cal_data, str):
        file_name = cal_data
        with open(file_name, 'r') as f:
            cal_data = json.load(f)

    # Sort increasingly the calibration data according to the input states (converted to int value)
    # Ref.: https://docs.python.org/2/library/collections.html?highlight=ordereddict#ordereddict-examples-and-recipes
    cal_data = dict(OrderedDict(sorted(cal_data.items(), key=lambda t: int(t[0], 2))))

    # Sort increasingly the calibration data according to the output states for each input state
    for in_state, out_info in cal_data.items():
        cal_data[in_state] = dict(OrderedDict(sorted(out_info.items(), key=lambda t: int(t[0], 2))))

    # If the input is the file, write the processed calibration data back
    if file_name is not None:
        with open(file_name, 'w') as fp:
            json.dump(cal_data, fp, indent=4)

    return cal_data


def check_cal_data(cal_data: Dict[str, Dict[str, int]]) -> bool:
    """
    Check the structural correctness of the calibration data.

    :param cal_data: Dict[str, Dict[str, int]], a dictionary of the calibration data
    :return: bool, if the calibration data satisfies the structural constraints, return True; otherwise, return False.
    """
    if not isinstance(cal_data, Dict):
        print("The calibration data under checking is not a dictionary!")
        return False

    # Check Point 1: all keys must have the same length
    if len(set(map(len, [*cal_data]))) != 1:
        print("The input states in the calibration data do not have the same length!")
        return False

    # Check Point 2: all keys' corresponding dictionary must have the same size
    counts = [len(out_info) for (in_state, out_info) in cal_data.items()]
    if len(set(counts)) != 1:
        print("The number of elements in the dictionary of each input state are not the same!")
        return False

    # Check Point 3: the length of the key and the keys in its corresponding dictionary must have the same size
    for in_state, out_info in cal_data.items():
        out_lens = [len(out_state) for (out_state, out_count) in out_info]
        if any([out_len != len(in_state)] for out_len in out_lens):
            print("The length of the key and the keys in its corresponding dictionary must have the same size!")
        return False

    # Check Point 4: cannot have any negative counts
    for in_state, out_info in cal_data.items():
        out_negatives = [out_count < 0 for (out_state, out_count) in out_info]
        if any(out_negatives):
            print("The output state counts cannot be negative!")
        return False

    return True


def state_labels(n: int) -> List[str]:
    """
    Create the full list of classical labels when measuring the n-qubit quantum state in the computational basis.

    References: https://www.py4u.net/discuss/199398

    :param n: int, the number of qubits
    :return: a list of labels

    **Example**

        >>> a = state_labels(3)
        >>> print(a)
        ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    labels = list()
    for i in range(2 ** n):
        s = str(bin(i))[2:]
        labels.append(s.zfill(n))
    return labels


def plot_cal_matrix(A: np.ndarray,
                    show_labels: bool = False,
                    title: str = None,
                    fig_name: str = None) -> None:
    r"""
    Visualize the calibration matrix.

    Reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    :param A: np.ndarray, a :math:`2^n \times 2^n` column-stochastic matrix.
    :param show_labels: bool, indicator for adding labels to the x and y axes or not.
            Notice that if A is very large (more than 5 qubits), then it is meaningless to add the labels.
    :param title: str, the file name for saving
    :param fig_name: str, the file name for saving
    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Function "plot_cal_matrix" requires matplotlib. Please run "pip install matplotlib" first.')

    # Compute the number of qubits
    n = int(np.log2(A.shape[0]))
    # Create the label list
    labels = state_labels(n)

    fig, ax = plt.subplots(figsize=(10, 8))

    min_value = np.amin(A)
    # For [0, 1], use RdPu, OrRd
    # For [-1, 1], use RdBu, cool, Reds, GnBu
    if min_value < 0:
        im = ax.imshow(A, cmap='RdBu')
    else:
        im = ax.imshow(A, cmap='RdPu')

    # Add the colorbar
    fig.colorbar(im, ax=ax)

    if show_labels:
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    if title is not None:
        ax.set_title(title)
    if fig_name is not None:
        plt.savefig(fig_name, dpi=400)

    plt.show()


def plot_histograms(counts: Union[np.ndarray, List[List[float]]],
                    legends: List[str] = None,
                    binary_labels: bool = True,
                    **kwargs) -> None:
    r"""
    Plot the histograms of a set of counts.
    Assume there are :math:`K` sets of count instances and each instance has :math:`N` elements, then
    + `counts` is a np.ndarray matrix of size :math:`K\times N`, where each row represents a count dict, and
    + `legends` is a list of strings of length math:`K`, where `legends[i]` is the legend label for `counts[i,:]`

    If the @counts is a list of counts, convert it to the above array type.
    In this case, each inner `List[float]` has :math:`N` elements and `len(counts) = K`.

    Reference: https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/

    :param counts: Union[np.ndarray, List[List[float]]], a :math:`K \times N` matrix
    :param legends: List[str], a list of strings of length math:`K`
    :param binary_labels: bool, indicator for adding binary labels to the x axis or not.
            Notice that if counts is very large (more than 5 qubits), then it is meaningless to add the labels.
    :param binary_labels:
    """
    if not HAS_SEABORN:
        raise ImportError('Function "plot_histograms" requires pandas and seaborn. '
                          'Please run "pip install pandas, seaborn" first.')

    labels = kwargs.get('labels', None)
    title = kwargs.get('title', None)
    fig_name = kwargs.get('fig_name', None)

    # If the input @counts is a list of counts, convert it to the array type
    if isinstance(counts, list):
        counts = np.array(counts)
    # Number of counts
    K = counts.shape[0]
    # Number of elements in each count
    N = counts.shape[1]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Set x axis labels
    if binary_labels:  # Compute the number of qubits and create the binary label list (ignore the @labels input)
        n = int(np.log2(N))
        labels = state_labels(n)
    if labels is None:
        labels = [str(i) for i in range(N)]

    # Set the legends
    if (legends is None) or (len(legends) != K):
        legends = [str(k) for k in range(K)]

    multi_legends = []
    for l in legends:
        multi_legends.extend([l]*N)
    df = pandas.DataFrame(zip(labels*K, multi_legends, counts.flatten()),
                          columns=["Output Sequences", "Category", "Counts"])

    # https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
    # http://man.hubwiz.com/docset/Seaborn.docset/Contents/Resources/Documents/tutorial/color_palettes.html
    # `color_palette` candidates: Set2, Paired, hls, husl
    seaborn.barplot(x="Output Sequences", hue="Category", y="Counts", data=df,
                    palette=seaborn.color_palette("husl"))
    # Rotate the tick labels and set their alignment
    if binary_labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        plt.setp(ax.get_xticklabels())
    # Put the legend out of the figure
    # https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    # Add figure title
    if title is not None:
        ax.set_title(title)

    # Save figure
    if fig_name is not None:
        plt.savefig(fig_name, dpi=400)

    plt.show()
