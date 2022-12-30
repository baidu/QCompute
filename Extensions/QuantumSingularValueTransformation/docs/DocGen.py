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
This scripts automatically generates the API Documentation configuration files.
Switch to `docs/` and run `python doc-generator.py` in the terminal. It does the following steps:

Step 1. Automatically scan the `qcompute_qsvt` module and generate `autogen.rst`.

Step 2. Automatically run `sphinx-autogen autogen.rst` to generate the rst configuration files in `docs/`.

Step 3. Automatically update the TOC in `index.rst`, according to the rst configuration files.

Step 4. Automatically build the documentation by running `sphinx-build -b html ./ _build `

Now you can find the API htmls in `_build`! This is really nice!
More details on this script can be found in: https://ku.baidu-int.com/d/y7L8cMRHvzv0gg.
"""

import os
from collections import defaultdict


def check_the_project(proj) -> defaultdict:
    r"""读取 qcompute_qsvt 目录下的每个文件，不包含 tests."""
    # defaultdict 是一个类似字典的数据结构，但是他的值可以修改为 list 形式
    project_files = defaultdict(list)

    def scan(path, exclude_files_or_paths=None):
        if exclude_files_or_paths is None:
            exclude_files_or_paths = []
        if os.path.isfile(path):
            path_list = path.split('/')
            # 去除文件名后缀
            path_list[-1] = path_list[-1].split('.')[0]
            # 恢复文件地址
            key = path.split('/')[1] + '.' + path.split('/')[2]
            new_path = '.'.join(path_list[1:])
            # 正常的文件扫描出来以后储存起来
            project_files[key].append(new_path)
            return
        for file in os.listdir(path):
            if file not in exclude_files_or_paths:
                scan(path + '/' + file, exclude_files_or_paths=exclude_files_or_paths)

    # 执行路径 path 可以根据文件位置调整
    # 不包含每个目录下存在着的一些隐藏目录 .DS_Store 和 __pycache__ 以及 __init__.py 文件。
    scan(proj, exclude_files_or_paths=['__init__.py', '__pycache__', '.DS_Store', 'arm', 'x64',
                                       'SymmetricQSPInternalCppSrc', 'SymmetricQSPInternalCpp.py'])
    return project_files


def autogen_file(file_name):
    r"""生成 autogen.rst 文件."""

    files = check_the_project('../qcompute_qsvt')
    with open(file_name, 'w+') as f:
        # autogen.rst 文件的标题
        f.write('Auto generate the API rst files \n')
        f.write('=============================== \n')
        f.write('\n')

        for k, files_arr in files.items():
            f.write('\n')
            f.write('.. autosummary:: \n')
            f.write('   :toctree: {} \n'.format(k))
            f.write('\n')
            for file in sorted(files_arr):
                f.write('   {}\n'.format(file))


def check_docs_files(docs) -> defaultdict:
    r"""Check files in docs."""

    docs_files = defaultdict(list)

    # index.rst 中不包含的目录以及文件。
    exclude_files_or_paths = ['_build', '_static', '_templates', '__pycache__', '.DS_Store',
                              'autogen.rst', 'conf.py', 'DocGen.py', 'index.rst']

    def scan(path):
        if os.path.isfile(path):
            # 正常的文件扫描出来以后储存起来
            key = path.split('/')[2]
            new_path = '.'.join(path.split('/')[3:])
            docs_files[key].append(new_path[:-4])
            return
        for file in os.listdir(path):
            if file not in exclude_files_or_paths:
                scan(path + '/' + file)

    # 执行路径 path 可以根据文件位置调整
    scan(docs)
    return docs_files


def index_file(file_name):
    r"""生成 index.rst 文件."""
    # 读取 docs 下的 .rst 文件，以此更新 index.rst
    files = check_docs_files('../docs')

    # 生成新的 toctree 内容
    def rewrite():
        data = ''
        for k, files_arr in sorted(files.items()):
            data += '\n'
            data += '.. toctree::\n'
            data += '   :maxdepth: 1\n'
            data += '   :glob:\n'
            data += '   :caption: {}\n'.format(k)
            data += '\n'
            for file in sorted(files_arr):
                data += '   {}/{}\n'.format(k, file)
        return data

    with open(file_name, 'r') as f:
        read_all = f.read()
        f.close()

    with open(file_name, 'w') as f:
        # 从第一个 toctree 开始更新
        start = read_all.find('.. toctree::')
        # 得到新的 toctree 列表
        rewrite_data = rewrite()
        # 将已有的内容部分与新的 toctree 列表拼接
        # read_tail = read_all[-91:]
        read_all = read_all[:start - 1] + rewrite_data + '\n' + read_all[-73:]
        # 将新内容写入 index.rst
        f.write(read_all)


def main():
    # 每次更新了 qcompute_qsvt 目录下的文件，就重新扫描 qcompute_qsvt 目录，重写覆盖 autogen.rst
    # autogen_file('autogen.rst')
    # 生成所有对应的 .rst 文件
    # os.system('sphinx-autogen autogen.rst')
    # 根据新生成的 .rst 文件更新 index.rst 文件
    # index_file('index.rst')
    # 生成网页
    # -Q option: Do not output anything on standard output, also suppress warnings. Only errors are shown.
    # https://www.sphinx-doc.org/en/master/man/sphinx-build.html
    os.system('sphinx-build -Q -b html ./ _build ')


if __name__ == '__main__':
    main()
