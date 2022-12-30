conda create -n qsvt_env python=3.9
conda activate qsvt_env
cd ..
setup_build_whl.bat
cd dist
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple qcompute_qsvt-0.1.0-py3-none-any.whl
cd ..
pip install sphinx
pip install sphinx_rtd_theme
pip install sphinx_markdown_builder
cd docs
python DocGen.py