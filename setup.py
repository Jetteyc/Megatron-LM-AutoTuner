# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "VERSION")) as f:
    __version__ = f.read().strip()

install_requires = [
]

extras_require = {
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="Megatron-LM-AutoTuner",
    version=__version__,
    packages=find_packages(include=["AutoTuner", "AutoTuner.*"]),
    url="https://github.com/ETOgaosion/Megatron-LM-AutoTuner",
    license="MIT License",
    author="ACS Frontier System Research Lab, ICT CAS",
    author_email="gaoziyuan19@mails.ucas.ac.cn",
    description="Megatron-LM-AutoTuner: An Automated Performance Tuning Framework for LLM PreTraining and PostTraining",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={"AutoTuner": ["testbench/profile/configs/*.json"]},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
