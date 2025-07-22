# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
import logging
import os
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from configparser import ConfigParser
from pathlib import Path

from .converter import AntaresStudyConverter
from .logger import Logger

DEFAULT: dict = {}
LOGGER_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data/logging.log"
)


class CreateFileIfMissing:
    """
    Argparse type that accepts a file path and creates the file if it doesn't exist.
    Fails if the path points to an existing directory or if the parent directory is missing.
    """

    def __call__(self, string: str) -> Path:
        path = Path(string).expanduser().resolve()

        if path.exists() and not path.is_file():
            raise ArgumentTypeError(f"Path exists and is not a file: '{path}'")

        if not path.parent.exists():
            raise ArgumentTypeError(f"Parent directory does not exist: '{path.parent}'")

        if not path.exists():
            try:
                path.touch()
            except Exception as e:
                raise ArgumentTypeError(f"Failed to create file: {e}")

        return path


class PathType:
    def __init__(
        self,
        exists: bool = False,
        file_ok: bool = False,
        dir_ok: bool = False,
    ) -> None:
        if not (file_ok or dir_ok):
            msg = "Either `file_ok` or `dir_ok` must be set at a minimum."
            raise ValueError(msg)
        self.exists = exists
        self.file_ok = file_ok
        self.dir_ok = dir_ok

    def __call__(self, string: str) -> Path:
        file_path = Path(string).expanduser()
        if not self.exists:
            return file_path
        if self.file_ok and self.dir_ok:
            if file_path.exists():
                return file_path
            msg = f"The file or directory path does not exist: '{file_path}'"
            raise ArgumentTypeError(msg)
        elif self.file_ok:
            if file_path.is_file():
                return file_path
            elif file_path.exists():
                msg = f"The path is not a regular file: '{file_path}'"
            else:
                msg = f"The file path does not exist: '{file_path}'"
            raise ArgumentTypeError(msg)
        elif self.dir_ok:
            if file_path.is_dir():
                return file_path
            elif file_path.exists():
                msg = f"The path is not a directory: '{file_path}'"
            else:
                msg = f"The directory path does not exist: '{file_path}'"
            raise ArgumentTypeError(msg)
        else:  # pragma: no cover
            raise NotImplementedError((self.file_ok, self.dir_ok))


def parse_commandline() -> Namespace:
    """Parse command-line arguments using argparse to specify configuration file paths,
    logging options, and version display. Returns the parsed arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        type=PathType(exists=True, file_ok=True),
        help="Give the path of the configuration file",
        default="data/config.ini",
    )
    parser.add_argument(
        "-l",
        "--logging",
        type=CreateFileIfMissing(),
        help="Give the path of the logger file",
        default=LOGGER_PATH,
    )
    parser.add_argument(
        "-i",
        "--study_path",
        type=PathType(exists=True, dir_ok=True),
        help="Give the path of the study_path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=PathType(exists=True, dir_ok=True),
        help="Give the path of the output path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    config: dict = {}
    args = parse_commandline()
    log_path = args.logging
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger: logging.Logger = Logger(__name__, log_path)

    config_parser = ConfigParser()

    # Load the default configuration dictionary into the config parser.
    config_parser.read_dict(DEFAULT)

    if args.conf:
        # Check if the specified config file exists, if not, exit with an error message.
        if not os.path.exists(args.conf):
            sys.exit(f"Aborting: missing config file at {args.conf}")
        else:
            config_parser.read(args.conf)

    if not config_parser.has_section("study"):
        config_parser.add_section("study")

    if args.study_path:
        config_parser.set("study", "study_path", str(args.study_path))
    if args.output_path:
        config_parser.set("study", "output_path", str(args.output_path))

    converter = AntaresStudyConverter(
        study_input=Path(config_parser["study"].get("study_path")), logger=logger
    )
    converter.process_all()
