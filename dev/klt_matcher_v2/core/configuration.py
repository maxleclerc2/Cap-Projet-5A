"""
Represents the configuration of the application.

Contains inputs, outputs and processings parameters.
"""

import csv
import glob
import json
import logging
import os
import sys
import re

package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(package_dir, 'core'))
from errors import ConfigurationError

LOGGER = logging.getLogger()

class Configuration:
    """Application configuration."""

    def __init__(self, arguments):
        """
        Initialize class and check for configuration files existence.

        Load the configuration as a dictionary if it exists.
            :param self: Instance of the class
            :param arguments: Arguments parsed from the command line
            :param output_path: Where to store outputs
        """
        self.values = {
            "output_directory": {"path": arguments.out},
            "working_image": {"path": arguments.mon},
            "reference_image": {"path": arguments.ref},
            "mask": {"path": arguments.mask},
            "configuration": {"path": arguments.conf}
        }

        # Read configuration file :
        file_content = \
            self.check_configuration_file(arguments.conf)

        self.load_configuration(file_content)

        #TODO: Eventualy add these functions if required
        #self.save_configuration()
        #self.configure_readers()


    def load_configuration(self,file_content):
        """
        # Retrieve parameters as dic
        """
        self.klt_configuration = file_content['processing_configuration']['klt_matching']
        self.accuracy_analysis = file_content['processing_configuration']['accuracy_analysis']
        self.plot_configuration= file_content['processing_configuration']['plot_configuration']
        self.output = file_content['processing_configuration']['outputs']

    def check_configuration_file(self, filepath):
        """
        Check that the provided configuration file exists and is valid.
        And load configuration (json)

            :param self:
            :param filepath: The path of the configuration file.
        """
        if os.path.exists(filepath):
            LOGGER.info("** Checking %s", filepath)
            try:
                with open(filepath) as json_file:
                    file_content = json.load(json_file)
            except json.JSONDecodeError as error:
                raise ConfigurationError(f"{filepath} is not a valid configuration file: {error}")
        else:
            LOGGER.error("%s does not exist.", filepath)
            raise ConfigurationError(f"{filepath} does not exist.")
        return file_content

        #self.save_configuration()
        #self.configure_readers()
