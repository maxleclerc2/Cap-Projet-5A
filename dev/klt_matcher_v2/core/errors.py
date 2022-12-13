class ConfigurationError(Exception):
    """Base class for Configuration errors."""

    def __init__(self, message):
        super(ConfigurationError, self).__init__(message)
