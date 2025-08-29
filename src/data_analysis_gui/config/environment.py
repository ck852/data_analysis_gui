import os
import sys

class Environment:
    @staticmethod
    def is_ci():
        return any([
            os.environ.get('CI'),
            os.environ.get('GITHUB_ACTIONS'),
            os.environ.get('JENKINS_HOME'),
            not os.environ.get('DISPLAY') and sys.platform != 'win32'
        ])
    
    @staticmethod
    def is_headless():
        return Environment.is_ci() or not os.environ.get('DISPLAY')